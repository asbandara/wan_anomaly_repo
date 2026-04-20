"""
PyTorch MLP Classifier (added in Checkpoint 2)
================================================
Implements a two-hidden-layer Multi-Layer Perceptron for binary anomaly
detection on tabular WAN telemetry features.

Architecture:
    Input (37 features) → Linear(64) → ReLU → Dropout(0.20)
                        → Linear(32) → ReLU → Dropout(0.20)
                        → Linear(1)  → (BCEWithLogitsLoss during training)
                                      → sigmoid → probability at inference

Key design decisions:
  - BCEWithLogitsLoss with pos_weight: addresses ~16% class imbalance by
    upweighting the minority (anomaly) class during loss computation.
    pos_weight = neg_count / pos_count ≈ 5.25 for a 16% anomaly rate.
  - StandardScaler applied before training: neural networks are sensitive to
    feature scale; tree models are not, which is why sklearn models skip this.
  - Best-epoch checkpointing: saves the model state with the lowest average
    training loss across epochs to avoid using an overfit final epoch.
  - CPU training: no GPU required; 25 epochs on 56k samples runs in ~30 sec.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class MLPResult:
    """Container for MLP evaluation metrics (mirrors ModelResult in train.py)."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    pred_anomaly_rate: float


class MLPNet(nn.Module):
    """Two-hidden-layer MLP for binary classification.

    The output is a single raw logit (not a probability). The caller applies
    sigmoid at inference time; during training BCEWithLogitsLoss combines
    sigmoid + BCE in a numerically stable way.

    Parameters
    ----------
    input_dim : int — number of input features (37 for this pipeline).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # first hidden layer: 64 units
            nn.ReLU(),
            nn.Dropout(0.20),          # dropout regularizes to prevent overfitting
            nn.Linear(64, 32),         # second hidden layer: 32 units
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),          # output: single logit for binary classification
        )

    def forward(self, x):
        # Pass input through the sequential stack; returns raw logit
        return self.net(x)


def _compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute evaluation metrics from predicted probabilities.

    Parameters
    ----------
    y_true    : np.ndarray of int — ground truth labels (0/1).
    y_prob    : np.ndarray of float — predicted anomaly probabilities in [0, 1].
    threshold : float — decision threshold (default 0.5).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
                    pred_anomaly_rate.
    """
    # Convert probabilities to hard predictions at the given threshold
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # Guard against degenerate test sets containing only one class
    roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    pr = (
        average_precision_score(y_true, y_prob)
        if len(set(y_true)) > 1
        else float("nan")
    )

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "pred_anomaly_rate": float(y_pred.mean()),
    }


def train_mlp_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    random_state: int = 42,
):
    """
    Train a PyTorch MLP on tabular WAN telemetry features.

    Parameters
    ----------
    X_train, y_train : training feature matrix and binary labels.
    X_test, y_test   : held-out test feature matrix and labels.
    epochs           : number of full passes through the training data.
    batch_size       : number of samples per gradient update step.
    lr               : Adam optimizer learning rate.
    random_state     : seed for torch and numpy random generators.

    Returns
    -------
    model : torch.nn.Module
        Best model (by training loss).
    metrics : dict
        Evaluation metrics on the test set.
    y_prob : np.ndarray
        Predicted anomaly probabilities for the test set.
    scaler : StandardScaler
        Fitted scaler used on the input features.
    """
    # Fix all random sources for reproducible training
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Use GPU if available; otherwise fall back to CPU (sufficient for this scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale features for neural network stability
    # Fit the scaler on training data only; transform both train and test
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train).astype(np.float32)
    X_test_np = scaler.transform(X_test).astype(np.float32)

    # Reshape labels to column vectors as required by BCEWithLogitsLoss
    y_train_np = y_train.astype(np.float32).values.reshape(-1, 1)
    y_test_np = y_test.astype(np.float32).values.reshape(-1, 1)

    # Wrap training data in a DataLoader for mini-batch iteration
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # shuffle each epoch to reduce gradient variance
    )

    model = MLPNet(input_dim=X_train_np.shape[1]).to(device)

    # Handle imbalance using positive class weighting
    # pos_weight upscales the loss contribution of anomaly samples by ~5.25×
    pos_count = float(y_train_np.sum())
    neg_count = float(len(y_train_np) - pos_count)
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track the best model state (lowest average training loss) across all epochs
    best_model = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()          # clear gradients from previous step
            logits = model(xb)             # forward pass: raw logits
            loss = criterion(logits, yb)   # compute weighted BCE loss
            loss.backward()                # backpropagation
            optimizer.step()               # update weights
            epoch_loss += loss.item()
            n_batches += 1

        # Save the model if this epoch has the lowest average loss so far
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model.state_dict())

    # Restore the best-epoch weights before running inference
    model.load_state_dict(best_model)
    model.eval()  # disable dropout for inference

    # Run inference on the test set without computing gradients
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        # Convert raw logits to probabilities via sigmoid
        y_prob = torch.sigmoid(logits).cpu().numpy().ravel()

    metrics = _compute_metrics(y_test.values, y_prob, threshold=0.5)

    return model, metrics, y_prob, scaler
