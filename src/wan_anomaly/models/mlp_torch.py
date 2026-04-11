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
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    pred_anomaly_rate: float


class MLPNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def _compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
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
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale features for neural network stability
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train).astype(np.float32)
    X_test_np = scaler.transform(X_test).astype(np.float32)

    y_train_np = y_train.astype(np.float32).values.reshape(-1, 1)
    y_test_np = y_test.astype(np.float32).values.reshape(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLPNet(input_dim=X_train_np.shape[1]).to(device)
    # Handle imbalance using positive class weighting
    pos_count = float(y_train_np.sum())
    neg_count = float(len(y_train_np) - pos_count)
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy().ravel()

    metrics = _compute_metrics(y_test.values, y_prob, threshold=0.5)

    return model, metrics, y_prob, scaler