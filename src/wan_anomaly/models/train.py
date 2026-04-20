"""
Classical ML Baseline Models
==============================
Trains and evaluates three scikit-learn classifiers on the WAN anomaly dataset:
  1. Logistic Regression  — linear baseline; wrapped in Pipeline with StandardScaler.
  2. Random Forest        — tree ensemble; handles feature interactions natively.
  3. SVM RBF              — kernel method; wrapped in Pipeline with StandardScaler.

All three models use class-weight balancing to handle the ~16% anomaly class
minority, which avoids the model collapsing to always predicting 'normal'.

The primary evaluation metric is PR-AUC (average_precision_score), which is
more informative than accuracy or ROC-AUC for imbalanced binary classification:
a high PR-AUC requires good precision AND recall simultaneously across all
decision thresholds.

Each trained model is serialized to a .joblib file for later inference and
SHAP explanation (Step 4).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ModelResult:
    """Container for per-model evaluation metrics returned by train_all."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    pred_anomaly_rate: float


def _metrics(y_true, y_prob, y_pred):
    """Compute all evaluation metrics for a single model.

    Parameters
    ----------
    y_true : array-like of int — ground truth labels (0/1).
    y_prob : array-like of float — predicted anomaly probabilities.
    y_pred : array-like of int — hard predictions at threshold 0.5.

    Returns
    -------
    acc, precision, recall, f1, roc_auc, pr_auc : floats
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # Guard against degenerate test sets with only one class
    roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return acc, p, r, f1, roc, pr


def train_all(X_train, y_train, X_test, y_test, out_dir: str, random_state: int = 42):
    """
    Train classical ML baseline models and save them.

    Models are trained in sequence; each is fitted on X_train / y_train,
    evaluated on X_test / y_test, and saved to out_dir as <name>.joblib.

    Parameters
    ----------
    X_train, y_train : training feature matrix and labels.
    X_test, y_test   : held-out test feature matrix and labels.
    out_dir          : directory path where .joblib files are written.
    random_state     : integer seed passed to all stochastic components.

    Returns
    -------
    metrics_df : pd.DataFrame
    rates_df : pd.DataFrame
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models = {}

    # --- Logistic Regression -------------------------------------------------
    # Wrapped in a Pipeline so StandardScaler is fit only on training data.
    # with_mean=False preserves sparsity if the feature matrix is sparse.
    # class_weight='balanced' upweights the minority (anomaly) class.
    models["logreg"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        )),
    ])

    # --- Random Forest -------------------------------------------------------
    # Does not need explicit scaling (tree splits are scale-invariant).
    # balanced_subsample recomputes class weights per bootstrap sample,
    # which is more robust than global 'balanced' for deep forests.
    # n_jobs=-1 uses all available CPU cores for parallel tree building.
    models["rf"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    # --- SVM with RBF Kernel -------------------------------------------------
    # Requires scaled features; StandardScaler in the Pipeline handles this.
    # probability=True enables predict_proba() for PR-AUC computation;
    # this uses Platt scaling internally and adds a small training overhead.
    models["svm_rbf"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        )),
    ])

    results = []        # collects ModelResult dataclass instances
    anomaly_rates = []  # collects pred vs true anomaly rate dicts

    for name, model in models.items():
        model.fit(X_train, y_train)
        # predict_proba returns [P(normal), P(anomaly)]; take column 1
        y_prob = model.predict_proba(X_test)[:, 1]
        # Hard threshold at 0.5 for accuracy / precision / recall / F1
        y_pred = (y_prob >= 0.5).astype(int)

        acc, p, r, f1, roc, pr = _metrics(y_test, y_prob, y_pred)

        results.append(ModelResult(
            name=name,
            accuracy=acc,
            precision=p,
            recall=r,
            f1=f1,
            roc_auc=roc,
            pr_auc=pr,
            pred_anomaly_rate=float(y_pred.mean()),
        ))

        anomaly_rates.append({
            "model": name,
            "pred_anomaly_rate": float(y_pred.mean()),
            "true_anomaly_rate": float(y_test.mean()),
        })

        # Serialize the fitted model for later use (SHAP, inference)
        dump(model, out_path / f"{name}.joblib")

    # Sort by PR-AUC descending so the best model appears at the top of the CSV
    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        "pr_auc", ascending=False
    )
    rates_df = pd.DataFrame(anomaly_rates)

    return metrics_df, rates_df
