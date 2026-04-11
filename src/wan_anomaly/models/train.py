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
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    pred_anomaly_rate: float


def _metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    return acc, p, r, f1, roc, pr


def train_all(X_train, y_train, X_test, y_test, out_dir: str, random_state: int = 42):
    """
    Train classical ML baseline models and save them.

    Returns
    -------
    metrics_df : pd.DataFrame
    rates_df : pd.DataFrame
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models = {}

    models["logreg"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        )),
    ])

    models["rf"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    models["svm_rbf"] = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        )),
    ])

    results = []
    anomaly_rates = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
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

        dump(model, out_path / f"{name}.joblib")

    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        "pr_auc", ascending=False
    )
    rates_df = pd.DataFrame(anomaly_rates)

    return metrics_df, rates_df