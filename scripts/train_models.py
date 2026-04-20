"""
Step 3 — Model Training & Evaluation
=====================================
Trains all four classifiers on the feature-engineered dataset produced by
build_dataset.py and saves metrics, anomaly-rate comparisons, and plots.

Models trained:
  1. Logistic Regression  (sklearn Pipeline + StandardScaler)
  2. Random Forest        (sklearn, class_weight='balanced_subsample')
  3. SVM RBF              (sklearn Pipeline + StandardScaler)
  4. PyTorch MLP          (custom MLPNet, BCEWithLogitsLoss + pos_weight)

Primary evaluation metric: PR-AUC (area under precision-recall curve).
PR-AUC is preferred over ROC-AUC for imbalanced datasets because it is
not inflated by the large number of true negatives.

All results are written to the --out directory (default: artifacts/):
  metrics_table.csv    — per-model accuracy, F1, ROC-AUC, PR-AUC, etc.
  anomaly_rates.csv    — predicted vs true anomaly rate per model
  run_summary.json     — split cutoff, set sizes, true anomaly rate
  plots/               — bar charts for PR-AUC, accuracy, anomaly rates

Usage:
    python scripts/train_models.py \
        --data data/processed/dataset.parquet \
        --out artifacts
"""

import os
import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

# Allow imports from src/ without installing the package
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from wan_anomaly.utils.io import read_table
from wan_anomaly.processing.features import make_ml_table
from wan_anomaly.processing.split import time_split
from wan_anomaly.models.train import train_all
from wan_anomaly.models.mlp_torch import train_mlp_classifier
from wan_anomaly.evaluation.reports import (
    plot_anomaly_rates,
    plot_metrics_table,
)


def main():
    # --- Argument parsing ---------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)                          # processed Parquet
    ap.add_argument("--out", required=True, help="Output directory, e.g., artifacts")
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    models_dir = out_dir / "models"  # saved .joblib and .pt checkpoints
    plots_dir = out_dir / "plots"    # PNG figures

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and parse dataset ---------------------------------------------
    df = read_table(args.data)
    df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg["time_col"]])

    # --- Time-aware train / test split (70 / 30) ----------------------------
    # Rows are split strictly by time: all training samples are EARLIER than
    # all test samples, preventing any look-ahead data leakage.
    train_idx, test_idx, cutoff = time_split(
        df,
        cfg["time_col"],
        train_frac=cfg["train_frac"],
    )

    # --- Build feature matrix -----------------------------------------------
    # Drop non-feature columns (identifiers and timestamp); keep 'anomaly' as label
    drop_cols = cfg["group_cols"] + [cfg["time_col"]]
    X, y = make_ml_table(df, drop_cols=drop_cols, label_col="anomaly")

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # ---- Classical ML baselines ----
    # Trains LogReg, RF, and SVM; saves .joblib files; returns metric DataFrames
    metrics_df, rates_df = train_all(
        X_train,
        y_train,
        X_test,
        y_test,
        out_dir=str(models_dir),
        random_state=cfg["random_state"],
    )

    # ---- PyTorch MLP ----
    # Returns the best model (by training loss), evaluation metrics on test set,
    # predicted probabilities, and the fitted StandardScaler
    mlp_model, mlp_metrics, mlp_probs, mlp_scaler = train_mlp_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=25,
        batch_size=256,
        lr=1e-3,
        random_state=cfg["random_state"],
    )

    # Save PyTorch model checkpoint
    # Stores state dict + metadata needed to reconstruct the model for inference
    torch.save(
        {
            "model_state_dict": mlp_model.state_dict(),
            "input_dim": X_train.shape[1],
            "feature_names": list(X_train.columns),
        },
        models_dir / "mlp_torch.pt",
    )

    # Append MLP results to metrics table
    # Builds a dict matching the columns of the sklearn metrics_df
    mlp_row = {
        "name": "mlp_torch",
        "accuracy": mlp_metrics["accuracy"],
        "precision": mlp_metrics["precision"],
        "recall": mlp_metrics["recall"],
        "f1": mlp_metrics["f1"],
        "roc_auc": mlp_metrics["roc_auc"],
        "pr_auc": mlp_metrics["pr_auc"],
        "pred_anomaly_rate": mlp_metrics["pred_anomaly_rate"],
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mlp_row])], ignore_index=True)

    # Append MLP anomaly-rate summary
    rates_row = {
        "model": "mlp_torch",
        "pred_anomaly_rate": mlp_metrics["pred_anomaly_rate"],
        "true_anomaly_rate": float(y_test.mean()),
    }
    rates_df = pd.concat([rates_df, pd.DataFrame([rates_row])], ignore_index=True)

    # Sort and overwrite CSV outputs
    # Primary sort by PR-AUC descending so the best model appears first
    metrics_df = metrics_df.sort_values("pr_auc", ascending=False)
    rates_df = rates_df.sort_values("pred_anomaly_rate", ascending=False)

    metrics_df.to_csv(out_dir / "metrics_table.csv", index=False)
    rates_df.to_csv(out_dir / "anomaly_rates.csv", index=False)

    # Refresh comparison plots with MLP included
    plot_anomaly_rates(
        str(out_dir / "anomaly_rates.csv"),
        str(plots_dir / "anomaly_rates.png"),
    )
    plot_metrics_table(
        str(out_dir / "metrics_table.csv"),
        str(plots_dir / "pr_auc.png"),
        metric="pr_auc",
    )
    plot_metrics_table(
        str(out_dir / "metrics_table.csv"),
        str(plots_dir / "accuracy.png"),
        metric="accuracy",
    )

    # Write a compact summary of the experiment configuration and split sizes
    summary = {
        "cutoff": str(cutoff),
        "n_train": int(train_idx.sum()),
        "n_test": int(test_idx.sum()),
        "true_anomaly_rate_test": float(y_test.mean()),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Done. See artifacts/metrics_table.csv and artifacts/plots/")


if __name__ == "__main__":
    main()
