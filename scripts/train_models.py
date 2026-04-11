import os
import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="Output directory, e.g., artifacts")
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    models_dir = out_dir / "models"
    plots_dir = out_dir / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.data)
    df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg["time_col"]])

    train_idx, test_idx, cutoff = time_split(
        df,
        cfg["time_col"],
        train_frac=cfg["train_frac"],
    )

    drop_cols = cfg["group_cols"] + [cfg["time_col"]]
    X, y = make_ml_table(df, drop_cols=drop_cols, label_col="anomaly")

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # ---- Classical ML baselines ----
    metrics_df, rates_df = train_all(
        X_train,
        y_train,
        X_test,
        y_test,
        out_dir=str(models_dir),
        random_state=cfg["random_state"],
    )

    # ---- PyTorch MLP ----
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
    torch.save(
        {
            "model_state_dict": mlp_model.state_dict(),
            "input_dim": X_train.shape[1],
            "feature_names": list(X_train.columns),
        },
        models_dir / "mlp_torch.pt",
    )

    # Append MLP results to metrics table
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