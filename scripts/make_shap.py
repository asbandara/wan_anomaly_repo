import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import load

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from wan_anomaly.utils.io import read_table
from wan_anomaly.processing.features import make_ml_table
from wan_anomaly.processing.split import time_split
from wan_anomaly.explain.shap_plots import (
    shap_summary_bar,
    shap_summary_beeswarm,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--max-samples", type=int, default=3000)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    # Load data
    df = read_table(args.data)
    df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], utc=True)

    # Train / test split
    train_idx, test_idx, _ = time_split(
        df, cfg["time_col"], train_frac=cfg["train_frac"]
    )

    drop_cols = cfg["group_cols"] + [cfg["time_col"]]
    X, y = make_ml_table(df, drop_cols=drop_cols, label_col="anomaly")

    # Build test matrix
    X_test = X.loc[test_idx].copy()
    if len(X_test) > args.max_samples:
        X_test = X_test.sample(args.max_samples, random_state=42)

    # Force numeric (SHAP requirement)
    X_test = (
        X_test.apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float64")
    )

    # Background for interventional SHAP
    background = X_test.sample(min(200, len(X_test)), random_state=42)

    # Load trained model
    model = load(args.model)

    # SHAP explainer (log-odds, anomaly class)
    explainer = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test)
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Output plots
    os.makedirs(args.outdir, exist_ok=True)
    shap_summary_bar(
        shap_values,
        X_test.columns.tolist(),
        f"{args.outdir}/shap_rf_bar.png",
    )
    shap_summary_beeswarm(
        shap_values,
        X_test,
        f"{args.outdir}/shap_rf_beeswarm.png",
    )

    print("SHAP plots generated in", args.outdir)

if __name__ == "__main__":
    main()