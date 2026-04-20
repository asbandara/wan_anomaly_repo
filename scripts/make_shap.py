"""
Step 4 — SHAP Feature Importance
==================================
Uses SHAP's TreeExplainer to explain the predictions of the trained Random Forest
model (or any compatible tree-based sklearn model).

Two plots are produced:
  shap_rf_bar.png       — mean |SHAP| bar chart: overall feature ranking
  shap_rf_beeswarm.png  — beeswarm plot: shows both importance and direction
                           of each feature's effect on the anomaly prediction

SHAP (SHapley Additive exPlanations) assigns each feature an importance score
for each individual prediction, grounded in cooperative game theory. Positive
SHAP values push the model toward predicting 'anomaly'; negative values push
toward 'normal'.

Usage:
    python scripts/make_shap.py \
        --data data/processed/dataset.parquet \
        --model artifacts/models/rf.joblib \
        --outdir artifacts/plots
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import load

# Allow imports from src/ without installing the package
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
    # --- Argument parsing ---------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)               # processed Parquet path
    ap.add_argument("--model", required=True)              # .joblib model path
    ap.add_argument("--outdir", required=True)             # output directory for plots
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--max-samples", type=int, default=3000)  # SHAP is O(n); cap for speed
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    # Load data
    df = read_table(args.data)
    df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], utc=True)

    # Train / test split
    # Re-create the exact same split used in training so SHAP runs only on unseen data
    train_idx, test_idx, _ = time_split(
        df, cfg["time_col"], train_frac=cfg["train_frac"]
    )

    # Build feature matrix (same preprocessing as training)
    drop_cols = cfg["group_cols"] + [cfg["time_col"]]
    X, y = make_ml_table(df, drop_cols=drop_cols, label_col="anomaly")

    # Build test matrix
    # Subsample if test set is large — SHAP computation is O(n * depth)
    X_test = X.loc[test_idx].copy()
    if len(X_test) > args.max_samples:
        X_test = X_test.sample(args.max_samples, random_state=42)

    # Force numeric (SHAP requirement)
    # SHAP's TreeExplainer requires a pure float64 array; coerce and fill any gaps
    X_test = (
        X_test.apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float64")
    )

    # Background for interventional SHAP
    # A small background set is used to estimate the expected feature value (baseline)
    background = X_test.sample(min(200, len(X_test)), random_state=42)

    # Load trained model (sklearn Pipeline wrapping the Random Forest)
    model = load(args.model)

    # SHAP explainer (log-odds, anomaly class)
    # tree_path_dependent perturbation avoids re-running the tree for each feature;
    # model_output="raw" returns log-odds so positive = predicted anomaly
    explainer = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test)
    # For binary classifiers, shap_values may be a list [class0, class1]; take class 1 (anomaly)
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
