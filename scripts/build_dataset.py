"""
Step 2 — Dataset Builder
========================
Reads raw telemetry CSV produced by make_synthetic_data.py and produces a
feature-engineered Parquet file ready for model training.

Pipeline stages (in order):
  1. Load raw CSV and parse timestamps.
  2. Label anomalies with the Tukey-IQR rule per device (site + link group).
  3. Add cyclical time features (hour sin/cos, day-of-week sin/cos).
  4. Compute rolling mean/std/max and lag/delta features per metric.
  5. Inject label noise (anomaly→normal flips) to simulate imperfect labeling.
  6. Write the final dataset as Parquet.

All parameters (windows, noise fraction, grouping columns) come from
config/config.json so the pipeline is fully config-driven.

Usage:
    python scripts/build_dataset.py \
        --in data/raw/telemetry.csv \
        --out data/processed/dataset.parquet
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Allow imports from src/ without installing the package
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from wan_anomaly.utils.io import read_table, write_table
from wan_anomaly.processing.label import label_anomalies_tukey
from wan_anomaly.processing.features import add_time_features, rolling_features


def main():
    # --- Argument parsing ---------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)        # input CSV path
    ap.add_argument("--out", required=True)                   # output Parquet path
    ap.add_argument("--config", default="config/config.json") # pipeline config
    args = ap.parse_args()

    # Load pipeline configuration (metrics, windows, noise fraction, etc.)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # --- Stage 1: Load raw data ---------------------------------------------
    df = read_table(args.inp)
    # Parse timestamps with UTC timezone awareness; drop any unparseable rows
    df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg["time_col"]])

    # --- Stage 2: Anomaly labeling (Tukey-IQR) ------------------------------
    # Label each row as 0 (normal) or 1 (anomalous) based on whether any
    # metric exceeds the per-device Tukey IQR upper fence.
    # label anomalies (Tukey-IQR)
    df = label_anomalies_tukey(
        df,
        metric_cols=cfg["metrics"],

        group_cols=cfg["group_cols"],
        k=cfg["labeling"]["k"],
        min_breach_metrics=cfg["labeling"]["min_breach_metrics"],
    )

    # --- Stage 3 & 4: Feature engineering -----------------------------------
    # Cyclical time features capture hour-of-day and day-of-week patterns
    # features
    df = add_time_features(df, cfg["time_col"])
    # Rolling statistics (mean, std, max) and lag/delta features per metric
    df = rolling_features(
        df,
        metric_cols=cfg["metrics"],
        group_cols=cfg["group_cols"],
        time_col=cfg["time_col"],
        windows_min=cfg["feature_windows_min"],
        freq_min=cfg["freq_min"],
    )

    # --- Stage 5: Label noise injection -------------------------------------
    # Label noise: flip a fraction of *anomaly* labels to normal to simulate
    # imperfect labeling (missed detections). This decouples the label from
    # raw feature values so models must learn temporal patterns.
    # We only flip anomaly→normal (not normal→anomaly) to keep the class
    # ratio realistic.
    label_noise = cfg["labeling"].get("label_noise_frac", 0.0)
    if label_noise > 0:
        # Use the same seeded RNG as all other stages for reproducibility
        rng = np.random.default_rng(cfg.get("random_state", 42))
        anomaly_idx = df.index[df["anomaly"] == 1]
        n_flip = int(len(anomaly_idx) * label_noise)
        # Randomly sample indices to flip; replace=False avoids double-flipping
        flip_idx = rng.choice(anomaly_idx, size=n_flip, replace=False)
        df.loc[flip_idx, "anomaly"] = 0

    # --- Stage 6: Write output ----------------------------------------------
    write_table(df, args.out)
    print(f"Wrote dataset -> {args.out} (rows={len(df):,}, anomaly_rate={df['anomaly'].mean():.3f})")

if __name__ == "__main__":
    main()
