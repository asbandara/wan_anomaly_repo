import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from wan_anomaly.utils.io import read_table, write_table
from wan_anomaly.processing.label import label_anomalies_tukey
from wan_anomaly.processing.features import add_time_features, rolling_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    df = read_table(args.inp)
    # standardize time
    df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg["time_col"]])

    # label anomalies (Tukey-IQR)
    df = label_anomalies_tukey(
        df,
        metric_cols=cfg["metrics"],

        group_cols=cfg["group_cols"],
        k=cfg["labeling"]["k"],
        min_breach_metrics=cfg["labeling"]["min_breach_metrics"],
    )

    # features
    df = add_time_features(df, cfg["time_col"])
    df = rolling_features(
        df,
        metric_cols=cfg["metrics"],
        group_cols=cfg["group_cols"],
        time_col=cfg["time_col"],
        windows_min=cfg["feature_windows_min"],
        freq_min=cfg["freq_min"],
    )

    # Label noise: flip a fraction of *anomaly* labels to normal to simulate
    # imperfect labeling (missed detections). This decouples the label from
    # raw feature values so models must learn temporal patterns.
    # We only flip anomaly→normal (not normal→anomaly) to keep the class
    # ratio realistic.
    label_noise = cfg["labeling"].get("label_noise_frac", 0.0)
    if label_noise > 0:
        rng = np.random.default_rng(cfg.get("random_state", 42))
        anomaly_idx = df.index[df["anomaly"] == 1]
        n_flip = int(len(anomaly_idx) * label_noise)
        flip_idx = rng.choice(anomaly_idx, size=n_flip, replace=False)
        df.loc[flip_idx, "anomaly"] = 0

    write_table(df, args.out)
    print(f"Wrote dataset -> {args.out} (rows={len(df):,}, anomaly_rate={df['anomaly'].mean():.3f})")
if __name__ == "__main__":
    main()
