from __future__ import annotations
import numpy as np
import pandas as pd

def tukey_iqr_flags(x: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean flags where x is an outlier by Tukey IQR rule."""
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (x < lo) | (x > hi)

def label_anomalies_tukey( df: pd.DataFrame, 
    metric_cols: list[str],
    group_cols: list[str],
    k: float = 1.5,
    min_breach_metrics: int = 1,
) -> pd.DataFrame:
    """Create an `anomaly` label using Tukey-IQR per metric within each group.

    A row is labeled anomalous if at least `min_breach_metrics` metrics are outliers.
    """
    df = df.copy()
    flags = []
    for m in metric_cols:
        f = df.groupby(group_cols, dropna=False)[m].transform(lambda s: tukey_iqr_flags(s, k=k)).astype(int)
        flags.append(f.rename(f"flag_{m}"))
    flags_df = pd.concat(flags, axis=1)
    df = pd.concat([df, flags_df], axis=1)
    df["anomaly"] = (flags_df.sum(axis=1) >= min_breach_metrics).astype(int)
    return df
