"""
Anomaly Labeling — Tukey-IQR Rule
===================================
Implements the statistical labeling strategy from Schummer et al. (2024).

Each metric is evaluated independently per device group (site_id + link_id).
A row is labeled anomalous (anomaly=1) when at least `min_breach_metrics`
metrics exceed their per-device Tukey IQR upper fence:

    upper_fence = Q3 + k * IQR    (default k=1.5)

Grouping by device is critical: normal operating ranges differ significantly
between MPLS, Broadband, and LTE/5G links. Without grouping, LTE high-jitter
values would be incorrectly flagged as anomalous when compared to MPLS norms.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def tukey_iqr_flags(x: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean flags where x is an outlier by Tukey IQR rule.

    Parameters
    ----------
    x : pd.Series — a single metric column (already grouped per device).
    k : float     — IQR multiplier; 1.5 is the standard Tukey threshold.

    Returns
    -------
    pd.Series of bool — True where the value is below the lower or above
    the upper fence (both directions flagged for generality).
    """
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr   # lower fence (abnormally low throughput, etc.)
    hi = q3 + k * iqr   # upper fence (abnormally high latency / loss)
    return (x < lo) | (x > hi)

def label_anomalies_tukey( df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str],
    k: float = 1.5,
    min_breach_metrics: int = 1,
) -> pd.DataFrame:
    """Create an `anomaly` label using Tukey-IQR per metric within each group.

    A row is labeled anomalous if at least `min_breach_metrics` metrics are outliers.

    Parameters
    ----------
    df                  : input DataFrame with raw telemetry columns.
    metric_cols         : list of column names to evaluate (e.g., ['latency_ms', 'loss_pct']).
    group_cols          : columns that identify a device (e.g., ['site_id', 'link_id']).
    k                   : Tukey multiplier (default 1.5 = standard rule).
    min_breach_metrics  : minimum number of metrics that must breach their fence
                          for a row to be labeled anomalous (default 1 = any breach).

    Returns
    -------
    df : original DataFrame with additional columns:
         - flag_<metric> (int 0/1) for each metric
         - anomaly (int 0/1) aggregated across all metric flags
    """
    df = df.copy()
    flags = []
    for m in metric_cols:
        # Apply the IQR rule per device group; transform keeps the original index alignment
        f = df.groupby(group_cols, dropna=False)[m].transform(lambda s: tukey_iqr_flags(s, k=k)).astype(int)
        flags.append(f.rename(f"flag_{m}"))
    flags_df = pd.concat(flags, axis=1)
    df = pd.concat([df, flags_df], axis=1)
    # A row is anomalous if the total number of breached metrics meets the threshold
    df["anomaly"] = (flags_df.sum(axis=1) >= min_breach_metrics).astype(int)
    return df
