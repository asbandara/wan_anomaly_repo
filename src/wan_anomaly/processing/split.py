"""
Time-Aware Train / Test Split
================================
Splits a time-series DataFrame into training and test sets using a
chronological (temporal) boundary rather than random sampling.

Why temporal splitting matters
-------------------------------
Random splitting of time-series data causes look-ahead leakage: rolling
features computed from future values can appear in the training set.
Temporal splitting guarantees that every training sample is strictly
EARLIER than every test sample, preserving causality.

The split point is determined by the `train_frac` quantile of the timestamp
distribution (default 0.7 = 70th percentile of time), so the split
adapts automatically to any date range in the dataset.
"""

from __future__ import annotations
import pandas as pd

def time_split(df: pd.DataFrame, time_col: str, train_frac: float = 0.7):
    """Split a time-series DataFrame into train and test index masks.

    Parameters
    ----------
    df         : DataFrame containing the timestamp column.
    time_col   : name of the timestamp column (UTC-aware datetime or string).
    train_frac : fraction of the time range to assign to training (default 0.7).

    Returns
    -------
    train_idx : pd.Series of bool — True for rows in the training set.
    test_idx  : pd.Series of bool — True for rows in the test set.
    cutoff    : pd.Timestamp     — the exact boundary timestamp.
                                   train <= cutoff < test.
    """
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    # The cutoff is the train_frac quantile of the timestamp distribution
    cutoff = t.quantile(train_frac)
    train_idx = t <= cutoff   # all rows up to and including the cutoff
    test_idx = t > cutoff     # all rows strictly after the cutoff
    return train_idx, test_idx, cutoff
