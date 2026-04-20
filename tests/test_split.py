"""
Unit Tests — Time-Aware Train / Test Split
===========================================
Verifies the correctness of time_split() by checking:
  1. Both sets are non-empty.
  2. Train and test sets are disjoint (no row appears in both).
  3. All rows are covered (no row is excluded from both sets).
  4. The temporal ordering is strict: max(train timestamps) ≤ cutoff
     and min(test timestamps) > cutoff.

These properties collectively guarantee no look-ahead data leakage.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from wan_anomaly.processing.split import time_split


def test_time_split_fractions():
    """Verify split produces non-empty, non-overlapping, exhaustive index masks."""
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
    })
    train_idx, test_idx, cutoff = time_split(df, "timestamp", train_frac=0.7)
    # Both sets must contain at least one row
    assert train_idx.sum() > 0
    assert test_idx.sum() > 0
    # No overlap: a row cannot be in both train and test
    assert (train_idx & test_idx).sum() == 0
    # Covers all rows: every row belongs to exactly one set
    assert (train_idx | test_idx).sum() == n


def test_time_split_cutoff_ordering():
    """Verify strict temporal ordering: all train timestamps ≤ cutoff < test timestamps."""
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
    })
    train_idx, test_idx, cutoff = time_split(df, "timestamp", train_frac=0.8)
    t = pd.to_datetime(df["timestamp"], utc=True)
    # No training sample should be later than the cutoff
    assert t[train_idx].max() <= cutoff
    # No test sample should be at or before the cutoff
    assert t[test_idx].min() > cutoff
