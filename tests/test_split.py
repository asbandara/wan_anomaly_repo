import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from wan_anomaly.processing.split import time_split


def test_time_split_fractions():
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
    })
    train_idx, test_idx, cutoff = time_split(df, "timestamp", train_frac=0.7)
    assert train_idx.sum() > 0
    assert test_idx.sum() > 0
    # No overlap
    assert (train_idx & test_idx).sum() == 0
    # Covers all rows
    assert (train_idx | test_idx).sum() == n


def test_time_split_cutoff_ordering():
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
    })
    train_idx, test_idx, cutoff = time_split(df, "timestamp", train_frac=0.8)
    t = pd.to_datetime(df["timestamp"], utc=True)
    assert t[train_idx].max() <= cutoff
    assert t[test_idx].min() > cutoff
