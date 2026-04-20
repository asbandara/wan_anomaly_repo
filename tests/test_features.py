"""
Unit Tests — Feature Engineering
===================================
Tests the three feature engineering functions:
  1. add_time_features   — cyclical time encoding (hour, day-of-week)
  2. rolling_features    — rolling mean/std/max and lag/delta per metric
  3. make_ml_table       — final feature matrix and label vector preparation

All tests use a shared _base_df() fixture with 40 timesteps at 15-min intervals,
two metric columns, and one clear anomaly in the last row.
"""

import sys, os
# Add src/ to the Python path so wan_anomaly can be imported without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import numpy as np
from wan_anomaly.processing.features import add_time_features, rolling_features, make_ml_table


def _base_df(n=40):
    """Shared test fixture: 40-row single-site DataFrame with two metrics."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "site_id": ["S000"] * n,
        "link_id": ["S000-L0"] * n,
        "latency_ms": np.linspace(20, 60, n),       # linearly increasing latency
        "loss_pct": [0.1] * (n - 1) + [5.0],        # one outlier in the last row
        "anomaly": [0] * (n - 1) + [1],             # only the last row is anomalous
    })


def test_add_time_features_columns():
    """Verify that all four cyclical time columns are produced."""
    df = _base_df()
    out = add_time_features(df, "timestamp")
    # All four cyclical columns must be present
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert col in out.columns, f"Missing column: {col}"


def test_rolling_features_shape():
    """Verify that rolling_features adds columns without changing row count."""
    df = _base_df()
    out = rolling_features(
        df,
        metric_cols=["latency_ms", "loss_pct"],
        group_cols=["site_id", "link_id"],
        time_col="timestamp",
        windows_min=[15, 60],   # one 1-step window and one 4-step window
        freq_min=15,
    )
    # Should have more columns than the original
    assert out.shape[1] > df.shape[1]
    # Row count must be identical to the input (rolling does not drop rows)
    assert out.shape[0] == df.shape[0]


def test_make_ml_table_drops_label():
    """Verify that 'anomaly' is in y but NOT in X, and lengths are consistent."""
    df = _base_df()
    X, y = make_ml_table(df, drop_cols=["site_id", "link_id", "timestamp"], label_col="anomaly")
    # The label column must not appear as a feature
    assert "anomaly" not in X.columns
    # Feature matrix and label vector must have the same number of rows
    assert len(y) == len(df)
    # Exactly one anomaly row was created in _base_df()
    assert y.sum() == 1
