import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import numpy as np
from wan_anomaly.processing.features import add_time_features, rolling_features, make_ml_table


def _base_df(n=40):
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC"),
        "site_id": ["S000"] * n,
        "link_id": ["S000-L0"] * n,
        "latency_ms": np.linspace(20, 60, n),
        "loss_pct": [0.1] * (n - 1) + [5.0],
        "anomaly": [0] * (n - 1) + [1],
    })


def test_add_time_features_columns():
    df = _base_df()
    out = add_time_features(df, "timestamp")
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert col in out.columns, f"Missing column: {col}"


def test_rolling_features_shape():
    df = _base_df()
    out = rolling_features(
        df,
        metric_cols=["latency_ms", "loss_pct"],
        group_cols=["site_id", "link_id"],
        time_col="timestamp",
        windows_min=[15, 60],
        freq_min=15,
    )
    # Should have more columns than the original
    assert out.shape[1] > df.shape[1]
    assert out.shape[0] == df.shape[0]


def test_make_ml_table_drops_label():
    df = _base_df()
    X, y = make_ml_table(df, drop_cols=["site_id", "link_id", "timestamp"], label_col="anomaly")
    assert "anomaly" not in X.columns
    assert len(y) == len(df)
    assert y.sum() == 1
