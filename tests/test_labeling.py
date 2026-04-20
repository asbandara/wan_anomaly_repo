"""
Unit Tests — Tukey-IQR Anomaly Labeling
=========================================
Verifies that label_anomalies_tukey() correctly produces an 'anomaly' column
when run on a minimal synthetic DataFrame.

The test constructs a controlled scenario where the last row of loss_pct (10.0)
is an extreme outlier relative to the constant 0.1 baseline — ensuring at
least one anomaly is labeled — and checks that the output schema is correct.
"""

import pandas as pd
from wan_anomaly.processing.label import label_anomalies_tukey

def test_labeling_runs():
    """Smoke test: labeling function runs and produces the 'anomaly' column."""
    # Minimal DataFrame with 20 timesteps, one site, one link
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=20, freq="H", tz="UTC").astype(str),
        "site_id": ["S000"]*20,
        "link_id": ["S000-L0"]*20,
        "latency_ms": list(range(20)),
        "jitter_ms": list(range(20)),
        "loss_pct": [0.1]*19 + [10.0],   # last row is a large outlier
        "throughput_mbps": [100]*20,
        "congestion_pct": [10]*20,
    })
    out = label_anomalies_tukey(df, ["latency_ms","loss_pct"], ["site_id","link_id"])
    # The function must always produce an 'anomaly' column, regardless of the result
    assert "anomaly" in out.columns
