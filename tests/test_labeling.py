import pandas as pd
from wan_anomaly.processing.label import label_anomalies_tukey

def test_labeling_runs():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=20, freq="H", tz="UTC").astype(str),
        "site_id": ["S000"]*20,
        "link_id": ["S000-L0"]*20,
        "latency_ms": list(range(20)),
        "jitter_ms": list(range(20)),
        "loss_pct": [0.1]*19 + [10.0],
        "throughput_mbps": [100]*20,
        "congestion_pct": [10]*20,
    })
    out = label_anomalies_tukey(df, ["latency_ms","loss_pct"], ["site_id","link_id"])
    assert "anomaly" in out.columns
