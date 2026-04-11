from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


# ---------------------------------------------------------------------------
# Anomaly difficulty tuning
# ---------------------------------------------------------------------------
# v1 (original): severity 1–3x, large deltas → PR-AUC ~1.0 (too easy)
#
# v2 (this version): realistic degradation that overlaps with normal
#   high-load periods. Anomalies are injected as gradual ramp-ups on a
#   SINGLE metric at a time (not all metrics simultaneously), so the
#   Tukey-IQR labeler only fires on borderline cases.
#   High baseline noise ensures normal/anomaly distributions overlap.
#   Target: LogReg PR-AUC ~0.65–0.75, RF PR-AUC ~0.80–0.88
# ---------------------------------------------------------------------------

def _inject_event(rng, arr, idx, dur, delta_lo, delta_hi, mode="add"):
    """Gradually ramp up then ramp down — more realistic than a hard step."""
    severity = float(rng.uniform(delta_lo, delta_hi))
    ramp = np.concatenate([
        np.linspace(0, severity, max(1, dur // 2)),
        np.linspace(severity, 0, dur - max(1, dur // 2)),
    ])
    end = min(idx + dur, len(arr))
    ramp = ramp[:end - idx]
    if mode == "add":
        arr[idx:end] += ramp
    elif mode == "mul":
        arr[idx:end] *= (1.0 - ramp / (severity + 1e-9) * 0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--sites", type=int, default=20)
    ap.add_argument("--links-per-site", type=int, default=3)
    ap.add_argument("--freq-min", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_steps = int((args.days * 24 * 60) // args.freq_min)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    ts = pd.date_range(start=start, periods=n_steps, freq=f"{args.freq_min}min")

    rows = []
    link_types = ["mpls", "broadband", "lte5g"]

    for s in range(args.sites):
        site_id = f"S{s:03d}"
        for lk in range(args.links_per_site):
            link_id = f"{site_id}-L{lk}"
            link_type = link_types[lk % len(link_types)]

            hour = (ts.hour + ts.minute / 60.0).to_numpy()
            diurnal = 0.5 + 0.5 * np.sin(2 * np.pi * hour / 24.0)

            base_latency = {"mpls": 25, "broadband": 45, "lte5g": 55}[link_type]
            base_loss    = {"mpls": 0.15, "broadband": 0.35, "lte5g": 0.6}[link_type]
            base_jitter  = {"mpls": 2.0, "broadband": 4.5, "lte5g": 7.0}[link_type]
            base_thr     = {"mpls": 200, "broadband": 300, "lte5g": 80}[link_type]

            # Larger noise std → normal/anomaly distributions overlap significantly
            latency    = base_latency + 8 * diurnal + rng.normal(0, 7.0, size=n_steps)
            jitter     = base_jitter  + 2.5 * diurnal + rng.normal(0, 2.5, size=n_steps)
            loss       = np.clip(base_loss + 0.4 * diurnal + rng.normal(0, 0.35, size=n_steps), 0, None)
            thr        = np.clip(base_thr - 60 * diurnal + rng.normal(0, 40, size=n_steps), 1, None)
            congestion = np.clip(15 + 60 * diurnal + rng.normal(0, 15, size=n_steps), 0, 100)

            # Inject events that affect only 1–2 metrics at a time (not all)
            # so Tukey-IQR only fires on the most extreme cases
            n_events = rng.integers(3, 7)
            for _ in range(n_events):
                idx = int(rng.integers(0, n_steps - 10))
                dur = int(rng.integers(3, 10))  # 45 min – 2.5 hrs

                # Pick which metric(s) to degrade — not all at once
                affected = rng.choice(["latency", "loss", "jitter", "thr"], size=2, replace=False)

                # Magnitudes cross the Tukey IQR fence (so labels fire)
                # but high baseline noise keeps single-row detection hard
                if "latency" in affected:
                    _inject_event(rng, latency, idx, dur, 18, 35)
                if "loss" in affected:
                    _inject_event(rng, loss, idx, dur, 0.5, 1.5)
                if "jitter" in affected:
                    _inject_event(rng, jitter, idx, dur, 4, 9)
                if "thr" in affected:
                    _inject_event(rng, thr, idx, dur, 0.15, 0.35, mode="mul")
                if "latency" in affected:
                    _inject_event(rng, congestion, idx, dur, 8, 20)

            rows.append(pd.DataFrame({
                "timestamp":       ts.astype(str),
                "site_id":         site_id,
                "link_id":         link_id,
                "link_type":       link_type,
                "latency_ms":      latency,
                "jitter_ms":       jitter,
                "loss_pct":        loss,
                "throughput_mbps": thr,
                "congestion_pct":  congestion,
            }))

    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
