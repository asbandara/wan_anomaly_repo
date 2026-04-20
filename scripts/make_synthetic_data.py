"""
Step 1 — Synthetic WAN Telemetry Generator
===========================================
Generates a self-contained CSV dataset that mimics real enterprise WAN telemetry
across three link types (MPLS, Broadband, LTE/5G).

Design goals:
  - ~16% anomaly rate after label noise, matching real-world imbalance.
  - Gradual ramp-up / ramp-down anomaly events (not hard steps) so that
    anomalous values overlap with the tail of the normal distribution.
  - Diurnal (time-of-day) variation in all metrics to simulate business-hours load.
  - Deterministic output when --seed is fixed (default 42).

Usage:
    python scripts/make_synthetic_data.py --out data/raw/telemetry.csv \
        --days 14 --sites 20 --freq-min 15
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/ without installing the package
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


# ---------------------------------------------------------------------------
# Anomaly difficulty tuning
# ---------------------------------------------------------------------------
# v3 (this version): targets ~19% pre-noise Tukey-IQR anomaly rate so that
#   after 15% label-noise flip the final dataset has ~16.4% anomalous rows.
#   Events are longer (3-6 h) and more frequent (15-25 per link) so they
#   cover enough timesteps while still overlapping the tail of the normal
#   distribution — keeping PR-AUC well below 1.0.
# ---------------------------------------------------------------------------

def _inject_event(rng, arr, idx, dur, delta_lo, delta_hi, mode="add"):
    """Gradually ramp up then ramp down — more realistic than a hard step.

    Parameters
    ----------
    rng        : numpy Generator — seeded random number generator.
    arr        : 1-D float array to modify in-place.
    idx        : int — starting timestep index for the event.
    dur        : int — event duration in timesteps.
    delta_lo   : float — minimum anomaly magnitude.
    delta_hi   : float — maximum anomaly magnitude.
    mode       : "add" raises the metric; "mul" reduces throughput by scaling.
    """
    # Draw a random severity uniformly between the low and high bounds
    severity = float(rng.uniform(delta_lo, delta_hi))
    # Build a triangular ramp: ramp up for the first half, ramp down for the second half
    ramp = np.concatenate([
        np.linspace(0, severity, max(1, dur // 2)),
        np.linspace(severity, 0, dur - max(1, dur // 2)),
    ])
    # Clip to array bounds in case the event extends past the end of the series
    end = min(idx + dur, len(arr))
    ramp = ramp[:end - idx]
    if mode == "add":
        # Additive injection: latency, jitter, loss — higher values are worse
        arr[idx:end] += ramp
    elif mode == "mul":
        # Multiplicative injection: reduces throughput proportionally to severity
        arr[idx:end] *= (1.0 - ramp / (severity + 1e-9) * 0.35)


def main():
    # --- Argument parsing ---------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)           # output CSV path
    ap.add_argument("--days", type=int, default=14)   # simulation duration
    ap.add_argument("--sites", type=int, default=20)  # number of network sites
    ap.add_argument("--links-per-site", type=int, default=3)  # link types per site
    ap.add_argument("--freq-min", type=int, default=15)       # polling interval (minutes)
    ap.add_argument("--seed", type=int, default=42)           # random seed for reproducibility
    args = ap.parse_args()

    # Seeded RNG ensures identical output for the same seed value
    rng = np.random.default_rng(args.seed)
    # Total number of timesteps in the simulation window
    n_steps = int((args.days * 24 * 60) // args.freq_min)
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    ts = pd.date_range(start=start, periods=n_steps, freq=f"{args.freq_min}min")

    rows = []  # will collect one DataFrame per (site, link) pair
    link_types = ["mpls", "broadband", "lte5g"]

    for s in range(args.sites):
        site_id = f"S{s:03d}"
        for lk in range(args.links_per_site):
            link_id = f"{site_id}-L{lk}"
            # Cycle through link types so each site has all three technologies
            link_type = link_types[lk % len(link_types)]

            # --- Diurnal pattern (0–1 scale, peaks at ~noon) ----------------
            # Simulates higher load during business hours
            hour = (ts.hour + ts.minute / 60.0).to_numpy()
            diurnal = 0.5 + 0.5 * np.sin(2 * np.pi * hour / 24.0)

            # --- Per-link-type baseline values ------------------------------
            # MPLS: lowest latency/loss (premium service)
            # Broadband: mid-tier
            # LTE/5G: highest variability (wireless)
            base_latency = {"mpls": 25, "broadband": 45, "lte5g": 55}[link_type]
            base_loss    = {"mpls": 0.15, "broadband": 0.35, "lte5g": 0.6}[link_type]
            base_jitter  = {"mpls": 2.0, "broadband": 4.5, "lte5g": 7.0}[link_type]
            base_thr     = {"mpls": 200, "broadband": 300, "lte5g": 80}[link_type]

            # --- Generate normal metric time series -------------------------
            # Larger noise std → normal/anomaly distributions overlap significantly
            latency    = base_latency + 8 * diurnal + rng.normal(0, 7.0, size=n_steps)
            jitter     = base_jitter  + 2.5 * diurnal + rng.normal(0, 2.5, size=n_steps)
            loss       = np.clip(base_loss + 0.4 * diurnal + rng.normal(0, 0.35, size=n_steps), 0, None)
            thr        = np.clip(base_thr - 60 * diurnal + rng.normal(0, 40, size=n_steps), 1, None)
            congestion = np.clip(15 + 60 * diurnal + rng.normal(0, 15, size=n_steps), 0, 100)

            # --- Inject anomaly events --------------------------------------
            # Inject 30–44 events per link, each 3–6 hours long, with small-to-moderate
            # magnitudes that just barely cross the Tukey IQR fence at the ramp peak.
            # High event count → ~16% anomaly rate.
            # Small magnitudes overlap normal high-load periods → PR-AUC ~0.30–0.35.
            # 30–44 events per link, each 3–6 hours, magnitudes reliably cross
            # the Tukey IQR fence → ~15.9% anomaly rate after 15% label noise.
            n_events = rng.integers(30, 44)
            for _ in range(n_events):
                # Choose a random start time at least 25 steps from the end
                idx = int(rng.integers(0, n_steps - 25))
                dur = int(rng.integers(12, 25))  # 3 h – 6.25 h

                # Randomly pick 2 of the 3 metrics to degrade simultaneously
                affected = rng.choice(["latency", "loss", "jitter"], size=2, replace=False)

                if "latency" in affected:
                    # Latency spike also drives up congestion (correlated degradation)
                    _inject_event(rng, latency, idx, dur, 40, 65)
                    _inject_event(rng, congestion, idx, dur, 12, 25)
                if "loss" in affected:
                    _inject_event(rng, loss, idx, dur, 1.5, 3.5)
                if "jitter" in affected:
                    _inject_event(rng, jitter, idx, dur, 10, 20)

            # Collect all metric columns for this (site, link) pair
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

    # Combine all (site, link) DataFrames into one flat CSV
    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
