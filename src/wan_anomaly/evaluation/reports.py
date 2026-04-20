"""
Evaluation Report Plots
========================
Generates comparison bar charts from the CSV files produced by train_models.py.

Two plots are produced:
  1. plot_anomaly_rates  — side-by-side bars showing predicted vs true anomaly
                           rate per model. Useful for detecting systematic over-
                           or under-prediction of the anomaly class.

  2. plot_metrics_table  — single bar per model for any scalar metric column
                           (e.g., pr_auc, accuracy). Enables quick visual
                           comparison of all models on the chosen metric.

Both functions read from CSV files so they can be regenerated independently
of the training run (e.g., after editing the CSV manually).
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_anomaly_rates(rates_csv: str, out_png: str):
    """Plot predicted vs true anomaly rate as side-by-side bars per model.

    Parameters
    ----------
    rates_csv : path to anomaly_rates.csv (columns: model, pred_anomaly_rate, true_anomaly_rate).
    out_png   : output PNG file path.
    """
    df = pd.read_csv(rates_csv)
    # Ensure the output directory exists before saving
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    x = range(len(df))
    plt.figure()
    # Offset the two bars by ±0.2 so they appear side-by-side for each model
    plt.bar([i - 0.2 for i in x], df["true_anomaly_rate"], width=0.4, label="True")
    plt.bar([i + 0.2 for i in x], df["pred_anomaly_rate"], width=0.4, label="Predicted")
    plt.xticks(list(x), df["model"], rotation=20)
    plt.ylabel("Anomaly rate")
    plt.title("Predicted vs True Anomaly Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()  # release matplotlib figure memory

def plot_metrics_table(metrics_csv: str, out_png: str, metric: str = "pr_auc"):
    """Plot a single scalar metric as a bar chart across all models.

    Parameters
    ----------
    metrics_csv : path to metrics_table.csv (must contain columns 'name' and `metric`).
    out_png     : output PNG file path.
    metric      : column name to plot (default 'pr_auc').
    """
    df = pd.read_csv(metrics_csv)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(df["name"], df[metric])
    plt.xticks(rotation=20)
    plt.ylabel(metric)
    plt.title(f"Model comparison: {metric}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
