from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_anomaly_rates(rates_csv: str, out_png: str):
    df = pd.read_csv(rates_csv)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    x = range(len(df))
    plt.figure()
    plt.bar([i - 0.2 for i in x], df["true_anomaly_rate"], width=0.4, label="True")
    plt.bar([i + 0.2 for i in x], df["pred_anomaly_rate"], width=0.4, label="Predicted")
    plt.xticks(list(x), df["model"], rotation=20)
    plt.ylabel("Anomaly rate")
    plt.title("Predicted vs True Anomaly Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_metrics_table(metrics_csv: str, out_png: str, metric: str = "pr_auc"):
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
