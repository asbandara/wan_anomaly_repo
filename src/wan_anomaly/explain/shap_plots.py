from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _to_2d(shap_values: np.ndarray) -> np.ndarray:
    """
    Ensure SHAP array is (n_samples, n_features).
    If it has extra dimensions (e.g., classes/outputs), collapse them.
    """
    sv = np.asarray(shap_values)
    if sv.ndim == 2:
        return sv
    if sv.ndim > 2:
        # Collapse any trailing dims into one by averaging magnitude
        # shape: (n_samples, n_features, ...)
        return np.mean(sv, axis=tuple(range(2, sv.ndim)))
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")
def _ensure_dir(out_png: str) -> None:
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)


def shap_summary_bar(shap_values: np.ndarray, feature_names: list[str], out_png: str, max_display: int = 20) -> None:
    """Mean(|SHAP|) bar chart."""
    shap_values = _to_2d(shap_values)
    imp = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(imp)[::-1][:max_display]

    plt.figure()
    plt.bar(range(len(order)), imp[order])
    plt.xticks(range(len(order)), [feature_names[i] for i in order], rotation=45, ha="right")
    plt.ylabel("mean(|SHAP|)")
    plt.title("SHAP Feature Importance (mean |value|)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def shap_summary_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, out_png: str, max_display: int = 20) -> None:
    """Beeswarm-style plot (simple implementation)."""
    shap_values = _to_2d(shap_values)
    imp = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(imp)[::-1][:max_display]
    feat_names = X.columns.to_list()

    plt.figure(figsize=(8, max(4, 0.35 * len(order))))
    y_positions = np.arange(len(order))[::-1]

    for yi, idx in enumerate(order):
        vals = shap_values[:, idx]
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        plt.scatter(vals, y_positions[yi] + jitter, s=6, alpha=0.5)

    plt.yticks(y_positions, [feat_names[i] for i in order])
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP value")
    plt.title("SHAP Summary (beeswarm-style)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()