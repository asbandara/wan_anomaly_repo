"""
SHAP Visualization Plots
==========================
Generates two types of SHAP summary plots for model interpretability.

SHAP (SHapley Additive exPlanations) assigns each feature a signed importance
value for each individual prediction:
  - Positive SHAP value → feature pushes prediction toward 'anomaly'.
  - Negative SHAP value → feature pushes prediction toward 'normal'.
  - Magnitude → strength of that feature's influence.

Plot types:
  shap_summary_bar      — Mean absolute SHAP per feature (overall ranking).
                          Useful for answering "which features matter most?"
  shap_summary_beeswarm — Scatter plot of raw SHAP values (one dot per sample).
                          Shows both importance AND direction of effect for
                          each feature, sorted by mean |SHAP|.

Both functions implement the plots from scratch using matplotlib rather than
calling shap.summary_plot(), for full control over styling and output format.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _to_2d(shap_values: np.ndarray) -> np.ndarray:
    """
    Ensure SHAP array is (n_samples, n_features).
    If it has extra dimensions (e.g., classes/outputs), collapse them.

    Some versions of SHAP return a 3-D array for multi-output models.
    This helper normalizes the shape so downstream code always sees 2-D.
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
    """Create the parent directory of out_png if it does not already exist."""
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)


def shap_summary_bar(shap_values: np.ndarray, feature_names: list[str], out_png: str, max_display: int = 20) -> None:
    """Mean(|SHAP|) bar chart showing overall feature importance ranking.

    Parameters
    ----------
    shap_values   : 2-D array (n_samples, n_features) of SHAP values.
    feature_names : list of feature name strings (length = n_features).
    out_png       : output PNG file path.
    max_display   : maximum number of features to show (top-k by mean |SHAP|).
    """
    shap_values = _to_2d(shap_values)
    # Compute mean absolute SHAP across all test samples for each feature
    imp = np.mean(np.abs(shap_values), axis=0)
    # Sort features by importance descending; take top max_display
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
    """Beeswarm-style plot (simple implementation).

    Each dot represents one test sample. The horizontal position is the SHAP
    value (positive = toward anomaly), and a small vertical jitter separates
    overlapping points within each feature row.

    Features are ordered top-to-bottom by mean absolute SHAP (most important
    at the top), consistent with the bar chart.

    Parameters
    ----------
    shap_values  : 2-D array (n_samples, n_features) of SHAP values.
    X            : DataFrame of test features (used for column names).
    out_png      : output PNG file path.
    max_display  : maximum number of features to show.
    """
    shap_values = _to_2d(shap_values)
    # Rank features by mean |SHAP|; select top max_display
    imp = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(imp)[::-1][:max_display]
    feat_names = X.columns.to_list()

    plt.figure(figsize=(8, max(4, 0.35 * len(order))))
    # Reverse y-positions so the most important feature appears at the top
    y_positions = np.arange(len(order))[::-1]

    for yi, idx in enumerate(order):
        vals = shap_values[:, idx]
        # Small random vertical jitter prevents overplotting of identical values
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        plt.scatter(vals, y_positions[yi] + jitter, s=6, alpha=0.5)

    plt.yticks(y_positions, [feat_names[i] for i in order])
    plt.axvline(0, linewidth=1)   # vertical line at SHAP=0 (decision boundary)
    plt.xlabel("SHAP value")
    plt.title("SHAP Summary (beeswarm-style)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
