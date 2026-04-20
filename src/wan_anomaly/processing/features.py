"""
Feature Engineering
====================
Constructs the 37-dimensional feature vector fed to all four classifiers.

Feature groups:
  1. Cyclical time features — hour and day-of-week encoded as sin/cos pairs
     (4 features). Cyclical encoding avoids the discontinuity between hour 23
     and hour 0 that a raw integer representation would introduce.

  2. Rolling statistics — for each of the 5 metrics × 2 windows (60 min, 240 min):
     mean, std, max per window (30 features).
     Long-horizon windows (4 h) capture sustained degradation events, which are
     a stronger anomaly signal than momentary spikes.

  3. Lag and delta features — for each metric: value at t-1 (lag1) and
     first-order difference (delta1 = value - lag1).

All rolling and lag computations are performed within each device group
(site_id + link_id) so that a link's rolling statistics are not contaminated
by values from other links.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Add cyclical hour-of-day and day-of-week encodings.

    Cyclical (sin/cos) encoding wraps time values around a circle so that
    the model sees hour 23 and hour 0 as adjacent rather than 23 units apart.

    Parameters
    ----------
    df       : DataFrame containing the timestamp column.
    time_col : name of the timestamp column (must be parseable as datetime).

    Returns
    -------
    DataFrame with four new columns: hour_sin, hour_cos, dow_sin, dow_cos.
    The intermediate integer columns (_hour, _dow) are dropped before returning.
    """
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["_hour"] = t.dt.hour.astype("int16")
    df["_dow"] = t.dt.dayofweek.astype("int16")
    # cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["_hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["_dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["_dow"] / 7.0)
    # Drop the raw integer columns; only the encoded versions are needed by models
    df.drop(columns=["_hour", "_dow"], inplace=True)
    return df

def rolling_features(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str],
    time_col: str,
    windows_min: list[int],
    freq_min: int,
) -> pd.DataFrame:
    """Compute rolling window statistics, lag, and delta features per metric.

    For each metric and each window size, three statistics are computed:
      - mean  : captures the average level over the window.
      - std   : captures volatility / instability.
      - max   : captures the worst-case value within the window.

    Additionally, for each metric:
      - lag1  : the value at the previous timestep (t-1).
      - delta1: the first-order difference (current - previous).

    All computations are grouped by device (site_id, link_id) to prevent
    values from one link contaminating another link's rolling statistics.

    Parameters
    ----------
    df          : DataFrame sorted or to be sorted by group + time.
    metric_cols : list of metric column names to compute features for.
    group_cols  : columns identifying a device (used for groupby).
    time_col    : timestamp column name.
    windows_min : list of window sizes in minutes (e.g., [60, 240]).
    freq_min    : sampling frequency in minutes (e.g., 15 for 15-min data).

    Returns
    -------
    DataFrame with all original columns plus the new feature columns.
    """
    # Rolling mean/std/max and lag/delta per metric and window.
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    # Sort within each group by time before computing rolling windows
    df.sort_values(group_cols + [time_col], inplace=True)
    # Convert window sizes from minutes to number of timesteps
    steps = [max(1, int(w // freq_min)) for w in windows_min]
    out = [df]  # start with original columns; append feature Series to this list

    for m in metric_cols:
        g = df.groupby(group_cols, dropna=False)[m]
        # lag1: previous timestep value (NaN for the first row of each group)
        out.append(g.shift(1).rename(f"{m}_lag1"))
        for wmin, k in zip(windows_min, steps):
            # Require at least min_periods non-NaN values to produce a result
            # (avoids NaN-only windows at the start of each group)
            min_p = min(k, max(2, k // 3))
            r = g.rolling(k, min_periods=min_p)
            out.append(r.mean().reset_index(level=group_cols, drop=True).rename(f"{m}_mean_{wmin}m"))
            out.append(r.std().reset_index(level=group_cols, drop=True).rename(f"{m}_std_{wmin}m"))
            out.append(r.max().reset_index(level=group_cols, drop=True).rename(f"{m}_max_{wmin}m"))
        # delta1: first-order difference (change from previous timestep)
        out.append((df[m] - g.shift(1)).rename(f"{m}_delta1"))
    return pd.concat(out, axis=1)

def make_ml_table(
    df: pd.DataFrame,
    drop_cols: list[str],
    label_col: str = "anomaly",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare the final feature matrix X and label vector y for model training.

    Steps:
      1. Drop identifier and timestamp columns (not useful as model inputs).
      2. Separate the label column into y.
      3. Replace infinite values with NaN.
      4. One-hot encode any remaining non-numeric columns (e.g., link_type).
      5. Fill NaN values with per-column medians, then zeros for any remaining.

    Parameters
    ----------
    df        : fully feature-engineered DataFrame.
    drop_cols : columns to exclude from X (e.g., group_cols + [time_col]).
    label_col : name of the binary target column (default 'anomaly').

    Returns
    -------
    X : pd.DataFrame  — numeric feature matrix, NaN-free, ready for sklearn.
    y : pd.Series     — integer label vector (0=normal, 1=anomalous).
    """
    X = df.drop(columns=drop_cols + [label_col], errors="ignore").copy()
    y = df[label_col].astype(int).copy()

    # Replace inf with nan
    # Infinite values arise from division in delta features when a group has only one row
    X = X.replace([np.inf, -np.inf], np.nan)

    # One-hot encode ALL non-numeric columns (object or category)
    # Handles any string columns that survived the drop (e.g., link_type if not dropped)
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

    # Fill missing values: numeric medians, then zeros for any remaining
    # Median imputation is robust to outliers; zeros handle any remaining edge cases
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y
