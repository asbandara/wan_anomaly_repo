from __future__ import annotations
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["_hour"] = t.dt.hour.astype("int16")
    df["_dow"] = t.dt.dayofweek.astype("int16")
    # cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["_hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["_dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["_dow"] / 7.0)
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
    # Rolling mean/std/max and lag/delta per metric and window.
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df.sort_values(group_cols + [time_col], inplace=True)
    steps = [max(1, int(w // freq_min)) for w in windows_min]
    out = [df]

    for m in metric_cols:
        g = df.groupby(group_cols, dropna=False)[m]
        out.append(g.shift(1).rename(f"{m}_lag1"))
        for wmin, k in zip(windows_min, steps):
            min_p = min(k, max(2, k // 3))
            r = g.rolling(k, min_periods=min_p)
            out.append(r.mean().reset_index(level=group_cols, drop=True).rename(f"{m}_mean_{wmin}m"))
            out.append(r.std().reset_index(level=group_cols, drop=True).rename(f"{m}_std_{wmin}m"))
            out.append(r.max().reset_index(level=group_cols, drop=True).rename(f"{m}_max_{wmin}m"))
        out.append((df[m] - g.shift(1)).rename(f"{m}_delta1"))
    return pd.concat(out, axis=1)

def make_ml_table(
    df: pd.DataFrame,
    drop_cols: list[str],
    label_col: str = "anomaly",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=drop_cols + [label_col], errors="ignore").copy()
    y = df[label_col].astype(int).copy()

    # Replace inf with nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # One-hot encode ALL non-numeric columns (object or category)
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

    # Fill missing values: numeric medians, then zeros for any remaining
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y
