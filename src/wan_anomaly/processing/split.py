from __future__ import annotations
import pandas as pd

def time_split(df: pd.DataFrame, time_col: str, train_frac: float = 0.7):
    df = df.copy()
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    cutoff = t.quantile(train_frac)
    train_idx = t <= cutoff
    test_idx = t > cutoff
    return train_idx, test_idx, cutoff
