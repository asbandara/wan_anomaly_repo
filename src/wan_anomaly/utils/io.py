from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")

def write_table(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet"}:
        df.to_parquet(p, index=False)
    elif p.suffix.lower() in {".csv"}:
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
