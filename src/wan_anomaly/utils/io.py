"""
Table I/O Utilities
====================
Thin wrappers around pandas read/write functions that dispatch on file extension.
Centralizing I/O here means the rest of the pipeline does not need to know
whether the dataset is stored as CSV or Parquet — just call read_table() and
write_table() with any supported path.

Supported formats:
  .csv     — comma-separated values (human-readable, larger on disk)
  .parquet — columnar binary format (smaller, faster I/O for wide DataFrames)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_table(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file into a pandas DataFrame.

    Parameters
    ----------
    path : str or path-like — file path with .csv or .parquet extension.

    Returns
    -------
    pd.DataFrame with the file contents.

    Raises
    ------
    ValueError if the file extension is not .csv or .parquet.
    """
    p = Path(path)
    if p.suffix.lower() in {".parquet"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")

def write_table(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a CSV or Parquet file.

    Creates any missing parent directories automatically.

    Parameters
    ----------
    df   : DataFrame to write.
    path : str or path-like — destination file path (.csv or .parquet).

    Raises
    ------
    ValueError if the file extension is not .csv or .parquet.
    """
    p = Path(path)
    # Ensure the destination directory exists before writing
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet"}:
        df.to_parquet(p, index=False)
    elif p.suffix.lower() in {".csv"}:
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
