"""Read-only profiles of the research CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "others files that can help"

FILES = {
    "api": "dynamic_api_call_sequence_per_malware_100_0_306.csv",
    "syscall": "feature_vectors_syscallsbinders_frequency_5_Cat.csv",
    "nids": "NF-UQ-NIDS-Dataset.csv",
}


def path_for(key: str) -> Path:
    return DATA_DIR / FILES[key]


def profile(key: str, sample_rows: int = 5000) -> dict:
    path = path_for(key)
    columns = list(pd.read_csv(path, nrows=0).columns)
    label = next((c for c in columns if str(c).lower() in {"label", "malware", "class", "attack"}), None)
    if label is None:
        raise ValueError(f"No label column in {path.name}")
    counts = pd.read_csv(path, usecols=[label])[label].value_counts(dropna=False)
    sample = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    nulls = {str(col): int(sample[col].isna().sum()) for col in sample.columns if int(sample[col].isna().sum())}
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "columns": [str(c) for c in columns],
        "n_columns": len(columns),
        "rows": int(counts.sum()),
        "label": str(label),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "sample_rows": int(len(sample)),
        "sample_missing_counts": nulls,
        "numeric_columns": [str(c) for c in sample.select_dtypes(include="number").columns],
    }
