"""Configurable dataset loaders (replaces hard-coded Kaggle paths)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "others files that can help"


@dataclass
class DatasetPaths:
    data_dir: Path = DEFAULT_DATA_DIR
    nids_name: str = "NF-UQ-NIDS-Dataset.csv"
    api_name: str = "dynamic_api_call_sequence_per_malware_100_0_306.csv"
    syscalls_name: str = "feature_vectors_syscallsbinders_frequency_5_Cat.csv"

    @property
    def nids(self) -> Path:
        return self.data_dir / self.nids_name

    @property
    def api(self) -> Path:
        return self.data_dir / self.api_name

    @property
    def syscalls(self) -> Path:
        return self.data_dir / self.syscalls_name


def file_sha256(path: Path, max_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = max_bytes
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _find_label(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_nids_dataset(
    paths: DatasetPaths | None = None,
    nrows: int | None = 150_000,
) -> dict:
    """Load NF-UQ-NIDS family CSV with notebook label rule ``label > 0``."""
    paths = paths or DatasetPaths()
    if not paths.nids.exists():
        raise FileNotFoundError(f"NIDS dataset missing: {paths.nids}")

    sample = pd.read_csv(paths.nids, nrows=5)
    label_col = _find_label(list(sample.columns), ["Label", "Attack", "label", "attack", "class", "Class"])
    if label_col is None:
        raise ValueError(f"No label column in {paths.nids}")

    feature_cols = [c for c in sample.columns if c != label_col]
    frame = pd.read_csv(paths.nids, usecols=feature_cols + [label_col], nrows=nrows, low_memory=False)
    labels = (pd.to_numeric(frame[label_col], errors="coerce").fillna(0).values > 0).astype(np.int64)
    features = frame.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0)
    missing = int(frame.isna().sum().sum())
    meta = {
        "path": str(paths.nids),
        "sha256_prefix": file_sha256(paths.nids),
        "nrows_requested": nrows,
        "nrows_loaded": int(len(features)),
        "n_features": int(features.shape[1]),
        "feature_names": list(features.columns),
        "label_column": label_col,
        "class_counts": {"normal": int((labels == 0).sum()), "attack": int((labels == 1).sum())},
        "missing_values": missing,
        "notebook_note": "Notebook used Kaggle NF-UQ-NIDS-v2.csv; local file may be NF-UQ-NIDS-Dataset.csv.",
    }
    return {"X": features.to_numpy(dtype=np.float64), "y": labels, "feature_names": list(features.columns), "meta": meta}


def load_api_dataset(paths: DatasetPaths | None = None) -> dict:
    paths = paths or DatasetPaths()
    if not paths.api.exists():
        raise FileNotFoundError(f"API dataset missing: {paths.api}")
    frame = pd.read_csv(paths.api)
    label_col = _find_label(list(frame.columns), ["malware", "label", "Label"])
    if label_col is None:
        raise ValueError("No API label column")
    labels = (pd.to_numeric(frame[label_col], errors="coerce").fillna(0).values > 0).astype(np.int64)
    features = frame.drop(columns=[label_col])
    meta = {
        "path": str(paths.api),
        "sha256_prefix": file_sha256(paths.api),
        "nrows_loaded": int(len(features)),
        "n_columns": int(features.shape[1]),
        "label_column": label_col,
        "class_counts": {"benign": int((labels == 0).sum()), "malware": int((labels == 1).sum())},
        "alignment_note": "Do not assume rows align with NIDS flows by application id.",
    }
    return {"frame": features, "y": labels, "meta": meta}


def load_syscalls_dataset(paths: DatasetPaths | None = None) -> dict:
    paths = paths or DatasetPaths()
    if not paths.syscalls.exists():
        raise FileNotFoundError(f"Syscalls dataset missing: {paths.syscalls}")
    frame = pd.read_csv(paths.syscalls)
    label_col = _find_label(list(frame.columns), ["Class", "class", "label", "Label"])
    meta = {
        "path": str(paths.syscalls),
        "sha256_prefix": file_sha256(paths.syscalls),
        "nrows_loaded": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "label_column": label_col,
        "used_by_final_ensemble": False,
    }
    return {"frame": frame, "meta": meta}


def dump_meta(meta: dict, path: Path) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
