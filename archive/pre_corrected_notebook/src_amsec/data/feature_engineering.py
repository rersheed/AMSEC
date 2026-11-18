"""Feature-matrix helpers used by experiment runners."""
from __future__ import annotations

import numpy as np


def select_columns(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(x)[:, np.asarray(indices)]


def subsample_rows(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_rows:
        return x, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), max_rows, replace=False)
    return x[idx], y[idx]
