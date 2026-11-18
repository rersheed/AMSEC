"""Feature attribution helpers for completed runs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def meta_feature_importance(coefficients: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    coef = np.asarray(coefficients).reshape(-1)
    if names is None:
        names = [f"meta_{i}" for i in range(len(coef))]
    return pd.DataFrame({"feature": names, "coefficient": coef}).sort_values("coefficient", key=np.abs, ascending=False)


def selected_feature_table(names: list[str], indices: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"index": indices, "feature": [names[i] if i < len(names) else f"idx_{i}" for i in indices]})
