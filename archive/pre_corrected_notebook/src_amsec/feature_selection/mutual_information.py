"""Mutual-information ranking from the PhD notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def mutual_info_prefilter(
    x: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Cell 2/4: SelectKBest(mutual_info_classif, k=min(100, n))."""
    n_features = x.shape[1]
    k_eff = min(k if k is not None else 100, n_features)
    selector = SelectKBest(mutual_info_classif, k=k_eff)
    # sklearn MI uses random_state via the callable; wrap for determinism
    def _mi(xx, yy):
        return mutual_info_classif(xx, yy, random_state=random_state)

    selector = SelectKBest(_mi, k=k_eff)
    transformed = selector.fit_transform(x, y)
    mask = selector.get_support()
    indices = np.where(mask)[0]
    return transformed, indices, selector


def mutual_info_ranking(
    frame: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cell 22-style ranking table."""
    scores = mutual_info_classif(frame.fillna(0).to_numpy(), y, random_state=random_state)
    table = pd.DataFrame({"feature": frame.columns, "mi_score": scores})
    return table.sort_values("mi_score", ascending=False).head(top_k).reset_index(drop=True)
