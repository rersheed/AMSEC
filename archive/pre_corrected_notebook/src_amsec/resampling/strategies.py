"""Notebook resampling strategies (including intentional duplicate SMOTE)."""
from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, TomekLinks

# Eight unique methods used by cell 2 (×2 models = 16 configs)
EIGHT_SAMPLERS = [
    "SMOTE",
    "ADASYN",
    "SMOTE-ENN",
    "SMOTE-Tomek",
    "SMOTE-NCR",
    "ADASYN-ENN",
    "ADASYN-Tomek",
    "ADASYN-NCR",
]

# Cell 4 nine-entry list — SMOTE appears twice by design
NINE_STACKING_METHODS = [
    "SMOTE",
    "ADASYN",
    "SMOTE-ENN",
    "SMOTE-Tomek",
    "SMOTE-NCR",
    "ADASYN-ENN",
    "ADASYN-Tomek",
    "ADASYN-NCR",
    "SMOTE",  # duplicate for 9th configuration
]


def apply_sampling_method(x: np.ndarray, y: np.ndarray, method_name: str, random_state: int = 42):
    """Apply the specified sampling method (notebook cell 4 logic)."""
    if method_name in (None, "None", "No Balancing", "no_balancing"):
        return x, y
    try:
        if method_name == "SMOTE":
            return SMOTE(random_state=random_state, k_neighbors=5).fit_resample(x, y)
        if method_name == "ADASYN":
            return ADASYN(random_state=random_state, n_neighbors=5).fit_resample(x, y)
        if method_name == "SMOTE-ENN":
            return SMOTEENN(random_state=random_state).fit_resample(x, y)
        if method_name == "SMOTE-Tomek":
            return SMOTETomek(random_state=random_state).fit_resample(x, y)
        if method_name == "SMOTE-NCR":
            x_s, y_s = SMOTE(random_state=random_state, k_neighbors=5).fit_resample(x, y)
            return NearMiss(version=1, n_neighbors=3).fit_resample(x_s, y_s)
        if method_name == "ADASYN-ENN":
            x_a, y_a = ADASYN(random_state=random_state, n_neighbors=5).fit_resample(x, y)
            return EditedNearestNeighbours(n_neighbors=3).fit_resample(x_a, y_a)
        if method_name == "ADASYN-Tomek":
            x_a, y_a = ADASYN(random_state=random_state, n_neighbors=5).fit_resample(x, y)
            return TomekLinks().fit_resample(x_a, y_a)
        if method_name == "ADASYN-NCR":
            x_a, y_a = ADASYN(random_state=random_state, n_neighbors=5).fit_resample(x, y)
            return NearMiss(version=1, n_neighbors=3).fit_resample(x_a, y_a)
        # cell 22 aliases
        if method_name in ("SMOTE+ENN", "SMOTE+Tomek", "SMOTE+NCR", "ADASYN+ENN", "ADASYN+Tomek", "ADASYN+NCR"):
            return apply_sampling_method(x, y, method_name.replace("+", "-"), random_state)
        return x, y
    except Exception:
        return SMOTE(random_state=random_state, k_neighbors=5).fit_resample(x, y)
