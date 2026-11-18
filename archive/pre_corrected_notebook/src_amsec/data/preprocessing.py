"""Preprocessing helpers matching notebook transforms."""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler


def variance_filter(x: np.ndarray, threshold: float = 1e-5) -> tuple[np.ndarray, VarianceThreshold]:
    selector = VarianceThreshold(threshold=threshold)
    transformed = selector.fit_transform(x)
    return transformed, selector


def notebook_scale_matrix(
    x: np.ndarray,
    *,
    fit: bool = True,
    scaler: RobustScaler | None = None,
    power: PowerTransformer | None = None,
    use_power: bool = True,
) -> tuple[np.ndarray, RobustScaler, PowerTransformer | None]:
    """RobustScaler (+ optional Yeo-Johnson) as in cells 2/4."""
    if fit or scaler is None:
        scaler = RobustScaler()
        scaled = scaler.fit_transform(x)
    else:
        scaled = scaler.transform(x)
    fitted_power = power
    if use_power:
        try:
            if fit or power is None:
                fitted_power = PowerTransformer(method="yeo-johnson", standardize=True)
                scaled = fitted_power.fit_transform(scaled)
            else:
                scaled = fitted_power.transform(scaled)
        except Exception:
            fitted_power = None
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    return scaled, scaler, fitted_power


def standard_scale(
    x: np.ndarray,
    *,
    fit: bool = True,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    if fit or scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(x).astype(np.float32), scaler
    return scaler.transform(x).astype(np.float32), scaler
