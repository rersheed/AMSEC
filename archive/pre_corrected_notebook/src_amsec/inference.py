"""Load the local AMSEC bundle and score rows that match its feature schema.

Missing columns are rejected. Columns are never positionally truncated or
filled with zeros to force a prediction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from amsec.models import LightweightCNN, MobileNetV4Style

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MANIFEST_PATH = ARTIFACTS / "model_manifest.json"

CLASS_NORMAL = 0
CLASS_ATTACK = 1


class InputValidationError(ValueError):
    """The uploaded table is not the schema this model was trained on."""


class ModelUnavailableError(FileNotFoundError):
    """No validated local bundle is present."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_available() -> bool:
    return MANIFEST_PATH.is_file()


def require_feature_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return the required columns in training order, or raise a validation error."""
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise InputValidationError("Input must be a table.")
    if frame.empty:
        raise InputValidationError("The table has no data rows.")
    if frame.columns.duplicated().any():
        duplicated = [str(name) for name in frame.columns[frame.columns.duplicated()].tolist()]
        raise InputValidationError(f"Duplicate column names: {duplicated}")
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise InputValidationError(
            "Required features are missing: "
            + ", ".join(missing)
            + ". This model does not zero-fill or truncate columns."
        )
    selected = frame.loc[:, feature_names].copy()
    for name in feature_names:
        coerced = pd.to_numeric(selected[name], errors="coerce")
        non_numeric = coerced.isna() & selected[name].notna()
        if bool(non_numeric.any()):
            raise InputValidationError(
                f"Column '{name}' has {int(non_numeric.sum())} non-numeric value(s)."
            )
        if bool(coerced.isna().any()):
            raise InputValidationError(
                f"Column '{name}' contains missing values. They are not filled with zeros."
            )
        selected[name] = coerced
    return selected


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise ModelUnavailableError(
            "No local model manifest at artifacts/model_manifest.json. "
            "The Kaggle HDF5 ensemble was not included in the source package."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class AmsecEnsemble:
    def __init__(self, manifest: dict | None = None):
        self.manifest = manifest or load_manifest()
        self.feature_names = list(self.manifest["feature_names"])
        self.threshold = float(self.manifest["decision_threshold"])
        self._check_files()
        bundle = joblib.load(ARTIFACTS / self.manifest["preprocessor_file"])
        self.scaler = bundle["scaler"]
        self.power = bundle["power"]
        self.meta = joblib.load(ARTIFACTS / self.manifest["meta_learner_file"])
        if list(getattr(self.meta, "classes_", [])) != [CLASS_NORMAL, CLASS_ATTACK]:
            raise ModelUnavailableError(
                f"Meta-learner class order {list(self.meta.classes_)} is not [0, 1]."
            )
        input_dim = len(self.feature_names)
        self.mobile = MobileNetV4Style(input_dim)
        self.light = LightweightCNN(input_dim)
        self.mobile.load_state_dict(torch.load(ARTIFACTS / self.manifest["mobilenet_file"], map_location="cpu", weights_only=True))
        self.light.load_state_dict(torch.load(ARTIFACTS / self.manifest["lightweight_file"], map_location="cpu", weights_only=True))
        self.mobile.eval()
        self.light.eval()

    def _check_files(self) -> None:
        expected = {
            self.manifest["preprocessor_file"]: self.manifest["hashes"]["preprocessor"],
            self.manifest["meta_learner_file"]: self.manifest["hashes"]["meta_learner"],
            self.manifest["mobilenet_file"]: self.manifest["hashes"]["mobilenet"],
            self.manifest["lightweight_file"]: self.manifest["hashes"]["lightweight"],
        }
        for name, digest in expected.items():
            path = ARTIFACTS / name
            if not path.is_file():
                raise ModelUnavailableError(f"Missing artifact: {name}")
            found = _sha256(path)
            if found != digest:
                raise ModelUnavailableError(f"SHA-256 mismatch for {name}")

    def validate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return require_feature_frame(frame, self.feature_names)

    def transform(self, selected: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(selected.to_numpy(dtype=np.float64))
        try:
            scaled = self.power.transform(scaled)
        except Exception as exc:
            raise InputValidationError(f"Power transform failed: {exc}") from exc
        scaled = np.nan_to_num(scaled, nan=np.nan, posinf=5.0, neginf=-5.0)
        if not np.isfinite(scaled).all():
            raise InputValidationError("Preprocessing produced non-finite values. The row was not scored.")
        return scaled.astype(np.float32)

    def _softmax_pair(self, module: torch.nn.Module, matrix: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            logits = module(torch.from_numpy(matrix))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.transform(self.validate_frame(frame))
        mobile = self._softmax_pair(self.mobile, matrix)
        light = self._softmax_pair(self.light, matrix)
        meta_features = np.hstack([mobile, light])
        probabilities = self.meta.predict_proba(meta_features)
        order = list(self.meta.classes_)
        if order != [CLASS_NORMAL, CLASS_ATTACK]:
            raise ModelUnavailableError(f"Unexpected class order {order}")
        return probabilities

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(frame)
        attack = probabilities[:, CLASS_ATTACK]
        labels = np.where(attack >= self.threshold, "attack", "normal")
        out = frame.copy()
        out["amsec_probability_normal"] = probabilities[:, CLASS_NORMAL]
        out["amsec_probability_attack"] = attack
        out["amsec_prediction"] = labels
        out["amsec_threshold"] = self.threshold
        out["amsec_model_id"] = self.manifest["experiment_id"]
        return out

    def attribute(self, frame: pd.DataFrame, epsilon: float = 1e-2) -> pd.DataFrame:
        """Finite-difference sensitivity of the deployed ensemble, not notebook LRP."""
        selected = self.validate_frame(frame)
        base = self.predict_proba(selected)[:, CLASS_ATTACK]
        columns = []
        for name in self.feature_names:
            shifted = selected.copy()
            shifted[name] = shifted[name].astype(float) + epsilon
            delta = (self.predict_proba(shifted)[:, CLASS_ATTACK] - base) / epsilon
            columns.append(pd.Series(delta, name=name, index=selected.index))
        return pd.concat(columns, axis=1)


_ENSEMBLE: AmsecEnsemble | None = None


def get_ensemble() -> AmsecEnsemble:
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = AmsecEnsemble()
    return _ENSEMBLE


def reset_ensemble() -> None:
    global _ENSEMBLE
    _ENSEMBLE = None
