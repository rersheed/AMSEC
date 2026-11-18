"""Malware/attack detection with models trained by a faithful notebook run.

Only cell-4 stacking ensembles (MobileNetV4 + Lightweight + LogisticRegression)
and cell-2 single models produced by an AMSEC run are servable. Preprocessing
replays the run's fitted objects exactly:

  raw NF-UQ-NIDS columns → training dtype casts (``optimize_memory`` result)
  → GA-selected columns → RobustScaler → Yeo-Johnson PowerTransformer
  → nan_to_num(nan=0, posinf=5, neginf=-5) → float32 → model(s)

Notes
* VarianceThreshold only drops columns, so selecting the GA columns by name is
  equivalent to ``vt.transform`` followed by index selection.
* Training applied ``optimize_memory``; float16 columns are rounded through
  float16 here as in training. Integer columns are kept as int64 (the training
  downcast was lossless) to avoid wrap-around on out-of-range inputs, which are
  reported instead.
* Inputs are network-flow records (NF-UQ-NIDS schema), not Android APKs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from amsec.lab.runs import RUNS_ROOT, list_runs, read_json


class InputValidationError(ValueError):
    pass


@dataclass
class ServableModel:
    run_id: str
    entry: int
    config: str
    experiment_id: str
    kind: str
    metrics: dict

    @property
    def label(self) -> str:
        return f"{self.run_id} · entry {self.entry} · {self.config} · F1 {self.metrics.get('f1_score', float('nan')):.4f} ({self.kind})"


def servable_models() -> list[ServableModel]:
    out = []
    for run in list_runs():
        if run.get("status") != "completed" or run["experiment_id"] not in ("cell04_stacking9", "cell02_lightweight16"):
            continue
        path = Path(run["path"])
        rows = read_json(path / "results.json") or []
        for row in rows:
            tag = path / "models" / f"entry{int(row['entry']):02d}"
            needed = ["mobilenetv4.keras", "lightweight.keras", "meta_learner.joblib"] if run["experiment_id"] == "cell04_stacking9" else ["model.keras"]
            if all((tag / n).exists() for n in needed) and (path / "models" / "preprocessing.joblib").exists():
                out.append(ServableModel(run["run_id"], int(row["entry"]), row["config"], run["experiment_id"], run["kind"],
                                         {k: row.get(k) for k in ("accuracy", "f1_score", "roc_auc", "balanced_accuracy", "mcc")}))
    # full reproductions first, then by F1
    order = {"full_reproduction": 0, "partial": 1, "smoke": 2}
    return sorted(out, key=lambda m: (order.get(m.kind, 3), -(m.metrics.get("f1_score") or 0)))


class NotebookDetector:
    def __init__(self, run_id: str, entry: int):
        import joblib
        from tensorflow import keras

        self.run_dir = RUNS_ROOT / run_id
        self.run = read_json(self.run_dir / "run.json") or {}
        self.entry = int(entry)
        self.pre = joblib.load(self.run_dir / "models" / "preprocessing.joblib")
        mdir = self.run_dir / "models" / f"entry{self.entry:02d}"
        self.kind = self.run.get("experiment_id")
        if self.kind == "cell04_stacking9":
            self.mobile = keras.models.load_model(mdir / "mobilenetv4.keras")
            self.light = keras.models.load_model(mdir / "lightweight.keras")
            self.meta = joblib.load(mdir / "meta_learner.joblib")
        else:
            self.single = keras.models.load_model(mdir / "model.keras")
        self.feature_names: list[str] = list(self.pre["selected_features"])
        self.dtypes: dict = self.pre["column_dtypes"]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if frame is None or len(frame) == 0:
            raise InputValidationError("No rows to score.")
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise InputValidationError(f"Missing required columns: {', '.join(missing)}")
        warnings: list[str] = []
        cols = []
        for name in self.feature_names:
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.isna().any():
                warnings.append(f"{name}: {int(values.isna().sum())} non-numeric value(s) treated as 0 (notebook fillna(0)).")
            values = values.fillna(0)
            dtype = self.dtypes.get(name, "float64")
            if dtype == "float16":
                values = values.astype(np.float16)
            elif dtype.startswith("int"):
                info = np.iinfo(np.dtype(dtype))
                out_of_range = ((values < info.min) | (values > info.max)).sum()
                if out_of_range:
                    warnings.append(f"{name}: {int(out_of_range)} value(s) outside the training dtype range ({dtype}).")
            cols.append(values.to_numpy(dtype=np.float64))
        x = np.column_stack(cols)
        x = self.pre["robust_scaler"].transform(x)
        if self.pre.get("power_transformer") is not None:
            x = self.pre["power_transformer"].transform(x)
        nt = self.pre["nan_to_num"]
        x = np.nan_to_num(x, nan=nt["nan"], posinf=nt["posinf"], neginf=nt["neginf"]).astype(np.float32)
        return x, warnings

    def predict_proba_matrix(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "cell04_stacking9":
            meta_x = np.hstack([self.mobile.predict(x, verbose=0), self.light.predict(x, verbose=0)])
            return self.meta.predict_proba(meta_x)
        return self.single.predict(x, verbose=0)

    def predict_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        x, warnings = self.transform(frame)
        proba = self.predict_proba_matrix(x)
        out = frame.copy()
        out["amsec_p_normal"] = proba[:, 0]
        out["amsec_p_attack"] = proba[:, 1]
        out["amsec_prediction"] = np.where(proba.argmax(axis=1) == 1, "attack", "normal")  # notebook: argmax
        out["amsec_model"] = f"{self.run_dir.name}/entry{self.entry:02d}"
        return out, warnings

    def lrp(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Notebook LRP (|x · ∂p/∂x| normalised) on the MobileNetV4 base model, per row."""
        from amsec.notebook.keras_common import LRPExplainer

        x, _ = self.transform(frame)
        model = self.mobile if self.kind == "cell04_stacking9" else self.single
        explainer = LRPExplainer(model)
        rows = []
        for i in range(len(x)):
            pred = int(np.argmax(self.predict_proba_matrix(x[i:i + 1])))
            rel = explainer.explain(x[i:i + 1], target_class=pred)["relevance_scores"]
            rows.append(dict(zip(self.feature_names, map(float, rel))))
        return pd.DataFrame(rows)
