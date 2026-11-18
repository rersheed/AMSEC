"""Per-run artifact directories under experiment_runs/."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "experiment_runs"


class RunStorage:
    def __init__(self, run_id: str, root: Path | None = None):
        self.run_id = run_id
        self.root = (root or RUNS_ROOT) / run_id
        self.charts = self.root / "charts"
        self.models = self.root / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.charts.mkdir(exist_ok=True)
        self.models.mkdir(exist_ok=True)

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.root / name
        frame.to_csv(path, index=False)
        return path

    def save_progress(self, payload: dict) -> None:
        self.write_json("progress.json", payload)

    def mark_status(self, status: str, **extra) -> None:
        payload = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        self.write_json("status.json", payload)

    def save_configuration(self, config: dict) -> None:
        self.write_json("configuration.json", config)

    def save_metadata(self, meta: dict) -> None:
        base = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        }
        base.update(meta)
        self.write_json("metadata.json", base)

    def save_metrics(self, metrics: dict) -> None:
        self.write_json("metrics.json", metrics)

    def save_history(self, history: dict) -> None:
        rows = []
        mobile = history.get("mobile_history") or history.get("history") or {}
        light = history.get("light_history") or {}
        n = max(len(mobile.get("loss", [])), len(light.get("loss", [])), 1)
        for i in range(n):
            row = {"epoch": i + 1}
            for prefix, hist in (("mobile", mobile), ("light", light)):
                for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
                    values = hist.get(key, [])
                    if i < len(values):
                        row[f"{prefix}_{key}"] = values[i]
            rows.append(row)
        self.write_csv("training_history.csv", pd.DataFrame(rows))

    def save_predictions(self, y_true, y_pred, y_proba) -> None:
        frame = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "p_class0": y_proba[:, 0],
                "p_class1": y_proba[:, 1],
            }
        )
        self.write_csv("predictions.csv", frame)

    def save_models(self, ensemble) -> None:
        torch.save(ensemble.mobilenet.state_dict(), self.models / "mobilenetv4.pt")
        torch.save(ensemble.lightweight.state_dict(), self.models / "lightweight.pt")
        if ensemble.meta_learner is not None:
            joblib.dump(ensemble.meta_learner, self.models / "meta_learner.joblib")

    def save_selected_features(self, names: list[str], indices: list[int]) -> None:
        self.write_json("selected_features.json", {"indices": indices, "names": names})
        self.write_csv(
            "feature_importance.csv",
            pd.DataFrame({"index": indices, "feature": names}),
        )
