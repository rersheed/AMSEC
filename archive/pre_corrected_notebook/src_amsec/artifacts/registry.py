"""Index of completed / running experiments."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "experiment_runs"
INDEX_PATH = RUNS_ROOT / "index.json"


class ExperimentRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or RUNS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _load(self) -> dict:
        if not self.index_path.exists():
            return {"runs": []}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def next_run_id(self) -> str:
        data = self._load()
        n = len(data["runs"]) + 1
        # also scan directories
        existing = {p.name for p in self.root.glob("run_*") if p.is_dir()}
        while f"run_{n:04d}" in existing:
            n += 1
        return f"run_{n:04d}"

    def register(self, entry: dict) -> None:
        data = self._load()
        data["runs"] = [r for r in data["runs"] if r.get("run_id") != entry.get("run_id")]
        data["runs"].append(entry)
        self._save(data)

    def list_runs(self, status: str | None = None) -> list[dict]:
        runs = self._load()["runs"]
        if status:
            runs = [r for r in runs if r.get("status") == status]
        return sorted(runs, key=lambda r: r.get("run_id", ""), reverse=True)

    def get(self, run_id: str) -> dict | None:
        for run in self._load()["runs"]:
            if run.get("run_id") == run_id:
                return run
        path = self.root / run_id / "metrics.json"
        if path.exists():
            return {"run_id": run_id, "metrics_path": str(path)}
        return None

    def load_metrics(self, run_id: str) -> dict | None:
        path = self.root / run_id / "metrics.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_configuration(self, run_id: str) -> dict | None:
        path = self.root / run_id / "configuration.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_predictions(self, run_id: str):
        import pandas as pd

        path = self.root / run_id / "predictions.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def load_history(self, run_id: str):
        import pandas as pd

        path = self.root / run_id / "training_history.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def completed_runs(self) -> list[dict]:
        return self.list_runs(status="completed")
