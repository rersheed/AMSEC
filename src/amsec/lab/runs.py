"""Auditable experiment runs under ``experiment_runs/run_XXXX``.

Each run folder contains
  run.json          experiment id, notebook cell + SHA-256, parameters, deviations,
                    run kind, environment (package versions), dataset fingerprints
  status.json       queued | running | completed | failed (+ timestamps, error)
  progress.json     last progress event
  log.txt           full console log
  …                 experiment outputs (results.csv / results.json, histories,
                    test predictions, confusion matrices, models/, …)

Run kinds
  full_reproduction  notebook parameters and the full configuration list
  partial            notebook parameters, subset of configurations
  smoke              any parameter reduced (plumbing check — never a result)
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from amsec.notebook.provenance import (
    DATA_DIR, DATASET_FILES, EXPERIMENTS, NOTEBOOK_NAME, NOTEBOOK_SHA256, ROOT, verify_notebook,
)

RUNS_ROOT = ROOT / "experiment_runs"
FINGERPRINT_CACHE = RUNS_ROOT / ".dataset_fingerprints.json"
EXECUTABLE = {k: v for k, v in EXPERIMENTS.items() if v.get("executable")}

SMOKE_OVERRIDES = {
    "cell02_lightweight16": {"max_rows": 4000, "chunk_size": 4000, "ga_sample": 2000, "ga_population": 6, "ga_generations": 2, "epochs": 1},
    "cell04_stacking9": {"max_rows": 4000, "chunk_size": 4000, "ga_sample": 2000, "ga_population": 6, "ga_generations": 2, "epochs": 1},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def environment() -> dict:
    import importlib

    versions = {}
    for name in ("numpy", "pandas", "sklearn", "imblearn", "torch", "tensorflow", "keras", "scipy"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def dataset_fingerprints() -> dict:
    cache = read_json(FINGERPRINT_CACHE) or {}
    out = {}
    for key, name in DATASET_FILES.items():
        path = DATA_DIR / name
        if not path.exists():
            out[key] = {"file": name, "present": False}
            continue
        stat = path.stat()
        token = f"{stat.st_size}:{int(stat.st_mtime)}"
        if cache.get(name, {}).get("token") == token:
            out[key] = cache[name]["info"]
            continue
        digest = hashlib.sha256()
        lines = 0
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                digest.update(block)
                lines += block.count(b"\n")
        info = {"file": name, "present": True, "bytes": stat.st_size, "sha256": digest.hexdigest(), "data_rows": lines - 1}
        cache[name] = {"token": token, "info": info}
        out[key] = info
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    _write(FINGERPRINT_CACHE, cache)
    return out


def next_run_id() -> str:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[4:]) for p in RUNS_ROOT.glob("run_*") if p.is_dir() and p.name[4:].isdigit()]
    return f"run_{(max(nums) + 1 if nums else 1):04d}"


def run_kind(experiment_id: str, overrides: dict, selection: list | None) -> str:
    if overrides:
        return "smoke"
    if selection:
        return "partial"
    return "full_reproduction"


def create_run(experiment_id: str, selection: list | None = None, overrides: dict | None = None, label: str = "") -> Path:
    """Create a queued run folder. ``selection`` = entry numbers (cells 2/4) or method keys (cell 22)."""
    if experiment_id not in EXECUTABLE:
        raise ValueError(f"{experiment_id} is not executable in AMSEC")
    if experiment_id == "cell22_dual_dataset" and overrides:
        raise ValueError("Cell 22 is already CPU-feasible; it has no smoke overrides")
    overrides = dict(overrides or {})
    from amsec.notebook.cell22_dual_dataset import ALL_METHODS
    from amsec.notebook.keras_pipeline import config_list

    if experiment_id == "cell22_dual_dataset":
        full = list(ALL_METHODS)
    else:
        full = [c["entry"] for c in config_list("cell02" if experiment_id.startswith("cell02") else "cell04")]
    if selection and sorted(map(str, selection)) == sorted(map(str, full)):
        selection = None
    run_id = next_run_id()
    path = RUNS_ROOT / run_id
    path.mkdir(parents=True)
    spec = EXPERIMENTS[experiment_id]
    record = {
        "run_id": run_id, "experiment_id": experiment_id, "cell": spec["cell"], "title": spec["title"],
        "notebook": {"name": NOTEBOOK_NAME, "sha256_recorded": NOTEBOOK_SHA256, **verify_notebook()},
        "selection": selection, "overrides": overrides, "kind": run_kind(experiment_id, overrides, selection),
        "label": label, "created_at": _now(), "environment": environment(), "datasets": dataset_fingerprints(),
    }
    _write(path / "run.json", record)
    _write(path / "status.json", {"status": "queued", "updated_at": _now()})
    return path


def execute_run(path: Path) -> dict:
    """Execute a queued run in-process (used by the worker subprocess)."""
    record = read_json(path / "run.json")
    if record is None:
        raise FileNotFoundError(path / "run.json")
    log_path = path / "log.txt"
    log_fh = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        log_fh.write(str(msg) + "\n"); log_fh.flush()

    def progress(payload: dict) -> None:
        _write(path / "progress.json", {**payload, "at": _now()})

    started = _now()
    _write(path / "status.json", {"status": "running", "started_at": started, "updated_at": started})
    import contextlib

    class _Tee:
        """Send notebook print()/Keras output to the run log as well as the console."""
        def __init__(self, stream):
            self.stream = stream
        def write(self, s):
            log_fh.write(s); log_fh.flush()
            try:
                return self.stream.write(s)
            except Exception:
                return len(s)
        def flush(self):
            log_fh.flush()
            try:
                self.stream.flush()
            except Exception:
                pass

    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout)))
    try:
        exp = record["experiment_id"]
        if exp == "cell22_dual_dataset":
            from amsec.notebook.cell22_dual_dataset import run_cell22

            result = run_cell22(DATA_DIR, path, record.get("selection"), log=log, progress=progress)
            summary = {"failures": result["failures"], "n_results": int(len(result["results"]))}
        else:
            from amsec.notebook.keras_pipeline import run_keras_experiment

            result = run_keras_experiment(
                "cell02" if exp.startswith("cell02") else "cell04", DATA_DIR, path,
                entries=[int(e) for e in record["selection"]] if record.get("selection") else None,
                overrides=record.get("overrides") or None, log=log, progress=progress,
            )
            summary = {"failures": result["metadata"]["failures"], "n_results": int(len(result["results"]))}
        status = "completed" if summary["n_results"] else "failed"
        _write(path / "status.json", {"status": status, "started_at": started, "finished_at": _now(), "updated_at": _now(), **summary})
        return summary
    except Exception as exc:
        log(traceback.format_exc())
        _write(path / "status.json", {"status": "failed", "started_at": started, "finished_at": _now(), "updated_at": _now(), "error": repr(exc)})
        raise
    finally:
        stack.close()
        log_fh.close()


def launch_background(path: Path) -> int:
    """Start the worker subprocess for a queued run; returns its PID."""
    worker = ROOT / "scripts" / "run_notebook_experiment.py"
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    with (path / "worker_stdout.txt").open("ab") as out:
        proc = subprocess.Popen([sys.executable, "-X", "utf8", str(worker), "--run-dir", str(path)],
                                cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, **kwargs)
    status = read_json(path / "status.json") or {}
    _write(path / "status.json", {**status, "pid": proc.pid, "updated_at": _now()})
    return proc.pid


def list_runs() -> list[dict]:
    rows = []
    for p in sorted(RUNS_ROOT.glob("run_*"), reverse=True):
        rec = read_json(p / "run.json")
        if not rec:
            continue
        rec["status"] = (read_json(p / "status.json") or {}).get("status", "unknown")
        rec["path"] = str(p)
        rows.append(rec)
    return rows


def load_results(path: Path):
    import pandas as pd

    for name in ("results.csv", "dual_dataset_balancing_results.csv"):
        f = Path(path) / name
        if f.exists():
            return pd.read_csv(f)
    return None
