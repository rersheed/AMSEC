"""Data access for the dashboard. Two sources only, never mixed silently:

* HISTORICAL — saved outputs of the corrected notebook
  (``research_results/corrected_notebook_historical.json``)
* RUNS — new executions under ``experiment_runs/run_XXXX`` produced by the
  faithful modules in :mod:`amsec.notebook`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from amsec.lab.runs import list_runs, read_json
from amsec.notebook.provenance import EXPERIMENTS, FIGURES_DIR, HISTORICAL_JSON, ROOT

METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1",
    "balanced_accuracy": "Balanced accuracy", "mcc": "MCC", "roc_auc": "ROC-AUC",
    "specificity": "Specificity", "fpr": "FPR", "fnr": "FNR",
}
CELL24_METRIC_NOTE = "Cells 2/4: precision, recall and F1 are support-weighted averages; ROC-AUC is weighted one-vs-rest."
CELL22_METRIC_NOTE = "Cell 22: precision, recall and F1 are for the malware class (binary); ROC-AUC on P(malware)."


@lru_cache(maxsize=1)
def historical() -> dict:
    return json.loads(HISTORICAL_JSON.read_text(encoding="utf-8"))


def historical_table(experiment_id: str, table_id: str) -> pd.DataFrame:
    t = historical()["experiments"].get(experiment_id, {}).get("tables", {}).get(table_id)
    return pd.DataFrame(t["rows"]) if t and "rows" in t else pd.DataFrame()


def historical_tables(experiment_id: str) -> dict[str, dict]:
    return historical()["experiments"].get(experiment_id, {}).get("tables", {})


def historical_figures(experiment_id: str) -> list[dict]:
    figs = historical()["experiments"].get(experiment_id, {}).get("figures", [])
    return [{**f, "path": FIGURES_DIR / f["file"]} for f in figs]


def runs_for(experiment_id: str | None = None, status: str | None = "completed") -> list[dict]:
    rows = list_runs()
    if experiment_id:
        rows = [r for r in rows if r["experiment_id"] == experiment_id]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return rows


def run_results(run: dict) -> pd.DataFrame:
    path = Path(run["path"])
    if run["experiment_id"] == "cell22_dual_dataset":
        f = path / "dual_dataset_balancing_results.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()
    rows = read_json(path / "results.json") or []
    df = pd.DataFrame(rows)
    if len(df):
        df["rank"] = df["f1_score"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    return df


def run_file(run: dict, name: str) -> Path:
    return Path(run["path"]) / name


def run_predictions(run: dict, key) -> pd.DataFrame | None:
    """Test-set probabilities. key = entry number (cells 2/4) or method (cell 22)."""
    path = Path(run["path"])
    if run["experiment_id"] == "cell22_dual_dataset":
        f = path / "test_predictions.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        df = df[df["method"] == key].rename(columns={"proba_malware": "proba"})
        return df[["y_true", "proba"]]
    f = path / f"test_predictions_entry{int(key):02d}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).rename(columns={"proba_attack": "proba"})


def run_history(run: dict, key) -> pd.DataFrame | None:
    path = Path(run["path"])
    if run["experiment_id"] == "cell22_dual_dataset":
        f = path / "training_history.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        df = df[df["method"] == key]
        long = []
        for _, r in df.iterrows():
            long += [{"model": "DualInputModel", "epoch": r.epoch, "series": "train", "loss": r.train_loss, "accuracy": r.train_acc},
                     {"model": "DualInputModel", "epoch": r.epoch, "series": "validation", "loss": r.val_loss, "accuracy": r.val_acc}]
        return pd.DataFrame(long)
    hist = read_json(path / f"history_entry{int(key):02d}.json")
    if not hist:
        return None
    long = []
    for model, h in hist.items():
        for i in range(len(h.get("loss", []))):
            long.append({"model": model, "epoch": i + 1, "series": "train", "loss": h["loss"][i], "accuracy": h.get("accuracy", [None] * 999)[i]})
            if "val_loss" in h:
                long.append({"model": model, "epoch": i + 1, "series": "validation", "loss": h["val_loss"][i], "accuracy": h.get("val_accuracy", [None] * 999)[i]})
    return pd.DataFrame(long)


def run_confusion(run: dict, key) -> list | None:
    if run["experiment_id"] == "cell22_dual_dataset":
        cms = read_json(run_file(run, "confusion_matrices.json")) or {}
        return cms.get(key)
    rows = read_json(run_file(run, "results.json")) or []
    row = next((r for r in rows if int(r["entry"]) == int(key)), None)
    return row["confusion_matrix"] if row else None


def compare_with_historical(experiment_id: str, run: dict) -> pd.DataFrame:
    """Per-entry historical vs reproduced comparison for cells 2/4."""
    table_id = {"cell04_stacking9": "results_9", "cell02_lightweight16": "results_16"}[experiment_id]
    hist = historical_table(experiment_id, table_id)
    log = pd.DataFrame(historical()["experiments"][experiment_id].get("config_log", []))
    new = run_results(run)
    if hist.empty or new.empty:
        return pd.DataFrame()
    rows = []
    for _, n in new.sort_values("entry").iterrows():
        h = hist[hist["entry"] == n["entry"]]
        lg = log[log["entry"] == n["entry"]] if len(log) else pd.DataFrame()
        if h.empty:
            continue
        h = h.iloc[0]
        row = {"entry": int(n["entry"]), "config": n["config"]}
        for m in ("accuracy", "f1_score", "roc_auc", "balanced_accuracy", "mcc"):
            row[f"{m}_notebook"] = h[m]
            row[f"{m}_reproduced"] = round(float(n[m]), 5)
            row[f"{m}_delta"] = round(float(n[m]) - float(h[m]), 5)
        row["parameters_match"] = int(h["parameters"]) == int(n["parameters"])
        if len(lg):
            g = lg.iloc[0]
            row["samples_after_sampling_notebook"] = g.get("samples_after_sampling")
            row["samples_after_sampling_reproduced"] = n.get("samples_after_sampling")
            row["split_sizes_match"] = (int(g.get("train_samples", -1)), int(g.get("val_samples", -1)), int(g.get("test_samples", -1))) == (
                int(n["train_samples"]), int(n["val_samples"]), int(n["test_samples"]))
        rows.append(row)
    return pd.DataFrame(rows)


def identity() -> dict:
    f = ROOT / "research_results" / "catalog.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))["identity"]
    except Exception:
        return {}


def experiment_options(executable_only: bool = False) -> dict[str, str]:
    return {k: f"Cell {v['cell']} — {v['short']}" for k, v in EXPERIMENTS.items() if v.get("executable") or not executable_only}
