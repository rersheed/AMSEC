"""Validate served detection models against the training pipeline.

1. Preprocessing parity: re-create the notebook's phase-1/2/3 matrix for the first
   150,000 rows (optimize_memory → VT → GA columns → RobustScaler → Yeo-Johnson → nan_to_num)
   from the run's fitted objects, and compare with ``NotebookDetector.transform`` applied
   to the *raw* CSV rows. Must be identical.
2. Out-of-sample scoring: rows 150,001–200,000 of NF-UQ-NIDS were never read by cells 2/4.
   Score them with every entry's model and report metrics (labels from the CSV).

Usage: python scripts/validate_detection.py --run run_0002 [--entries 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score  # noqa: E402

from amsec.detection import NotebookDetector  # noqa: E402
from amsec.lab.runs import RUNS_ROOT, read_json  # noqa: E402
from amsec.notebook import keras_common as kc  # noqa: E402
from amsec.notebook.provenance import DATA_DIR, DATASET_FILES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--entries", default="")
    ap.add_argument("--holdout-start", type=int, default=150_000)
    ap.add_argument("--holdout-rows", type=int, default=50_000)
    a = ap.parse_args()
    run_dir = RUNS_ROOT / a.run
    rows = read_json(run_dir / "results.json") or []
    entries = [int(e) for e in a.entries.split(",") if e] or [int(r["entry"]) for r in rows]
    csv = DATA_DIR / DATASET_FILES["nids"]
    report: dict = {"run": a.run, "entries": {}}

    # ---- 1. preprocessing parity on the training rows
    det = NotebookDetector(a.run, entries[0])
    pre = det.pre
    raw = pd.read_csv(csv, nrows=150_000, low_memory=False)
    nids_df = kc.optimize_memory(raw.drop(columns=["Label"]).copy())
    X = nids_df.select_dtypes(include=[np.number]).fillna(0).values
    X = pre["variance_threshold"].transform(X)[:, pre["selected_indices_post_vt"]]
    X = pre["robust_scaler"].transform(X)
    if pre.get("power_transformer") is not None:
        X = pre["power_transformer"].transform(X)
    X_train_pipeline = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    X_served, warns = det.transform(raw)
    diff = float(np.max(np.abs(X_train_pipeline - X_served)))
    report["preprocessing_parity"] = {"rows": int(len(raw)), "max_abs_diff": diff, "identical": diff == 0.0, "warnings": warns[:5]}
    print("preprocessing parity:", report["preprocessing_parity"], flush=True)

    # ---- 2. out-of-sample scoring
    hold = pd.read_csv(csv, skiprows=range(1, a.holdout_start + 1), nrows=a.holdout_rows, low_memory=False)
    y = (hold["Label"].to_numpy() > 0).astype(int)
    report["holdout"] = {"first_row": a.holdout_start + 1, "rows": int(len(hold)), "normal": int((y == 0).sum()), "attack": int((y == 1).sum())}
    for e in entries:
        d = NotebookDetector(a.run, e)
        x, w = d.transform(hold)
        p = d.predict_proba_matrix(x)
        pred = p.argmax(axis=1)
        m = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
             "f1_attack": f1_score(y, pred), "f1_weighted": f1_score(y, pred, average="weighted"),
             "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, p[:, 1]), "warnings": len(w)}
        report["entries"][e] = {k: (round(float(v), 5) if not isinstance(v, int) else v) for k, v in m.items()}
        print(e, report["entries"][e], flush=True)
    out = run_dir / "detection_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0 if report["preprocessing_parity"]["identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
