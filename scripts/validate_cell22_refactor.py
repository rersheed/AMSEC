"""Prove the cell-22 refactor is equivalent to the notebook's own code.

1. Executes the *verbatim* cell-22 source from the authoritative notebook in a
   subprocess; only the two path literals (BASE, OUT) are substituted.
2. Runs ``amsec.notebook.cell22_dual_dataset.run_cell22`` with the same inputs.
3. Compares the result tables metric-by-metric and writes a JSON report.

Usage: python scripts/validate_cell22_refactor.py [--methods no_balancing,smote]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.notebook.cell22_dual_dataset import ALL_METHODS, run_cell22  # noqa: E402
from amsec.notebook.provenance import DATA_DIR, cell_source  # noqa: E402

METRICS = ["train_samples", "test_samples", "accuracy", "precision", "recall", "f1_score", "roc_auc",
           "balanced_accuracy", "mcc", "specificity", "fpr", "fnr", "parameters"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--out", default=str(ROOT / "experiment_runs" / "_validation" / "cell22_refactor"))
    args = ap.parse_args()
    out = Path(args.out)
    verbatim_out = out / "verbatim"
    module_out = out / "module"
    verbatim_out.mkdir(parents=True, exist_ok=True)

    src = cell_source(22)
    base_literal = "BASE=Path('/mnt/data/experiments_ii/Experiments II')"
    out_literal = "OUT=Path('/mnt/data/experiments_ii/results_dual_dataset')"
    assert base_literal in src and out_literal in src, "cell-22 path literals changed; update validator"
    patched = src.replace(base_literal, f"BASE=Path({str(DATA_DIR)!r})").replace(
        out_literal, f"OUT=Path({str(verbatim_out)!r})"
    )
    # The cell's API file name is identical; its NIDS name is the local file name already.
    script = verbatim_out / "cell22_verbatim.py"
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + patched, encoding="utf-8")
    env = dict(os.environ, METHODS=args.methods, PYTHONIOENCODING="utf-8")
    print("[1/2] Running verbatim notebook cell 22 ...", flush=True)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(verbatim_out), env=env,
                          capture_output=True, text=True, encoding="utf-8")
    (verbatim_out / "stdout.txt").write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        return proc.returncode
    tag = args.methods.replace(",", "_") if "METHODS" in env else "all"
    verbatim_csv = verbatim_out / f"dual_dataset_balancing_results_{tag}.csv"
    verbatim = pd.read_csv(verbatim_csv)

    print("[2/2] Running refactored module ...", flush=True)
    result = run_cell22(DATA_DIR, module_out, args.methods.split(","), log=lambda s: None, save_models=False)
    module = result["results"]

    merged = verbatim.merge(module, on="method", suffixes=("_verbatim", "_module"))
    report = {"methods": args.methods.split(","), "rows_compared": int(len(merged)), "per_method": {}, "all_equal": True}
    for _, row in merged.iterrows():
        diffs = {}
        for m in METRICS:
            a, b = float(row[f"{m}_verbatim"]), float(row[f"{m}_module"])
            diffs[m] = {"verbatim": a, "module": b, "abs_diff": abs(a - b)}
            if not np.isclose(a, b, rtol=0, atol=1e-12):
                report["all_equal"] = False
        report["per_method"][row["method"]] = diffs
    meta_v = json.loads((verbatim_out / f"dual_dataset_run_metadata_{tag}.json").read_text())
    report["metadata_equal"] = {k: meta_v.get(k) == result["metadata"].get(k) for k in meta_v}
    (out / "refactor_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"all_equal": report["all_equal"], "rows": report["rows_compared"],
                      "metadata_equal": all(report["metadata_equal"].values())}, indent=2))
    return 0 if report["all_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
