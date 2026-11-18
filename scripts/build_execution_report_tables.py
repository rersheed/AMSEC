"""Generate the data tables of docs/EXECUTION_REPORT.md from run artifacts.

Writes docs/validation/execution_tables.md. Every number comes from
experiment_runs/ or the historical registry — nothing is typed by hand.
Usage: python scripts/build_execution_report_tables.py --cell4 run_0005 --cell2 run_0006 --cell22 run_0001,run_0004
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.dashboard.data import compare_with_historical, historical  # noqa: E402
from amsec.lab.runs import RUNS_ROOT, list_runs, read_json  # noqa: E402


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".5f") if len(df) else "_no data_"


def run_rec(run_id: str) -> dict:
    return next(r for r in list_runs() if r["run_id"] == run_id)


def keras_section(exp_id: str, run_id: str, title: str) -> str:
    run = run_rec(run_id)
    meta = read_json(Path(run["path"]) / "run_metadata.json") or {}
    feats = meta.get("features", {})
    hist = historical()["experiments"][exp_id]
    fs = hist["feature_selection"]
    ga = pd.read_csv(Path(run["path"]) / "ga_fitness.csv")
    chk = []
    for c in fs["checkpoints"]:
        g = c["generation"]
        chk.append({"generation": g, "notebook": c["best_fitness"], "reproduced": round(float(ga.loc[ga.generation == g, "best_fitness"].iloc[0]), 4)})
    chk = pd.DataFrame(chk)
    chk["match"] = (chk["notebook"] - chk["reproduced"]).abs() < 1e-9
    cmp = compare_with_historical(exp_id, run)
    exact_cols = ["entry", "config", "samples_after_sampling_notebook", "samples_after_sampling_reproduced", "split_sizes_match", "parameters_match"]
    metric_cols = ["entry", "config"] + [f"{m}_{s}" for m in ("accuracy", "f1_score", "roc_auc") for s in ("notebook", "reproduced", "delta")]
    status = read_json(Path(run["path"]) / "status.json") or {}
    out = [f"### {title} — `{run_id}` ({run['kind']}, status {status.get('status')})",
           f"Environment: {json.dumps(run['environment']['packages'])}",
           f"\n**Feature selection.** NIDS shape {meta.get('nids_shape')} (notebook {fs['nids_shape']}); labels {meta.get('label_counts')} "
           f"(notebook {fs['label_counts']}); after VT {len(feats.get('after_variance_threshold', []))} (notebook {fs['after_variance_threshold']}); "
           f"GA kept {len(feats.get('selected_features', []))} (notebook {fs['selected']}): {feats.get('selected_features')}\n",
           md(chk), "\n**Deterministic checks (exact).**\n", md(cmp[[c for c in exact_cols if c in cmp]]) if len(cmp) else "_none_",
           "\n**Metrics (training is not bit-reproducible across GPU→CPU).**\n", md(cmp[[c for c in metric_cols if c in cmp]]) if len(cmp) else "_none_"]
    if len(cmp):
        d = cmp["f1_score_delta"]
        out.append(f"\nF1 delta: mean {d.mean():+.5f}, max |Δ| {d.abs().max():.5f}, entries {len(cmp)}.")
        h = pd.DataFrame(hist["tables"]["results_9" if exp_id == "cell04_stacking9" else "results_16"]["rows"])
        new = cmp.sort_values("f1_score_reproduced", ascending=False)["entry"].tolist()
        old = h.sort_values("f1_score", ascending=False)["entry"].tolist()
        out.append(f"Top entry: notebook {old[0]}, reproduced {new[0]}. Ranking (notebook): {old}; (reproduced): {new}.")
    post = meta.get("post_phases", {})
    if post:
        out.append(f"\nPhases 7–9: best = {post.get('best_result')}; API evaluation = {post.get('api_evaluation')}.")
    dv = read_json(Path(run["path"]) / "detection_validation.json")
    if dv:
        out.append(f"\n**Detection validation.** Preprocessing parity on {dv['preprocessing_parity']['rows']} training rows: "
                   f"max |Δ| = {dv['preprocessing_parity']['max_abs_diff']} (identical = {dv['preprocessing_parity']['identical']}). "
                   f"Hold-out rows from #{dv['holdout']['first_row']} ({dv['holdout']['rows']} rows; normal {dv['holdout']['normal']}, attack {dv['holdout']['attack']}):\n")
        out.append(md(pd.DataFrame([{"entry": k, **v} for k, v in dv["entries"].items()])))
    return "\n".join(out)


def cell22_section(run_ids: list[str]) -> str:
    out = ["### Cell 22 — corrected dual-dataset benchmark (no saved notebook outputs)"]
    eq = read_json(RUNS_ROOT / "_validation" / "cell22_refactor" / "refactor_equivalence.json")
    if eq:
        out.append(f"Refactor equivalence vs verbatim cell (current environment): all_equal = **{eq['all_equal']}** over {eq['rows_compared']} methods.\n")
    for rid in run_ids:
        run = run_rec(rid)
        df = pd.read_csv(Path(run["path"]) / "dual_dataset_balancing_results.csv")
        out.append(f"**`{rid}`** — sklearn {run['environment']['packages']['sklearn']}, imblearn {run['environment']['packages']['imblearn']}\n")
        out.append(md(df[["rank", "method", "train_samples", "accuracy", "f1_score", "roc_auc", "balanced_accuracy", "specificity", "mcc"]]))
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell4"); ap.add_argument("--cell2"); ap.add_argument("--cell22", default="")
    a = ap.parse_args()
    parts = ["<!-- generated by scripts/build_execution_report_tables.py -->"]
    if a.cell4:
        parts.append(keras_section("cell04_stacking9", a.cell4, "Cell 4 — 9-entry stacking"))
    if a.cell2:
        parts.append(keras_section("cell02_lightweight16", a.cell2, "Cell 2 — 16-configuration lightweight models"))
    if a.cell22:
        parts.append(cell22_section([r for r in a.cell22.split(",") if r]))
    out = ROOT / "docs" / "validation" / "execution_tables.md"
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8")[:6000])


if __name__ == "__main__":
    main()
