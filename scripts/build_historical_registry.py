"""Build the historical-results registry from the corrected notebook's SAVED outputs.

Writes
  research_results/corrected_notebook_historical.json   tables, GA logs, figure index
  research_results/notebook_figures/cell_XX_img_N.png    embedded PNG outputs

Nothing is typed in by hand: every number is parsed from the notebook JSON,
and the notebook SHA-256 is recorded so the registry can be re-validated.
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.notebook.provenance import (  # noqa: E402
    EXPERIMENTS, FIGURES_DIR, HISTORICAL_JSON, NOTEBOOK_NAME, NON_EXPERIMENT_CELLS,
    experiment_for_cell, load_notebook, notebook_sha256,
)

MULTIWORD = {
    "Sampling Method": "method", "Balanced Acc": "balanced_accuracy", "Training Time (s)": "training_time",
    "Inference Time (s)": "inference_time", "FLOPs (M)": "flops_m", "Inference (ms)": "inference_ms",
    "Throughput (s/s)": "throughput", "F1-Score": "f1_score", "AUC-ROC": "roc_auc", "ROC-AUC": "roc_auc",
}
SIMPLE = {
    "Config": "config", "Model": "model_type", "Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
    "MCC": "mcc", "Parameters": "parameters", "FLOPs": "flops", "Rank": "rank", "Method": "method",
    "Train_Benign": "train_benign", "Train_Malware": "train_malware", "Test_Accuracy": "accuracy",
    "Test_Precision": "precision", "Test_Recall": "recall", "Test_F1": "f1_score", "Test_AUC": "roc_auc",
    "Test_MCC": "mcc", "Test_Specificity": "specificity", "Test_FPR": "fpr", "Test_FNR": "fnr",
    "Test_Balanced_Accuracy": "balanced_accuracy", "FLOPs_M": "flops_m", "Inference_ms": "inference_ms",
    "Throughput_s_s": "throughput",
}
TEXT_COLUMNS = {"config", "method", "model_type"}

# (cell, table title regex, header regex, table id, description)
TABLE_SPECS = [
    (2, r"COMPREHENSIVE RESULTS - ALL 16 CONFIGURATIONS", r"^Sampling Method\s+Model", "results_16", "16-configuration results, ranked by weighted F1"),
    (4, r"COMPREHENSIVE RESULTS - ALL 9 CONFIGURATIONS", r"^\s*Config\s+Sampling Method", "results_9", "9-entry stacking results, ranked by weighted F1"),
]
for c in (6, 7, 9):
    TABLE_SPECS.append((c, r"FINAL RESULTS - API SEQUENCES DATASET", r"^Sampling Method Accuracy", "api_results", "API-sequence evaluation"))
    TABLE_SPECS.append((c, r"FINAL RESULTS - NF-UQ-NIDS-v2 DATASET", r"^Sampling Method Accuracy", "nids_results", "NF-UQ-NIDS-v2 evaluation"))
for c in (11, 13):
    TABLE_SPECS.append((c, r"COMPLETE COMPARISON TABLE", r"^\s*Method\s+Train_Benign", "comparison", "No balancing vs balancing (test split)"))
for c in (15, 17, 19):
    TABLE_SPECS.append((c, r"COMPLETE COMPARISON TABLE - RANKED", r"^\s*Rank\s+Method", "comparison", "Ranked by test F1"))


def cell_outputs(cell: dict) -> tuple[str, list[tuple[int, str, str]]]:
    """Return concatenated text and [(image_no, context_text, b64)]."""
    parts: list[str] = []
    images: list[tuple[int, str, str]] = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            parts.append("".join(out.get("text", [])))
        elif kind in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                context = "\n".join([ln for ln in "".join(parts).splitlines() if ln.strip()][-4:])
                images.append((len(images) + 1, context, data["image/png"]))
                parts.append(f"\n[IMAGE {len(images)}]\n")
            elif "text/plain" in data:
                parts.append("".join(data["text/plain"]))
    return "".join(parts), images


def header_columns(line: str) -> list[str]:
    s = line.strip()
    for k, v in MULTIWORD.items():
        s = s.replace(k, v)
    return [SIMPLE.get(tok, tok) for tok in s.split()]


def to_number(tok: str):
    t = tok.replace(",", "")
    try:
        return int(t) if re.fullmatch(r"-?\d+", t) else float(t)
    except ValueError:
        return None


def parse_rows(lines: list[str], columns: list[str]) -> list[dict]:
    text_pos = [i for i, c in enumerate(columns) if c in TEXT_COLUMNS]
    n_text = len(text_pos)
    lead = text_pos[0] if text_pos else 0  # numeric columns before the first text column
    lead_cols = columns[:lead]
    trail_cols = [c for c in columns[lead:] if c not in TEXT_COLUMNS]
    text_cols = [columns[i] for i in text_pos]
    rows = []
    for line in lines:
        if not line.strip() or line.strip().startswith("="):
            break
        toks = line.split()
        lead_vals = [to_number(t) for t in toks[:lead]]
        trail_toks = toks[len(toks) - len(trail_cols):]
        texts = toks[lead: len(toks) - len(trail_cols)]
        values = [to_number(t) for t in trail_toks]
        if any(v is None for v in values + lead_vals):
            break
        if n_text == 1:
            text_vals = [" ".join(texts)]
        else:
            text_vals = texts if len(texts) == n_text else [" ".join(texts[:-1]), texts[-1]]
        row = dict(zip(lead_cols, lead_vals))
        row.update(dict(zip(text_cols, text_vals)))
        row.update(dict(zip(trail_cols, values)))
        rows.append(row)
    return rows


def find_table(text: str, title_re: str, header_re: str) -> tuple[list[str], list[dict]] | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(title_re, line):
            for j in range(i + 1, min(i + 8, len(lines))):
                if re.search(header_re, lines[j]):
                    cols = header_columns(lines[j])
                    return cols, parse_rows(lines[j + 1:], cols)
    return None


def parse_ga(text: str) -> dict:
    gens = [{"generation": int(g), "best_fitness": float(f), "n_features": int(n)}
            for g, f, n in re.findall(r"Gen (\d+): Fitness=([\d.]+), Features=(\d+)", text)]
    m = re.search(r"Selected (\d+) from (\d+) features", text)
    lab = re.search(r"Label distribution: Normal=(\d+), Attack=(\d+)", text)
    vt = re.search(r"After variance threshold: (\d+) features", text)
    shape = re.search(r"Loaded NIDS: \((\d+), (\d+)\)", text)
    return {
        "checkpoints": gens,
        "selected": int(m.group(1)) if m else None, "candidates": int(m.group(2)) if m else None,
        "after_variance_threshold": int(vt.group(1)) if vt else None,
        "nids_shape": [int(shape.group(1)), int(shape.group(2))] if shape else None,
        "label_counts": {"normal": int(lab.group(1)), "attack": int(lab.group(2))} if lab else None,
    }


def parse_config_log(text: str) -> list[dict]:
    """Per-entry log of cells 2/4: entry order, post-sampling size, split shapes, metrics."""
    lines = text.splitlines()
    entries: list[dict] = []
    current: dict | None = None
    for line in lines:
        m = re.match(r"^Config (\d+)/(\d+): (.+)$", line.strip())
        if m:
            current = {"entry": int(m.group(1)), "of": int(m.group(2)), "config": m.group(3).strip()}
            entries.append(current)
            continue
        if current is None:
            continue
        if (m := re.search(r"After (.+?): (\d+) samples", line)):
            current["samples_after_sampling"] = int(m.group(2))
        elif (m := re.search(r"Normal=(\d+), Attack=(\d+)", line)) and "class_counts_after_sampling" not in current:
            current["class_counts_after_sampling"] = {"0": int(m.group(1)), "1": int(m.group(2))}
        elif (m := re.search(r"Training: \((\d+), (\d+)\), Validation: \((\d+), \d+\), Test: \((\d+), \d+\)", line)):
            current.update(train_samples=int(m.group(1)), n_features=int(m.group(2)), val_samples=int(m.group(3)), test_samples=int(m.group(4)))
        elif (m := re.search(r"Accuracy: ([\d.]+), F1: ([\d.]+), ROC-AUC: ([\d.]+)", line)) and "accuracy" not in current:
            current.update(accuracy=float(m.group(1)), f1_score=float(m.group(2)), roc_auc=float(m.group(3)))
    return entries


def main() -> None:
    nb = load_notebook()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    registry = {
        "source": {"notebook": f"notebooks/{NOTEBOOK_NAME}", "sha256": notebook_sha256(),
                   "note": "Saved outputs of the authoritative notebook. Historical — not re-executed by AMSEC."},
        "experiments": {}, "non_experiment_cells": {str(k): v for k, v in NON_EXPERIMENT_CELLS.items()},
    }
    texts: dict[int, str] = {}
    for idx, cell in enumerate(nb["cells"]):
        text, images = cell_outputs(cell)
        texts[idx] = text
        exp_id = experiment_for_cell(idx)
        if exp_id is None:
            continue
        entry = {"cell": idx, "execution_count": cell.get("execution_count"), "has_source": bool("".join(cell["source"]).strip()),
                 "n_outputs": len(cell.get("outputs", [])), "tables": {}, "figures": []}
        for no, context, b64 in images:
            name = f"cell_{idx:02d}_img_{no}.png"
            (FIGURES_DIR / name).write_bytes(base64.b64decode(b64))
            entry["figures"].append({"file": name, "context": context})
        registry["experiments"][exp_id] = entry
    for cell, title_re, header_re, table_id, desc in TABLE_SPECS:
        found = find_table(texts[cell], title_re, header_re)
        exp_id = experiment_for_cell(cell)
        if not found:
            registry["experiments"][exp_id]["tables"][table_id] = {"description": desc, "error": "table not found"}
            continue
        cols, rows = found
        registry["experiments"][exp_id]["tables"][table_id] = {"description": desc, "columns": cols, "rows": rows}
    for cell, table_id in ((2, "results_16"), (4, "results_9")):
        exp = registry["experiments"][experiment_for_cell(cell)]
        exp["feature_selection"] = parse_ga(texts[cell])
        exp["config_log"] = parse_config_log(texts[cell])
        # Attach the entry number to each ranked-table row (match on config name + F1 printed in the log).
        table_rows = exp["tables"][table_id]["rows"]
        unused = list(exp["config_log"])
        for row in table_rows:
            name = row.get("config") or f"{row['method']} + GA + {row['model_type']}"
            match = next((e for e in unused if e["config"] == name and abs(e.get("f1_score", -1) - row["f1_score"]) < 1.5e-5), None)
            if match:
                row["entry"] = match["entry"]
                unused.remove(match)
        exp["tables"][table_id]["entry_mapping"] = "entry = position in the notebook loop, matched via the per-config log (config name + printed F1)"
    for exp_id, spec in EXPERIMENTS.items():
        registry["experiments"].setdefault(exp_id, {"cell": spec["cell"], "tables": {}, "figures": []})
    HISTORICAL_JSON.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    for exp_id, e in registry["experiments"].items():
        print(exp_id, {k: len(v.get("rows", [])) for k, v in e["tables"].items()}, "figures:", len(e["figures"]))


if __name__ == "__main__":
    main()
