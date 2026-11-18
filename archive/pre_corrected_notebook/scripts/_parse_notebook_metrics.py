"""Parse stored notebook tables into verified experiment inventory JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "docs" / "extracted"
OUT = ROOT / "research_results" / "notebook_experiment_inventory.json"


def read(name: str) -> str:
    return (EXTRACTED / name).read_text(encoding="utf-8", errors="replace")


def grab_metric_block(text: str, label: str) -> dict:
    # Find blocks like "Accuracy: 0.97" near a config name
    pattern = re.compile(
        rf"{re.escape(label)}.*?Accuracy[:\s]+([0-9.]+).*?"
        rf"(?:Precision[:\s]+([0-9.]+).*?)?"
        rf"(?:Recall[:\s]+([0-9.]+).*?)?"
        rf"(?:F1[-\s]?[Ss]core[:\s]+([0-9.]+)|F1[:\s]+([0-9.]+)).*?"
        rf"(?:Balanced Accuracy[:\s]+([0-9.]+).*?)?"
        rf"(?:MCC[:\s]+([0-9.]+).*?)?"
        rf"(?:ROC-?AUC[:\s]+([0-9.]+))?",
        re.S | re.I,
    )
    match = pattern.search(text)
    if not match:
        return {}
    groups = match.groups()
    f1 = groups[3] or groups[4]
    keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "roc_auc"]
    values = [groups[0], groups[1], groups[2], f1, groups[5], groups[6], groups[7]]
    return {k: float(v) for k, v in zip(keys, values) if v}


def parse_rank_table(text: str) -> list[dict]:
    """Parse lines like rank tables printed in cell 4."""
    rows = []
    # Common pattern: ConfigName ... 0.97789
    for line in text.splitlines():
        if "SMOTE" in line or "ADASYN" in line or "No" in line:
            nums = re.findall(r"\d+\.\d+", line)
            name_match = re.search(
                r"(SMOTE(?:-ENN|-Tomek|-NCR)?|ADASYN(?:-ENN|-Tomek|-NCR)?|No[-_ ]?[Bb]alanc\w*)"
                r"[^\s]*",
                line,
            )
            if name_match and nums:
                rows.append({"line": line.strip()[:200], "name_hint": name_match.group(0), "numbers": [float(n) for n in nums[:10]]})
    return rows[:40]


def main() -> None:
    inventory = {
        "notebook": "others files that can help/Final phd-Experiment-android-malware.ipynb",
        "cells": {
            "2": {
                "family": "mobilenetv4_16_config",
                "description": "8 samplers × MobileNetV4 and MobileNetV4+Attention = 16 configs",
                "source_file": "phd_cell_2_source.py",
                "stream_file": "phd_cell_2_stream.txt",
                "stream_chars": len(read("phd_cell_2_stream.txt")),
            },
            "4": {
                "family": "ensemble_stacking_9_config",
                "description": "9 stacking configs; SMOTE appears twice by design",
                "source_file": "phd_cell_4_source.py",
                "stream_file": "phd_cell_4_stream.txt",
                "stream_chars": len(read("phd_cell_4_stream.txt")),
                "best_known": {
                    "id": "SMOTE-ENN_GA_Ensemble_Stacking",
                    "accuracy": 0.97789,
                    "f1": 0.97789,
                    "balanced_accuracy": 0.97775,
                    "mcc": 0.95582,
                    "roc_auc": 0.99767,
                    "parameters": 75004,
                    "flops_notebook": 150008,
                },
            },
            "22": {
                "family": "cnn_bilstm_benchmark",
                "description": "API-sequence CNN–BiLSTM with multiple resamplers",
                "source_file": "phd_cell_22_source.py",
                "stream_file": "phd_cell_22_stream.txt",
            },
        },
        "sampling_methods_declared": [
            "SMOTE",
            "ADASYN",
            "SMOTE-ENN",
            "SMOTE-Tomek",
            "SMOTE-NCR",
            "ADASYN-ENN",
            "ADASYN-Tomek",
            "ADASYN-NCR",
        ],
        "nine_config_note": (
            "Cell 4 declares 8 sampling methods + an extra SMOTE duplicate to reach 9 "
            "configurations. The repeated SMOTE is intentional in the notebook template."
        ),
        "datasets": {
            "nids": {
                "notebook_path": "/kaggle/input/.../NF-UQ-NIDS-v2.csv",
                "local_path": "others files that can help/NF-UQ-NIDS-Dataset.csv",
                "cell4_nrows": 150000,
                "label": "Label",
                "label_rule": "label > 0 -> attack",
            },
            "api": {
                "notebook_path": "dynamic_api_call_sequence_per_malware_100_0_306.csv",
                "local_path": "others files that can help/dynamic_api_call_sequence_per_malware_100_0_306.csv",
                "rows": 43876,
                "label": "malware",
            },
            "syscalls": {
                "local_path": "others files that can help/feature_vectors_syscallsbinders_frequency_5_Cat.csv",
                "used_by_final_ensemble": False,
            },
        },
        "leakage_in_notebook_reproduction": [
            "RobustScaler/PowerTransformer fit before split (cells 2/4)",
            "MI + GA feature selection before split",
            "Resampling applied before split",
            "Meta-learner fit on in-sample base predictions (cell 4)",
        ],
    }
    for cell in ("2", "4", "22"):
        stream = read(f"phd_cell_{cell}_stream.txt")
        inventory["cells"][cell]["rank_hints"] = parse_rank_table(stream)[:25]
        if cell == "4":
            inventory["cells"][cell]["best_block"] = grab_metric_block(stream, "SMOTE-ENN")
    OUT.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("cell4 best_block", inventory["cells"]["4"].get("best_block"))
    print("cell4 rank hints", len(inventory["cells"]["4"]["rank_hints"]))


if __name__ == "__main__":
    main()
