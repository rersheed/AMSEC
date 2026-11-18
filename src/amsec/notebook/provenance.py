"""Provenance for the sole authoritative experimental source.

Every experiment, historical table and chart in AMSEC is keyed to a cell of
``notebooks/Final_phd-Experiment-android-malware_CORRECTED.ipynb``.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_NAME = "Final_phd-Experiment-android-malware_CORRECTED.ipynb"
NOTEBOOK_PATH = ROOT / "notebooks" / NOTEBOOK_NAME
NOTEBOOK_SHA256 = "9e4724f7af4a8438bf7aea02b8768cc1579a50afc0be935b1f24b08581a0e8fa"
NOTEBOOK_KERNEL = {"name": "python3", "language_version": "3.12.13", "tensorflow": "2.20.0 (printed by cells 2 and 4)"}
# Library versions with which the cell-2/4 GA reproduces the notebook's printed checkpoints exactly
# (verified 2026-11-18 with scripts/diagnose_ga_environment.py: sklearn 1.2.2 and 1.6.1 match; 1.9.1 does not).
REPRODUCIBLE_PINS = {"sklearn": "1.6.1", "imblearn": "0.13.0"}
DATA_DIR = Path(os.environ.get("AMSEC_DATA_DIR", ROOT / "others files that can help"))
HISTORICAL_JSON = ROOT / "research_results" / "corrected_notebook_historical.json"
FIGURES_DIR = ROOT / "research_results" / "notebook_figures"

DATASET_FILES = {
    "nids": "NF-UQ-NIDS-Dataset.csv",
    "api": "dynamic_api_call_sequence_per_malware_100_0_306.csv",
    "syscall": "feature_vectors_syscallsbinders_frequency_5_Cat.csv",
}

# ---------------------------------------------------------------------------
# Cell → experiment map. ``executable`` means AMSEC can re-run the cell's
# procedure through a faithful module; otherwise only saved outputs are shown.
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "cell02_lightweight16": {
        "cell": 2,
        "title": "Lightweight-model experiment — 16 configurations",
        "short": "16-config MobileNetV4 / MobileNetV4+Attention",
        "framework": "TensorFlow/Keras 2.20",
        "datasets": ["NF-UQ-NIDS-v2 (first 150,000 rows)"],
        "design": "8 resampling methods × {MobileNetV4, MobileNetV4+Attention}; Hybrid MI→GA feature selection (5 of 41 kept); 30 epochs.",
        "executable": True,
        "module": "amsec.notebook.keras_pipeline (experiment='cell02')",
    },
    "cell04_stacking9": {
        "cell": 4,
        "title": "Ensemble-stacking experiment — 9 entries",
        "short": "9-entry MobileNetV4 + Lightweight stacking",
        "framework": "TensorFlow/Keras 2.20 + scikit-learn LogisticRegression meta-learner",
        "datasets": ["NF-UQ-NIDS-v2 (first 150,000 rows)"],
        "design": "9 entries = 8 resampling methods + SMOTE listed twice; MI→GA selection; MobileNetV4 + Lightweight base models; LR meta-learner; 30 epochs.",
        "executable": True,
        "module": "amsec.notebook.keras_pipeline (experiment='cell04')",
    },
    "cell06_trimsec_v1": {
        "cell": 6,
        "title": "TRIM-SEC CNN–BiLSTM benchmark — implementation A (≈95% target)",
        "short": "CNN–BiLSTM benchmark A",
        "framework": "PyTorch",
        "datasets": ["API call sequences", "NF-UQ-NIDS-v2 (reduced)"],
        "design": "9 sampling settings incl. None; separate API-sequence and NIDS evaluations; 10 epochs.",
        "executable": False,
    },
    "cell07_orphan": {
        "cell": 7,
        "title": "Orphaned outputs (cell has no source code)",
        "short": "Orphaned outputs",
        "framework": "unknown",
        "datasets": [],
        "design": "Cell 7 has an empty source but saved outputs (execution_count 7). The code that produced them is not in the notebook.",
        "executable": False,
        "untraceable": True,
    },
    "cell09_trimsec_v2": {
        "cell": 9,
        "title": "TRIM-SEC CNN–BiLSTM benchmark — implementation B (Mutambik 2025)",
        "short": "CNN–BiLSTM benchmark B",
        "framework": "PyTorch",
        "datasets": ["API call sequences", "NF-UQ-NIDS-v2 (up to 500,000 rows)"],
        "design": "9 sampling settings incl. None; 30 epochs.",
        "executable": False,
    },
    "cell11_cnnbilstm_v1": {
        "cell": 11, "title": "CNN–BiLSTM (Torch) — No Balancing vs balancing, variant 1", "short": "CNN–BiLSTM variant 1",
        "framework": "PyTorch", "datasets": ["API call sequences (static branch = API MI-top features)"],
        "design": "NIDS loaded (200,000 rows) but not used by the model; 8 epochs; dropout 0.65.", "executable": False,
    },
    "cell13_cnnbilstm_v2": {
        "cell": 13, "title": "CNN–BiLSTM (Torch) — variant 2", "short": "CNN–BiLSTM variant 2",
        "framework": "PyTorch", "datasets": ["API call sequences"],
        "design": "NIDS loaded but not used; 6 epochs; dropout 0.72.", "executable": False,
    },
    "cell15_cnnbilstm_v3": {
        "cell": 15, "title": "CNN–BiLSTM (Torch) — variant 3 (ranked table)", "short": "CNN–BiLSTM variant 3",
        "framework": "PyTorch", "datasets": ["API call sequences"],
        "design": "NIDS loaded but not used; 6 epochs; ranked by F1.", "executable": False,
    },
    "cell17_cnnbilstm_v4": {
        "cell": 17, "title": "CNN–BiLSTM (Torch) — variant 4 (+specificity/FPR/FNR)", "short": "CNN–BiLSTM variant 4",
        "framework": "PyTorch", "datasets": ["API call sequences"],
        "design": "NIDS loaded but not used; 6 epochs; adds specificity, FPR, FNR, balanced accuracy.", "executable": False,
    },
    "cell19_cnnbilstm_v5": {
        "cell": 19, "title": "CNN–BiLSTM (Torch) — variant 5 (+complexity)", "short": "CNN–BiLSTM variant 5",
        "framework": "PyTorch", "datasets": ["API call sequences"],
        "design": "NIDS loaded but not used; sequence length 40; top-5 API MI features; adds params/FLOPs/latency.", "executable": False,
    },
    "cell22_dual_dataset": {
        "cell": 22,
        "title": "Corrected TRIM-SEC dual-dataset CNN–BiLSTM benchmark",
        "short": "Corrected dual-dataset CNN–BiLSTM (final)",
        "framework": "PyTorch",
        "datasets": ["API call sequences (200 benign + 1,000 malware subset)", "NF-UQ-NIDS (first 50,000 rows)"],
        "design": "Label-matched cross-dataset pairing; split before MI/scaling; top-5 NIDS MI features; 9 balancing settings; 3 epochs; CPU-feasible.",
        "executable": True,
        "module": "amsec.notebook.cell22_dual_dataset",
        "no_saved_outputs": True,
    },
}

NON_EXPERIMENT_CELLS = {
    0: "Kaggle boilerplate (lists /kaggle/input). No experiment.",
    1: "Empty.", 3: "Empty.", 5: "Empty.", 8: "Empty.", 10: "Empty.", 12: "Empty.",
    14: "Empty.", 16: "Empty.", 18: "Empty.", 20: "Empty.", 21: "Empty.",
}


@lru_cache(maxsize=1)
def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_sha256() -> str:
    return hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest()


def verify_notebook() -> dict:
    present = NOTEBOOK_PATH.exists()
    digest = notebook_sha256() if present else None
    return {"path": str(NOTEBOOK_PATH.relative_to(ROOT)), "present": present, "sha256": digest,
            "matches_recorded": digest == NOTEBOOK_SHA256}


def cell_source(index: int) -> str:
    return "".join(load_notebook()["cells"][index]["source"])


def experiment_for_cell(cell: int) -> str | None:
    return next((k for k, v in EXPERIMENTS.items() if v["cell"] == cell), None)
