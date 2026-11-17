"""Central research-results registry for the AMSEC dashboard."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "research_results" / "catalog.json"
VERIFIED_PATH = ROOT / "data" / "verified_results.json"
LOCAL_EVAL_PATH = ROOT / "artifacts" / "local_evaluation.json"
MANIFEST_PATH = ROOT / "artifacts" / "model_manifest.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_verified() -> dict[str, Any]:
    return json.loads(VERIFIED_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_local_evaluation() -> dict[str, Any] | None:
    if not LOCAL_EVAL_PATH.is_file():
        return None
    return json.loads(LOCAL_EVAL_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def experiments() -> list[dict[str, Any]]:
    return list(load_catalog()["experiments"])


def experiment_by_id(experiment_id: str) -> dict[str, Any] | None:
    for item in experiments():
        if item["id"] == experiment_id:
            return item
    return None


def derive_binary_rates(matrix: list[list[int]]) -> dict[str, float]:
    """Confusion matrix [[TN, FP], [FN, TP]]."""
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tpr": tp / max(tp + fn, 1),
        "tnr": tn / max(tn + fp, 1),
        "precision_attack": tp / max(tp + fp, 1),
        "recall_attack": tp / max(tp + fn, 1),
        "precision_normal": tn / max(tn + fn, 1),
        "recall_normal": tn / max(tn + fp, 1),
        "error_rate": (fp + fn) / max(tn + fp + fn + tp, 1),
    }


def local_metrics_enriched() -> dict[str, Any] | None:
    evaluation = load_local_evaluation()
    if evaluation is None:
        return None
    metrics = dict(evaluation["test_metrics"])
    matrix = metrics["confusion_matrix"]["matrix"]
    metrics.update(derive_binary_rates(matrix))
    metrics["selected_features"] = len(evaluation["selection"]["selected_features"])
    metrics["dataset_rows_used"] = evaluation["rows_loaded"]
    manifest = load_manifest()
    if manifest:
        params = manifest.get("architecture_parameter_counts_keras_style", {})
        metrics["parameters"] = int(params.get("mobilenetv4_style", 0)) + int(
            params.get("lightweight_cnn", 0)
        )
        metrics["feature_names"] = list(manifest.get("feature_names", []))
        metrics["flops"] = None
        metrics["flops_note"] = "Not profiled for the local PyTorch retrain"
    return metrics


def notebook_final_metrics() -> dict[str, Any]:
    experiment = experiment_by_id("notebook_cell4_stack_smoteenn")
    assert experiment is not None
    metrics = dict(experiment["metrics"])
    metrics["fpr"] = None
    metrics["fnr"] = None
    return metrics


def comparison_rows() -> pd.DataFrame:
    notebook = notebook_final_metrics()
    local = local_metrics_enriched()
    rows = [
        {
            "configuration": "Notebook cell-4 SMOTE-ENN stack",
            "source": "Original executed notebook",
            "equivalent_protocols": "No — resample-before-split; unnamed 5 features; TensorFlow",
            "accuracy": notebook["accuracy"],
            "precision": notebook["precision"],
            "recall": notebook["recall"],
            "f1": notebook["f1"],
            "balanced_accuracy": notebook["balanced_accuracy"],
            "mcc": notebook["mcc"],
            "roc_auc": notebook["roc_auc"],
            "fpr": notebook["fpr"],
            "fnr": notebook["fnr"],
            "parameters": notebook["parameters"],
            "selected_features": notebook["selected_features"],
            "dataset_rows": notebook["dataset_rows_used"],
        }
    ]
    if local:
        rows.append(
            {
                "configuration": "Local leakage-corrected retrain",
                "source": "Verified saved model evaluation",
                "equivalent_protocols": "No — split-before-fit; 19 named features; PyTorch",
                "accuracy": local["accuracy"],
                "precision": local["precision"],
                "recall": local["recall"],
                "f1": local["f1"],
                "balanced_accuracy": local["balanced_accuracy"],
                "mcc": local["mcc"],
                "roc_auc": local["roc_auc"],
                "fpr": local["fpr"],
                "fnr": local["fnr"],
                "parameters": local.get("parameters"),
                "selected_features": local["selected_features"],
                "dataset_rows": local["dataset_rows_used"],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) == 2:
        for column in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "mcc",
            "roc_auc",
            "fpr",
            "fnr",
        ]:
            if pd.notna(frame.loc[0, column]) and pd.notna(frame.loc[1, column]):
                frame.loc[1, f"delta_vs_notebook_{column}"] = (
                    float(frame.loc[1, column]) - float(frame.loc[0, column])
                )
    return frame


def stacking_ablations() -> pd.DataFrame:
    return pd.DataFrame(load_verified()["stacking_ablations_cell4"])


def mobilenet_ablations() -> pd.DataFrame:
    return pd.DataFrame(load_verified()["mobilenet_ablations_cell2"])


def cnn_bilstm_benchmark() -> pd.DataFrame:
    return pd.DataFrame(load_verified()["cnn_bilstm_benchmark_cell22"])


def complexity_table() -> pd.DataFrame:
    rows = [
        {
            "model": "Notebook stack (MobileNet + Light + meta)",
            "experiment": "cell4 SMOTE-ENN",
            "source": "Original executed notebook",
            "parameters": 75004,
            "flops_m": 0.150008,
            "flops_note": "2×params",
            "f1": 0.97789,
            "balanced_accuracy": 0.97775,
            "training_seconds": 326.63078,
            "inference_seconds": 7.15040,
        },
        {
            "model": "MobileNetV4+Attention",
            "experiment": "cell2 SMOTE-ENN",
            "source": "Original executed notebook",
            "parameters": 87990,
            "flops_m": 0.175980,
            "flops_note": "2×params",
            "f1": 0.97058,
            "balanced_accuracy": 0.97039,
            "training_seconds": 195.86988,
            "inference_seconds": 3.57949,
        },
        {
            "model": "MobileNetV4",
            "experiment": "cell2 SMOTE-ENN",
            "source": "Original executed notebook",
            "parameters": 62710,
            "flops_m": 0.125420,
            "flops_note": "2×params",
            "f1": 0.96900,
            "balanced_accuracy": 0.96909,
            "training_seconds": 157.41778,
            "inference_seconds": 2.86387,
        },
        {
            "model": "CNN–BiLSTM benchmark",
            "experiment": "cell22 No Balancing",
            "source": "Original executed notebook",
            "parameters": 1108931,
            "flops_m": 16.633,
            "flops_note": "notebook estimate",
            "f1": 0.9815,
            "balanced_accuracy": 0.9366,
            "training_seconds": None,
            "inference_seconds": None,
        },
    ]
    local = local_metrics_enriched()
    if local and local.get("parameters"):
        rows.append(
            {
                "model": "Local PyTorch stack",
                "experiment": "local SMOTE-ENN",
                "source": "Verified saved model evaluation",
                "parameters": local["parameters"],
                "flops_m": None,
                "flops_note": "not profiled",
                "f1": local["f1"],
                "balanced_accuracy": local["balanced_accuracy"],
                "training_seconds": (load_local_evaluation() or {}).get("elapsed_seconds"),
                "inference_seconds": None,
            }
        )
    return pd.DataFrame(rows)


def artifact_file_sizes() -> pd.DataFrame:
    names = [
        "mobilenetv4.pt",
        "lightweight.pt",
        "meta_learner.joblib",
        "preprocessor.joblib",
        "model_manifest.json",
        "local_evaluation.json",
    ]
    rows = []
    for name in names:
        path = ROOT / "artifacts" / name
        rows.append(
            {
                "file": name,
                "available": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return pd.DataFrame(rows)


def unavailable(message: str) -> str:
    return f"Unavailable — {message}"
