"""Curve / matrix tables derived only from real prediction arrays."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def confusion_frames(cm: list[list[int]] | np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    mat = np.asarray(cm, dtype=float)
    raw = pd.DataFrame(mat, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm = pd.DataFrame(mat / row_sums, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    return raw, norm


def roc_pr_frames(y_true: np.ndarray, y_score: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr = pd.DataFrame({"recall": recall, "precision": precision})
    summary = {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(recall, precision))}
    return roc, pr, summary
