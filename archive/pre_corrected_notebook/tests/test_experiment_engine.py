"""Unit tests for notebook experiment engine (no full matrix training)."""
from __future__ import annotations

import numpy as np
import pytest

from amsec.evaluation.metrics import classification_metrics
from amsec.models import LightweightCNN, MobileNetV4Style, keras_style_parameter_count
from amsec.notebook_models.keras_builders import keras_available, notebook_flops
from amsec.notebook_models.mobilenetv4 import build_mobilenet
from amsec.resampling.strategies import NINE_STACKING_METHODS, apply_sampling_method
from amsec.feature_selection.genetic_algorithm import GeneticFeatureSelector


def test_nine_stacking_methods_preserve_duplicate_smote():
    assert len(NINE_STACKING_METHODS) == 9
    assert NINE_STACKING_METHODS.count("SMOTE") == 2
    assert NINE_STACKING_METHODS[0] == "SMOTE"
    assert NINE_STACKING_METHODS[-1] == "SMOTE"


def test_notebook_parameter_counts_at_five_features():
    mobile = MobileNetV4Style(5)
    light = LightweightCNN(5)
    assert keras_style_parameter_count(mobile) == 62_710
    assert keras_style_parameter_count(light) == 12_294
    assert notebook_flops(62_710) == 125_420


def test_attention_variant_builds_and_forwards():
    model = build_mobilenet(8, use_attention=True)
    import torch

    out = model(torch.randn(4, 8))
    assert out.shape == (4, 2)


def test_sampling_smote_changes_counts():
    rng = np.random.RandomState(0)
    x = rng.randn(100, 4)
    y = np.array([0] * 80 + [1] * 20)
    xb, yb = apply_sampling_method(x, y, "SMOTE")
    assert len(yb) >= len(y)


def test_ga_smoke_selects_features():
    rng = np.random.RandomState(0)
    x = rng.randn(120, 10)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        y[:60] = 0
        y[60:] = 1
    ga = GeneticFeatureSelector(x, y, population_size=4, generations=2, n_estimators=10, random_state=0)
    selected, history = ga.run()
    assert selected.size >= 3
    assert len(history) == 2


def test_metrics_confusion_shape():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert "roc_auc" in m


def test_keras_optional():
    # Environment may or may not have TF; just ensure the probe does not crash.
    assert isinstance(keras_available(), bool)
