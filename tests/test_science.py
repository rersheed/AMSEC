import json
from pathlib import Path

import pandas as pd
import pytest

from amsec.datasets import profile
from amsec.inference import InputValidationError, require_feature_frame
from amsec.models import LightweightCNN, MobileNetV4Style, keras_style_parameter_count
from amsec.selection import GeneticFeatureSelector
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_metrics_are_the_transcribed_cell4_numbers():
    data = json.loads((ROOT / "data" / "verified_results.json").read_text(encoding="utf-8"))
    metrics = data["final_ensemble"]["metrics"]
    assert metrics["accuracy"] == 0.97789
    assert metrics["balanced_accuracy"] == 0.97775
    assert metrics["f1"] == 0.97789
    assert data["final_ensemble"]["weights_in_package"] is False
    assert len(data["stacking_ablations_cell4"]) == 9
    assert len(data["mobilenet_ablations_cell2"]) == 16
    assert len(data["cnn_bilstm_benchmark_cell22"]) == 9
    assert data["cnn_bilstm_benchmark_cell22"][0]["fpr"] == 0.6111
    assert "0.9847" not in json.dumps(data)


def test_architecture_parameter_counts_match_notebook():
    assert keras_style_parameter_count(MobileNetV4Style(5)) == 62710
    assert keras_style_parameter_count(LightweightCNN(5)) == 12294


def test_feature_frame_rejects_missing_extra_padding_and_blanks():
    names = ["MIN_IP_PKT_LEN", "PROTOCOL", "L4_DST_PORT"]
    with pytest.raises(InputValidationError, match="missing"):
        require_feature_frame(pd.DataFrame({"PROTOCOL": [6]}), names)
    with pytest.raises(InputValidationError, match="no data"):
        require_feature_frame(pd.DataFrame(columns=names), names)
    with pytest.raises(InputValidationError, match="missing values"):
        require_feature_frame(pd.DataFrame({"MIN_IP_PKT_LEN": [1.0], "PROTOCOL": [None], "L4_DST_PORT": [80]}), names)
    with pytest.raises(InputValidationError, match="non-numeric"):
        require_feature_frame(pd.DataFrame({"MIN_IP_PKT_LEN": ["x"], "PROTOCOL": [6], "L4_DST_PORT": [80]}), names)
    frame = pd.DataFrame({"extra": [9], "L4_DST_PORT": [80], "PROTOCOL": [6], "MIN_IP_PKT_LEN": [40]})
    selected = require_feature_frame(frame, names)
    assert list(selected.columns) == names
    assert selected.iloc[0].tolist() == [40, 6, 80]


def test_genetic_search_is_deterministic_on_a_fixed_sample():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(180, 6))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    kwargs = dict(population_size=4, generations=2, n_estimators=8, random_state=1)
    first, _ = GeneticFeatureSelector(x, y, **kwargs).run()
    second, _ = GeneticFeatureSelector(x, y, **kwargs).run()
    assert list(first) == list(second)
    assert len(first) >= 3


def test_api_and_syscall_schemas():
    api = profile("api")
    syscall = profile("syscall")
    assert api["n_columns"] == 102
    assert api["label"] == "malware"
    assert api["label_counts"]["1"] == 42797
    assert api["label_counts"]["0"] == 1079
    assert api["rows"] == 43876
    assert syscall["label"] == "Class"
    assert syscall["rows"] == 11598
    assert syscall["n_columns"] == 471
    assert set(syscall["label_counts"]) == {"1", "2", "3", "4", "5"}


def test_nids_schema_and_row_count():
    nids = profile("nids", sample_rows=200)
    assert nids["label"] == "Label"
    assert nids["n_columns"] == 46
    assert nids["rows"] == 1_048_575
    assert set(nids["label_counts"]) == {"0", "1"}
