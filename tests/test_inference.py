import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amsec.inference import get_ensemble, reset_ensemble

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model_manifest.json").is_file(),
    reason="Local ensemble bundle has not been trained",
)


def test_real_batch_matches_training_parity_file():
    reset_ensemble()
    model = get_ensemble()
    batch = pd.read_csv(ARTIFACTS / "parity_batch.csv")
    expected = np.load(ARTIFACTS / "parity_proba.npy")
    scored = model.predict_frame(batch)
    np.testing.assert_allclose(scored["amsec_probability_normal"], expected[:, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scored["amsec_probability_attack"], expected[:, 1], rtol=1e-5, atol=1e-5)
    again = model.predict_proba(batch)
    np.testing.assert_allclose(again, model.predict_proba(batch))
    single = model.predict_proba(batch.iloc[[0]])
    assert single.shape == (1, 2)
    assert abs(single.sum() - 1) < 1e-5
    assert list(model.manifest["label_mapping"]) == ["0", "1"] or set(model.manifest["label_mapping"]) == {"0", "1"}


def test_malformed_uploads_do_not_score():
    reset_ensemble()
    model = get_ensemble()
    with pytest.raises(Exception):
        model.predict_proba(pd.DataFrame({"not_a_feature": [1, 2, 3]}))
    empty = pd.DataFrame(columns=model.feature_names)
    with pytest.raises(Exception):
        model.predict_proba(empty)


def test_manifest_records_local_retrain_not_historical_accuracy():
    manifest = json.loads((ARTIFACTS / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "local_retrain_not_kaggle_weights"
    assert "not the missing Kaggle" in manifest["claim"]
    assert manifest["historical_reference_not_this_model"]["accuracy"] == 0.97789
    assert set(manifest["local_test_metrics"]) >= {"accuracy", "f1", "balanced_accuracy", "roc_auc", "confusion_matrix"}
    assert manifest["training"]["split_before_fit"] is True
    assert len(manifest["feature_names"]) >= 5
