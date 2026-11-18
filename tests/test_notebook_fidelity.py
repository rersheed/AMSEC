"""Fidelity: AMSEC's experiment code must match the authoritative notebook.

* The notebook file must be the approved one (SHA-256).
* Every class/function AMSEC copies from the notebook must have an AST
  identical to the notebook definition (docstrings and formatting included
  via ast.dump of the parsed definitions).
"""
from __future__ import annotations

import ast
import inspect
import json

import pytest

from amsec.notebook import provenance
from amsec.notebook.provenance import NOTEBOOK_SHA256, ROOT, cell_source


def _notebook_defs(cell: int) -> dict[str, str]:
    tree = ast.parse(cell_source(cell))
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _module_defs(path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_notebook_is_the_approved_file():
    assert provenance.NOTEBOOK_PATH.exists()
    assert provenance.notebook_sha256() == NOTEBOOK_SHA256


def test_notebook_has_expected_cell_layout():
    nb = provenance.load_notebook()
    assert len(nb["cells"]) == 23
    for exp in provenance.EXPERIMENTS.values():
        assert nb["cells"][exp["cell"]]["cell_type"] == "code"
    assert cell_source(7).strip() == ""  # orphaned outputs, no code


NB_DIR = ROOT / "src" / "amsec" / "notebook"


@pytest.mark.parametrize(
    "module_file,cell,names",
    [
        ("keras_common.py", 4, ["set_seed", "GeneticFeatureSelector", "apply_sampling_method", "get_model_flops", "optimize_memory", "LRPExplainer"]),
        ("keras_common.py", 2, ["set_seed", "GeneticFeatureSelector", "get_model_flops", "optimize_memory", "LRPExplainer"]),
        ("cell02_models.py", 2, ["MobileNetV4Model"]),
        ("cell04_models.py", 4, ["MobileNetV4Model", "LightweightModel", "EnsembleStackingModel"]),
        ("cell02_api.py", 2, ["process_api_dataset"]),
        ("cell04_api.py", 4, ["process_api_dataset"]),
        ("cell22_dual_dataset.py", 22, ["PairDataset", "CNNBiLSTM", "DualInputModel", "apply_balancing", "evaluate"]),
    ],
)
def test_definitions_are_verbatim(module_file, cell, names):
    nb = _notebook_defs(cell)
    mod = _module_defs(NB_DIR / module_file)
    for name in names:
        assert name in nb, f"{name} missing from notebook cell {cell}"
        assert name in mod, f"{name} missing from {module_file}"
        assert mod[name] == nb[name], f"{module_file}:{name} differs from notebook cell {cell}"


def test_cell2_sampler_differs_from_cell4_only_by_docstring():
    a = ast.parse(cell_source(2)); b = ast.parse(cell_source(4))
    fa = next(n for n in a.body if getattr(n, "name", "") == "apply_sampling_method")
    fb = next(n for n in b.body if getattr(n, "name", "") == "apply_sampling_method")
    fb.body = fb.body[1:]  # drop docstring
    assert ast.dump(fa) == ast.dump(fb)


def test_cell22_script_constants_match_notebook():
    src = cell_source(22)
    for literal in ["SEED=42", "torch.set_num_threads(3)", "seq_len=80", "quota=min(200,", "mal_quota=1000",
                    "nrows=50_000", "test_size=.30", "test_size=.50", "head(5)", "for ep in range(3)",
                    "lr=.0018,weight_decay=8e-4", "batch_size=512", "patience=1,factor=.5"]:
        assert literal in src, literal
    from amsec.notebook import cell22_dual_dataset as c22

    module_src = inspect.getsource(c22.run_cell22)
    for literal in ["seq_len=80", "quota=min(200,", "mal_quota=1000", "nrows=50_000", "test_size=.30",
                    "test_size=.50", "head(5)", "for ep in range(3)", "lr=.0018,weight_decay=8e-4", "batch_size=512",
                    "patience=1,factor=.5", "shuffle=(method=='no_balancing')"]:
        assert literal in module_src, literal


def test_cell4_and_cell2_script_constants_match_pipeline():
    from amsec.notebook import keras_pipeline as kp

    assert kp.CELL04_SAMPLING == ["SMOTE", "ADASYN", "SMOTE-ENN", "SMOTE-Tomek", "SMOTE-NCR", "ADASYN-ENN", "ADASYN-Tomek", "ADASYN-NCR", "SMOTE"]
    assert kp.CELL04_SAMPLING.count("SMOTE") == 2  # duplicate preserved
    for cell in (2, 4):
        src = cell_source(cell)
        for literal in ["max_rows = 150000", "chunk_size = 50000", "VarianceThreshold(threshold=1e-5)", "20000",
                        "population_size=40, generations=30, mutation_rate=0.12, crossover_rate=0.9",
                        "k_mi = min(100", "> 180000", "test_size=0.15", "test_size=0.12", "epochs=30, batch_size=128",
                        "test_size=0.2", "posinf=5.0, neginf=-5.0"]:
            assert literal in src, (cell, literal)
    d = kp.NOTEBOOK_DEFAULTS
    assert (d["max_rows"], d["chunk_size"], d["ga_sample"], d["ga_population"], d["ga_generations"], d["epochs"], d["batch_size"], d["balanced_cap"]) == (
        150000, 50000, 20000, 40, 30, 30, 128, 180000)


def test_historical_registry_is_from_this_notebook():
    reg = json.loads(provenance.HISTORICAL_JSON.read_text(encoding="utf-8"))
    assert reg["source"]["sha256"] == NOTEBOOK_SHA256
    rows = reg["experiments"]["cell04_stacking9"]["tables"]["results_9"]["rows"]
    assert len(rows) == 9 and sum(r["method"] == "SMOTE" for r in rows) == 2
    assert len(reg["experiments"]["cell02_lightweight16"]["tables"]["results_16"]["rows"]) == 16
    assert reg["experiments"]["cell22_dual_dataset"]["tables"] == {}  # no saved outputs in the notebook
