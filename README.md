# AMSEC

**AMSEC — An Explainable Lightweight Hybrid Deep Learning Framework for Android Malware Detection.**

A single-mode Streamlit platform for **research, experimentation, analytics and malware detection**. There is no presentation mode.

## Authoritative experimental source

`notebooks/Final_phd-Experiment-android-malware_CORRECTED.ipynb` (SHA-256 `9e4724f7…e8fa`) is the **sole** experimental source. AMSEC checks its hash at start-up. Earlier notebooks are archived in `archive/pre_corrected_notebook/notebooks/` and are not used.

| Cell | Experiment | AMSEC |
|---|---|---|
| 2 | 16-configuration lightweight-model experiment (MobileNetV4 / +Attention × 8 samplers) — Keras | executable + historical |
| 4 | 9-entry ensemble-stacking experiment (SMOTE listed twice) — Keras + LR meta-learner | executable + historical; serves detection models |
| 6, 7, 9, 11–19 | earlier CNN–BiLSTM benchmark implementations (cell 7: outputs without code) | historical only |
| 22 | corrected dual-dataset CNN–BiLSTM benchmark (label-matched pairing, CPU-feasible) — PyTorch | executable (no saved outputs in notebook) |

Details, pairing assumptions and methodological concerns: [docs/NOTEBOOK_EXPERIMENT_AUDIT.md](docs/NOTEBOOK_EXPERIMENT_AUDIT.md). Validation results: [docs/EXECUTION_REPORT.md](docs/EXECUTION_REPORT.md).

## How fidelity is enforced

* Notebook classes/functions are copied **verbatim** (`scripts/generate_notebook_modules.py`) and AST-compared with the notebook (`tests/test_notebook_fidelity.py`).
* Cell 22's refactor is proven equal to the unmodified cell by execution (`scripts/validate_cell22_refactor.py`).
* Historical numbers are **parsed** from the notebook's saved outputs (`scripts/build_historical_registry.py` → `research_results/corrected_notebook_historical.json`); nothing is typed in.
* New results live in `experiment_runs/run_XXXX/` with parameters, deviations, environment, dataset fingerprints and the notebook hash. Historical and new results are shown separately and compared only in an explicit "Historical vs reproduced" view.

## Pages

Research Overview · Experiment Laboratory · Results & Analytics · Notebook Outputs (Historical) · Malware Detection · Explainable AI · Dataset Explorer · Reproducibility & Audit · Methodology & Limitations

## Requirements

* **Python 3.11** (validated with CPython 3.11.9).
* TensorFlow-CPU 2.20 (cells 2/4, detection) and PyTorch 2.6 CPU (cell 22) — see `requirements.txt`.
* Datasets (not in git) in `others files that can help/` or `AMSEC_DATA_DIR`:
  `NF-UQ-NIDS-Dataset.csv`, `dynamic_api_call_sequence_per_malware_100_0_306.csv`, `feature_vectors_syscallsbinders_frequency_5_Cat.csv`.

```bat
py -3.11 -m venv .venv311
.venv311\Scripts\python -m pip install -r requirements.txt
start_amsec.bat
```

## Running experiments from the command line

```bat
.venv311\Scripts\python scripts\run_notebook_experiment.py --experiment cell22_dual_dataset
.venv311\Scripts\python scripts\run_notebook_experiment.py --experiment cell04_stacking9
.venv311\Scripts\python scripts\run_notebook_experiment.py --experiment cell02_lightweight16 --select 5,6
.venv311\Scripts\python scripts\run_notebook_experiment.py --experiment cell04_stacking9 --smoke
```

Run kinds: `full_reproduction` (notebook parameters, all configurations), `partial` (subset of configurations), `smoke` (reduced parameters — plumbing only, never a result).

## Validation

```bat
.venv311\Scripts\python -m pytest -q
.venv311\Scripts\python scripts\validate_cell22_refactor.py
.venv311\Scripts\python scripts\build_historical_registry.py
```

## Layout

| Path | Role |
|---|---|
| `notebooks/` | the authoritative notebook |
| `src/amsec/notebook/` | faithful modules (generated verbatim definitions + pipelines), provenance |
| `src/amsec/lab/` | run management |
| `src/amsec/detection.py` | serving of Keras models trained by runs |
| `src/amsec/dashboard/` | Streamlit pages |
| `research_results/` | historical registry + notebook figures |
| `experiment_runs/` | auditable runs |
| `docs/extracted_corrected/` | per-cell source and text output extracted from the notebook |
| `archive/pre_corrected_notebook/` | superseded notebooks, code, docs and model bundle (not used) |

## Deployment

Not deployed or pushed automatically. On Streamlit Community Cloud select **Python 3.11** in Advanced settings (see `docs/STREAMLIT_CLOUD_PYTHON.md`).
