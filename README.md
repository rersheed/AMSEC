# AMSEC

**AMSEC — An Explainable Lightweight Hybrid Deep Learning Framework for Android Malware Detection.**

Local Streamlit research console for the AMSEC PhD work. It ships two operating modes:

1. **Research Dashboard** — overview, analytics, single/batch detection, dataset explorer, model comparison, resampling experiments, explainability, methodology, and limitations.
2. **PhD Viva Presentation Mode** — a 21-slide viva deck with speaker notes and a live detection hand-off that reuses the same inference path.

Predictions come only from a model trained in this workspace (PyTorch cell-4 stack retrain). Historical notebook metrics are shown as transcribed evidence and are **not** claimed as the served model’s scores.

Repository: https://github.com/rersheed/AMSEC

## Features

- Switchable Research Dashboard / Viva Presentation modes
- Research Overview and Analytics (Plotly)
- Malware Detection and Batch Detection (trained ensemble)
- Dataset Explorer with schema/profile checks
- Model Comparison and Resampling Experiments views
- Explainability panel (feature-level scores from the served pipeline)
- Local artifact bundle under `artifacts/` (preprocessor, MobileNetV4 / lightweight towers, meta-learner, evaluation JSON)
- Verified research registry (`research_results/`, `data/verified_results.json`)

## Methodology (summary)

The notebook final ensemble (`SMOTE-ENN_GA_Ensemble_Stacking`, cell 4 of `Final phd-Experiment-android-malware.ipynb`) reported accuracy ≈ **0.97789** on a Kaggle run. Those Keras `.h5` weights are **not** in this package.

This console retrains an equivalent **PyTorch** layer stack locally via `scripts/train_local_ensemble.py`, writes `artifacts/`, and serves that bundle. Local accuracy is lower (~0.966 on the local protocol) and must not be presented as the notebook figure. See `docs/LOCAL_VALIDATION_REPORT.md` and `docs/SOURCE_AUDIT.md`.

## Requirements

- Python **3.11** recommended (`.python-version`). Local validation also ran on Python 3.14 with the pinned PyTorch stack.
- TensorFlow is **not** required for the served console.
- Large CSV datasets under `others files that can help/` stay local and are gitignored.

## Installation

```bat
git clone https://github.com/rersheed/AMSEC.git
cd AMSEC
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

If `artifacts/` is incomplete on a fresh machine, retrain:

```bat
.venv\Scripts\python scripts\train_local_ensemble.py
```

Place research CSVs (optional for Dataset Explorer) under `others files that can help/` without committing them.

## Run

```bat
start_amsec.bat
```

Or:

```bat
.venv\Scripts\python -m streamlit run app.py
```

Open http://localhost:8501

Production-oriented Streamlit settings live in `.streamlit/config.toml`. Do not commit `.streamlit/secrets.toml`.

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit entry (mode router + dashboard pages) |
| `src/amsec/` | Inference, datasets, research registry, dashboard/viva UI |
| `artifacts/` | Served local PyTorch ensemble + evaluation metadata |
| `research_results/` | Catalog + viva slide content |
| `scripts/` | Train, audit, viva checks |
| `tests/` | Integrity, science schema, inference parity |
| `docs/` | Validation report, source audit, extracted notebook evidence |

## Tests

```bat
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\check_viva.py
```

## Model artifacts

| Artifact | Status |
| --- | --- |
| Local PyTorch bundle in `artifacts/` | Included (small; required for inference) |
| Original Kaggle Keras `.h5` ensemble | **Missing** — obtain only from the authorised Kaggle/`/kaggle/working/models` run; not redistributed here |
| Large NIDS / API CSVs | Local only; see validation report for schemas |

Do not enable Git LFS unless the deployment target (e.g. Streamlit Community Cloud) is confirmed to support it.

## Deployment notes

- This repository push does **not** deploy to Streamlit Community Cloud.
- For a future cloud deploy: pin Python 3.11, ensure `artifacts/` is present or rebuildable, set secrets only via Streamlit secrets / env vars, and keep CSVs out of the repo.

## Licence / academic integrity

Treat notebook metrics and local retrain metrics as separate evidence streams. Do not substitute local scores for the published notebook figures.
