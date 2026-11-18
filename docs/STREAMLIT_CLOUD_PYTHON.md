# Streamlit Community Cloud — Python 3.11

Community Cloud ignores `runtime.txt` and `.python-version`. Select the interpreter explicitly:

1. Deploy (or delete + redeploy) the app at https://share.streamlit.io
2. **Advanced settings → Python version = 3.11**
3. Deploy `app.py`

Docs: [Upgrade your app's Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)

## Dependency policy

* The authoritative notebook uses **TensorFlow/Keras 2.20** (cells 2 and 4) and **PyTorch** (cell 22). AMSEC runs the notebook's own architectures, so both are required: `tensorflow-cpu==2.20.0` and `torch==2.6.0` (CPU wheel index).
* TensorFlow 2.20 has no wheels for Python 3.14 — this is why Python 3.11 must be selected.
* `scikit-learn==1.9.1` must match the version that pickled the run artifacts (`preprocessing.joblib`, `meta_learner.joblib`).
* Both frameworks together make a large image (~1 GB installed). Deployment has **not** been validated on Community Cloud; only local Windows / Python 3.11.9 is validated (see `docs/EXECUTION_REPORT.md`).

## What a deployment needs from the repository

* `notebooks/Final_phd-Experiment-android-malware_CORRECTED.ipynb` (hash-verified at start-up)
* `research_results/corrected_notebook_historical.json` + `research_results/notebook_figures/`
* `experiment_runs/run_XXXX/` for completed runs (results + Keras models used by Malware Detection)
* The CSV datasets are **not** in git; on Cloud the Laboratory and Dataset Explorer therefore cannot run, while historical results, completed-run analytics and detection work.
