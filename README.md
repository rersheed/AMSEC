# AMSEC local console

Local research console for the AMSEC notebooks. Predictions are produced only by a model trained in this workspace. The older GitHub dashboard's hand-written scoring functions are not used.

The notebook's Keras weights were not in the source package, and TensorFlow does not install on this machine's Python 3.14. `scripts/train_local_ensemble.py` trains the cell-4 layer stack in PyTorch, writes `artifacts/`, and records that the run is not the Kaggle 0.97789 result.

## Run

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\train_local_ensemble.py
start_amsec.bat
```
 
The console listens on http://localhost:8501.

See `docs/LOCAL_VALIDATION_REPORT.md` and `docs/SOURCE_AUDIT.md` for validation notes. Large CSV datasets under `others files that can help/` are kept local and are not committed.
