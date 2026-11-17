# AMSEC local validation report

Date: 2026-11-16 / 2026-11-17 (local Windows).  
Remote GitHub and Streamlit Cloud were **not** modified.

## Verdict

The local console runs at `http://localhost:8501` with **real trained-model inference**.  
Notebook historical metrics are shown only as transcribed evidence. The served weights are a **local PyTorch retrain**, not the missing Kaggle HDF5 ensemble.

| Area | Status |
| --- | --- |
| Source audit | **PASS** |
| Final ensemble identification (cell 4 SMOTE-ENN stack) | **PASS** |
| Heuristic prediction removed from live detection | **PASS** |
| Local model training + artifact bundle | **PASS** (local retrain; not Kaggle weights) |
| Inference parity vs saved batch | **PASS** |
| Automated tests (`pytest`) | **PASS** (9/9) |
| Streamlit startup on :8501 | **PASS** |
| Browser navigation + single-sample score | **PASS** |
| Browser CSV file-picker upload gesture | **BLOCKED** (widget present; automation could not drive Streamlit’s native file dialog; batch scoring verified by CLI on the same CSV) |
| Reload of original Kaggle `.h5` stack | **BLOCKED** (weights absent; TensorFlow unavailable on Python 3.14) |
| Reproduction of notebook 0.97789 accuracy | **BLOCKED** (cannot claim without those weights / same protocol) |

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 10 (build 26200) |
| Python | 3.14.7 |
| Virtualenv | `.venv` |
| Start command | `python -m streamlit run app.py` or `start_amsec.bat` |
| URL | http://localhost:8501 |
| Hardware used for training | CPU (8 logical processors, ~32 GB RAM) |

Pinned console packages are in `requirements.txt`. Full freeze: `docs/pip_freeze.txt`.

Notable installed versions:

- torch 2.15.0  
- streamlit 1.67.0  
- scikit-learn 1.9.1  
- imbalanced-learn 0.14.2  
- pandas 3.1.1  
- numpy 2.5.5  
- plotly 7.1.0  
- pytest 9.2.0  

**TensorFlow:** not installable for this interpreter (`No matching distribution found for tensorflow`). The notebook used TensorFlow 2.20.0.

## Dataset audit

Files under `others files that can help/`:

| File | Local rows | Columns | Label | Role |
| --- | ---: | ---: | --- | --- |
| `dynamic_api_call_sequence_per_malware_100_0_306.csv` | 43,876 | 102 | `malware` (0=1,079; 1=42,797) | Cell-22 benchmark only |
| `feature_vectors_syscallsbinders_frequency_5_Cat.csv` | 11,598 | 471 | `Class` (1–5) | Not used by final ensemble |
| `NF-UQ-NIDS-Dataset.csv` | 1,048,575 | 46 | `Label` | Local source for cell-4 family; notebook used Kaggle `NF-UQ-NIDS-v2.csv` first 150k |

Verified by `tests/test_science.py` schema tests.

## Final research model (notebook)

From `Final phd-Experiment-android-malware.ipynb` cell 4:

- Config: `SMOTE-ENN_GA_Ensemble_Stacking`  
- Accuracy 0.97789, F1 0.97789, balanced accuracy 0.97775, MCC 0.95582, ROC-AUC 0.99767  
- Parameters 75,004; FLOPs reported as 150,008 (= 2 × params)  
- Fit on 150,000 NF-UQ-NIDS-v2 rows only; API CSV was **not** loaded in that executed run  
- Five GA features selected; **names never printed**  
- Weights saved only under `/kaggle/working/models/*.h5` — **not in this package**

Other experiments are labelled separately in `data/verified_results.json` (cell 2 MobileNet ablations; cell 22 CNN–BiLSTM).

## Model used for local inference

| Field | Value |
| --- | --- |
| Experiment id | `local_retrain_cell4_stack_smoteenn` |
| Status | `local_retrain_not_kaggle_weights` |
| Base learners | PyTorch MobileNetV4-style Conv1D + Lightweight Conv1D |
| Meta-learner | `LogisticRegression` on `[p_mobile_0, p_mobile_1, p_light_0, p_light_1]` |
| Threshold | 0.5 on attack probability |
| Features (19) | `L4_SRC_PORT`, `L4_DST_PORT`, `IN_PKTS`, `OUT_BYTES`, `TCP_FLAGS`, `FLOW_DURATION_MILLISECONDS`, `MIN_TTL`, `MAX_TTL`, `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `DST_TO_SRC_SECOND_BYTES`, `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_OUT_BYTES`, `DST_TO_SRC_AVG_THROUGHPUT`, `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `DNS_QUERY_ID`, `FTP_COMMAND_RET_CODE` |
| Artifacts | `artifacts/mobilenetv4.pt`, `lightweight.pt`, `meta_learner.joblib`, `preprocessor.joblib`, `model_manifest.json` |
| SHA-256 | recorded in `artifacts/model_manifest.json` |

### Local held-out metrics (clean split; not the notebook score)

| Metric | Value |
| --- | ---: |
| Accuracy | 0.96627 |
| Precision | 0.99305 |
| Recall | 0.95639 |
| F1 | 0.97437 |
| Balanced accuracy | 0.97138 |
| MCC | 0.92648 |
| ROC-AUC | 0.99473 |
| Confusion [[TN,FP],[FN,TP]] | [[7311,101],[658,14430]] |

Train/val/test sizes before resampling: 105,000 / 22,500 / 22,500.  
SMOTE-ENN on **train only** → 136,537 rows.

### Deviations from the notebook (documented, not silent)

1. PyTorch instead of TensorFlow 2.20 (TF unavailable on Python 3.14; Kaggle HDF5 absent).  
2. Selection, scaling, and SMOTE-ENN fit **after** the split (leakage corrected).  
3. GA shortened (population 12, generations 5, 25 trees, 8,000-row sample) → **19** features, not 5 unnamed ones.  
4. Epochs capped at 8.  
5. Parameter counts at 19 inputs: MobileNet-style 62,766; Lightweight 12,350 (notebook 5-input counts 62,710 / 12,294 still match the architecture test).

## Tests

```bat
.venv\Scripts\python.exe -m pytest tests -v
```

Result: **9 passed** in ~15 s.

Covered:

- Transcribed notebook metrics integrity (no GitHub 0.9847 defaults)  
- Architecture parameter counts at 5 inputs  
- Strict schema validation (missing / non-numeric / blank / empty)  
- GA determinism on a fixed toy matrix  
- API / syscall / NIDS dataset schemas  
- Manifest status and local-retrain claim  
- Malformed upload rejection  
- Parity batch: inference probabilities match `artifacts/parity_proba.npy` within 1e-5  

## Application / browser checks

| Check | Result |
| --- | --- |
| `streamlit run app.py` on localhost:8501 | **PASS** |
| Research Overview shows notebook 0.97789 and local 0.9663 separately | **PASS** |
| Malware Detection scores with trained stack (browser: attack, P=0.9953 on zero vector) | **PASS** |
| Batch Detection page + CSV uploader present | **PASS** |
| Batch scoring of `artifacts/parity_batch.csv` via inference API (32 rows, 20 attack) | **PASS** |
| Model Comparison tables / experiment separation | **PASS** |
| Dataset Explorer / Resampling / Explainability / Limitations pages reachable | **PASS** (navigated; Dataset Explorer reads live CSVs) |
| Heuristic `_predict_*` paths | **PASS** (not used by this app) |

## Remaining limitations

- No Android on-device package scanner.  
- Final notebook ensemble is network-flow only; API pairing by app id does not exist.  
- Notebook LRP JSON is unavailable; explainability page uses local sensitivity / meta coefficients and marks notebook LRP as Unavailable.  
- Upstream GitHub `enhanced_model*.h5` is an older unrelated dashboard artifact and is not loaded.  
- Cloud readiness: Torch + large NIDS CSV make Streamlit Community Cloud heavy; use a Python ≤3.12 image if TF replay is required later.

## Readiness for Streamlit Community Cloud

**Not ready to deploy yet** without an explicit go-ahead.

Blockers / decisions needed:

1. Confirm whether Cloud should ship the **local retrain** or wait for original Kaggle weights.  
2. Prefer a Python 3.11/3.12 runtime if TensorFlow reload is required.  
3. Decide whether the ~181 MB NIDS CSV belongs in the Cloud repo or is explored only locally.  
4. Human review of `docs/SOURCE_AUDIT.md` and this report.

No push and no Cloud deploy were performed.
