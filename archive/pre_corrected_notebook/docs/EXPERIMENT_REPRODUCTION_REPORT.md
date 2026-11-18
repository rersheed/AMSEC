# Experiment reproduction report

**Date:** 2026-11-17  
**Notebook:** `others files that can help/Final phd-Experiment-android-malware.ipynb`  
**Application:** local AMSEC Streamlit console  
**GitHub:** https://github.com/rersheed/AMSEC (not redeployed in this workstream)  
**Cloud deploy:** not performed

---

## 1. Source notebook audit

See `docs/NOTEBOOK_EXPERIMENT_AUDIT.md` for the full inventory.

Highlights:

| Family | Cell | Configs | Notes |
| --- | ---: | ---: | --- |
| MobileNetV4 ± Attention | 2 | 16 | 8 samplers × 2 models |
| Ensemble stacking | 4 | 9 | **SMOTE duplicated** intentionally |
| CNN–BiLSTM | 22 | 9 balancers | PyTorch dual-input; NIDS loaded unused |

Authoritative final ensemble: cell 4 `SMOTE-ENN_GA_Ensemble_Stacking` (stored accuracy 0.97789).

---

## 2. Implemented experiment engines

Package layout under `src/amsec/`:

- `data/` — configurable loaders, hashing, scaling, alignment policies  
- `feature_selection/` — MI + genetic algorithm (notebook hyperparameters)  
- `resampling/` — eight strategies + nine-entry stacking list with duplicate SMOTE  
- `notebook_models/` — Keras builders (optional TF) + PyTorch ports + stacking trainer + CNN–BiLSTM defs  
- `experiments/` — `notebook_reproduction`, `leakage_controlled`, runner, subprocess worker  
- `evaluation/` — metrics, curves, complexity, explainability helpers  
- `artifacts/` — `experiment_runs/run_XXXX` storage + registry  

Streamlit:

- **Experiment Laboratory** — configure / run / cancel / reload  
- **Advanced Research Analytics** — plots from saved run artifacts when a run is selected  
- **Experiment Reproducibility** — notebook stored metrics vs new execution  
- Single-mode research UI only (Viva Presentation Mode removed)  

---

## 3. Actually executed experiments (this machine)

| Run ID | Protocol | Sampling | nrows | epochs | Status | Accuracy | F1 |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| `run_0001` | leakage_controlled | SMOTE | 4000 | 1 | **completed** | 0.86833 | 0.87103 |

Artifacts written under `experiment_runs/run_0001/` (configuration, metadata, metrics, history, predictions, confusion, ROC/PR, GA fitness, selected features, complexity, model weights).

**Not claimed:** full 9-config stacking matrix, full 16-config MobileNet matrix, or cell-22 CNN–BiLSTM matrix. Those require deliberate long batch runs.

---

## 4. Original vs reproduced

| Metric | Notebook SMOTE-ENN (stored) | `run_0001` smoke (leakage-controlled SMOTE) | Comparable? |
| --- | ---: | ---: | --- |
| Accuracy | 0.97789 | 0.86833 | **No** — different protocol, sampler, nrows, epochs, backend |
| F1 | 0.97789 | 0.87103 | No |
| Features | 5 unnamed | 10 named (smoke GA) | No |

A fair notebook-reproduction comparison requires: `protocol=notebook_reproduction`, `sampling=SMOTE-ENN`, `nrows=150000`, `epochs=30`, GA pop/gen 40/30, and ideally TensorFlow 2.20. That full run was **not** completed here.

---

## 5. Methodological differences (labelled)

1. **Backend:** PyTorch notebook port (`backend=pytorch_notebook_port`) because TensorFlow has no wheel for local Python 3.14. Keras builders are present and activate when TF is installed.  
2. **Data file:** local `NF-UQ-NIDS-Dataset.csv` vs Kaggle `NF-UQ-NIDS-v2.csv`.  
3. **Smoke profile:** reduced rows/epochs/GA for CI-scale validation.  
4. **Leakage-controlled mode** deliberately differs from notebook order (split-first).  
5. Cell-22 and MobileNet ablation **training runners** are scaffolded; Laboratory currently trains `ensemble_stacking` only.

---

## 6. Dataset compatibility

| Dataset | Local path | Validated |
| --- | --- | --- |
| NIDS | `others files that can help/NF-UQ-NIDS-Dataset.csv` | Yes (loader + smoke) |
| API | `.../dynamic_api_call_sequence_per_malware_100_0_306.csv` | Present; not used by stacking smoke |
| Syscalls | `.../feature_vectors_syscallsbinders_frequency_5_Cat.csv` | Present; not used by final ensemble |

Alignment policy: stacking = NIDS-only; cell 22 = API self-truncate; independent row-joins forbidden.

---

## 7. Model artifact status

| Artifact | Status |
| --- | --- |
| Kaggle Keras `.h5` bases | **Missing** |
| Local serving bundle `artifacts/` | Present (prior leakage-corrected retrain) |
| Laboratory `experiment_runs/run_0001/models/` | Present (smoke weights) |
| Notebook LRP JSON | Missing |

---

## 8. Tests performed

| Test | Result |
| --- | --- |
| `pytest` (science, inference, integrity, experiment engine) | **16 passed** |
| Smoke E2E `scripts/run_experiment.py --smoke` | **completed** `run_0001` |
| Architecture param counts @ 5 inputs | 62710 / 12294 |
| Duplicate SMOTE list integrity | Pass |
| Streamlit pages wired | Laboratory + Reproducibility + Analytics run artifacts |

---

## 9. Remaining issues / blockers

- No TensorFlow on Python 3.14 → cannot execute bit-identical Keras graphs locally.  
- Full matrix training needs hours of CPU and large RAM for SMOTE-ENN on 150k rows.  
- MobileNet ablation and CNN–BiLSTM Laboratory runners still raise `NotImplementedError` until dedicated runners are finished.  
- Original Kaggle weights unavailable → cannot reload historical HDF5 ensemble.  
- Streamlit Community Cloud: keep training disabled by default; view saved runs + inference only.

---

## 10. Readiness

| Target | Ready? |
| --- | --- |
| Local research use (Laboratory smoke + detection + analytics from artifacts) | **Yes** |
| Claim full notebook metric reproduction | **No** |
| Streamlit Cloud deploy | **Not yet** (per project rule; needs Python 3.11 image, artifact policy, no auto-train) |

Do not push until the researcher reviews smoke results and optionally launches a full `notebook_reproduction` SMOTE-ENN run.
