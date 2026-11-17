# AMSEC source audit

Audit date: 2026-11-16. Nothing was pushed and nothing was deployed.

The numbers below are copied from notebook outputs or counted from local files. They are not a new experiment.

## What is in the workspace

| Path | Role |
| --- | --- |
| `others files that can help/Final phd-Experiment-android-malware.ipynb` | 23 cells. This is the notebook that contains the final stacking run. |
| `others files that can help/notebook48605fb318 (3).ipynb` | 11 cells. A prefix of the same notebook (cells 0–10). It does not contain the final CNN–BiLSTM cell. |
| `others files that can help/notebook48605fb318%20(3).html` | HTML render of that shorter notebook. |
| `others files that can help/dynamic_api_call_sequence_per_malware_100_0_306.csv` | API-call sequences. |
| `others files that can help/feature_vectors_syscallsbinders_frequency_5_Cat.csv` | Five-class system-call and binder frequencies. |
| `others files that can help/NF-UQ-NIDS-Dataset.csv` | Local network-flow table. The notebooks load a Kaggle file named `NF-UQ-NIDS-v2.csv`. |
| `AMSEC_Streamlit_Rebuild.zip` (project parent and Downloads) | The earlier evidence-only dashboard: `app.py`, `requirements.txt`, `data/verified_results.json`, one test. No model weights. |
| `upstream/` | Shallow clone of `mctelex-lab/HYBRID-ANDROID-MALWARE-DETECTION`. Not modified. |

## Which experiment is the final ensemble

Cell 4 of `Final phd-Experiment-android-malware.ipynb` is the final AMSEC ensemble.

It ranks nine stacking runs by F1. Rank 1 is `SMOTE-ENN_GA_Ensemble_Stacking`:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.97789 |
| Precision | 0.97795 |
| Recall | 0.97789 |
| F1 | 0.97789 |
| Balanced accuracy | 0.97775 |
| MCC | 0.95582 |
| ROC-AUC | 0.99767 |
| Parameters | 75,004 |
| FLOPs | 150,008 |
| Training seconds | 326.63078 |
| Reported inference seconds | 7.15040 |

FLOPs in that notebook are `2 * parameter count`, not a profiled multiply-add count.

The base learners are a MobileNetV4-style Conv1D network (62,710 parameters) and a lightweight Conv1D network (12,294 parameters). Logistic regression is the meta-learner. Its inputs are the four softmax probabilities, in the order `[mobile class 0, mobile class 1, lightweight class 0, lightweight class 1]`. Class 0 is normal and class 1 is attack, because the notebook binarised the flow label with `label > 0`. The decision used at evaluation is `argmax`, which is a 0.5 threshold on the attack probability when the class order is `[0, 1]`.

TensorFlow in that run was 2.20.0, on two Tesla T4 GPUs.

### What that run actually trained on

The executed cell loaded 150,000 rows of NF-UQ-NIDS-v2: 49,414 normal and 100,586 attack, 45 feature columns after the label was removed. Variance threshold left 41 features. Mutual information was asked for `min(100, 41)` features, so it kept all 41. The genetic algorithm then kept 5. **The five column names were never printed.**

The API CSV was not found. Phase 8 printed `API dataset not available`. The published 0.97789 score is a network-flow score. It is not a paired API-plus-network score.

Saved weights were written only to `/kaggle/working/models/mobilenetv4_*.h5` and `lightweight_*.h5`. The logistic meta-learner was not serialised. None of those files are in this package.

## Other experiments, kept separate

**Cell 2** is an ablation, not the stack. Sixteen runs cross eight samplers with `MobileNetV4` (62,710 parameters) and `MobileNetV4+Attention` (87,990 parameters). The best row is SMOTE-ENN with attention: accuracy 0.97059, F1 0.97058, balanced accuracy 0.97039, ROC-AUC 0.99787. The full table is in `data/verified_results.json`.

**Cell 22** is the CNN–BiLSTM benchmark (1,108,931 parameters, 16.63 million FLOPs in the notebook's own units). It encodes API calls to length 80 and selects five API columns by mutual information: `t_86`, `t_82`, `t_90`, `t_92`, `t_75`. NF-UQ-NIDS is loaded (200,000 rows) and then not used in the alignment. The split is 30,713 / 6,581 / 6,582 from the 43,876 API rows. ADASYN has the highest F1 (0.9916) and a false-positive rate of 0.6111. No-balancing accuracy is 0.9644 with balanced accuracy 0.9366. SMOTE-ENN in this benchmark falls to accuracy 0.0255. Those figures must not be merged with the cell-4 stack.

Cells 6, 7 and 9 are earlier TRIM-SEC runs. Several of them report accuracy 1.0 after aligning independent tables by row position and changing the label counts (for example 14,532 / 29,344). They are not the final ensemble.

`notebook48605fb318 (3).ipynb` stops before cell 22. It repeats cells 2, 4, 6, 7 and 9.

## Dataset schemas

Counts below are what the notebooks printed, or what an earlier local count recorded. The test suite recomputes the local CSV counts.

| File | Notebook or prior count | Label | Role in the final ensemble |
| --- | --- | --- | --- |
| `dynamic_api_call_sequence_per_malware_100_0_306.csv` | 43,876 rows, 102 columns. Malware 42,797, benign 1,079. | `malware` | Not loaded by the executed cell-4 run. Used by the cell-22 benchmark. |
| `feature_vectors_syscallsbinders_frequency_5_Cat.csv` | 11,598 rows, 471 columns. Classes 1–5. | `Class` | Not used by cell 4 or cell 22. |
| `NF-UQ-NIDS-Dataset.csv` | Prior dashboard recorded 1,048,575 rows and 46 columns, which is the Excel row ceiling. The notebook's Kaggle file was named `NF-UQ-NIDS-v2.csv`. | `Label` | Source family for the final ensemble. The notebook used the first 150,000 rows only. |

The nine names previously listed as "selected NIDS features from the thesis" (`MIN_IP_PKT_LEN`, `PROTOCOL`, and seven others) are **not** the five features chosen in cell 4. Cell 4 did not print names. Those nine names are not used as model inputs.

## GitHub model artifacts (older, not AMSEC cell 4)

`upstream/Models/` contains:

- `enhanced_model.h5`
- `enhanced_model_best.h5`
- `enhanced_aggregate_results.json`
- `scaler.pkl`
- `label_encoder.pkl`
- `selected_features.pkl`
- `api_optimal_threshold.pkl`

`enhanced_aggregate_results.json` records CICMalDroid accuracy 0.9290 and API-sequence accuracy 0.5. The Streamlit file in that repository ignores this JSON unless it sits on a Kaggle path, and otherwise displays hardcoded defaults (aggregate accuracy 0.9530, CICMalDroid 0.9847). Prediction buttons call `_predict_cicmal` and `_predict_api`, which are averages of feature slices, not a loaded network. The analytics page draws learning curves, confusion matrices and ROC curves from `numpy` random generators.

Those files are retained in `upstream/` for inspection. The local console does not load them and does not call those formulas.

## Missing dependencies for the original training stack

- TensorFlow 2.20 / Keras, required to reload the missing `.h5` files. `pip` reports no TensorFlow distribution for the installed CPython 3.14.
- The Kaggle output directory: base-model HDF5 files, the unsaved logistic regression, GA feature indices, scalers, and `lrp_results_best.json`.
- A record of the five selected flow-feature names.
- Paired API and network rows that share an application id.
- `thop` was missing in some earlier cells; cell 22 installed it. FLOPs there are still the notebook's own estimate.

## Leakage that is documented and not silently "fixed" in the historical tables

- Cell 4 fits `RobustScaler` and `PowerTransformer` before the split.
- Cell 4 resamples with SMOTE-ENN (and the other samplers) before the split.
- Cell 4 selects features with mutual information and the genetic algorithm on rows that are later split.
- The meta-learner is fit on in-sample base predictions.
- Cell 22 fits mutual information and `StandardScaler` before the split.
- Earlier cells align unrelated tables by row number.

The historical metrics in `data/verified_results.json` are unchanged. The local serving model, if trained, fits selection and resampling on the training fold only and is labelled as a retrain.

## Second notebook and HTML

`notebook48605fb318 (3).ipynb` and the HTML file do not add a later ensemble. They are an earlier export of the same Kaggle session.

## Local follow-up (after this audit)

A leakage-corrected PyTorch retrain was produced under `artifacts/` and documented in `docs/LOCAL_VALIDATION_REPORT.md`. It is labelled `local_retrain_not_kaggle_weights` and is **not** a claim that the notebook 0.97789 result was reproduced.
