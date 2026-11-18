# Notebook experiment audit — `Final_phd-Experiment-android-malware_CORRECTED.ipynb`

**Sole authoritative experimental source for AMSEC.**

| Item | Value |
|---|---|
| File | `notebooks/Final_phd-Experiment-android-malware_CORRECTED.ipynb` |
| SHA-256 | `9e4724f7af4a8438bf7aea02b8768cc1579a50afc0be935b1f24b08581a0e8fa` |
| Kernel | Python 3.12.13 (Kaggle image); TensorFlow 2.20.0 printed by cells 2 and 4; 2 GPUs listed |
| Cells | 23 (all code cells) |
| Superseded notebooks | `Final phd-Experiment-android-malware.ipynb`, `notebook48605fb318 (3).ipynb` → moved to `archive/pre_corrected_notebook/notebooks/`; not used anywhere in the active pipeline |

## What changed relative to the superseded notebook

Cells 0–21 are byte-identical to the superseded `Final phd-Experiment-android-malware.ipynb` (source **and** saved outputs). Only **cell 22** differs: the old 827-line Keras/Torch CNN–BiLSTM cell (with saved outputs) was replaced by a 156-line *corrected dual-dataset* PyTorch benchmark **with no saved outputs**. Consequently every "cell 22" number used by earlier AMSEC builds (e.g. ADASYN accuracy 0.9836, 1,108,931 parameters) came from code that no longer exists in the authoritative notebook and has been removed from AMSEC (`archive/pre_corrected_notebook/data/verified_results.json`).

## Cell map

| Cell | Role | Framework | Saved outputs | AMSEC status |
|---|---|---|---|---|
| 0 | Kaggle boilerplate | – | none | ignored |
| 1, 3, 5, 8, 10, 12, 14, 16, 18, 20, 21 | empty | – | none | ignored |
| **2** | **Lightweight-model experiment, 16 configurations** (8 samplers × MobileNetV4 / MobileNetV4+Attention) | TF/Keras | 148 outputs, 8 figures, ranked table | **Executable** (`amsec.notebook.keras_pipeline`, `cell02`) + historical |
| **4** | **Ensemble-stacking experiment, 9 entries** (8 samplers + SMOTE listed twice; MobileNetV4 + Lightweight → LogisticRegression) | TF/Keras + sklearn | 34 outputs, 8 figures, ranked table | **Executable** (`cell04`) + historical |
| 6 | TRIM-SEC CNN–BiLSTM benchmark A (separate API / NIDS evaluations, 10 epochs) | PyTorch | 29 outputs, 8 figures | historical only |
| 7 | **Empty source** with 29 saved outputs (execution count 7) | unknown | tables differ from cell 6 | historical, flagged **untraceable** |
| 9 | TRIM-SEC CNN–BiLSTM benchmark B ("Mutambik 2025"; up to 500k NIDS rows; 30 epochs) | PyTorch | 31 outputs, 9 figures | historical only |
| 11, 13, 15, 17, 19 | Five successive CNN–BiLSTM (Torch) variants: "No Balancing vs all balancing", progressively reduced capacity/epochs; 15+ rank by F1; 17 adds specificity/FPR/FNR; 19 adds params/FLOPs/latency | PyTorch | 31–33 outputs each | historical only |
| **22** | **Corrected TRIM-SEC dual-dataset CNN–BiLSTM benchmark** (label-matched cross-dataset pairing, CPU-feasible sampling) | PyTorch | **none** | **Executable** (`amsec.notebook.cell22_dual_dataset`) |

The CNN–BiLSTM cells are successive *implementations* of one benchmark idea, not independent replications; AMSEC keeps each cell's results separate and never merges or cross-ranks them.

## Executable experiments — what AMSEC runs

### Cells 2 and 4 (shared pipeline, TensorFlow/Keras)

Definitions are copied **verbatim** by `scripts/generate_notebook_modules.py` into `keras_common.py`, `cell02_models.py`, `cell04_models.py`, `cell02_api.py`, `cell04_api.py`; `tests/test_notebook_fidelity.py` asserts AST equality with the notebook. `GeneticFeatureSelector`, `LRPExplainer`, `optimize_memory`, `get_model_flops`, `set_seed` are identical in both cells; `apply_sampling_method` differs only by a docstring.

`keras_pipeline.run_keras_experiment` replays the top-level script in notebook order (single `set_seed(42)`; global NumPy RNG consumed identically for the full configuration list):

1. NF-UQ-NIDS first 150,000 rows in 50,000-row chunks → `optimize_memory` (int down-casting, float16 where in range) → binary label `Label > 0`.
2. Numeric columns → `VarianceThreshold(1e-5)` → 20,000-row random sample → `RobustScaler` → `SelectKBest(mutual_info_classif, k=min(100, n))` → GA (population 40, 30 generations, mutation 0.12, crossover 0.9; fitness = (0.55·acc + 0.45·weighted-F1)·(1 − 0.06·k/n) with a RandomForest(100, depth 14) on a 75/25 split).
3. `RobustScaler` + Yeo-Johnson `PowerTransformer` on the full selected matrix; `nan_to_num(0, ±5)`.
4. Per entry: resample full matrix → cap 180,000 rows (global RNG) → 85/15 stratified split → 88/12 train/val → train (30 epochs, batch 128, AdamW, EarlyStopping, ReduceLROnPlateau, class weights) → evaluate (weighted P/R/F1, balanced accuracy, MCC, weighted OvR ROC-AUC).
5. Rank by weighted F1 (`rank(method='min')`).
6. Phases 7–9: best entry → re-train the same architecture on an **un-resampled** 80/20 split (test split passed as validation data) → API evaluation (`process_api_dataset`) → LRP on 3 random test rows.

Only file-system side effects are redirected (Kaggle `/kaggle/working` paths). Additional artifacts (history, test probabilities, confusion matrices, selected feature **names**, fitted preprocessing, Keras models, meta-learner) are written without touching the computation.

**Cell-4 nine-entry list (preserved):** `SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek, SMOTE-NCR, ADASYN-ENN, ADASYN-Tomek, ADASYN-NCR, SMOTE`. Entries 1 and 9 are both SMOTE and are kept as two separate entries; the historical registry maps each table row to its entry using the per-configuration log (entry 1 F1 0.95596, entry 9 F1 0.95421).

**API evaluation.** In the saved run the API file path was missing (`API dataset not available`), so phase 8's API evaluation never executed. AMSEC defaults to the same state (`api_dataset_available=False`); enabling it is a recorded deviation.

### Cell 22 (corrected dual-dataset benchmark, PyTorch)

`PairDataset`, `CNNBiLSTM`, `DualInputModel`, `apply_balancing`, `evaluate` are verbatim (AST-tested). `run_cell22` is the cell script with paths parameterised. **Equivalence is proven by execution:** `scripts/validate_cell22_refactor.py` runs the unmodified cell source (only the `BASE`/`OUT` literals substituted) and the module on the same machine; all metrics for all nine methods are identical (`experiment_runs/_validation/cell22_refactor/refactor_equivalence.json`).

> A first validation attempt found the module's extra test-probability pass shifted torch's global RNG (creating a `DataLoader` iterator draws a base seed even with `shuffle=False`), changing every method after the first. The pass now saves/restores the RNG state; equivalence is exact.

**Pairing assumptions (documented, not altered).**

* The API-call file and NF-UQ-NIDS share **no sample/application identifier**.
* API subset: seeded (`default_rng(42)`) draw of 200 benign + 1,000 malware rows from 43,876 (1,079 benign available), first 80 of 100 tokens.
* NIDS: first 50,000 rows; for each API row one NIDS row **with the same binary label** is drawn without replacement (`default_rng(42)`), 1,200 unique NIDS rows.
* The paired NIDS row thus carries the API row's label by construction. The static branch learns NIDS attack/normal separability; the model is not observing two modalities of the same application. Results characterise a synthetic fusion, not app-level multimodal detection.
* Split 70/15/15 stratified **before** MI (top-5 NIDS features on the training split, ≤5,000 rows) and `StandardScaler` (fit on train). The API token vocabulary is built from all 1,200 rows (unsupervised).
* Resampling acts on the 5 static features; resampled rows that exactly match an original keep its sequence, synthetic rows get a random same-class sequence. Resampled training sets are **not shuffled** (`shuffle=(method=='no_balancing')`).
* Method names vs. behaviour: `adasyn_enn` = ADASYN→SMOTEENN, `adasyn_tomek` = ADASYN→SMOTETomek, `smote_ncr` = SMOTE→NeighbourhoodCleaningRule (a real NCR, unlike cells 2/4).

**How it differs from the other benchmark sections.** Cells 11–19 load NF-UQ-NIDS but build the static branch from API `t_*` columns (NIDS unused); cells 6/7/9 evaluate API and NIDS separately. Cell 22 is the only benchmark in which NIDS features enter the model, and the only one designed for CPU.

## Methodological concerns (documented; reproduction procedure unchanged)

Cells 2/4
1. Scaling, MI and GA are fitted before the train/test split; resampling is applied to the full matrix before splitting — synthetic neighbours of test points can be in training, and test sets are balanced rather than reflecting the 33/67 source ratio.
2. `SMOTE-NCR`/`ADASYN-NCR` call **NearMiss-1**, not the Neighbourhood Cleaning Rule. After SMOTE the classes are equal, so NearMiss removes nothing: SMOTE-NCR trains on the same resampled set as SMOTE (both 201,172 rows). The F1 spread across entries 1, 5 and 9 (0.95421–0.95596) is therefore a direct estimate of run-to-run noise.
3. Phase 8 uses the test split as Keras validation data (early stopping / LR schedule).
4. Cell-4 meta-learner fitted on in-sample base-model predictions (not out-of-fold).
5. "FLOPs" = 2 × parameters.
6. GA-selected feature names were never printed; AMSEC records them.
7. Phase-8 API evaluation (if enabled) feeds 11 summary statistics of token IDs, padded/truncated to 5 slots, through the NIDS scaler/model and picks the threshold on the evaluation labels.

Cell 22 — see pairing assumptions above; small test set (180 rows, 30 benign) → wide uncertainty; 3 epochs.

Cells 6–19 — near-perfect (cells 6/7/9) or degenerate (F1≈0.99 with specificity 0; F1 = 0 for SMOTE+ENN/ADASYN+ENN in cell 19) results; cell 7 has no code.

## Dataset provenance

| File | Rows | SHA-256 (prefix) | Notes |
|---|---|---|---|
| `NF-UQ-NIDS-Dataset.csv` | 1,048,575 | `b1641766…` | Truncated copy of NF-UQ-NIDS-v2 (exactly Excel's row limit). First 150,000 rows: normal 49,414 / attack 100,586 = notebook's printed counts. |
| `dynamic_api_call_sequence_per_malware_100_0_306.csv` | 43,876 | `11005ff6…` | benign 1,079 / malware 42,797 |
| `feature_vectors_syscallsbinders_frequency_5_Cat.csv` | – | – | not used by any executable experiment |

Full fingerprints are recorded in every run's `run.json`.
