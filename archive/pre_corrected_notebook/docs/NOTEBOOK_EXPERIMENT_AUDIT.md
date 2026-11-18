# Notebook experiment audit

**Source notebook:** `others files that can help/Final phd-Experiment-android-malware.ipynb`  
**Also referenced as:** `Final phd-Experiment-android-malware(1).ipynb` (same content supplied by the researcher)  
**Audit date:** 2026-11-17  
**Cells:** 23 (code + empty placeholders). Full sources extracted under `docs/extracted/phd_cell_*`.

This audit describes **what the notebook actually implements**. Historical metrics are transcribed from stored outputs / `data/verified_results.json`. They are **not** regenerated numbers.

---

## Experiment inventory

| Cell | Family | Configurations | Backend in notebook | Primary dataset |
| ---: | --- | ---: | --- | --- |
| 2 | MobileNetV4 ± Attention ablation | **16** (8 samplers × 2 models) | TensorFlow / Keras | NF-UQ-NIDS-v2, first **150,000** rows |
| 4 | Ensemble stacking (MobileNetV4 + Lightweight + LR meta) | **9** (8 samplers + **duplicate SMOTE**) | TensorFlow / Keras + sklearn LR | NF-UQ-NIDS-v2, first **150,000** rows |
| 6–19 | Earlier TRIM-SEC / alignment variants | Multiple | Mixed | API ± NIDS (row-aligned in some cells) |
| 22 | CNN–BiLSTM dual-input benchmark | **9** balancing methods | **PyTorch** | API sequences (43,876); NIDS loaded (`nrows=200000`) but **not used in alignment** |

**Authoritative final AMSEC ensemble:** cell **4**, config `SMOTE-ENN_GA_Ensemble_Stacking`.

---

## 1. Sixteen-configuration MobileNetV4 / Attention (cell 2)

### Design
- Samplers (8): `SMOTE`, `ADASYN`, `SMOTE-ENN`, `SMOTE-Tomek`, `SMOTE-NCR`, `ADASYN-ENN`, `ADASYN-Tomek`, `ADASYN-NCR`
- Models (2): `MobileNetV4` (no attention) and `MobileNetV4+Attention` (`MultiHeadAttention`, 4 heads, `key_dim=16`)
- Total: **16** training runs, 30 epochs, batch 128, seed 42

### Architecture (Keras)
Shared Conv1D tower: 48 → 64 → 96 filters (k=3, same padding, L2 `5e-5`), GAP+GMP concat, Dense 128→64, softmax 2.  
Attention branch: reshape `(1,64)` → MHA → flatten → LayerNorm → concat → Dense 64 → softmax.  
Notebook parameter counts: **62,710** (plain) / **87,990** (attention). FLOPs printed as `2 × params`.

### Preprocessing / selection (notebook order — leakage present)
1. Load NIDS numeric features; `VarianceThreshold(1e-5)`
2. Optional 20k-row subsample for GA
3. `RobustScaler` → MI `SelectKBest(k=min(100, n))` → `GeneticFeatureSelector(pop=40, gen=30)`
4. Fit `RobustScaler` (+ optional `PowerTransformer`) on **full selected matrix**
5. Resample **full** matrix → then `train_test_split` (test 0.15, then val 0.12 of remainder)

### Best stored output
`SMOTE-ENN` + `MobileNetV4+Attention`: accuracy **0.97059**, F1 **0.97058**, bal-acc **0.97039**, ROC-AUC **0.99787**.

Full ranking: `data/verified_results.json` → `mobilenet_ablations_cell2`.

---

## 2. Nine-entry ensemble stacking (cell 4)

### Sampling list (preserve duplicate SMOTE)
```text
['SMOTE', 'ADASYN', 'SMOTE-ENN', 'SMOTE-Tomek', 'SMOTE-NCR',
 'ADASYN-ENN', 'ADASYN-Tomek', 'ADASYN-NCR', 'SMOTE']
```
The **ninth entry is intentionally a second SMOTE run**, not a distinct resampling algorithm. Stored ranks treat them as separate executions (rank 3 vs rank 5 in verified JSON).

### Architecture
- `MobileNetV4Model`: same Conv1D stack as cell 2 without attention (**62,710** params at 5 inputs)
- `LightweightModel`: Conv1D 32→48 (k=2), GAP+GMP, Dense 64→32 (**12,294** params)
- Meta-learner: `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')` on stacked softmax probs `[p_m0, p_m1, p_l0, p_l1]`
- Total reported params **75,004**; FLOPs **150,008** (`2 × params`)

### Training
- 30 epochs, batch 128; AdamW (mobile `1.2e-3`, light `1.5e-3`); EarlyStopping / ReduceLROnPlateau
- Meta-learner fit on **in-sample** base predictions (train+val predict → fit)

### Dataset reality for the executed cell
- NIDS: `max_rows = 150000`, label `(Label > 0)`
- API CSV path checked; executed output recorded **API dataset not available**
- Final published score is **network-flow only**

### Best stored output
`SMOTE-ENN_GA_Ensemble_Stacking`: accuracy **0.97789**, F1 **0.97789**, bal-acc **0.97775**, MCC **0.95582**, ROC-AUC **0.99767**.

---

## 3. Lightweight + MobileNetV4 stacking

Implemented only inside cell 4’s `EnsembleStackingModel` (see above). Cell 2 trains MobileNet variants **without** the lightweight tower or meta-learner.

---

## 4. CNN–BiLSTM benchmark (cell 22)

### Pipeline
1. Load API CSV (`malware` label); encode `t_*` sequences length **80**
2. MI top-**5** numeric API columns (stored names: `t_86`, `t_82`, `t_90`, `t_92`, `t_75`)
3. `StandardScaler` on those five (fit before split — leakage)
4. Align by **row truncation** to `min(len)` of selected features / sequences / labels — **not** by application id
5. Split 70/15/15; then apply balancing **on train only** in this cell’s loops
6. Dual-input PyTorch model: static MLP + CNN–BiLSTM sequence branch + fusion head
7. Epochs **6**; device forced to **CPU** in the notebook

### NIDS in cell 22
Loaded with `nrows=200000` “for completeness” then **not fused** into the dual-input tensors.

### Stored highlights
- Best F1: ADASYN **0.9916** (FPR **0.6111**)
- No Balancing: accuracy **0.9644**, bal-acc **0.9366**
- SMOTE+ENN collapses to accuracy **0.0255**

Parameters **1,108,931**; notebook FLOPs ~**16.633 M** (thop when available).

---

## 5. Resampling strategies

| Name | Implementation in notebook |
| --- | --- |
| SMOTE | `SMOTE(random_state=42, k_neighbors=5)` |
| ADASYN | `ADASYN(random_state=42, n_neighbors=5)` |
| SMOTE-ENN | `SMOTEENN(random_state=42)` |
| SMOTE-Tomek | `SMOTETomek(random_state=42)` |
| SMOTE-NCR | SMOTE then `NearMiss(version=1, n_neighbors=3)` |
| ADASYN-ENN | ADASYN then `EditedNearestNeighbours(n_neighbors=3)` |
| ADASYN-Tomek | ADASYN then `TomekLinks()` |
| ADASYN-NCR | ADASYN then NearMiss v1 |
| None / No Balancing | Cell 22 only |

On failure, notebook **falls back to SMOTE**.

---

## 6. Mutual information

- Cells 2/4: `SelectKBest(mutual_info_classif, k=min(100, n_features))` after RobustScaler on GA subsample
- Cell 22: `mutual_info_classif` ranking; keep top 5; plots saved as MI bars / cumulative curve

---

## 7. Genetic algorithm

`GeneticFeatureSelector` (cells 2/4):
- `population_size=40`, `generations=30`, `mutation_rate=0.12`, `crossover_rate=0.9`
- Fitness: RF (`n_estimators=100`, `max_depth=14`, balanced) on internal 75/25 split; `0.55*acc + 0.45*f1` × sparsity penalty
- Tournament selection size 4; single-point crossover; bit-flip mutation
- **Selected feature names were never printed** in cell 4

---

## 8. Explainability

Cells 2/4 include LRP-style plotting helpers and write under `/kaggle/working/xai_analysis`. `lrp_results_best.json` is **not** in this package. Local console currently uses meta-coefficients / input sensitivity and marks notebook LRP as Unavailable until artifacts exist.

---

## 9. Complexity accounting

- Keras cells: `flops = count_params() * 2` (not profiled MACs)
- Cell 22: optional `thop.profile`; falls back if missing
- Timing: wall-clock train / inference seconds printed per config

---

## Dataset alignment policy (critical)

| Experiment | Combination rule |
| --- | --- |
| Cell 2 / 4 | NIDS only for the scored run; API optional and was missing in the executed cell-4 session |
| Cell 22 | API static MI features + API sequences; NIDS loaded but unused |
| Earlier cells (6–19) | Some **row-index** alignments of independent tables — **do not treat as paired apps** |

Independently sourced API and network-flow rows **must not** be assumed to represent the same Android application.

---

## Leakage checklist (notebook reproduction mode)

Preserve and **label**, do not silently “fix” when reproducing cell 2/4:

1. Scaler / PowerTransformer fit before split  
2. MI + GA on data that later enters the test fold  
3. Resampling before split (cells 2/4)  
4. Meta-learner trained on in-sample base predictions  

Leakage-controlled mode implements the corrected protocol separately.

---

## Artifacts the notebook wrote (Kaggle only)

- `/kaggle/working/models/mobilenetv4_*.h5`, `lightweight_*.h5`
- Plots under `/kaggle/working/plots/`
- Results / XAI JSON under `/kaggle/working/results` and `xai_analysis`
- Meta-learner **not** serialised in the reviewed cell-4 code path

None of the HDF5 weights are in the local repository.

---

## Local engine implications

- TensorFlow 2.20 is required for bit-identical Keras graphs; current local interpreter is **Python 3.14** (no TF wheel). The experiment engine therefore:
  - Keeps notebook Keras builders behind an optional TF import
  - Executes trainable runs with **faithful PyTorch ports** of the same layer widths / stacking protocol, labelled `backend=pytorch_notebook_port`
- Full 9× / 16× matrices are expensive; the Laboratory defaults to **single-config** runs and smoke profiles (`nrows` / epoch caps) unless the researcher starts a deliberate batch.
