# AMSEC execution report — corrected-notebook implementation

**Date:** 2026-11-19 · **Machine:** Windows 11, 8 logical CPUs, no GPU · **Python:** 3.11.9 (`.venv311`)
**Authoritative source:** `notebooks/Final_phd-Experiment-android-malware_CORRECTED.ipynb`, SHA-256 `9e4724f7…e8fa` (verified at app start and in every run record).

Full per-entry tables (generated from run artifacts, no hand-typed numbers): [`docs/validation/execution_tables.md`](validation/execution_tables.md).
Cell-by-cell audit and methodological concerns: [`docs/NOTEBOOK_EXPERIMENT_AUDIT.md`](NOTEBOOK_EXPERIMENT_AUDIT.md).

## 1. Summary

| Check | Result |
|---|---|
| Notebook identity | SHA-256 verified; cells 0–21 byte-identical to the superseded notebook, only cell 22 changed (and has no saved outputs) |
| Verbatim code (cells 2, 4, 22) | AST of every copied class/function equals the notebook — `tests/test_notebook_fidelity.py` passes |
| Cell-22 refactor ≡ notebook cell | Verbatim cell vs AMSEC module: **all metrics identical** for all 9 methods (sklearn 1.9.1 and 1.6.1 environments) |
| Cell-4 data loading + GA | **Exact**: shape, label counts, 41 post-VT features, all 6 printed GA checkpoints, 5 selected features — *after pinning scikit-learn 1.6.1* |
| Cell-4 resampling counts | **Exact for SMOTE, SMOTE-NCR, ADASYN-NCR (entries 1, 5, 8, 9)**; differ for ADASYN / ENN / Tomek variants (§4) |
| Cell-4 metrics | Same winner (entry 3, SMOTE-ENN). F1 Δ mean +0.0025, max \|Δ\| 0.0066 (notebook's own SMOTE-equivalent spread: 0.0018) |
| Detection | Serving preprocessing ≡ training matrix (max \|Δ\| = 0.0 on 150,000 rows); hold-out accuracy 0.943–0.961 |
| Cell 2 (16 configs) | **Running** (`run_0006`) — results to be added |
| Cell 22 | Executed (`run_0001`, `run_0004`); no notebook outputs exist to compare against |
| Tests | 23/23 pass (`pytest`): fidelity, page rendering (all 9 pages), no presentation mode |
| UI | Single-mode, 9 pages; dark theme fixed for text contrast; screenshots checked in Chrome |

## 2. Environment and the scikit-learn pin

The notebook ran on Kaggle (Python 3.12.13, TensorFlow 2.20.0, two GPUs). Locally: TensorFlow-CPU 2.20.0, Keras 3.15.1, PyTorch 2.6.0 CPU, NumPy 2.1.3, pandas 2.2.3, SciPy 1.17.1.

**Discrepancy found and resolved.** With scikit-learn 1.9.1 the cell-4 GA selected **7** features (fitness 0.9702) instead of the notebook's **5** (0.9715). Cells 2 and 4 print identical GA checkpoints in the notebook, so the GA is deterministic; the drift was environmental. Running only the notebook's own phase 1–2 code (`scripts/diagnose_ga_environment.py`):

| scikit-learn | NumPy | Gen 5 | Final |
|---|---|---|---|
| notebook | – | 0.9624 / 13 features | 0.9715 / 5 |
| 1.2.2 | 1.26.4 | **0.9624 / 13** | – |
| 1.6.1 | 1.26.4 | **0.9624 / 13** | – |
| 1.6.1 | 2.1.3 | **0.9624 / 13** | **0.9715 / 5 (all 6 checkpoints exact)** |
| 1.9.1 | 2.1.3 | 0.9606 / 13 | 0.9702 / 7 |

The global RNG state after mutual information was identical in all environments; the RandomForest fitness changed after scikit-learn 1.6. AMSEC therefore pins `scikit-learn==1.6.1`, `imbalanced-learn==0.13.0`; the Laboratory warns in any other environment. Evidence: `docs/validation/ga_environment_diagnosis.txt`. The sklearn-1.9.1 cell-4 run (`run_0002`) was stopped after 4 entries and is kept, labelled *stopped*, as evidence.

## 3. Cell 4 — 9-entry stacking (`run_0005`, full reproduction, pinned environment)

All 9 entries in notebook order (SMOTE duplicated as entries 1 and 9), phases 1–9, 30 epochs, notebook parameters. Wall time 21.3 h (includes host sleep; GA ≈ 20 min of compute).

| Entry | Sampler | Samples after resampling (notebook → AMSEC) | F1 notebook | F1 AMSEC | Δ |
|---|---|---|---|---|---|
| 1 | SMOTE | 201,172 → 201,172 ✓ | 0.95596 | 0.96081 | +0.00485 |
| 2 | ADASYN | 202,084 → 201,771 | 0.94272 | 0.94176 | −0.00096 |
| 3 | SMOTE-ENN | 191,389 → 189,710 | **0.97789** | **0.98030** | +0.00241 |
| 4 | SMOTE-Tomek | 200,606 → 200,604 | 0.95910 | 0.96267 | +0.00357 |
| 5 | SMOTE-NCR | 201,172 → 201,172 ✓ | 0.95451 | 0.96107 | +0.00656 |
| 6 | ADASYN-ENN | 176,601 → 178,065 | 0.94992 | 0.94960 | −0.00032 |
| 7 | ADASYN-Tomek | 201,140 → 200,818 | 0.94380 | 0.94302 | −0.00078 |
| 8 | ADASYN-NCR | 201,172 → 201,172 ✓ | 0.94106 | 0.94272 | +0.00166 |
| 9 | SMOTE | 201,172 → 201,172 ✓ | 0.95421 | 0.95932 | +0.00511 |

* Ranking (notebook): 3, 4, 1, 5, 9, 6, 7, 2, 8 — (AMSEC): 3, 4, 5, 1, 9, 6, 7, 8, 2. Same best and second-best; the swaps are within noise.
* Parameter counts (75,004) and FLOPs (150,008) identical; split sizes identical for 8/9 entries (entry 6 differs because its resampled size differs).
* Entries 1, 5 and 9 train on the same resampled data (NearMiss removes nothing after SMOTE). AMSEC's spread among them (0.95932–0.96107) and the notebook's (0.95421–0.95596) each measure run-to-run training noise; AMSEC is ≈ +0.005 higher on this group, a plausible CPU-vs-GPU / Keras-version training difference, not a methodological one.
* Phases 7–9: best = entry 3 (as in the notebook); phase-8 re-train meta-learner validation accuracy 0.94920 (notebook 0.95837); API evaluation skipped as in the notebook. LRP explained rows 7213, 11510, 22708 (notebook: 5602, 28029, 21071) — the random rows differ because the global RNG stream diverges once resampled sizes differ.

## 4. Resampling counts — investigated, not reproducible bit-for-bit

Neighbour-based samplers (ADASYN, ENN, Tomek) do not reproduce the notebook's counts. Investigation (`scripts/diagnose_resampling_environment.py`, evidence in `docs/validation/resampling_environment_diagnosis.txt`):

* On these 5 features the 150,000 rows collapse to **17,825 unique rows** — neighbour searches are dominated by ties.
* Counts are **identical** on Windows across scikit-learn 1.2.2/1.6.1, imbalanced-learn 0.10.1/0.11.0/0.12.4/0.13.0, SciPy 1.11.4/1.15.3/1.17.1, NumPy 1.26.4/2.1.3 — library versions are not the cause.
* On Linux (WSL2; newer stack) the Yeo-Johnson λ values change by ~10⁻⁸, the unique-row count becomes 17,829, and every neighbour-based count changes again (e.g. ADASYN 201,994) — different from both Windows and the notebook.
* The local `NF-UQ-NIDS-Dataset.csv` is an **Excel re-save** of NF-UQ-NIDS-v2 (1,048,575 rows = Excel's limit; CRLF; 70 values in `SRC_TO_DST_SECOND_BYTES`/`DST_TO_SRC_SECOND_BYTES` truncated to 6 significant digits in the first 150,000 rows). Those columns are not among the 5 selected features, but the file is not byte-identical to Kaggle's.

**Conclusion.** These counts are chaotically sensitive to last-bit floating-point differences. Exact reproduction would require Kaggle's binary stack and the original Kaggle CSV, neither available. Entries 1, 5, 8, 9 are exact replications of the data pipeline; entries 2, 3, 4, 6, 7 are faithful re-executions of the notebook's code whose resampled sets differ by 0.001–0.9 %.

## 5. Detection (models from `run_0005`)

* **Preprocessing parity:** re-creating the notebook's matrix for the first 150,000 rows vs `NotebookDetector.transform` on raw CSV rows — max |Δ| = **0.0**.
* **Out-of-sample:** rows 150,001–200,000 (never read by cells 2/4; normal 16,351 / attack 33,649, the natural ratio):

| Entry | Accuracy | Balanced acc. | F1 (attack) | MCC | ROC-AUC |
|---|---|---|---|---|---|
| 1 SMOTE | 0.95790 | 0.95998 | 0.96825 | 0.90672 | 0.99390 |
| 2 ADASYN | 0.94152 | 0.95020 | 0.95514 | 0.87521 | 0.99187 |
| **3 SMOTE-ENN** (default served) | 0.95712 | 0.96156 | 0.96751 | 0.90607 | 0.99015 |
| 4 SMOTE-Tomek | 0.96096 | 0.96306 | 0.97058 | 0.91341 | 0.99421 |
| 5 SMOTE-NCR | 0.96104 | 0.96093 | 0.97077 | 0.91280 | 0.99419 |
| 6 ADASYN-ENN | 0.94798 | 0.95480 | 0.96031 | 0.88772 | 0.99234 |
| 7 ADASYN-Tomek | 0.94356 | 0.95179 | 0.95677 | 0.87923 | 0.99245 |
| 8 ADASYN-NCR | 0.94268 | 0.95125 | 0.95605 | 0.87759 | 0.99156 |
| 9 SMOTE | 0.95276 | 0.95816 | 0.96410 | 0.89715 | 0.99419 |

**Finding.** The notebook's in-split test scores (balanced, post-resampling, leakage-affected) overstate performance on natural-distribution data: SMOTE-ENN scores F1 0.980 on its own split but 0.957 accuracy on unseen rows, and is not the best entry there (SMOTE-Tomek / SMOTE-NCR ≈ 0.961). The hold-out is a contiguous block of the same file, so it is out-of-sample but not out-of-distribution.

## 6. Cell 22 — corrected dual-dataset benchmark

No saved notebook outputs exist; nothing historical to compare.

* **Refactor equivalence:** the unmodified notebook cell (only `BASE`/`OUT` paths substituted) and `amsec.notebook.cell22_dual_dataset` produce identical metrics for all 9 methods. A first attempt failed because an extra test-probability pass consumed torch's global RNG (DataLoader base seed); fixed by save/restore; re-verified in both environments.
* **Results are highly environment-sensitive:** `run_0001` (scikit-learn 1.9.1 / imbalanced-learn 0.14.2) and `run_0004` (1.6.1 / 0.13.0) differ because resampled training-set sizes differ slightly; with 180 test rows (30 benign) this flips outcomes (e.g. `adasyn_enn`: F1 0.909 → 0.000).
* **Substantive result (both runs):** accuracy ≈ 0.83–0.84 with balanced accuracy 0.50–0.59 and specificity 0–0.20 — the models label almost every benign test sample as malware. High F1 (≈ 0.91) reflects the 5:1 malware majority, not discrimination. Combined with the label-matched synthetic pairing (audit §Cell 22), these numbers should not be read as evidence of multimodal app-level detection.

## 7. Cell 2 — 16-configuration experiment

`run_0006` (full reproduction, pinned environment) is running. Expected from cell 4: exact GA and SMOTE/NCR counts; ADASYN/ENN/Tomek counts differing as in §4. This section will be completed when it finishes.

## 8. Historical outputs

All results tables (cells 2, 4, 6, 7, 9, 11, 13, 15, 17, 19), GA logs, per-entry logs and 94 figures are parsed from the notebook JSON (`scripts/build_historical_registry.py`) into `research_results/corrected_notebook_historical.json`; each table row is mapped to its loop entry. They are displayed as *historical* and never merged with new runs. Cell 7 (outputs without source) is flagged untraceable.

## 9. Changes since the previous build

* Old notebook references, PyTorch look-alike models, hand-typed "verified" metrics (including numbers from the old cell 22) and the PyTorch serving bundle were archived to `archive/pre_corrected_notebook/` (not deleted, not used).
* Viva/presentation mode removed (verified by test).
* Dark-on-dark text fixed (`.streamlit/config.toml` was a light theme under a dark CSS background).

## 10. Open items

1. Complete cell 2 (§7) and regenerate `docs/validation/execution_tables.md` with `--cell2 run_0006`.
2. Deployment on Streamlit Community Cloud is untested: requires **Python 3.11**, TensorFlow-CPU 2.20 + PyTorch (large install); datasets are not in git, so the Laboratory cannot run there.
3. Root-level duplicate notebook `Final_phd-Experiment-android-malware_CORRECTED.ipynb` (committed separately) — canonical copy is `notebooks/`.
4. Exact neighbour-based resampling replication would need the original Kaggle CSV and a Linux environment matching Kaggle's.
