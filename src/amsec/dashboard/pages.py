"""Remaining pages: historical outputs, detection, XAI, datasets, reproducibility, methodology."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from amsec.dashboard.data import historical, historical_figures, historical_tables, runs_for
from amsec.dashboard.theme import style_fig, thin_bars
from amsec.lab.runs import dataset_fingerprints, list_runs, read_json
from amsec.notebook.provenance import (
    DATA_DIR, DATASET_FILES, EXPERIMENTS, NON_EXPERIMENT_CELLS, NOTEBOOK_KERNEL, NOTEBOOK_NAME, ROOT, verify_notebook,
)


# ---------------------------------------------------------------- historical outputs
def render_historical() -> None:
    st.title("Notebook Outputs (Historical)")
    st.caption(f"Tables and figures saved inside `{NOTEBOOK_NAME}`. Parsed directly from the notebook JSON — not re-executed, not edited.")
    options = {k: f"Cell {v['cell']} — {v['short']}" for k, v in EXPERIMENTS.items()}
    exp_id = st.selectbox("Cell", list(options), format_func=options.get)
    spec = EXPERIMENTS[exp_id]
    st.markdown(f"**{spec['title']}** · {spec['framework']}  \n{spec['design']}")
    if spec.get("untraceable"):
        st.error("Orphaned outputs: this cell's source is empty, so these outputs cannot be tied to code. They are not used anywhere else in AMSEC.")
    if spec.get("no_saved_outputs"):
        st.info("This cell has no saved outputs in the corrected notebook. Its results exist only as AMSEC runs.")
    for table_id, t in historical_tables(exp_id).items():
        if t.get("rows"):
            st.markdown(f"**Table `{table_id}`** — {t['description']}")
            st.dataframe(pd.DataFrame(t["rows"]), hide_index=True, width="stretch")
    figs = historical_figures(exp_id)
    if figs:
        st.markdown(f"**Figures ({len(figs)})** — captions are the console text printed just before each figure.")
        cols = st.columns(2)
        for i, f in enumerate(figs):
            with cols[i % 2]:
                st.image(str(f["path"]), width="stretch")
                st.caption(f"`{f['file']}` · " + f["context"].replace("\n", " · ")[-220:])
    with st.expander("Non-experiment cells"):
        st.json(NON_EXPERIMENT_CELLS)


# ---------------------------------------------------------------- detection
@st.cache_resource(show_spinner="Loading Keras models…")
def _detector(run_id: str, entry: int):
    from amsec.detection import NotebookDetector

    return NotebookDetector(run_id, entry)


def _pick_model():
    try:
        from amsec.detection import servable_models
    except Exception as exc:  # TensorFlow missing etc.
        st.error(f"Detection unavailable: {exc}")
        return None
    models = servable_models()
    if not models:
        st.error("No servable model. Complete a cell-4 (or cell-2) run in the Experiment Laboratory first — "
                 "AMSEC serves only models trained by the notebook's own code.")
        return None
    m = st.selectbox("Trained model", models, format_func=lambda m: m.label, key="det_model")
    if m.kind == "smoke":
        st.warning("This model comes from a smoke run (reduced data/epochs). Use only for testing the interface.")
    try:
        return _detector(m.run_id, m.entry), m
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return None


def render_detection() -> None:
    st.title("Malware Detection")
    st.info("Scores **network-flow records** (NF-UQ-NIDS schema) with a stacking ensemble trained by the notebook's cell-4 code "
            "(MobileNetV4 + Lightweight → LogisticRegression). This is not an APK scanner.")
    picked = _pick_model()
    if not picked:
        return
    det, meta = picked
    st.caption(f"Required columns (GA-selected, model order): {', '.join(det.feature_names)} · decision = argmax of the meta-learner (notebook).")
    single, batch = st.tabs(["Single flow", "Batch CSV"])
    with single:
        cols = st.columns(min(3, len(det.feature_names)))
        values = {n: cols[i % len(cols)].number_input(n, value=0.0, format="%.4f", key=f"det_{n}") for i, n in enumerate(det.feature_names)}
        if st.button("Score flow", type="primary"):
            try:
                out, warns = det.predict_frame(pd.DataFrame([values]))
            except Exception as exc:
                st.error(str(exc)); return
            for w in warns:
                st.warning(w)
            r = out.iloc[0]
            c = st.columns(3)
            c[0].metric("Prediction", r["amsec_prediction"])
            c[1].metric("P(normal)", f"{r['amsec_p_normal']:.4f}")
            c[2].metric("P(attack)", f"{r['amsec_p_attack']:.4f}")
    with batch:
        up = st.file_uploader("CSV containing the required columns (extra columns are ignored)", type=["csv"])
        if up is not None:
            if up.size > 80 * 1024 * 1024:
                st.error("Uploads are limited to 80 MiB."); return
            try:
                frame = pd.read_csv(up, low_memory=False)
                out, warns = det.predict_frame(frame)
            except Exception as exc:
                st.error(str(exc)); return
            for w in warns:
                st.warning(w)
            n_att = int((out["amsec_prediction"] == "attack").sum())
            c = st.columns(3)
            c[0].metric("Rows", f"{len(out):,}"); c[1].metric("Predicted attack", f"{n_att:,}")
            if "Label" in frame.columns:
                y = (pd.to_numeric(frame["Label"], errors="coerce").fillna(0) > 0).astype(int)
                c[2].metric("Accuracy vs Label column", f"{((out['amsec_prediction'] == 'attack').astype(int) == y).mean():.4f}")
            st.dataframe(out.head(200), width="stretch")
            st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"), "amsec_scored.csv", "text/csv")


# ---------------------------------------------------------------- XAI
def render_xai() -> None:
    st.title("Explainable AI — LRP")
    st.markdown("The notebook's `LRPExplainer` computes |x · ∂p/∂x| for the predicted class on the **MobileNetV4 base model** and "
                "normalises it to sum to 1. It is gradient×input, labelled LRP in the notebook; AMSEC keeps the notebook's method and name.")
    runs = [r for r in runs_for() if (Path(r["path"]) / "lrp_explanations.json").exists()]
    if runs:
        run = st.selectbox("Run (phase-9 explanations)", runs, format_func=lambda r: f"{r['run_id']} · cell {r['cell']} · {r['kind']}")
        ex = json.loads((Path(run["path"]) / "lrp_explanations.json").read_text(encoding="utf-8"))
        for e in ex:
            d = pd.DataFrame({"feature": e["feature_names"], "relevance": e["relevance_scores"]}).sort_values("relevance")
            fig = px.bar(d, x="relevance", y="feature", orientation="h")
            st.plotly_chart(thin_bars(style_fig(fig, f"Sample {e['sample_idx']} · true {e['true_label']} · predicted {e['predicted_label']} "
                                                     f"(p={e['prediction_proba']:.3f}) · {run['run_id']}", 300)), width="stretch", key=f"lrp_{run['run_id']}_{e['sample_idx']}")
        st.caption("Phase 9 explains the phase-8 model (re-trained on an un-resampled 80/20 split), on 3 random rows of its test split — as in the notebook.")
    else:
        st.info("No run with phase-9 LRP explanations yet (cell 2 or 4 full/partial run).")
    hist = historical()["experiments"]["cell04_stacking9"]["figures"]
    lrp_figs = [f for f in hist if "LRP" in f["context"] or "lrp" in f["context"].lower()]
    if lrp_figs:
        st.subheader("Notebook LRP figures (historical, cell 4)")
        for f in lrp_figs:
            st.image(str(Path(ROOT / "research_results" / "notebook_figures" / f["file"])), width="stretch")
    st.subheader("Explain your own flow")
    picked = _pick_model()
    if picked:
        det, _ = picked
        cols = st.columns(min(3, len(det.feature_names)))
        values = {n: cols[i % len(cols)].number_input(n, value=0.0, format="%.4f", key=f"xai_{n}") for i, n in enumerate(det.feature_names)}
        if st.button("Compute LRP relevance"):
            rel = det.lrp(pd.DataFrame([values])).T.reset_index()
            rel.columns = ["feature", "relevance"]
            fig = px.bar(rel.sort_values("relevance"), x="relevance", y="feature", orientation="h")
            st.plotly_chart(thin_bars(style_fig(fig, "LRP relevance (MobileNetV4 base model)", 300)), width="stretch")


# ---------------------------------------------------------------- datasets
@st.cache_data(show_spinner="Fingerprinting datasets…")
def _fingerprints() -> dict:
    return dataset_fingerprints()


@st.cache_data(show_spinner=False)
def _label_counts(name: str, label: str, nrows: int | None) -> dict:
    s = pd.read_csv(DATA_DIR / name, usecols=[label], nrows=nrows)[label]
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def render_datasets() -> None:
    st.title("Dataset Explorer")
    st.caption(f"Data directory: `{DATA_DIR}` (override with `AMSEC_DATA_DIR`).")
    fp = _fingerprints()
    st.dataframe(pd.DataFrame(fp.values()), hide_index=True, width="stretch")
    st.markdown("""
| Experiment | Dataset slice actually read |
|---|---|
| Cells 2, 4 | first **150,000** rows of NF-UQ-NIDS (50,000-row chunks); API file absent in the saved run |
| Cell 22 | API: all 43,876 rows → seeded subset **200 benign + 1,000 malware**; NIDS: first **50,000** rows |
| Cells 6–19 (historical) | API sequences; NIDS loaded (200k / 500k rows) — cells 11–19 do not feed NIDS to the model |
""")
    st.warning("The local `NF-UQ-NIDS-Dataset.csv` holds 1,048,575 data rows — exactly Excel's row limit — so it is a truncated copy of "
               "NF-UQ-NIDS-v2 (~75M rows). The notebook experiments read only the first 150,000 rows, whose label counts "
               "(normal 49,414 / attack 100,586) match the notebook's printed counts exactly.")
    for key, label, nrows, title in (("nids", "Label", 150_000, "NF-UQ-NIDS · first 150,000 rows (cells 2/4)"),
                                    ("nids", "Label", 50_000, "NF-UQ-NIDS · first 50,000 rows (cell 22)"),
                                    ("api", "malware", None, "API call sequences · all rows")):
        if not fp.get(key, {}).get("present"):
            continue
        counts = _label_counts(DATASET_FILES[key], label, nrows)
        d = pd.DataFrame({"class": list(counts), "rows": list(counts.values())})
        fig = px.bar(d, x="class", y="rows", text="rows")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(thin_bars(style_fig(fig, title, 300)), width="stretch")


# ---------------------------------------------------------------- reproducibility
def render_reproducibility() -> None:
    st.title("Reproducibility & Audit")
    nb = verify_notebook()
    st.markdown(f"**Authoritative notebook:** `{nb['path']}`  \n**SHA-256:** `{nb['sha256']}` — "
                + ("verified ✓" if nb["matches_recorded"] else "**MISMATCH**") + f"  \n**Notebook kernel:** {NOTEBOOK_KERNEL}")
    st.subheader("Fidelity evidence")
    st.markdown("""
* `tests/test_notebook_fidelity.py` — AST of every copied class/function equals the notebook definition (cells 2, 4, 22); notebook hash; constants.
* `scripts/generate_notebook_modules.py` — regenerates the Keras definition modules byte-for-byte from the notebook.
* `scripts/validate_cell22_refactor.py` — runs the **verbatim** cell-22 source and the AMSEC module; compares every metric.
""")
    eq = ROOT / "experiment_runs" / "_validation" / "cell22_refactor" / "refactor_equivalence.json"
    if eq.exists():
        r = json.loads(eq.read_text(encoding="utf-8"))
        (st.success if r["all_equal"] else st.error)(
            f"Cell-22 refactor equivalence: all_equal = {r['all_equal']} over {r['rows_compared']} methods; metadata equal = {all(r['metadata_equal'].values())}.")
    st.subheader("Runs")
    runs = list_runs()
    if not runs:
        st.info("No runs.")
        return
    run = st.selectbox("Run record", runs, format_func=lambda r: f"{r['run_id']} · cell {r['cell']} · {r['kind']} · {r['status']}")
    meta = read_json(Path(run["path"]) / "run_metadata.json") or read_json(Path(run["path"]) / "dual_dataset_run_metadata.json") or {}
    c = st.columns(3)
    c[0].metric("Kind", run["kind"]); c[1].metric("Status", run["status"]); c[2].metric("Notebook hash match", str(run["notebook"]["matches_recorded"]))
    st.markdown("**Deviations from the notebook**")
    st.json(meta.get("deviations") or {"none": "notebook parameters and full configuration list" if run["kind"] == "full_reproduction" else run.get("overrides")})
    with st.expander("Run record (parameters, environment, dataset fingerprints)"):
        st.json({k: v for k, v in run.items() if k != "path"})
    with st.expander("Experiment metadata"):
        st.json(meta)


# ---------------------------------------------------------------- methodology
CONCERNS = {
    "Cells 2 and 4 (shared pipeline)": [
        "RobustScaler, PowerTransformer, mutual information and the GA are fitted before the train/test split (information leakage).",
        "Resampling is applied to the full matrix before splitting, so synthetic/near-duplicate points reach the test fold; test sets are balanced and do not reflect the original 33/67 class ratio.",
        "'SMOTE-NCR' and 'ADASYN-NCR' run NearMiss-1, not the Neighbourhood Cleaning Rule. After SMOTE the classes are already equal, so NearMiss removes nothing: SMOTE-NCR trains on the same data as SMOTE.",
        "The 180,000-row cap and the GA sample use the global NumPy RNG, so results depend on loop order.",
        "Precision/recall/F1 are support-weighted; ROC-AUC is weighted one-vs-rest.",
        "FLOPs are reported as 2 × parameters (a parameter count, not a measured FLOP count).",
        "Feature names of the five GA-selected features were not printed; AMSEC logs them.",
        "Phase 8 re-trains the 'best' model on an un-resampled 80/20 split using the test split as validation data (early stopping/LR schedule see the test fold); LRP explains that model.",
        "Phase 8's API evaluation maps token IDs to 11 summary statistics padded/truncated to 5 slots and tunes the threshold on the evaluation labels; it did not run in the saved notebook (API file absent).",
        "Cell 4: the meta-learner is fitted on base-model predictions of the training rows (not out-of-fold).",
        "Cell 4 lists SMOTE twice (entries 1 and 9); AMSEC keeps both as separate entries.",
    ],
    "Cell 22 (corrected dual-dataset benchmark)": [
        "API and NIDS rows share no identifier. Pairing draws, for each API row, a random NIDS row with the same label. The static branch therefore receives label-conditioned NIDS features: performance reflects NIDS separability plus API sequences, not two views of one application.",
        "CPU-feasible subset: 200 of 1,079 benign and 1,000 of 42,797 malware API rows; test split = 180 rows (30 benign) — metrics have wide uncertainty.",
        "Split happens before MI and scaling (no leakage there); the API vocabulary is built from all 1,200 rows (unsupervised).",
        "'adasyn_enn' = ADASYN → SMOTEENN and 'adasyn_tomek' = ADASYN → SMOTETomek (not ENN/Tomek alone).",
        "Resampled training sets are not shuffled (shuffle only for no_balancing); synthetic static rows receive a random same-class API sequence.",
        "3 epochs; model selection on validation accuracy; Gaussian noise (σ=0.08) on static features during training.",
        "Torch RNG state carries across methods, so running a subset of methods changes later methods' initialisation.",
    ],
    "Cells 6–19 (earlier CNN–BiLSTM implementations, historical only)": [
        "Cells 11–19 load NF-UQ-NIDS but build the 'static' branch from API columns; NIDS is not used by the model.",
        "Several configurations report F1 ≈ 0.99 with specificity ≈ 0 (benign class missed) or F1 = 0 (SMOTE+ENN, ADASYN+ENN in cell 19).",
        "Cell 6 reports exactly 1.0000 on every metric for both datasets; cells 7 and 9 report 0.9994–1.0000 — near-perfect scores that warrant a leakage check.",
        "Cell 7 has outputs but no source code.",
    ],
}


def render_methodology() -> None:
    st.title("Methodology & Limitations")
    st.markdown("""
**Pipeline (cells 2/4).** First 150,000 NF-UQ-NIDS flows → `optimize_memory` dtype down-casting → numeric columns →
VarianceThreshold(1e-5) → 20,000-row sample → RobustScaler → SelectKBest(MI, k=min(100, n)) → GA (population 40, 30 generations,
RandomForest fitness 0.55·acc + 0.45·F1 with a size penalty) → RobustScaler + Yeo-Johnson on the full matrix → resampling →
cap 180,000 → 85/15 then 88/12 stratified splits → Keras models (30 epochs, batch 128, AdamW, early stopping, class weights).

**Cell 22.** Label-matched API/NIDS pairing → 70/15/15 split → MI top-5 NIDS features on train → StandardScaler →
dual-input CNN–BiLSTM (embedding 128, Conv1D 64/128, 2-layer BiLSTM 128 with attention, static MLP, fusion MLP; dropout 0.72) → 3 epochs.

AMSEC reproduces these procedures **as written**. The concerns below are documented, not corrected.
""")
    for group, items in CONCERNS.items():
        st.subheader(group)
        for item in items:
            st.write("• " + item)
    st.subheader("Platform limitations")
    for item in [
        "Detection scores network-flow records; the notebook does not contain an Android APK feature extractor.",
        "Historical metrics come from GPU runs (2 GPUs, TF 2.20, Kaggle); AMSEC runs on CPU. Bit-exact equality is not expected for Keras training; deterministic steps (sampling sizes, splits, parameter counts, GA) are checked exactly.",
        "No repeated-seed statistics or confidence intervals exist in the notebook.",
    ]:
        st.write("• " + item)
