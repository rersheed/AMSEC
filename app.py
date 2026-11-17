"""AMSEC PhD research console — Research Dashboard and Viva Presentation modes."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from amsec.dashboard.analytics import render_analytics
from amsec.dashboard.overview import render_overview
from amsec.dashboard.session import MODE_DASHBOARD, MODE_VIVA, init_session, set_mode


def _resume_viva() -> None:
    set_mode(MODE_VIVA)


def _return_viva_from_detect() -> None:
    set_mode(MODE_VIVA)
from amsec.dashboard.theme import inject, style_fig
from amsec.dashboard.viva import render_mode_sidebar_header, render_viva
from amsec.datasets import DATA_DIR, FILES, profile
from amsec.inference import (
    InputValidationError,
    ModelUnavailableError,
    get_ensemble,
    manifest_available,
)
from amsec.research import load_catalog, load_local_evaluation, load_manifest, load_verified

PAGES = [
    "Research Overview",
    "Research Analytics",
    "Malware Detection",
    "Batch Detection",
    "Dataset Explorer",
    "Model Comparison",
    "Resampling Experiments",
    "Explainability",
    "Architecture and Methodology",
    "Research Limitations",
]

st.set_page_config(
    page_title="AMSEC | Android Malware Security Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()
init_session()


@st.cache_data(show_spinner=False)
def cached_profile(key: str) -> dict:
    return profile(key)


@st.cache_resource(show_spinner=False)
def cached_manifest_probe():
    """Load manifest once per process without resetting weights on every rerun."""
    if not manifest_available():
        return None
    try:
        return get_ensemble().manifest
    except (ModelUnavailableError, Exception) as exc:
        return {"error": str(exc)}


def local_manifest():
    return cached_manifest_probe()


def _score_form(model):
    values = {}
    columns = st.columns(min(3, len(model.feature_names)))
    for index, name in enumerate(model.feature_names):
        with columns[index % len(columns)]:
            values[name] = st.number_input(name, value=0.0, format="%.6f", key=f"feat_{name}")
    if st.button("Score this flow", type="primary"):
        frame = pd.DataFrame([values])
        try:
            scored = model.predict_frame(frame)
        except InputValidationError as exc:
            st.error(str(exc))
            return
        row = scored.iloc[0]
        st.session_state["last_single_prediction"] = row.to_dict()
        st.metric("Prediction", row["amsec_prediction"])
        left, right = st.columns(2)
        left.metric("P(normal)", f"{row['amsec_probability_normal']:.4f}")
        right.metric("P(attack)", f"{row['amsec_probability_attack']:.4f}")
        st.caption(
            f"Decision threshold {row['amsec_threshold']} on the attack probability. "
            f"Model id: {row['amsec_model_id']}."
        )


def page_detect():
    st.title("Malware Detection")
    if st.session_state.get("operating_mode", MODE_DASHBOARD) == MODE_DASHBOARD:
        c1, c2 = st.columns([3, 1])
        with c2:
            st.button(
                "Return to Viva slide",
                width="stretch",
                on_click=_return_viva_from_detect,
                key="detect_return_viva",
            )
    st.info(
        "Scores network-flow features with the local trained stack. "
        "This is not an APK scanner and does not use heuristic formulas from the older GitHub app."
    )
    manifest = local_manifest()
    if not manifest or "error" in (manifest or {}):
        st.error("Unavailable. Train the local bundle with scripts/train_local_ensemble.py before scoring.")
        if manifest and "error" in manifest:
            st.write(manifest["error"])
        return
    model = get_ensemble()
    st.write("Required columns, in model order:")
    st.code(", ".join(model.feature_names))
    _score_form(model)


def page_batch():
    st.title("Batch Detection")
    manifest = local_manifest()
    if not manifest or "error" in (manifest or {}):
        st.error("Unavailable. A trained bundle is required before a CSV can be scored.")
        return
    model = get_ensemble()
    upload = st.file_uploader("CSV with the trained feature names", type=["csv"], key="batch_uploader")
    if upload is None:
        summary = st.session_state.get("batch_last_summary")
        if summary:
            st.info(
                f"Previous batch in this session: {summary['rows']:,} rows · "
                f"{summary['attack']:,} predicted attack (upload again to rescore)."
            )
        st.caption("Upload a CSV. Empty files and tables missing the trained columns are rejected.")
        return
    if upload.size == 0:
        st.error("The file is empty.")
        return
    if upload.size > 80 * 1024 * 1024:
        st.error("This console accepts uploads up to 80 MiB.")
        return
    try:
        frame = pd.read_csv(upload)
        scored = model.predict_frame(frame)
    except Exception as exc:
        st.error(str(exc))
        return
    extra = [column for column in frame.columns if column not in model.feature_names]
    if extra:
        st.warning(
            "Extra columns were ignored and were not used to pad the model: "
            + ", ".join(map(str, extra[:30]))
        )
    attack = int((scored["amsec_prediction"] == "attack").sum())
    st.session_state.batch_last_summary = {"rows": len(scored), "attack": attack}
    left, right = st.columns(2)
    left.metric("Rows scored", f"{len(scored):,}")
    right.metric("Predicted attack", f"{attack:,}")
    st.dataframe(scored.head(100), width="stretch")
    st.download_button(
        "Download annotated CSV",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="amsec_batch_predictions.csv",
        mime="text/csv",
    )


def page_datasets():
    st.title("Dataset Explorer")
    st.caption(f"Files are read from `{DATA_DIR}`.")
    notes = {
        "api": "Dynamic API-call sequences. Used by the CNN–BiLSTM benchmark. Not an input to the executed cell-4 ensemble.",
        "syscall": "Five-class system-call and binder frequencies. Separate from the binary API file. Not used by cell 4.",
        "nids": "Network-flow table for the final ensemble family. Notebook used Kaggle NF-UQ-NIDS-v2.csv; local file is NF-UQ-NIDS-Dataset.csv.",
    }
    for key, note in notes.items():
        with st.spinner(f"Reading {FILES[key]}"):
            item = cached_profile(key)
        with st.expander(f"{item['file']} — {item['rows']:,} rows, {item['n_columns']} columns", expanded=key == "nids"):
            st.write(note)
            st.write(f"Label column: `{item['label']}`")
            counts = pd.DataFrame({"class": list(item["label_counts"]), "count": list(item["label_counts"].values())})
            st.plotly_chart(style_fig(px.bar(counts, x="class", y="count", text="count"), height=300), width="stretch")
            if item["sample_missing_counts"]:
                st.write(f"Missing values inside the first {item['sample_rows']} rows:")
                st.json(item["sample_missing_counts"])
            else:
                st.write(f"No missing values in the first {item['sample_rows']} rows.")


def page_comparison():
    st.title("Model Comparison")
    st.info("Each table is one notebook or local experiment. Ranks are valid only inside that table.")
    st.info("For interactive scorecards and radar plots open **Research Analytics → Performance**.")
    verified = load_verified()
    st.subheader("Final ensemble — cell 4, SMOTE-ENN stack")
    st.dataframe(pd.DataFrame([verified["final_ensemble"]["metrics"]]), hide_index=True, width="stretch")
    st.subheader("Same cell, other resampling runs")
    st.dataframe(pd.DataFrame(verified["stacking_ablations_cell4"]), hide_index=True, width="stretch")
    st.subheader("Cell 2 ablation — MobileNet variants")
    st.dataframe(pd.DataFrame(verified["mobilenet_ablations_cell2"]), hide_index=True, width="stretch")
    st.subheader("Cell 22 benchmark — CNN–BiLSTM")
    st.dataframe(pd.DataFrame(verified["cnn_bilstm_benchmark_cell22"]), hide_index=True, width="stretch")
    manifest = local_manifest()
    st.subheader("Local retrain")
    if manifest and "error" not in manifest:
        st.dataframe(pd.DataFrame([manifest["local_test_metrics"]]), hide_index=True, width="stretch")
        st.caption(manifest["claim"])
    else:
        st.error("Local retrain metrics: Unavailable.")


def page_resampling():
    st.title("Resampling Experiments")
    st.info("Full criterion selector and FPR scatter live in **Research Analytics → Resampling**.")
    verified = load_verified()
    stack = pd.DataFrame(verified["stacking_ablations_cell4"])
    st.subheader("Cell 4 stacking ensemble")
    st.plotly_chart(style_fig(px.bar(stack, x="sampling", y=["f1", "balanced_accuracy"], barmode="group")), width="stretch")
    benchmark = pd.DataFrame(verified["cnn_bilstm_benchmark_cell22"])
    st.subheader("Cell 22 CNN–BiLSTM benchmark")
    st.plotly_chart(style_fig(px.bar(benchmark, x="method", y=["f1", "balanced_accuracy", "fpr"], barmode="group")), width="stretch")
    st.warning(
        "In cell 22, ADASYN raises accuracy from 0.9644 to 0.9836 while balanced accuracy falls from 0.9366 to 0.6937 "
        "and FPR rises from 0.0926 to 0.6111."
    )


def page_explain():
    st.title("Explainability")
    st.info("Extended XAI case studies are in **Research Analytics → Explainability**.")
    st.markdown(
        """
        Cell 4 names its explainer LRP (absolute input × gradient). The Kaggle JSON is **Unavailable**.
        Modality-level API versus network attribution is **Not validated** for the executed final ensemble.
        """
    )
    evaluation = load_local_evaluation()
    if evaluation is None or not manifest_available():
        st.error("Local feature attribution: Unavailable until the trained bundle is present.")
        return
    st.subheader("Local meta-learner coefficients")
    st.json(evaluation["meta_coefficients"])
    model = get_ensemble()
    values = {name: st.number_input(name, value=0.0, format="%.6f", key=f"xai_{name}") for name in model.feature_names}
    if st.button("Compute sensitivity"):
        try:
            table = model.attribute(pd.DataFrame([values]))
        except InputValidationError as exc:
            st.error(str(exc))
            return
        melted = table.T.reset_index()
        melted.columns = ["feature", "sensitivity"]
        st.plotly_chart(style_fig(px.bar(melted, x="sensitivity", y="feature", orientation="h")), width="stretch")


def page_method():
    st.title("Architecture and Methodology")
    verified = load_verified()
    final = verified["final_ensemble"]
    st.markdown(
        f"""
        **Data.** Cell 4 read 150,000 rows from `NF-UQ-NIDS-v2.csv`
        (normal {final['label_counts_before_resampling']['normal']:,}, attack {final['label_counts_before_resampling']['attack']:,}).
        The API path was missing in that run.

        **Representation.** Numeric flows → variance threshold → mutual information → genetic algorithm.
        Notebook kept five unnamed features; the local retrain logs nineteen named features.

        **Models.** MobileNetV4-style and lightweight Conv1D stacks with a logistic meta-learner.

        **Cost.** Notebook reported {final['metrics']['parameters']:,} parameters and {final['metrics']['flops']:,} FLOPs
        where FLOPs = 2 × parameters.

        **Local serving.** Same layer widths in PyTorch, with split-before-fit preprocessing.
        """
    )
    st.subheader("Leakage notes")
    for note in verified["leakage_notes"]:
        st.write("• " + note)


def page_limits():
    st.title("Research Limitations")
    for item in [
        "The final executed ensemble classifies network flows, not Android application packages.",
        "API-call rows and network-flow rows are not linked by an application identifier.",
        "The system-call frequency file has five classes and a different schema.",
        "Notebook test scores were measured after resampling the full matrix.",
        "Cell 22's highest F1 scores coincide with high false-positive rates.",
        "Kaggle HDF5 base models and the logistic meta-learner were not saved into this workspace.",
        "The older Streamlit app scored uploads with hand-written formulas and drew synthetic curves — not used here.",
        "No Android on-device latency, battery, or package-manager benchmark is in the supplied notebooks.",
        "Repeated-run statistical testing is Not established.",
        "Full ROC/PR curves are Unavailable without exported probability vectors.",
    ]:
        st.write("• " + item)


# ── Mode router ──────────────────────────────────────────────────────────────
with st.sidebar:
    mode = render_mode_sidebar_header()

if mode == MODE_VIVA:
    with st.sidebar:
        st.markdown("---")
        ready = local_manifest()
        if ready and "error" not in ready:
            st.success("Model bundle active")
        else:
            st.error("Model bundle unavailable")
    render_viva()
else:
    with st.sidebar:
        st.markdown("---")
        st.button(
            "Resume Viva Presentation",
            width="stretch",
            on_click=_resume_viva,
            key="resume_viva_btn",
        )
        page = st.radio("Section", PAGES, key="dashboard_page", label_visibility="collapsed")
        st.markdown("---")
        ready = local_manifest()
        if ready and "error" not in ready:
            st.success("Model bundle active")
            st.caption(ready.get("experiment_id", ""))
        else:
            st.error("Model bundle unavailable")
        identity = load_catalog()["identity"]
        st.caption(f"{identity['researcher']} · {identity['institution']}")

    {
        "Research Overview": render_overview,
        "Research Analytics": render_analytics,
        "Malware Detection": page_detect,
        "Batch Detection": page_batch,
        "Dataset Explorer": page_datasets,
        "Model Comparison": page_comparison,
        "Resampling Experiments": page_resampling,
        "Explainability": page_explain,
        "Architecture and Methodology": page_method,
        "Research Limitations": page_limits,
    }[page]()
