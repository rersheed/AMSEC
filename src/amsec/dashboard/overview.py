"""Research Overview — executive PhD landing page."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from amsec.dashboard.theme import kpi_html, source_chip, style_fig, unavailable_box
from amsec.datasets import profile
from amsec.research import (
    comparison_rows,
    load_catalog,
    load_local_evaluation,
    load_manifest,
    local_metrics_enriched,
    notebook_final_metrics,
)


def _fmt(value, digits=5, pct=False):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if pct:
        return f"{float(value) * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value:,}"


def render_overview() -> None:
    catalog = load_catalog()
    identity = catalog["identity"]
    notebook = notebook_final_metrics()
    local = local_metrics_enriched()
    manifest = load_manifest()
    ready = bool(manifest and manifest.get("status") == "local_retrain_not_kaggle_weights")

    st.markdown(
        f"""
        <div class="amsec-hero">
          <div class="amsec-badge">{identity['designation']} · {identity['student_id']}</div>
          <div class="amsec-title">{identity['brand']} <span>— {identity['full_name']}</span></div>
          <p class="amsec-sub">{identity['subtitle']}</p>
          <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:0.9rem;max-width:920px;">
            <div style="background:rgba(11,18,32,0.9);border:1px solid rgba(58,123,253,0.2);border-radius:12px;padding:0.9rem 1rem;">
              <div style="font-family:Syne,sans-serif;font-size:0.58rem;color:#6d7f9e;letter-spacing:0.1em;text-transform:uppercase;">Researcher</div>
              <div style="font-family:Syne,sans-serif;font-weight:700;color:#e8edf5;margin-top:0.25rem;">{identity['researcher']}</div>
              <div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#3a7bfd;margin-top:0.2rem;">{identity['department']} · {identity['faculty']}</div>
              <div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#8a9bbf;">{identity['institution']}</div>
            </div>
            <div style="background:rgba(11,18,32,0.9);border:1px solid rgba(58,123,253,0.2);border-radius:12px;padding:0.9rem 1rem;">
              <div style="font-family:Syne,sans-serif;font-size:0.58rem;color:#6d7f9e;letter-spacing:0.1em;text-transform:uppercase;">Supervisory committee</div>
              {''.join(f"<div style='font-family:DM Mono,monospace;font-size:0.7rem;margin-top:0.28rem;color:#cfe0f5;'>{s['name']} <span style='color:#4a5878;'>({s['role']})</span></div>" for s in identity['supervisors'])}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.2, 1])
    with left:
        st.markdown(
            """
            **Research summary.** AMSEC studies an explainable hybrid detection stack that combines
            mutual-information and genetic feature selection with lightweight convolutional base learners
            and a logistic stacking meta-learner. The executed final notebook configuration is a
            **network-flow** ensemble on NF-UQ-NIDS; the API-call modality appears in separate benchmark cells.
            """
        )
        st.caption("Version: local PhD console · Experimental configuration: notebook cell-4 SMOTE-ENN stack (reference) + leakage-corrected local retrain (served).")
    with right:
        if ready:
            st.success("Local model readiness: **Ready** — schema-validated inference enabled.")
            st.caption(f"Bundle `{manifest['experiment_id']}`")
        else:
            st.error("Local model readiness: **Unavailable**")

    nav1, nav2, nav3, nav4 = st.columns(4)
    nav1.info("→ Malware Detection")
    nav2.info("→ Research Analytics")
    nav3.info("→ Explainability")
    nav4.info("→ Methodology")

    st.subheader("Key research indicators")
    st.caption(source_chip("mixed"))
    primary = st.columns(4)
    primary[0].markdown(kpi_html("Notebook accuracy", _fmt(notebook["accuracy"]), "Original executed notebook", "#00e5cc"), unsafe_allow_html=True)
    primary[1].markdown(kpi_html("Notebook F1", _fmt(notebook["f1"]), "Original executed notebook", "#3a7bfd"), unsafe_allow_html=True)
    primary[2].markdown(kpi_html("Notebook bal. acc.", _fmt(notebook["balanced_accuracy"]), "Original executed notebook", "#8b5cf6"), unsafe_allow_html=True)
    if local:
        primary[3].markdown(kpi_html("Local retrain accuracy", _fmt(local["accuracy"]), "Verified saved model evaluation", "#22c55e"), unsafe_allow_html=True)
    else:
        primary[3].markdown(kpi_html("Local retrain accuracy", "—", "Unavailable", "#ef4444"), unsafe_allow_html=True)

    secondary = st.columns(4)
    secondary[0].markdown(kpi_html("Precision (notebook)", _fmt(notebook["precision"]), "Original executed notebook"), unsafe_allow_html=True)
    secondary[1].markdown(kpi_html("Recall (notebook)", _fmt(notebook["recall"]), "Original executed notebook"), unsafe_allow_html=True)
    secondary[2].markdown(kpi_html("MCC (notebook)", _fmt(notebook["mcc"]), "Original executed notebook"), unsafe_allow_html=True)
    secondary[3].markdown(kpi_html("ROC-AUC (notebook)", _fmt(notebook["roc_auc"]), "Original executed notebook"), unsafe_allow_html=True)

    tertiary = st.columns(5)
    tertiary[0].markdown(kpi_html("Parameters (notebook)", _fmt(notebook["parameters"], 0), "Original executed notebook · FLOPs=2×params", "#ff6b35"), unsafe_allow_html=True)
    tertiary[1].markdown(kpi_html("FLOPs (notebook)", _fmt(notebook["flops"], 0), "Original executed notebook estimate"), unsafe_allow_html=True)
    tertiary[2].markdown(kpi_html("Selected features (notebook)", str(notebook["selected_features"]), "Names not logged"), unsafe_allow_html=True)
    if local:
        tertiary[3].markdown(kpi_html("FPR (local holdout)", _fmt(local["fpr"]), "Derived from verified confusion matrix"), unsafe_allow_html=True)
        tertiary[4].markdown(kpi_html("FNR (local holdout)", _fmt(local["fnr"]), "Derived from verified confusion matrix"), unsafe_allow_html=True)
    else:
        tertiary[3].markdown(kpi_html("FPR", "—", "Unavailable"), unsafe_allow_html=True)
        tertiary[4].markdown(kpi_html("FNR", "—", "Unavailable"), unsafe_allow_html=True)

    st.subheader("Original research versus local retraining")
    st.warning(
        "These configurations are **not equivalent**. The notebook resampled before the split and did not log "
        "feature names. The local model splits first, selects 19 named features, and is served in PyTorch."
    )
    compare = comparison_rows()
    st.dataframe(compare, hide_index=True, width="stretch")
    if local:
        deltas = {
            "Accuracy Δ": local["accuracy"] - notebook["accuracy"],
            "F1 Δ": local["f1"] - notebook["f1"],
            "Balanced accuracy Δ": local["balanced_accuracy"] - notebook["balanced_accuracy"],
            "MCC Δ": local["mcc"] - notebook["mcc"],
            "ROC-AUC Δ": local["roc_auc"] - notebook["roc_auc"],
        }
        cols = st.columns(len(deltas))
        for column, (label, value) in zip(cols, deltas.items()):
            column.metric(label, f"{value:+.5f}")
        fig = go.Figure()
        metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "roc_auc"]
        fig.add_bar(name="Notebook", x=metrics, y=[notebook[m] for m in metrics], marker_color="#3a7bfd")
        fig.add_bar(name="Local retrain", x=metrics, y=[local[m] for m in metrics], marker_color="#00e5cc")
        st.plotly_chart(style_fig(fig, "Scalar metrics — separate experiments", 380), width="stretch")

    st.subheader("Research dataset profile")
    dataset_cols = st.columns(3)
    for column, key, title in zip(
        dataset_cols,
        ["nids", "api", "syscall"],
        ["Network flows (NF-UQ-NIDS)", "API call sequences", "Syscalls / binders (5-class)"],
    ):
        with column:
            item = profile(key, sample_rows=2000 if key != "nids" else 3000)
            counts = item["label_counts"]
            if key == "nids":
                benign = int(counts.get("0", 0))
                mal = int(counts.get("1", 0))
                ratio = mal / max(benign, 1)
                modality = "Network-flow modality · used by final ensemble family"
            elif key == "api":
                benign = int(counts.get("0", 0))
                mal = int(counts.get("1", 0))
                ratio = mal / max(benign, 1)
                modality = "API-call modality · benchmark / not in executed cell-4 fit"
            else:
                benign = None
                mal = None
                ratio = None
                modality = "Separate five-class frequency dataset · not cell-4"
            st.markdown(f"**{title}**")
            st.caption(modality)
            st.write(f"File: `{item['file']}`")
            st.write(f"Rows: **{item['rows']:,}** · Columns: **{item['n_columns']}**")
            if ratio is not None:
                st.write(f"Imbalance (malware:benign): **{ratio:.2f}:1**")
            frame = pd.DataFrame({"class": list(counts), "count": list(counts.values())})
            chart = px.bar(frame, x="class", y="count", text="count", color="class")
            st.plotly_chart(style_fig(chart, height=280), width="stretch")
            missing = item["sample_missing_counts"]
            if missing:
                st.caption(f"Missing values in first {item['sample_rows']} rows: {len(missing)} columns affected.")
            else:
                st.caption(f"No missing values in the first {item['sample_rows']} rows.")

    st.subheader("Framework architecture")
    stages = catalog["pipeline_stages"]
    html = '<div class="amsec-pipe">'
    for index, stage in enumerate(stages, start=1):
        accent = "#22c55e" if stage["local_status"] == "implemented" else ("#ff6b35" if stage["local_status"] == "partial" else "#ef4444")
        html += (
            f'<div class="amsec-pipe-item"><div class="num" style="color:{accent}">{index:02d}</div>'
            f'<div class="lab">{stage["label"]}</div>'
            f'<div class="note">{stage["local_status"]} · {stage["note"]}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Research contribution summary")
    cards = st.columns(3)
    for index, contribution in enumerate(catalog["contributions"]):
        with cards[index % 3]:
            st.markdown(
                f'<div class="amsec-card"><h4>{contribution["title"]}</h4><p>{contribution["text"]}</p></div>',
                unsafe_allow_html=True,
            )

    evaluation = load_local_evaluation()
    if evaluation is None:
        unavailable_box("Local evaluation JSON is missing; local KPI cards may be incomplete.")
