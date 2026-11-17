"""PhD Viva Presentation Mode — slide-style defence interface."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from amsec.dashboard.session import (
    MODE_DASHBOARD,
    MODE_STATE_KEY,
    MODE_UI_KEY,
    clamp_slide,
    current_mode,
    set_mode,
)
from amsec.dashboard.theme import style_fig, unavailable_box
from amsec.datasets import profile
from amsec.inference import InputValidationError, get_ensemble, manifest_available
from amsec.research import (
    ROOT,
    cnn_bilstm_benchmark,
    comparison_rows,
    complexity_table,
    load_catalog,
    load_local_evaluation,
    load_manifest,
    local_metrics_enriched,
    mobilenet_ablations,
    notebook_final_metrics,
    stacking_ablations,
)

SLIDES_PATH = ROOT / "research_results" / "viva_slides.json"
VIVA_CSS = """
.viva-shell { max-width: 1180px; margin: 0 auto; }
.viva-progress { height: 6px; background: rgba(58,123,253,0.15); border-radius: 999px; overflow: hidden; margin: 0.4rem 0 0.9rem; }
.viva-progress > div { height: 100%; background: linear-gradient(90deg,#3a7bfd,#00e5cc); }
.viva-slide, .viva-figure-panel {
  background: linear-gradient(160deg,#07111f 0%,#0b1730 55%,#07111f 100%);
  border: 1px solid rgba(0,229,204,0.22); border-radius: 18px; padding: 1.6rem 1.9rem 1.4rem;
  box-shadow: 0 18px 40px rgba(0,0,0,0.35); margin-bottom: 0.75rem;
}
.viva-slide { min-height: 52vh; }
.viva-kicker { font-family: Syne,sans-serif; font-size: 0.62rem; letter-spacing: 0.14em; text-transform: uppercase; color: #00e5cc; }
.viva-h1 { font-family: Syne,sans-serif; font-size: 2.05rem; font-weight: 800; color: #e8edf5; margin: 0.35rem 0 0.35rem; line-height: 1.15; }
.viva-sub { font-family: Lora,serif; font-style: italic; color: #8a9bbf; margin: 0 0 1rem; font-size: 1.05rem; }
.viva-bullets { font-family: DM Mono,monospace; color: #cfe0f5; font-size: 0.92rem; line-height: 1.55; margin: 0; padding-left: 1.15rem; }
.viva-bullets li { margin: 0.45rem 0; }
.viva-review {
  border: 1px solid rgba(255,107,53,0.45); background: rgba(255,107,53,0.08); color: #ffd7c2;
  border-radius: 10px; padding: 0.65rem 0.8rem; font-family: DM Mono,monospace; font-size: 0.72rem; margin-bottom: 0.8rem;
}
.viva-count { font-family: DM Mono,monospace; color: #6d7f9e; font-size: 0.75rem; }
.viva-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.55rem; margin-top:0.9rem; }
.viva-metric { background:rgba(11,18,32,0.95); border:1px solid rgba(58,123,253,0.22); border-radius:12px; padding:0.7rem 0.8rem; }
.viva-metric .lab { font-family:Syne,sans-serif; font-size:0.55rem; letter-spacing:0.08em; text-transform:uppercase; color:#6d7f9e; }
.viva-metric .val { font-family:Syne,sans-serif; font-size:1.2rem; font-weight:800; color:#00e5cc; margin-top:0.2rem; }
"""


def _load_slides() -> list[dict]:
    payload = json.loads(SLIDES_PATH.read_text(encoding="utf-8"))
    return list(payload["slides"])


def _pct(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _mode_switcher() -> str:
    options = ["Research Dashboard", "Viva Presentation"]
    st.markdown("**Operating mode**")
    # Widget key MODE_UI_KEY is updated only via set_mode() callbacks or user clicks.
    if hasattr(st, "segmented_control"):
        chosen = st.segmented_control(
            "Operating mode",
            options,
            key=MODE_UI_KEY,
            label_visibility="collapsed",
        )
    else:
        chosen = st.radio(
            "Operating mode",
            options,
            horizontal=True,
            key=MODE_UI_KEY,
            label_visibility="collapsed",
        )
    if chosen is None:
        chosen = current_mode()
    st.session_state[MODE_STATE_KEY] = chosen
    return chosen


def render_mode_sidebar_header() -> str:
    """Render branding + mode switcher; return selected mode."""
    identity = load_catalog()["identity"]
    st.markdown(
        """
        <div style="text-align:center;padding:0.6rem 0 0.35rem;">
          <div style="width:58px;height:58px;border-radius:50%;margin:0 auto 0.55rem;
          background:linear-gradient(135deg,#3a7bfd,#00e5cc);display:flex;align-items:center;
          justify-content:center;font-size:1.45rem;">🛡️</div>
          <div style="font-family:Syne,sans-serif;font-weight:800;color:#e8edf5;">AMSEC</div>
          <div style="font-family:DM Mono,monospace;font-size:0.58rem;color:#4a5878;">PhD research console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode = _mode_switcher()
    st.caption(f"{identity['researcher']} · {identity['institution']}")
    return mode


def _shift_slide(delta: int) -> None:
    """Callback runs before widgets render, so selectbox key updates safely."""
    total = len(_load_slides())
    current = int(st.session_state.get("viva_slide_select", st.session_state.get("viva_slide_index", 0)))
    nxt = clamp_slide(current + delta, total)
    st.session_state.viva_slide_select = nxt
    st.session_state.viva_slide_index = nxt


def _return_to_dashboard() -> None:
    set_mode(MODE_DASHBOARD)


def _open_detection_page() -> None:
    set_mode(MODE_DASHBOARD)
    st.session_state.dashboard_page = "Malware Detection"


def render_viva() -> None:
    st.markdown(f"<style>{VIVA_CSS}</style>", unsafe_allow_html=True)
    slides = _load_slides()
    total = len(slides)
    if "viva_slide_select" not in st.session_state:
        st.session_state.viva_slide_select = clamp_slide(st.session_state.viva_slide_index, total)

    with st.sidebar:
        st.button(
            "⟵ Return to Research Dashboard",
            width="stretch",
            type="primary",
            on_click=_return_to_dashboard,
            key="viva_return_dashboard",
        )
        titles = [f"{i + 1:02d}. {s['title']}" for i, s in enumerate(slides)]
        st.selectbox(
            "Jump to slide",
            options=list(range(total)),
            format_func=lambda i: titles[i],
            key="viva_slide_select",
        )
        index = clamp_slide(st.session_state.viva_slide_select, total)
        st.session_state.viva_slide_index = index
        slide = slides[index]
        st.toggle("Show presenter notes", key="viva_show_notes")
        with st.expander("Presentation guidance", expanded=False):
            st.markdown(
                """
                - Use **Previous / Next** or the slide menu.
                - Press **F11** (or your browser’s fullscreen control) for projector view.
                - Slide index is preserved if you temporarily open the Research Dashboard.
                - Keyboard shortcuts are not injected via custom JS to avoid Streamlit conflicts.
                """
            )
        st.download_button(
            "Download viva HTML pack",
            _export_html(slides),
            file_name="amsec_viva_presentation.html",
            mime="text/html",
            width="stretch",
        )
        st.caption("PowerPoint export is not implemented in this build.")

    index = clamp_slide(st.session_state.viva_slide_select, total)
    st.session_state.viva_slide_index = index
    slide = slides[index]
    progress = (index + 1) / total
    has_figure = slide.get("figure") not in (None, "none")

    nav_l, nav_c, nav_r = st.columns([1, 2, 1])
    with nav_l:
        st.button(
            "◀ Previous",
            disabled=index <= 0,
            width="stretch",
            on_click=_shift_slide,
            args=(-1,),
            key="viva_prev",
        )
    with nav_c:
        st.markdown(
            f'<div class="viva-count" style="text-align:center;">Slide {index + 1} / {total} · {html.escape(slide["title"])}</div>'
            f'<div class="viva-progress"><div style="width:{progress * 100:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )
    with nav_r:
        st.button(
            "Next ▶",
            disabled=index >= total - 1,
            width="stretch",
            type="primary",
            on_click=_shift_slide,
            args=(1,),
            key="viva_next",
        )

    # Streamlit closes each markdown HTML fragment independently, so the full
    # slide chrome + copy must be emitted as one complete element.
    slide_html = _build_slide_html(slide)
    if has_figure:
        body_col, fig_col = st.columns([1.05, 1])
        with body_col:
            st.markdown(slide_html, unsafe_allow_html=True)
        with fig_col:
            _render_figure(slide["figure"])
    else:
        st.markdown(slide_html, unsafe_allow_html=True)

    if st.session_state.get("viva_show_notes"):
        notes = slide.get("notes", {})
        with st.expander("Presenter notes", expanded=True):
            st.markdown(f"**Main point.** {notes.get('main_point', '')}")
            st.markdown(f"**Figure.** {notes.get('figure', '')}")
            st.markdown(f"**Contribution.** {notes.get('contribution', '')}")
            st.markdown(f"**Method.** {notes.get('method', '')}")
            st.markdown(f"**Evidence.** `{notes.get('evidence', '')}`")
            qs = notes.get("examiner_questions") or []
            if qs:
                st.markdown("**Possible examiner questions**")
                for q in qs:
                    st.write(f"• {q}")


def _build_slide_html(slide: dict) -> str:
    bullets = "".join(f"<li>{html.escape(item)}</li>" for item in slide.get("bullets", []))
    parts = [
        '<div class="viva-slide">',
        f'<div class="viva-kicker">AMSEC PhD viva · {html.escape(slide["id"])}</div>',
        f'<div class="viva-h1">{html.escape(slide["heading"])}</div>',
    ]
    if slide.get("subtitle"):
        parts.append(f'<div class="viva-sub">{html.escape(slide["subtitle"])}</div>')
    if slide.get("review_status") == "for_review":
        note = slide.get("review_note", "This section requires alignment with the approved thesis text.")
        parts.append(f'<div class="viva-review"><strong>For review</strong> — {html.escape(note)}</div>')
    parts.append(f'<ul class="viva-bullets">{bullets}</ul>')
    if slide.get("id") == "performance":
        nb = notebook_final_metrics()
        loc = local_metrics_enriched()
        local_acc = _pct(loc["accuracy"]) if loc else "—"
        parts.append(
            '<div class="viva-metrics">'
            f'<div class="viva-metric"><div class="lab">Notebook accuracy</div><div class="val">{_pct(nb["accuracy"])}</div></div>'
            f'<div class="viva-metric"><div class="lab">Notebook F1</div><div class="val">{nb["f1"]:.5f}</div></div>'
            f'<div class="viva-metric"><div class="lab">Local accuracy</div><div class="val">{local_acc}</div></div>'
            "</div>"
            '<p style="font-family:DM Mono,monospace;font-size:0.68rem;color:#8a9bbf;margin:0.7rem 0 0;">'
            "Sources kept separate — not a unified leaderboard.</p>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_figure(kind: str) -> None:
    if kind == "architecture":
        stages = load_catalog()["pipeline_stages"]
        html_parts = ['<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.45rem;">']
        for i, stage in enumerate(stages, start=1):
            color = "#22c55e" if stage["local_status"] == "implemented" else "#ff6b35"
            html_parts.append(
                f'<div style="background:#0b1220;border:1px solid rgba(58,123,253,0.2);border-radius:10px;padding:0.55rem;">'
                f'<div style="font-family:Syne,sans-serif;color:{color};font-weight:800;">{i:02d}</div>'
                f'<div style="font-family:Syne,sans-serif;color:#e8edf5;font-size:0.82rem;">{html.escape(stage["label"])}</div>'
                f'<div style="font-family:DM Mono,monospace;color:#8a9bbf;font-size:0.58rem;">{html.escape(stage["local_status"])}</div></div>'
            )
        html_parts.append("</div>")
        st.markdown("".join(html_parts), unsafe_allow_html=True)
        return

    if kind == "dataset_dist":
        frames = []
        for key, title in [("nids", "NF-UQ-NIDS"), ("api", "API sequences"), ("syscall", "Syscalls")]:
            item = profile(key, sample_rows=1500)
            part = pd.DataFrame({"class": list(item["label_counts"]), "count": list(item["label_counts"].values())})
            part["dataset"] = title
            frames.append(part)
        data = pd.concat(frames, ignore_index=True)
        fig = px.bar(data, x="class", y="count", color="dataset", barmode="group", log_y=True)
        st.plotly_chart(style_fig(fig, "Dataset class counts (log scale)", 360), width="stretch")
        return

    if kind == "mi_ga":
        evaluation = load_local_evaluation()
        if not evaluation:
            unavailable_box("Local MI–GA logs unavailable.")
            return
        selection = evaluation["selection"]
        mi = pd.DataFrame(selection["top_mutual_information"]).head(12)
        st.plotly_chart(
            style_fig(px.bar(mi, x="score", y="feature", orientation="h"), "Local MI ranking", 340),
            width="stretch",
        )
        fitness = pd.DataFrame(
            {
                "generation": list(range(1, len(selection["ga_fitness_history"]) + 1)),
                "best_fitness": selection["ga_fitness_history"],
            }
        )
        st.plotly_chart(
            style_fig(px.line(fitness, x="generation", y="best_fitness", markers=True), "Local GA fitness", 240),
            width="stretch",
        )
        return

    if kind == "resampling":
        stack = stacking_ablations()
        st.plotly_chart(
            style_fig(px.bar(stack, x="sampling", y=["f1", "balanced_accuracy"], barmode="group"), "Cell-4 stacking", 320),
            width="stretch",
        )
        return

    if kind == "mobilenet_ablation":
        mobile = mobilenet_ablations()
        fig = px.scatter(mobile, x="parameters", y="f1", color="model", hover_data=["sampling", "balanced_accuracy"])
        st.plotly_chart(style_fig(fig, "Cell-2 · F1 vs parameters", 360), width="stretch")
        return

    if kind == "stacking_diagram":
        for i, step in enumerate(
            [
                "Named flow features → RobustScaler + PowerTransformer",
                "MobileNetV4-style Conv1D → class probabilities",
                "Lightweight Conv1D → class probabilities",
                "Stack 4 probabilities → LogisticRegression",
                "Threshold 0.5 → normal / attack",
            ],
            start=1,
        ):
            st.markdown(f"`{i:02d}`  {step}")
        evaluation = load_local_evaluation()
        if evaluation:
            coef = evaluation["meta_coefficients"]
            frame = pd.DataFrame({"component": coef["feature_order"], "coefficient": coef["coef"][0]})
            st.plotly_chart(
                style_fig(px.bar(frame, x="component", y="coefficient"), "Local meta coefficients", 300),
                width="stretch",
            )
        return

    if kind == "performance_compare":
        nb = notebook_final_metrics()
        loc = local_metrics_enriched()
        metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "roc_auc"]
        fig = go.Figure()
        fig.add_bar(name="Notebook cell-4", x=metrics, y=[nb[m] for m in metrics], marker_color="#3a7bfd")
        if loc:
            fig.add_bar(name="Local retrain", x=metrics, y=[loc[m] for m in metrics], marker_color="#00e5cc")
        st.plotly_chart(style_fig(fig, "Scalar metrics — separate experiments", 360), width="stretch")
        st.dataframe(comparison_rows()[["configuration", "source", "accuracy", "f1", "balanced_accuracy", "mcc"]], hide_index=True, width="stretch")
        return

    if kind == "bilstm_compare":
        bench = cnn_bilstm_benchmark()
        fig = px.scatter(
            bench,
            x="fpr",
            y="balanced_accuracy",
            size="f1",
            color="method",
            hover_data=["accuracy", "fnr"],
        )
        st.plotly_chart(style_fig(fig, "Cell-22 · balanced accuracy vs FPR", 360), width="stretch")
        return

    if kind == "complexity":
        table = complexity_table().dropna(subset=["parameters", "f1"])
        fig = px.scatter(
            table,
            x="parameters",
            y="f1",
            color="source",
            hover_data=["model", "experiment", "balanced_accuracy"],
            log_x=True,
        )
        st.plotly_chart(style_fig(fig, "Performance vs parameters", 360), width="stretch")
        return

    if kind == "explain":
        evaluation = load_local_evaluation()
        unavailable_box("Notebook LRP JSON unavailable; modality-level API vs network attribution Not validated for cell 4.")
        if evaluation:
            coef = evaluation["meta_coefficients"]
            frame = pd.DataFrame({"component": coef["feature_order"], "coefficient": coef["coef"][0]})
            st.plotly_chart(
                style_fig(px.bar(frame, x="component", y="coefficient"), "Local meta-learner coefficients", 320),
                width="stretch",
            )
        return

    if kind == "live_demo":
        _live_demo()
        return

    if kind == "confusion":
        local = local_metrics_enriched()
        if not local:
            unavailable_box("Local confusion matrix unavailable.")
            return
        matrix = np.array(local["confusion_matrix"]["matrix"], dtype=float)
        labels = local["confusion_matrix"]["labels"]
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=[f"Pred {x}" for x in labels],
                y=[f"True {y}" for y in labels],
                text=matrix.astype(int),
                texttemplate="%{text}",
                colorscale=[[0, "#0b1220"], [0.5, "#3a7bfd"], [1, "#00e5cc"]],
            )
        )
        st.plotly_chart(style_fig(fig, "Local hold-out confusion matrix", 360), width="stretch")
        return


def _live_demo() -> None:
    if not manifest_available():
        st.error("Live demonstration cannot run — trained model bundle unavailable.")
        return
    try:
        model = get_ensemble()
    except Exception as exc:
        st.error(f"Live demonstration cannot run — {exc}")
        return
    manifest = load_manifest() or {}
    st.success(f"Model ready · `{manifest.get('experiment_id', 'local')}`")
    choice = st.selectbox(
        "Supported sample",
        ["Parity batch row 0", "Parity batch row 1", "Zeros vector"],
        key="viva_demo_sample",
    )
    if choice == "Zeros vector":
        row = pd.DataFrame([{name: 0.0 for name in model.feature_names}])
    else:
        batch = pd.read_csv(ROOT / "artifacts" / "parity_batch.csv")
        idx = 0 if choice.endswith("0") else 1
        row = batch.iloc[[idx]][model.feature_names]
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run genuine inference", type="primary", key="viva_run_infer"):
            try:
                scored = model.predict_frame(row)
                st.session_state.viva_demo_result = scored.iloc[0].to_dict()
            except InputValidationError as exc:
                st.error(str(exc))
    with c2:
        st.button(
            "Open full Malware Detection page",
            key="viva_open_detect",
            on_click=_open_detection_page,
        )
    result = st.session_state.get("viva_demo_result")
    if result:
        st.metric("Prediction", result["amsec_prediction"])
        a, b = st.columns(2)
        a.metric("P(normal)", f"{float(result['amsec_probability_normal']):.4f}")
        b.metric("P(attack)", f"{float(result['amsec_probability_attack']):.4f}")
        st.caption(
            f"Genuine model output · threshold {result.get('amsec_threshold')} · id {result.get('amsec_model_id')}"
        )
        if st.button("Compute local sensitivity for this sample", key="viva_attrib"):
            attrib = model.attribute(row).T.reset_index()
            attrib.columns = ["feature", "sensitivity"]
            attrib = attrib.sort_values("sensitivity", key=lambda s: s.abs(), ascending=False).head(12)
            st.plotly_chart(
                style_fig(px.bar(attrib, x="sensitivity", y="feature", orientation="h"), "Local sensitivity", 360),
                width="stretch",
            )


def _export_html(slides: list[dict]) -> bytes:
    identity = load_catalog()["identity"]
    nb = notebook_final_metrics()
    loc = local_metrics_enriched()
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>AMSEC PhD Viva Pack</title>",
        "<style>body{font-family:Segoe UI,sans-serif;background:#07111f;color:#e8edf5;padding:24px;}",
        "h1,h2{font-family:Georgia,serif;} .slide{border:1px solid #234;border-radius:12px;padding:16px;margin:16px 0;background:#0b1220;}",
        ".review{color:#ffb089;} .src{color:#8a9bbf;font-size:12px;}</style></head><body>",
        f"<h1>{html.escape(identity['brand'])} — Viva presentation pack</h1>",
        f"<p>{html.escape(identity['subtitle'])}</p>",
        f"<p>{html.escape(identity['researcher'])} · {html.escape(identity['institution'])} · {html.escape(identity['student_id'])}</p>",
        f"<p class='src'>Notebook cell-4 accuracy: {_pct(nb['accuracy'])} (0.97789). "
        f"Local retrain accuracy: {_pct(loc['accuracy']) if loc else 'Unavailable'}. Sources kept separate.</p>",
        "<p class='src'>Provenance: research_results/viva_slides.json, data/verified_results.json, artifacts/local_evaluation.json</p>",
    ]
    for i, slide in enumerate(slides, start=1):
        review = " <span class='review'>[For review]</span>" if slide.get("review_status") == "for_review" else ""
        parts.append(f"<div class='slide'><h2>{i}. {html.escape(slide['heading'])}{review}</h2>")
        if slide.get("subtitle"):
            parts.append(f"<p><em>{html.escape(slide['subtitle'])}</em></p>")
        parts.append("<ul>")
        for bullet in slide.get("bullets", []):
            parts.append(f"<li>{html.escape(bullet)}</li>")
        parts.append("</ul>")
        evidence = (slide.get("notes") or {}).get("evidence", "")
        parts.append(f"<p class='src'>Evidence: {html.escape(str(evidence))}</p></div>")
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")
