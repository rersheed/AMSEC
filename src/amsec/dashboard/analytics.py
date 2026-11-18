"""Results & Analytics — every chart names its experiment (cell) and its source."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from amsec.dashboard.data import (
    CELL22_METRIC_NOTE, CELL24_METRIC_NOTE, METRIC_LABELS, compare_with_historical, historical,
    historical_tables, run_confusion, run_file, run_history, run_predictions, run_results, runs_for,
)
from amsec.dashboard.theme import SOURCE_COLORS, color_map, style_fig, thin_bars
from amsec.notebook.provenance import EXPERIMENTS

TEXT_COLS = ("method", "model_type", "config")


def _label_col(df: pd.DataFrame) -> str:
    if "config" in df.columns and "model_type" in df.columns and df["model_type"].nunique() > 1:
        df["label"] = df["method"].astype(str) + " · " + df["model_type"].astype(str)
    elif "entry" in df.columns and "config" in df.columns:
        df["label"] = df["entry"].astype(int).astype(str) + ". " + df.get("sampling_method", df.get("method")).astype(str)
    elif "entry" in df.columns:
        df["label"] = df["entry"].astype(int).astype(str) + ". " + df["method"].astype(str)
    else:
        df["label"] = df["method"].astype(str)
    return "label"


def metric_bar(df: pd.DataFrame, metric: str, title: str, color_by: str | None = None):
    d = df.copy()
    lab = _label_col(d)
    d = d.sort_values(metric, ascending=True)
    kwargs = {}
    if color_by and color_by in d.columns and d[color_by].nunique() > 1:
        kwargs = dict(color=color_by, color_discrete_map=color_map(d[color_by].unique().tolist()))
    fig = px.bar(d, x=metric, y=lab, orientation="h", text=d[metric].map(lambda v: f"{v:.4f}"),
                 hover_data=[c for c in d.columns if c in ("accuracy", "f1_score", "balanced_accuracy", "roc_auc", "mcc", "fpr")], **kwargs)
    fig.update_traces(textposition="outside", cliponaxis=False)
    lo = max(0.0, float(d[metric].min()) - 0.05) if d[metric].min() > 0.5 else 0.0
    fig.update_xaxes(range=[lo, min(1.02, float(d[metric].max()) + 0.03)], title=METRIC_LABELS.get(metric, metric))
    fig.update_yaxes(title="")
    return thin_bars(style_fig(fig, title, height=max(320, 34 * len(d) + 120)))


def _metric_choices(df: pd.DataFrame) -> list[str]:
    return [m for m in METRIC_LABELS if m in df.columns]


def render_analytics() -> None:
    st.title("Results & Analytics")
    options = {k: f"Cell {v['cell']} — {v['short']}" for k, v in EXPERIMENTS.items()}
    exp_id = st.selectbox("Experiment", list(options), format_func=options.get, index=list(options).index("cell04_stacking9"))
    spec = EXPERIMENTS[exp_id]
    st.caption(f"{spec['title']} · {spec['framework']} · {spec['design']}")
    st.caption(CELL22_METRIC_NOTE if exp_id == "cell22_dual_dataset" else (CELL24_METRIC_NOTE if exp_id in ("cell02_lightweight16", "cell04_stacking9") else ""))
    tabs = ["Notebook (historical)", "AMSEC runs (new)"]
    if exp_id in ("cell02_lightweight16", "cell04_stacking9"):
        tabs.append("Historical vs reproduced")
    t = st.tabs(tabs)
    with t[0]:
        _historical(exp_id)
    with t[1]:
        _runs(exp_id)
    if len(t) > 2:
        with t[2]:
            _comparison(exp_id)


# ---------------------------------------------------------------- historical
def _historical(exp_id: str) -> None:
    spec = EXPERIMENTS[exp_id]
    if spec.get("untraceable"):
        st.error("Cell 7 has **no source code**; its saved outputs cannot be traced to an implementation. Shown for audit only.")
    if spec.get("no_saved_outputs"):
        st.info("The corrected notebook stores **no outputs** for cell 22. There is nothing historical to show — see AMSEC runs.")
        return
    tables = historical_tables(exp_id)
    if not tables:
        st.info("No results table in the saved outputs.")
        return
    for table_id, t in tables.items():
        if not t.get("rows"):
            continue
        df = pd.DataFrame(t["rows"])
        st.markdown(f"**{t['description']}** — source: notebook cell {spec['cell']} saved output (`{table_id}`)")
        metric = st.selectbox("Metric", _metric_choices(df), index=_metric_choices(df).index("f1_score"), key=f"h_{exp_id}_{table_id}")
        st.plotly_chart(metric_bar(df, metric, f"Cell {spec['cell']} · {METRIC_LABELS[metric]} · notebook", color_by="model_type"),
                        width="stretch", key=f"hc_{exp_id}_{table_id}")
        if {"f1_score", "balanced_accuracy"} <= set(df.columns) and exp_id not in ("cell02_lightweight16", "cell04_stacking9"):
            fig = px.scatter(df, x="balanced_accuracy", y="f1_score", text="method", hover_data=[c for c in ("fpr", "specificity", "accuracy") if c in df.columns])
            fig.update_traces(marker=dict(size=11, line=dict(width=2, color="#0b1220")), textposition="top center")
            fig.update_xaxes(title="Balanced accuracy"); fig.update_yaxes(title="F1 (malware class)")
            st.plotly_chart(style_fig(fig, f"Cell {spec['cell']} · F1 vs balanced accuracy · notebook", 400), width="stretch", key=f"hs_{exp_id}_{table_id}")
            st.caption("High F1 with balanced accuracy near 0.5 means the benign minority is mostly misclassified.")
        st.dataframe(df, hide_index=True, width="stretch")
    fs = historical()["experiments"][exp_id].get("feature_selection")
    if fs and fs.get("checkpoints"):
        st.markdown("**Genetic algorithm convergence (printed every 5 generations)** — notebook")
        g = pd.DataFrame(fs["checkpoints"])
        fig = px.line(g, x="generation", y="best_fitness", markers=True, hover_data=["n_features"])
        fig.update_traces(line_width=2, marker_size=9)
        st.plotly_chart(style_fig(fig, f"Cell {spec['cell']} · GA best fitness · notebook", 320), width="stretch", key=f"ga_{exp_id}")
        st.caption(f"NIDS shape {fs['nids_shape']}, labels {fs['label_counts']}, {fs['after_variance_threshold']} features after variance threshold, "
                   f"GA kept {fs['selected']} of {fs['candidates']}. Feature names were not printed by the notebook.")


# ---------------------------------------------------------------- runs
def _runs(exp_id: str) -> None:
    runs = runs_for(exp_id)
    if not runs:
        st.info("No completed AMSEC run for this experiment." + (" Launch one in the Experiment Laboratory." if EXPERIMENTS[exp_id].get("executable") else " This experiment is historical-only."))
        return
    run = st.selectbox("Run", runs, format_func=lambda r: f"{r['run_id']} · {r['kind']} · {r['created_at']}", key=f"run_{exp_id}")
    if run["kind"] == "smoke":
        st.warning("Smoke run — reduced parameters. Not a scientific result.")
    elif run["kind"] == "partial":
        st.info(f"Partial run — configurations: {run['selection']}.")
    df = run_results(run)
    if df.empty:
        st.error("Run has no results.")
        return
    metric = st.selectbox("Metric", _metric_choices(df), index=_metric_choices(df).index("f1_score"), key=f"rm_{run['run_id']}")
    st.plotly_chart(metric_bar(df, metric, f"Cell {run['cell']} · {METRIC_LABELS[metric]} · {run['run_id']}", color_by="model_type"),
                    width="stretch", key=f"rc_{run['run_id']}")
    st.dataframe(df.drop(columns=[c for c in ("confusion_matrix", "class_counts_after_sampling") if c in df.columns]), hide_index=True, width="stretch")

    keys = df["method"].tolist() if exp_id == "cell22_dual_dataset" else df.sort_values("entry")["entry"].astype(int).tolist()
    fmt = (lambda k: k) if exp_id == "cell22_dual_dataset" else (lambda k: f"{k}. {df[df['entry'] == k]['config'].iloc[0]}")
    key = st.selectbox("Configuration detail", keys, format_func=fmt, key=f"rk_{run['run_id']}")
    c1, c2 = st.columns(2)
    with c1:
        cm = run_confusion(run, key)
        if cm:
            names = ["benign/normal", "malware/attack"]
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, colorscale=[[0, "#0b1220"], [1, "#3987e5"]], text=cm, texttemplate="%{text}",
                                       hovertemplate="true %{y}<br>predicted %{x}<br>%{z}<extra></extra>", showscale=False))
            fig.update_xaxes(title="Predicted"); fig.update_yaxes(title="True", autorange="reversed")
            st.plotly_chart(style_fig(fig, f"Confusion matrix · {run['run_id']}", 340), width="stretch", key=f"cm_{run['run_id']}_{key}")
    with c2:
        pred = run_predictions(run, key)
        if pred is not None and pred["y_true"].nunique() == 2:
            fpr, tpr, _ = roc_curve(pred["y_true"], pred["proba"])
            prec, rec, _ = precision_recall_curve(pred["y_true"], pred["proba"])
            curves = pd.concat([
                pd.DataFrame({"x": fpr, "y": tpr, "curve": f"ROC (AUC {auc(fpr, tpr):.4f})"}),
                pd.DataFrame({"x": rec, "y": prec, "curve": f"Precision–recall (AP area {auc(rec, prec):.4f})"}),
            ])
            fig = px.line(curves, x="x", y="y", color="curve", color_discrete_sequence=["#3987e5", "#d95926"])
            fig.update_traces(line_width=2)
            fig.update_xaxes(title="FPR / recall", range=[0, 1]); fig.update_yaxes(title="TPR / precision", range=[0, 1.02])
            st.plotly_chart(style_fig(fig, f"Curves from saved test probabilities · {run['run_id']}", 340), width="stretch", key=f"roc_{run['run_id']}_{key}")
    hist = run_history(run, key)
    if hist is not None and len(hist):
        for measure in ("loss", "accuracy"):
            h = hist.dropna(subset=[measure]).copy()
            h["series_label"] = h["model"] + " · " + h["series"]
            fig = px.line(h, x="epoch", y=measure, color="series_label", markers=True, color_discrete_map=color_map(h["series_label"].unique().tolist()))
            fig.update_traces(line_width=2, marker_size=8)
            st.plotly_chart(style_fig(fig, f"Training {measure} · {run['run_id']}", 320), width="stretch", key=f"h_{measure}_{run['run_id']}_{key}")
    ga = run_file(run, "ga_fitness.csv")
    if ga.exists():
        g = pd.read_csv(ga).assign(source="AMSEC run (new)")
        hist_fs = historical()["experiments"][exp_id].get("feature_selection", {}).get("checkpoints", [])
        hg = pd.DataFrame(hist_fs).assign(source="Notebook (historical)") if hist_fs else pd.DataFrame()
        fig = px.line(pd.concat([g, hg]), x="generation", y="best_fitness", color="source", markers=True, color_discrete_map=SOURCE_COLORS)
        fig.update_traces(line_width=2, marker_size=8)
        st.plotly_chart(style_fig(fig, "GA best fitness — run vs notebook checkpoints", 320), width="stretch", key=f"ga_{run['run_id']}")
        feats = run_file(run, "selected_features.json")
        if feats.exists():
            import json
            fr = json.loads(feats.read_text(encoding="utf-8"))
            st.markdown(f"**Selected features ({len(fr['selected_features'])}):** " + ", ".join(f"`{f}`" for f in fr["selected_features"]))
    mi = run_file(run, "nids_mutual_information_scores.csv")
    if mi.exists():
        m = pd.read_csv(mi).head(15).sort_values("mi_score")
        fig = px.bar(m, x="mi_score", y="feature", orientation="h")
        st.plotly_chart(thin_bars(style_fig(fig, f"NIDS mutual information (train split, top 15) · {run['run_id']}", 460)), width="stretch", key=f"mi_{run['run_id']}")


# ---------------------------------------------------------------- comparison
def _comparison(exp_id: str) -> None:
    runs = [r for r in runs_for(exp_id) if r["kind"] != "smoke"]
    if not runs:
        st.info("No non-smoke run to compare with the notebook.")
        return
    run = st.selectbox("Run", runs, format_func=lambda r: f"{r['run_id']} · {r['kind']}", key=f"cmp_{exp_id}")
    cmp = compare_with_historical(exp_id, run)
    if cmp.empty:
        st.info("Nothing to compare.")
        return
    metric = st.selectbox("Metric", ["f1_score", "accuracy", "roc_auc", "balanced_accuracy", "mcc"], key=f"cmpm_{exp_id}")
    long = pd.concat([
        pd.DataFrame({"entry": cmp["entry"].astype(str) + ". " + cmp["config"], "value": cmp[f"{metric}_notebook"], "source": "Notebook (historical)"}),
        pd.DataFrame({"entry": cmp["entry"].astype(str) + ". " + cmp["config"], "value": cmp[f"{metric}_reproduced"], "source": "AMSEC run (new)"}),
    ])
    fig = px.bar(long, x="value", y="entry", color="source", barmode="group", orientation="h", color_discrete_map=SOURCE_COLORS)
    lo = max(0.0, float(long["value"].min()) - 0.03)
    fig.update_xaxes(range=[lo, min(1.0, float(long["value"].max()) + 0.01)], title=METRIC_LABELS[metric]); fig.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(thin_bars(style_fig(fig, f"Cell {EXPERIMENTS[exp_id]['cell']} · {METRIC_LABELS[metric]} · notebook vs {run['run_id']}", 34 * 2 * len(cmp) + 140)),
                    width="stretch", key=f"cmpc_{exp_id}")
    d = cmp[f"{metric}_delta"]
    c = st.columns(3)
    c[0].metric("Mean Δ (run − notebook)", f"{d.mean():+.5f}")
    c[1].metric("Max |Δ|", f"{d.abs().max():.5f}")
    if "split_sizes_match" in cmp:
        c[2].metric("Split sizes identical", f"{int(cmp['split_sizes_match'].sum())}/{len(cmp)}")
    st.dataframe(cmp, hide_index=True, width="stretch")
    st.caption("Exact checks (sample counts after resampling, split sizes, parameter counts) test the deterministic part of the pipeline. "
               "Metric deltas include GPU-vs-CPU and Keras-version training nondeterminism; the notebook itself shows ~0.002 F1 spread "
               "between its SMOTE-equivalent entries (1, 5, 9).")
