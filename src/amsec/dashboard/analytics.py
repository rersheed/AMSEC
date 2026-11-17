"""Advanced Research Analytics workspace — ten scientific tabs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from amsec.dashboard.theme import source_chip, style_fig, unavailable_box
from amsec.inference import get_ensemble, manifest_available
from amsec.research import (
    ROOT,
    artifact_file_sizes,
    cnn_bilstm_benchmark,
    complexity_table,
    comparison_rows,
    load_catalog,
    load_local_evaluation,
    load_manifest,
    load_verified,
    local_metrics_enriched,
    mobilenet_ablations,
    notebook_final_metrics,
    stacking_ablations,
)


METRIC_HELP = {
    "accuracy": "Overall correct predictions / total. Misleading under severe imbalance.",
    "precision": "Of predicted positives, how many are true positives.",
    "recall": "Of actual positives, how many were found (sensitivity / TPR).",
    "f1": "Harmonic mean of precision and recall.",
    "balanced_accuracy": "Mean of recall for each class — preferred under imbalance.",
    "mcc": "Matthews correlation; robust for binary imbalance.",
    "roc_auc": "Area under ROC. Scalar only here unless probability curves were exported.",
    "fpr": "False positive rate = FP / (FP + TN).",
    "fnr": "False negative rate = FN / (FN + TP).",
}


def _fmt(value, digits=5):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return f"{value:,}"


def render_analytics() -> None:
    st.title("Research Analytics")
    st.caption("Publication-oriented workspace. Every chart is gated on an explicit source. Fabricated curves are not shown.")
    experiment = st.selectbox(
        "Primary experiment focus",
        [
            "Notebook cell-4 SMOTE-ENN stack",
            "Local leakage-corrected retrain",
            "Cell-2 MobileNet ablations",
            "Cell-22 CNN–BiLSTM benchmark",
        ],
    )
    tabs = st.tabs(
        [
            "1 · Performance",
            "2 · Training",
            "3 · Confusion",
            "4 · ROC / PR",
            "5 · MI–GA",
            "6 · Resampling",
            "7 · Complexity",
            "8 · Explainability",
            "9 · Statistics",
            "10 · Reproducibility",
        ]
    )
    with tabs[0]:
        _tab_performance(experiment)
    with tabs[1]:
        _tab_training()
    with tabs[2]:
        _tab_confusion()
    with tabs[3]:
        _tab_roc_pr()
    with tabs[4]:
        _tab_feature_selection()
    with tabs[5]:
        _tab_resampling()
    with tabs[6]:
        _tab_complexity()
    with tabs[7]:
        _tab_explain()
    with tabs[8]:
        _tab_statistics()
    with tabs[9]:
        _tab_reproducibility()


def _tab_performance(experiment: str) -> None:
    st.subheader("Experimental performance")
    with st.expander("Metric definitions", expanded=False):
        for key, text in METRIC_HELP.items():
            st.markdown(f"**{key}** — {text}")

    st.markdown(f"**Focus:** {experiment}")
    st.caption(source_chip("mixed"))

    notebook = notebook_final_metrics()
    local = local_metrics_enriched()
    scorecards = st.columns(4)
    scorecards[0].metric("Notebook accuracy", _fmt(notebook["accuracy"]))
    scorecards[1].metric("Notebook F1", _fmt(notebook["f1"]))
    if local:
        scorecards[2].metric("Local accuracy", _fmt(local["accuracy"]))
        scorecards[3].metric("Local F1", _fmt(local["f1"]))
    else:
        scorecards[2].metric("Local accuracy", "—")
        scorecards[3].metric("Local F1", "—")

    st.markdown("##### Notebook versus local (non-equivalent protocols)")
    st.dataframe(comparison_rows(), hide_index=True, width="stretch")

    stack = stacking_ablations()
    st.markdown("##### Cell-4 stacking ablations (same notebook protocol)")
    st.caption(source_chip("notebook"))
    metric = st.selectbox("Metric for ranking chart", ["f1", "balanced_accuracy", "accuracy", "mcc", "roc_auc"], key="perf_metric")
    fig = px.bar(stack.sort_values(metric, ascending=False), x="sampling", y=metric, color="sampling", text=metric)
    st.plotly_chart(style_fig(fig, f"Cell-4 stacking · {metric}", 420), width="stretch")

    radar_metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "roc_auc"]
    if local:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=[notebook[m] for m in radar_metrics] + [notebook["accuracy"]], theta=radar_metrics + ["accuracy"], fill="toself", name="Notebook"))
        fig_r.add_trace(go.Scatterpolar(r=[local[m] for m in radar_metrics] + [local["accuracy"]], theta=radar_metrics + ["accuracy"], fill="toself", name="Local retrain"))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.9, 1.0])))
        st.plotly_chart(style_fig(fig_r, "Radar of scalar metrics (protocols differ)", 420), width="stretch")

    st.markdown("##### Cell-2 MobileNet ablations")
    st.caption(source_chip("notebook"))
    mobile = mobilenet_ablations()
    st.dataframe(mobile, hide_index=True, width="stretch")
    fig_m = px.scatter(mobile, x="parameters", y="f1", color="model", hover_data=["sampling", "balanced_accuracy", "roc_auc"], size="roc_auc")
    st.plotly_chart(style_fig(fig_m, "Cell-2 · F1 versus parameters", 420), width="stretch")

    st.markdown("##### Cell-22 CNN–BiLSTM benchmark")
    st.caption(source_chip("notebook"))
    bench = cnn_bilstm_benchmark()
    st.dataframe(bench, hide_index=True, width="stretch")
    st.download_button("Export performance tables (CSV)", _bundle_performance_csv(), "amsec_performance_tables.csv", "text/csv")


def _bundle_performance_csv() -> bytes:
    parts = [
        comparison_rows().assign(table="notebook_vs_local"),
        stacking_ablations().assign(table="cell4_stack"),
        mobilenet_ablations().assign(table="cell2_mobilenet"),
        cnn_bilstm_benchmark().assign(table="cell22_bilstm"),
    ]
    return pd.concat(parts, ignore_index=True).to_csv(index=False).encode("utf-8")


def _tab_training() -> None:
    st.subheader("Training behaviour")
    evaluation = load_local_evaluation()
    if evaluation is None:
        unavailable_box("No local training history. Train with scripts/train_local_ensemble.py.")
        return
    st.caption(source_chip("verified") + " · Loss histories only (accuracy per epoch was not logged).")
    mobile = pd.DataFrame(evaluation["mobile_history"])
    light = pd.DataFrame(evaluation["lightweight_history"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=["MobileNet-style loss", "Lightweight CNN loss"])
    fig.add_trace(go.Scatter(x=mobile["epoch"], y=mobile["train_loss"], name="Mobile train", line=dict(color="#3a7bfd")), row=1, col=1)
    fig.add_trace(go.Scatter(x=mobile["epoch"], y=mobile["val_loss"], name="Mobile val", line=dict(color="#00e5cc", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=light["epoch"], y=light["train_loss"], name="Light train", line=dict(color="#8b5cf6")), row=1, col=2)
    fig.add_trace(go.Scatter(x=light["epoch"], y=light["val_loss"], name="Light val", line=dict(color="#ff6b35", dash="dot")), row=1, col=2)
    st.plotly_chart(style_fig(fig, "Local retrain epoch histories", 420), width="stretch")

    lr_fig = go.Figure()
    lr_fig.add_trace(go.Scatter(x=mobile["epoch"], y=mobile["lr"], name="MobileNet LR", mode="lines+markers"))
    lr_fig.add_trace(go.Scatter(x=light["epoch"], y=light["lr"], name="Lightweight LR", mode="lines+markers"))
    st.plotly_chart(style_fig(lr_fig, "Learning-rate progression", 320), width="stretch")

    best_m = int(mobile["val_loss"].idxmin())
    best_l = int(light["val_loss"].idxmin())
    st.info(
        f"MobileNet-style best validation loss at epoch {mobile.loc[best_m, 'epoch']} "
        f"({mobile.loc[best_m, 'val_loss']:.4f}). Lightweight best at epoch {light.loc[best_l, 'epoch']} "
        f"({light.loc[best_l, 'val_loss']:.4f}). Early-stop restored those checkpoints."
    )
    gap_m = float(mobile.iloc[-1]["val_loss"] - mobile.iloc[-1]["train_loss"])
    gap_l = float(light.iloc[-1]["val_loss"] - light.iloc[-1]["train_loss"])
    st.write(f"Final train–val loss gap · MobileNet-style: **{gap_m:.4f}** · Lightweight: **{gap_l:.4f}**")
    unavailable_box(
        "Notebook training/validation accuracy curves were not exported from Kaggle. "
        "Do not invent exponential learning curves. Re-run the notebook with history export to populate this tab for the original experiment."
    )
    st.download_button(
        "Export local training histories",
        pd.concat([mobile.assign(model="mobilenet"), light.assign(model="lightweight")]).to_csv(index=False),
        "amsec_local_training_history.csv",
        "text/csv",
    )


def _tab_confusion() -> None:
    st.subheader("Confusion matrix and classification errors")
    local = local_metrics_enriched()
    mode = st.radio("Matrix display", ["Raw counts", "Row-normalised percentages"], horizontal=True)
    if local is None:
        unavailable_box("Local hold-out confusion matrix unavailable.")
    else:
        st.caption(source_chip("verified") + " · Local leakage-corrected hold-out test set.")
        matrix = np.array(local["confusion_matrix"]["matrix"], dtype=float)
        labels = local["confusion_matrix"]["labels"]
        z = matrix.copy()
        text = matrix.astype(int).astype(str)
        if mode.startswith("Row"):
            z = matrix / matrix.sum(axis=1, keepdims=True)
            text = np.vectorize(lambda v: f"{v:.1%}")(z)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[f"Pred {x}" for x in labels],
                y=[f"True {y}" for y in labels],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#0b1220"], [0.5, "#3a7bfd"], [1, "#00e5cc"]],
                showscale=True,
            )
        )
        st.plotly_chart(style_fig(fig, "Local hold-out confusion matrix", 420), width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TN", local["tn"])
        c2.metric("FP", local["fp"])
        c3.metric("FN", local["fn"])
        c4.metric("TP", local["tp"])
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("FPR", _fmt(local["fpr"]))
        d2.metric("FNR", _fmt(local["fnr"]))
        d3.metric("Attack precision", _fmt(local["precision_attack"]))
        d4.metric("Attack recall", _fmt(local["recall_attack"]))
        err = pd.DataFrame(
            {
                "rate": ["FPR", "FNR", "Error rate"],
                "value": [local["fpr"], local["fnr"], local["error_rate"]],
            }
        )
        st.plotly_chart(style_fig(px.bar(err, x="rate", y="value", text="value"), "Local error rates", 320), width="stretch")

    unavailable_box(
        "Notebook cell-4 SMOTE-ENN confusion matrix arrays were not serialised into this package. "
        "Only scalar metrics were transcribed."
    )
    st.markdown("##### Cell-22 benchmark false-alarm profile")
    st.caption(source_chip("notebook"))
    bench = cnn_bilstm_benchmark()[["method", "accuracy", "f1", "balanced_accuracy", "fpr", "fnr", "specificity"]]
    st.dataframe(bench, hide_index=True, width="stretch")
    st.plotly_chart(
        style_fig(px.bar(bench, x="method", y=["fpr", "balanced_accuracy"], barmode="group"), "Cell-22 FPR vs balanced accuracy", 420),
        width="stretch",
    )


def _tab_roc_pr() -> None:
    st.subheader("ROC and precision–recall analysis")
    notebook = notebook_final_metrics()
    local = local_metrics_enriched()
    c1, c2 = st.columns(2)
    c1.metric("Notebook ROC-AUC (scalar)", _fmt(notebook["roc_auc"]))
    c1.caption(source_chip("notebook"))
    if local:
        c2.metric("Local ROC-AUC (scalar)", _fmt(local["roc_auc"]))
        c2.caption(source_chip("verified"))
    unavailable_box(
        "Full ROC and precision–recall curves require exported test probability vectors. "
        "Those arrays were not saved for the notebook run or the local hold-out. "
        "Scalar AUC values above are transcribed / verified; curves are not fabricated from a single AUC."
    )
    unavailable_box(
        "Precision/recall versus threshold and TPR/FPR versus threshold plots are Not validated here for the same reason."
    )
    st.info(
        "Class imbalance implication: high accuracy with elevated FPR (see cell-22 ADASYN) can look strong on ROC-AUC "
        "while harming specificity. Prefer balanced accuracy, MCC and FPR together."
    )


def _tab_feature_selection() -> None:
    st.subheader("Feature engineering and MI–GA")
    evaluation = load_local_evaluation()
    verified = load_verified()
    if evaluation:
        selection = evaluation["selection"]
        st.caption(source_chip("verified") + " · Local retrain selection.")
        a, b, c, d = st.columns(4)
        a.metric("After variance threshold", selection["features_after_variance_threshold"])
        b.metric("MI pool (k)", selection["mutual_information_k"])
        c.metric("Selected features", len(selection["selected_features"]))
        reduction = 1 - len(selection["selected_features"]) / max(selection["features_after_variance_threshold"], 1)
        d.metric("Dimensionality reduction", f"{reduction:.1%}")
        mi = pd.DataFrame(selection["top_mutual_information"])
        st.plotly_chart(
            style_fig(px.bar(mi, x="score", y="feature", orientation="h", title=""), "Local MI ranking (top 15)", 480),
            width="stretch",
        )
        st.markdown("**Selected local feature names**")
        st.code(", ".join(selection["selected_features"]))
        fitness = pd.DataFrame({"generation": list(range(1, len(selection["ga_fitness_history"]) + 1)), "best_fitness": selection["ga_fitness_history"]})
        st.plotly_chart(style_fig(px.line(fitness, x="generation", y="best_fitness", markers=True), "Local GA fitness convergence", 320), width="stretch")
        st.download_button("Export local MI ranking", mi.to_csv(index=False), "amsec_local_mi_ranking.csv", "text/csv")
    else:
        unavailable_box("Local MI–GA logs missing.")

    st.markdown("##### Notebook cell-4 selection")
    st.caption(source_chip("notebook"))
    st.write("Variance threshold retained 41 features; GA kept **5**. Feature names were **not printed**.")
    unavailable_box("Notebook MI ranking table and GA convergence series were not exported as machine-readable files.")

    st.markdown("##### Cell-22 API MI features")
    st.caption(source_chip("notebook"))
    api_mi = pd.DataFrame(verified["cell22_api_mi_features"])
    st.dataframe(api_mi, hide_index=True, width="stretch")
    st.plotly_chart(style_fig(px.bar(api_mi, x="feature", y="mi_score", text="mi_score"), "Cell-22 API MI top-5", 320), width="stretch")


def _tab_resampling() -> None:
    st.subheader("Class imbalance and resampling")
    criterion = st.selectbox(
        "Best-strategy criterion",
        ["f1", "balanced_accuracy", "accuracy", "mcc", "roc_auc"],
        help="The winner can change with the chosen criterion.",
    )
    st.markdown("##### Cell-4 stacking ensemble")
    st.caption(source_chip("notebook") + " · Notebook applied resampling before the split.")
    stack = stacking_ablations().sort_values(criterion, ascending=False)
    st.dataframe(stack, hide_index=True, width="stretch")
    st.success(f"Best under **{criterion}**: `{stack.iloc[0]['sampling']}` ({criterion}={stack.iloc[0][criterion]:.5f})")
    st.plotly_chart(
        style_fig(px.bar(stack, x="sampling", y=["accuracy", "balanced_accuracy", "f1"], barmode="group"), "Cell-4 resampling comparison", 420),
        width="stretch",
    )

    evaluation = load_local_evaluation()
    if evaluation:
        st.markdown("##### Local retrain class counts")
        st.caption(source_chip("verified") + " · SMOTE-ENN on training fold only.")
        split = evaluation["split"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train (pre-resample)", f"{split['train']:,}")
        c2.metric("Validation", f"{split['validation']:,}")
        c3.metric("Test", f"{split['test']:,}")
        c4.metric("Train after SMOTE-ENN", f"{split['resampled_train']:,}")

    st.markdown("##### Cell-22 CNN–BiLSTM resampling")
    st.caption(source_chip("notebook"))
    bench = cnn_bilstm_benchmark().sort_values(criterion if criterion in cnn_bilstm_benchmark().columns else "f1", ascending=False)
    # cell22 may lack mcc in sort for some criteria - use available
    sort_key = criterion if criterion in bench.columns else "f1"
    bench = cnn_bilstm_benchmark().sort_values(sort_key, ascending=False)
    st.dataframe(bench, hide_index=True, width="stretch")
    st.warning(
        "Under raw F1, ADASYN wins in cell 22, but balanced accuracy falls to 0.6937 and FPR rises to 0.6111. "
        "SMOTE-ENN collapses. Criterion choice matters."
    )
    fig = px.scatter(bench, x="fpr", y="balanced_accuracy", size="f1", color="method", hover_data=["accuracy", "fnr"])
    st.plotly_chart(style_fig(fig, "Cell-22 · balanced accuracy versus FPR", 420), width="stretch")


def _tab_complexity() -> None:
    st.subheader("Model architecture and complexity")
    table = complexity_table()
    st.dataframe(table, hide_index=True, width="stretch")
    plot_df = table.dropna(subset=["parameters", "f1"])
    fig = px.scatter(
        plot_df,
        x="parameters",
        y="f1",
        color="source",
        hover_data=["model", "experiment", "balanced_accuracy", "flops_m"],
        size=[20] * len(plot_df),
        log_x=True,
    )
    st.plotly_chart(style_fig(fig, "Performance versus parameter complexity", 420), width="stretch")
    st.caption("FLOPs use notebook definitions (often 2×parameters). Local PyTorch FLOPs were not profiled. No Android on-device latency is claimed.")

    sizes = artifact_file_sizes()
    st.markdown("##### Local artifact sizes")
    st.caption(source_chip("verified"))
    st.dataframe(sizes, hide_index=True, width="stretch")

    st.markdown("##### Ensemble architecture (served local model)")
    stages = [
        "Input features (named NIDS columns)",
        "RobustScaler + PowerTransformer",
        "MobileNetV4-style Conv1D",
        "Lightweight Conv1D",
        "Stack 4 class probabilities",
        "LogisticRegression meta-learner",
        "Threshold 0.5 → normal / attack",
    ]
    for index, stage in enumerate(stages, start=1):
        st.markdown(f"`{index:02d}`  {stage}")


def _tab_explain() -> None:
    st.subheader("Explainable AI")
    evaluation = load_local_evaluation()
    unavailable_box(
        "Notebook LRP JSON (`xai_analysis/lrp_results_best.json`) is not in this package. "
        "API versus network modality attribution is Not validated for the executed final ensemble "
        "(API features were not loaded in that run)."
    )
    if evaluation is None or not manifest_available():
        unavailable_box("Local explainability requires the trained artifact bundle.")
        return
    st.caption(source_chip("verified"))
    coef = evaluation["meta_coefficients"]
    frame = pd.DataFrame({"component": coef["feature_order"], "coefficient": coef["coef"][0]})
    st.plotly_chart(style_fig(px.bar(frame, x="component", y="coefficient", text="coefficient"), "Meta-learner coefficients (global)", 360), width="stretch")
    st.write(f"Intercept: `{coef['intercept'][0]:.6f}` · Classes: {coef['classes']}")

    st.markdown("##### Local sample sensitivity")
    model = get_ensemble()
    sample = st.selectbox("Case study sample", ["Zeros vector", "Parity batch row 0", "Parity batch row 1"])
    if sample == "Zeros vector":
        row = pd.DataFrame([{name: 0.0 for name in model.feature_names}])
    else:
        batch = pd.read_csv(ROOT / "artifacts" / "parity_batch.csv")
        idx = 0 if sample.endswith("0") else 1
        row = batch.iloc[[idx]][model.feature_names]
    scored = model.predict_frame(row)
    st.write(scored[["amsec_prediction", "amsec_probability_normal", "amsec_probability_attack"]])
    if st.button("Compute finite-difference sensitivity for selected sample"):
        attrib = model.attribute(row).T.reset_index()
        attrib.columns = ["feature", "sensitivity"]
        attrib = attrib.sort_values("sensitivity", key=lambda s: s.abs(), ascending=False)
        st.plotly_chart(style_fig(px.bar(attrib, x="sensitivity", y="feature", orientation="h"), "Local sensitivity · ΔP(attack)/Δfeature", 520), width="stretch")
        st.download_button("Export sensitivity CSV", attrib.to_csv(index=False), "amsec_local_sensitivity.csv", "text/csv")


def _tab_statistics() -> None:
    st.subheader("Statistical and scientific validation")
    st.error("**Not established — repeated experiments required.**")
    st.markdown(
        """
        No repeated independent training runs with saved folds were supplied for the notebook ensemble
        or the local retrain. Therefore this console does **not** compute:

        - means / standard deviations across runs  
        - confidence intervals from repeated trials  
        - paired t-tests or Wilcoxon tests  
        - effect sizes from multi-run designs  

        A t-test must not be calculated from unrelated single accuracy scores (for example notebook 0.97789 versus local 0.9663),
        because those figures come from non-equivalent protocols.
        """
    )
    st.info(
        "Evaluation methodology note: notebook cell-4 metrics follow a resample-before-split design with leakage risk. "
        "Local metrics follow split-before-fit on 22,500 held-out rows. Sample sizes are documented in the Reproducibility tab."
    )


def _tab_reproducibility() -> None:
    st.subheader("Reproducibility and export")
    catalog = load_catalog()
    manifest = load_manifest()
    evaluation = load_local_evaluation()
    verified = load_verified()

    st.markdown("##### Dataset and notebook sources")
    st.write(f"- Notebook: `{catalog['experiments'][0]['notebook']}`")
    st.write("- Local dataset file: `others files that can help/NF-UQ-NIDS-Dataset.csv`")
    st.write("- Verified transcription: `data/verified_results.json`")
    st.write("- Experiment registry: `research_results/catalog.json`")

    if manifest:
        st.markdown("##### Local model manifest")
        st.caption(source_chip("verified"))
        st.json(
            {
                "experiment_id": manifest["experiment_id"],
                "status": manifest["status"],
                "claim": manifest["claim"],
                "feature_names": manifest["feature_names"],
                "decision_threshold": manifest["decision_threshold"],
                "hashes": manifest["hashes"],
                "software": manifest["software"],
                "training": manifest["training"],
            }
        )
    else:
        unavailable_box("model_manifest.json missing.")

    if evaluation:
        st.markdown("##### Local evaluation protocol")
        st.json({"split": evaluation["split"], "rows_loaded": evaluation["rows_loaded"], "label_column": evaluation["label_column"]})

    st.markdown("##### Leakage notes (historical)")
    for note in verified["leakage_notes"]:
        st.write(f"• {note}")

    report = {
        "identity": catalog["identity"],
        "notebook_metrics": notebook_final_metrics(),
        "local_metrics": local_metrics_enriched(),
        "manifest_hashes": (manifest or {}).get("hashes"),
    }
    st.download_button(
        "Download reproducibility JSON",
        json.dumps(report, indent=2),
        "amsec_reproducibility_snapshot.json",
        "application/json",
    )
    html = f"""
    <html><head><meta charset='utf-8'><title>AMSEC reproducibility</title></head>
    <body style='font-family:Segoe UI,sans-serif;background:#07111f;color:#e8edf5;padding:24px;'>
    <h1>AMSEC reproducibility snapshot</h1>
    <p>Notebook accuracy: {notebook_final_metrics()['accuracy']}</p>
    <p>Local accuracy: {(local_metrics_enriched() or {}).get('accuracy')}</p>
    <pre>{json.dumps(report, indent=2)}</pre>
    </body></html>
    """
    st.download_button("Download HTML summary", html.encode("utf-8"), "amsec_reproducibility.html", "text/html")
