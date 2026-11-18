"""Notebook vs new-execution comparison page."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from amsec.artifacts.registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[3]
VERIFIED = ROOT / "data" / "verified_results.json"


def _notebook_reference(sampling: str = "SMOTE-ENN") -> dict:
    data = json.loads(VERIFIED.read_text(encoding="utf-8"))
    for row in data.get("stacking_ablations_cell4", []):
        if row.get("sampling") == sampling:
            return {
                "source": "notebook_cell4_stored_output",
                "config": f"{sampling}_GA_Ensemble_Stacking",
                **{k: row[k] for k in ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "roc_auc") if k in row},
            }
    final = data["final_ensemble"]["metrics"]
    return {"source": "notebook_final_ensemble", "config": data["final_ensemble"]["config"], **final}


def render_reproducibility() -> None:
    st.title("Experiment Reproducibility")
    st.caption(
        "Compares stored notebook outputs with metrics from a completed Laboratory run. "
        "Differences are explained — outputs are never adjusted to force a match. "
        "Original Notebook Reproduction and Leakage-Controlled Validation remain separate protocols."
    )

    registry = ExperimentRegistry()
    completed = [r for r in registry.list_runs() if r.get("status") == "completed"]
    if not completed:
        st.warning("No completed experiment runs yet. Launch a run from Experiment Laboratory.")
        nb = _notebook_reference()
        st.subheader("Notebook reference (historical)")
        st.json(nb)
        return

    run_id = st.selectbox("Completed run", [r["run_id"] for r in completed])
    metrics = registry.load_metrics(run_id) or {}
    cfg = registry.load_configuration(run_id) or {}
    sampling = cfg.get("sampling_method", "SMOTE-ENN")
    nb = _notebook_reference(sampling)

    keys = ["accuracy", "f1", "balanced_accuracy", "mcc", "roc_auc"]
    rows = []
    for key in keys:
        left = nb.get(key)
        right = metrics.get(key)
        delta = None
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta = float(right) - float(left)
        rows.append({"Metric": key, "Original Notebook": left, "New Execution": right, "Difference": delta})
    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Protocol context")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Notebook (historical)**")
        st.write(
            "- Protocol: original cell-4 order (scaler/MI/GA/resample before split)\n"
            "- Backend: TensorFlow 2.20 Keras on Kaggle\n"
            "- Features: 5 unnamed GA features\n"
            "- Data: NF-UQ-NIDS-v2 first 150k rows"
        )
        st.json({k: nb.get(k) for k in ("source", "config")})
    with c2:
        st.markdown("**This execution**")
        st.write(
            f"- Protocol: `{cfg.get('protocol')}`\n"
            f"- Backend: `{metrics.get('backend')}`\n"
            f"- Features ({metrics.get('n_features_selected')}): {metrics.get('selected_features')}\n"
            f"- Smoke: `{cfg.get('smoke')}` · nrows `{cfg.get('nrows')}` · epochs `{cfg.get('epochs')}`"
        )
        if metrics.get("leakage_preserved"):
            st.warning("Notebook reproduction mode preserved known leakage steps.")
        if metrics.get("leakage_controls"):
            st.success("Leakage-controlled safeguards were applied.")

    st.subheader("Why values may differ")
    for item in [
        "Local CSV may be NF-UQ-NIDS-Dataset.csv rather than Kaggle NF-UQ-NIDS-v2.csv.",
        "PyTorch notebook port vs TensorFlow 2.20 graphs (and missing original HDF5 weights).",
        "GA is stochastic beyond the seed; selected feature names were never printed in the notebook.",
        "Smoke profiles use fewer rows/epochs and are not comparable to the full Kaggle run.",
        "Leakage-controlled protocol is scientifically different from notebook reproduction — not a failure of reproduction.",
        "Nondeterministic GPU kernels on Kaggle vs local CPU.",
    ]:
        st.write("• " + item)

    if st.button("Explore in Analytics"):
        st.session_state["analytics_run_id"] = run_id
        st.session_state["dashboard_page"] = "Advanced Research Analytics"
        st.rerun()
