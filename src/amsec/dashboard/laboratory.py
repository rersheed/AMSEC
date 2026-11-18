"""Experiment Laboratory — launch faithful notebook experiments as auditable runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from amsec.lab.runs import EXECUTABLE, SMOKE_OVERRIDES, create_run, launch_background, list_runs, read_json
from amsec.notebook.cell22_dual_dataset import ALL_METHODS, METHOD_SEMANTICS, NOTEBOOK_HYPERPARAMETERS
from amsec.notebook.keras_pipeline import NOTEBOOK_DEFAULTS, SAMPLER_SEMANTICS, config_list

ESTIMATES = {
    "cell22_dual_dataset": "≈1–2 min on a laptop CPU (all 9 methods).",
    "cell04_stacking9": "≈1–2 h on CPU for all 9 entries + phases 7–9 (GA ≈ 10–20 min). Notebook ran on 2 GPUs.",
    "cell02_lightweight16": "≈1–2 h on CPU for all 16 configurations + phases 7–9.",
}


def _keras_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def render_laboratory() -> None:
    st.title("Experiment Laboratory")
    st.caption("Runs execute the notebook's own code (verbatim definitions, same order, same seeds) in a background worker. "
               "Every run is written to `experiment_runs/run_XXXX` with its parameters, deviations, environment and dataset fingerprints.")
    labels = {k: f"Cell {v['cell']} — {v['title']}" for k, v in EXECUTABLE.items()}
    exp_id = st.selectbox("Experiment", list(labels), format_func=labels.get, index=list(labels).index("cell22_dual_dataset"))
    spec = EXECUTABLE[exp_id]
    st.markdown(f"**Design.** {spec['design']}  \n**Framework.** {spec['framework']} · **Module.** `{spec['module']}`")
    if exp_id != "cell22_dual_dataset" and not _keras_available():
        st.error("TensorFlow is not installed in this environment, so cells 2 and 4 cannot run here. "
                 "Install `requirements-experiments.txt` (Python 3.11) — AMSEC never substitutes a PyTorch look-alike.")
        return

    import imblearn
    import sklearn

    from amsec.notebook.provenance import REPRODUCIBLE_PINS

    if (sklearn.__version__, imblearn.__version__) != (REPRODUCIBLE_PINS["sklearn"], REPRODUCIBLE_PINS["imblearn"]):
        st.warning(f"This environment has scikit-learn {sklearn.__version__} / imbalanced-learn {imblearn.__version__}. "
                   f"The notebook's GA feature selection reproduces exactly only with scikit-learn {REPRODUCIBLE_PINS['sklearn']} "
                   f"(imbalanced-learn {REPRODUCIBLE_PINS['imblearn']}); runs here may select different features.")
    with st.expander("Notebook hyperparameters and sampler semantics", expanded=False):
        if exp_id == "cell22_dual_dataset":
            st.json(NOTEBOOK_HYPERPARAMETERS)
            st.dataframe(pd.DataFrame([{"method": k, "executes": v} for k, v in METHOD_SEMANTICS.items()]), hide_index=True, width="stretch")
        else:
            st.json(NOTEBOOK_DEFAULTS)
            st.dataframe(pd.DataFrame([{"label": k, "executes": v} for k, v in SAMPLER_SEMANTICS.items()]), hide_index=True, width="stretch")

    if exp_id == "cell22_dual_dataset":
        selection = st.multiselect("Balancing methods", ALL_METHODS, default=ALL_METHODS,
                                   help="The notebook runs all nine in this order. Torch RNG state carries across methods, "
                                        "so a subset can give different numbers for the later methods.")
        smoke = False
    else:
        entries = config_list("cell02" if exp_id.startswith("cell02") else "cell04")
        names = {c["entry"]: f"{c['entry']}. {c['config']}" for c in entries}
        picked = st.multiselect("Configuration entries (notebook order)", list(names), default=list(names), format_func=names.get,
                                help="Cell 4 lists SMOTE twice (entries 1 and 9) — preserved as two separate entries.")
        selection = picked
        smoke = st.checkbox("Smoke run (reduced rows / GA / epochs — plumbing check only, never a result)", value=False)
        if smoke:
            st.json(SMOKE_OVERRIDES[exp_id])
    if exp_id == "cell22_dual_dataset":
        full = sorted(selection) == sorted(ALL_METHODS)
    else:
        full = len(selection) == len(entries)
    kind = "smoke" if smoke else ("full_reproduction" if full else "partial")
    st.markdown(f"Run kind: **{kind}** · Estimated time: {ESTIMATES[exp_id]}")
    label = st.text_input("Run label (optional)")
    running = [r for r in list_runs() if r.get("status") in ("running", "queued")]
    if running:
        st.warning(f"{len(running)} run(s) already active: {', '.join(r['run_id'] for r in running)}. Parallel runs compete for CPU.")
    if st.button("Launch run", type="primary", disabled=not selection):
        path = create_run(exp_id, selection, SMOKE_OVERRIDES.get(exp_id) if smoke else None, label)
        pid = launch_background(path)
        st.session_state["lab_watch"] = path.name
        st.success(f"Started {path.name} (worker PID {pid}).")

    st.divider()
    _monitor()


@st.fragment(run_every=5)
def _monitor() -> None:
    runs = list_runs()
    st.subheader("Runs")
    if not runs:
        st.info("No runs yet.")
        return
    table = pd.DataFrame([{
        "run": r["run_id"], "cell": r["cell"], "experiment": r["experiment_id"], "kind": r["kind"], "status": r["status"],
        "selection": ", ".join(map(str, r["selection"])) if r.get("selection") else "all", "created": r["created_at"], "label": r.get("label", ""),
    } for r in runs])
    st.dataframe(table, hide_index=True, width="stretch")
    ids = [r["run_id"] for r in runs]
    default = st.session_state.get("lab_watch", ids[0])
    watch = st.selectbox("Inspect run", ids, index=ids.index(default) if default in ids else 0, key="lab_inspect")
    path = Path(next(r["path"] for r in runs if r["run_id"] == watch))
    status = read_json(path / "status.json") or {}
    prog = read_json(path / "progress.json") or {}
    c1, c2 = st.columns(2)
    c1.metric("Status", status.get("status", "?"))
    c2.metric("Last phase", prog.get("phase", "—"))
    if prog:
        st.caption(" · ".join(f"{k}: {v}" for k, v in prog.items() if k not in ("phase",)))
    if status.get("error"):
        st.error(status["error"])
    log = path / "log.txt"
    if log.exists():
        lines = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if "━" not in ln and "step -" not in ln]
        st.code("\n".join(lines[-40:]) or "(empty)", language="text")
