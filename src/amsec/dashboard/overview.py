"""Research Overview — provenance first, then the experiment map."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from amsec.dashboard.data import historical, historical_table, identity, run_results, runs_for
from amsec.dashboard.theme import kpi_html
from amsec.notebook.provenance import EXPERIMENTS, NOTEBOOK_NAME, verify_notebook


def render_overview() -> None:
    ident = identity()
    st.markdown(
        f"""<div class="amsec-hero"><span class="amsec-badge">PhD research platform · single mode</span>
        <div class="amsec-title">AMSEC <span>·</span> {ident.get('full_name', 'Android Malware Security Framework')}</div>
        <p class="amsec-sub">{ident.get('subtitle', '')}</p></div>""",
        unsafe_allow_html=True,
    )
    nb = verify_notebook()
    if nb["matches_recorded"]:
        st.success(f"Sole authoritative experimental source: `{NOTEBOOK_NAME}` — SHA-256 verified ({nb['sha256'][:16]}…).")
    else:
        st.error(f"Authoritative notebook missing or modified ({nb}). Results below cannot be trusted until this is resolved.")

    st.subheader("Headline numbers — by source")
    cols = st.columns(4)
    h4 = historical_table("cell04_stacking9", "results_9")
    if len(h4):
        best = h4.sort_values("f1_score", ascending=False).iloc[0]
        cols[0].markdown(kpi_html("Cell 4 · notebook best F1", f"{best['f1_score']:.5f}", f"{best['method']} · historical saved output"), unsafe_allow_html=True)
    runs4 = [r for r in runs_for("cell04_stacking9") if r["kind"] == "full_reproduction"]
    if runs4:
        df = run_results(runs4[0])
        b = df.iloc[0]
        cols[1].markdown(kpi_html("Cell 4 · reproduced best F1", f"{b['f1_score']:.5f}", f"{b['sampling_method']} · {runs4[0]['run_id']} (new)", "#d95926"), unsafe_allow_html=True)
    else:
        cols[1].markdown(kpi_html("Cell 4 · reproduced", "—", "No full reproduction run yet", "#d95926"), unsafe_allow_html=True)
    h2 = historical_table("cell02_lightweight16", "results_16")
    if len(h2):
        b2 = h2.sort_values("f1_score", ascending=False).iloc[0]
        cols[2].markdown(kpi_html("Cell 2 · notebook best F1", f"{b2['f1_score']:.5f}", f"{b2['method']} · {b2['model_type']} · historical"), unsafe_allow_html=True)
    runs22 = [r for r in runs_for("cell22_dual_dataset") if r["kind"] == "full_reproduction"]
    if runs22:
        d22 = run_results(runs22[0]).iloc[0]
        cols[3].markdown(kpi_html("Cell 22 · best F1 (balanced acc.)", f"{d22['f1_score']:.4f} ({d22['balanced_accuracy']:.3f})",
                                  f"{d22['method']} · {runs22[0]['run_id']} (new; notebook has no saved output)", "#d95926"), unsafe_allow_html=True)
    else:
        cols[3].markdown(kpi_html("Cell 22", "—", "Notebook saved no outputs; run it in the Laboratory", "#d95926"), unsafe_allow_html=True)
    st.caption("Historical = parsed from the notebook's saved outputs. New = executed by AMSEC's faithful modules. "
               "They are never merged into one leaderboard; ranks are valid only within one experiment.")

    st.subheader("Experiment map — corrected notebook")
    reg = historical()["experiments"]
    rows = []
    for exp_id, spec in EXPERIMENTS.items():
        e = reg.get(exp_id, {})
        n_runs = len(runs_for(exp_id))
        rows.append({
            "Cell": spec["cell"], "Experiment": spec["title"], "Framework": spec["framework"],
            "Historical tables": sum(1 for t in e.get("tables", {}).values() if t.get("rows")),
            "Historical figures": len(e.get("figures", [])),
            "Executable in AMSEC": "yes" if spec.get("executable") else "no (historical only)",
            "Completed AMSEC runs": n_runs,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info(
        "These are **distinct** experiments. Cells 6–19 are successive CNN–BiLSTM benchmark implementations, not independent "
        "replications; cell 7 holds orphaned outputs with no source code; cell 22 is the corrected dual-dataset benchmark."
    )
    sup = ", ".join(f"{s['name']} ({s['role']})" for s in ident.get("supervisors", []))
    if ident:
        st.caption(f"{ident.get('researcher', '')} · {ident.get('department', '')}, {ident.get('institution', '')} · Supervisors: {sup}")
