"""AMSEC — single-mode PhD research, experimentation, analytics and detection platform.

Sole experimental source: notebooks/Final_phd-Experiment-android-malware_CORRECTED.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from amsec.dashboard.analytics import render_analytics  # noqa: E402
from amsec.dashboard.data import identity  # noqa: E402
from amsec.dashboard.laboratory import render_laboratory  # noqa: E402
from amsec.dashboard.overview import render_overview  # noqa: E402
from amsec.dashboard.pages import (  # noqa: E402
    render_datasets, render_detection, render_historical, render_methodology, render_reproducibility, render_xai,
)
from amsec.dashboard.session import PAGES, init_session  # noqa: E402
from amsec.dashboard.theme import inject  # noqa: E402
from amsec.notebook.provenance import verify_notebook  # noqa: E402

st.set_page_config(page_title="AMSEC | Research Platform", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
inject()
init_session()

RENDERERS = {
    "Research Overview": render_overview,
    "Experiment Laboratory": render_laboratory,
    "Results & Analytics": render_analytics,
    "Notebook Outputs (Historical)": render_historical,
    "Malware Detection": render_detection,
    "Explainable AI": render_xai,
    "Dataset Explorer": render_datasets,
    "Reproducibility & Audit": render_reproducibility,
    "Methodology & Limitations": render_methodology,
}

with st.sidebar:
    st.markdown("### AMSEC Research Platform")
    st.caption("Experimentation · analytics · detection")
    page = st.radio("Navigation", PAGES, key="dashboard_page")
    st.markdown("---")
    nb = verify_notebook()
    (st.success if nb["matches_recorded"] else st.error)("Corrected notebook verified" if nb["matches_recorded"] else "Notebook hash mismatch")
    ident = identity()
    if ident:
        st.caption(f"{ident.get('researcher', '')} · {ident.get('institution', '')}")

RENDERERS[page]()
