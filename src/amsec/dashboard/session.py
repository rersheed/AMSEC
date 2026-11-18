"""Session-state helpers for the AMSEC research platform (single mode)."""
from __future__ import annotations

import streamlit as st

DEFAULT_PAGE = "Research Overview"

PAGES = [
    "Research Overview",
    "Experiment Laboratory",
    "Results & Analytics",
    "Notebook Outputs (Historical)",
    "Malware Detection",
    "Explainable AI",
    "Dataset Explorer",
    "Reproducibility & Audit",
    "Methodology & Limitations",
]

_OBSOLETE_KEYS = (
    "operating_mode", "operating_mode_ui", "viva_slide_index", "viva_slide_select", "viva_show_notes",
    "viva_demo_result", "viva_return_requested", "experiment_focus", "analytics_run_id",
)


def init_session() -> None:
    st.session_state.setdefault("dashboard_page", DEFAULT_PAGE)
    for key in _OBSOLETE_KEYS:
        st.session_state.pop(key, None)
    if st.session_state.get("dashboard_page") not in PAGES:
        st.session_state.dashboard_page = DEFAULT_PAGE
