"""Session-state helpers for Research Dashboard / Viva Presentation modes."""
from __future__ import annotations

import streamlit as st

MODE_DASHBOARD = "Research Dashboard"
MODE_VIVA = "Viva Presentation"
MODES = (MODE_DASHBOARD, MODE_VIVA)

DEFAULT_PAGE = "Research Overview"


MODE_STATE_KEY = "operating_mode"
MODE_UI_KEY = "operating_mode_ui"


def init_session() -> None:
    defaults = {
        MODE_STATE_KEY: MODE_DASHBOARD,
        MODE_UI_KEY: MODE_DASHBOARD,
        "dashboard_page": DEFAULT_PAGE,
        "viva_slide_index": 0,
        "viva_show_notes": False,
        "viva_demo_result": None,
        "viva_return_requested": False,
        "batch_last_summary": None,
        "experiment_focus": "Notebook cell-4 SMOTE-ENN stack",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_mode() -> str:
    return st.session_state.get(MODE_STATE_KEY, MODE_DASHBOARD)


def set_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(mode)
    st.session_state[MODE_STATE_KEY] = mode
    st.session_state[MODE_UI_KEY] = mode


def go_dashboard(page: str | None = None) -> None:
    set_mode(MODE_DASHBOARD)
    if page:
        st.session_state.dashboard_page = page


def go_viva(slide_index: int | None = None) -> None:
    set_mode(MODE_VIVA)
    if slide_index is not None:
        st.session_state.viva_slide_index = max(0, int(slide_index))


def clamp_slide(index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(int(index), total - 1))
