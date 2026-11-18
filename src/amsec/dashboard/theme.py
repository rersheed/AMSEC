"""Shared visual theme for the AMSEC PhD research console."""
from __future__ import annotations

import streamlit as st

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,18,32,0.85)",
    font=dict(family="DM Mono, Consolas, monospace", color="#9bb0d0", size=12),
    margin=dict(l=48, r=28, t=56, b=48),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c5d4ea")),
    # Validated dark-mode categorical order (scripts/validate_palette.js, surface #0b1220: all checks pass)
    colorway=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
    hoverlabel=dict(bgcolor="#0b1220", bordercolor="#3987e5", font=dict(color="#e8edf5")),
    xaxis=dict(gridcolor="rgba(58,123,253,0.12)", zerolinecolor="rgba(58,123,253,0.18)"),
    yaxis=dict(gridcolor="rgba(58,123,253,0.12)", zerolinecolor="rgba(58,123,253,0.18)"),
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=DM+Mono:wght@400;500&family=Lora:ital@0;1&display=swap');
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #12284a 0%, #07111f 45%, #050b14 100%) !important; color:#e8edf5 !important; }
[data-testid="stSidebar"] { background:#080d18 !important; border-right:1px solid rgba(58,123,253,0.18) !important; }
[data-testid="stMetric"] { background:#0b1220 !important; border:1px solid rgba(58,123,253,0.22) !important; border-radius:12px !important; padding:0.85rem 1rem !important; }
[data-testid="stMetricLabel"] { font-family:Syne,sans-serif !important; letter-spacing:0.06em !important; text-transform:uppercase !important; color:#6d7f9e !important; font-size:0.68rem !important; }
[data-testid="stMetricValue"] { font-family:Syne,sans-serif !important; color:#00e5cc !important; }
.stTabs [data-baseweb="tab-list"] { background:#0b1220; border-radius:12px; border:1px solid rgba(58,123,253,0.18); padding:4px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#3a7bfd,#00e5cc) !important; color:#fff !important; }
.amsec-hero { background:linear-gradient(135deg,#04080f 0%,#0a1528 55%,#04080f 100%); border:1px solid rgba(0,229,204,0.22); border-radius:18px; padding:1.7rem 2rem; margin-bottom:1.2rem; }
.amsec-badge { display:inline-block; background:linear-gradient(135deg,#3a7bfd,#00e5cc); color:#fff; padding:5px 14px; border-radius:999px; font-family:Syne,sans-serif; font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.amsec-title { font-family:Syne,sans-serif; font-size:2.15rem; font-weight:800; margin:0.7rem 0 0.15rem; }
.amsec-title span { color:#3a7bfd; }
.amsec-sub { font-family:Lora,serif; font-style:italic; color:#8a9bbf; margin:0 0 1.1rem; }
.amsec-card { background:#0b1220; border:1px solid rgba(58,123,253,0.18); border-top:3px solid #3a7bfd; border-radius:14px; padding:0.95rem 1rem; height:100%; }
.amsec-card h4 { font-family:Syne,sans-serif; margin:0 0 0.35rem; color:#e8edf5; font-size:0.95rem; }
.amsec-card p { font-family:DM Mono,monospace; color:#8a9bbf; font-size:0.72rem; margin:0; line-height:1.55; }
.amsec-source { font-family:DM Mono,monospace; font-size:0.62rem; color:#4a5878; margin-top:0.45rem; }
.amsec-kpi { background:#0b1220; border:1px solid rgba(58,123,253,0.18); border-radius:14px; padding:0.9rem 1rem; border-top:3px solid var(--accent,#00e5cc); }
.amsec-kpi .label { font-family:Syne,sans-serif; font-size:0.58rem; letter-spacing:0.08em; text-transform:uppercase; color:#6d7f9e; }
.amsec-kpi .value { font-family:Syne,sans-serif; font-size:1.35rem; font-weight:800; color:var(--accent,#00e5cc); margin-top:0.2rem; }
.amsec-kpi .src { font-family:DM Mono,monospace; font-size:0.58rem; color:#8a9bbf; margin-top:0.25rem; }
.amsec-pipe { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0.55rem; }
.amsec-pipe-item { background:#0b1220; border:1px solid rgba(58,123,253,0.16); border-radius:12px; padding:0.75rem; text-align:center; }
.amsec-pipe-item .num { font-family:Syne,sans-serif; font-weight:800; color:#3a7bfd; }
.amsec-pipe-item .lab { font-family:Syne,sans-serif; font-size:0.78rem; color:#e8edf5; margin-top:0.2rem; }
.amsec-pipe-item .note { font-family:DM Mono,monospace; font-size:0.58rem; color:#8a9bbf; margin-top:0.25rem; }
@media (max-width: 1100px) { .amsec-pipe { grid-template-columns:repeat(2,minmax(0,1fr)); } .amsec-title { font-size:1.7rem; } }
"""


def inject() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def style_fig(fig, title: str = "", height: int = 420):
    layout = dict(PLOTLY_LAYOUT)
    layout["height"] = height
    if title:
        layout["title"] = dict(text=title, font=dict(family="Syne, sans-serif", size=14, color="#e8edf5"), x=0.01)
    fig.update_layout(**layout)
    return fig


CATEGORICAL = list(PLOTLY_LAYOUT["colorway"])
# Source identity is fixed: notebook (historical) vs AMSEC run (reproduced).
SOURCE_COLORS = {"Notebook (historical)": "#3987e5", "AMSEC run (new)": "#d95926"}


def color_map(keys: list[str]) -> dict[str, str]:
    """Stable entity → colour (follows the entity, never its rank)."""
    ordered = sorted(dict.fromkeys(map(str, keys)))
    return {k: CATEGORICAL[i % len(CATEGORICAL)] for i, k in enumerate(ordered)}


def thin_bars(fig):
    fig.update_traces(selector=dict(type="bar"), marker_line_width=2, marker_line_color="#0b1220", marker_cornerradius=4)
    fig.update_layout(bargap=0.35, bargroupgap=0.08)
    return fig


def kpi_html(label: str, value: str, source: str, accent: str = "#00e5cc") -> str:
    return (
        f'<div class="amsec-kpi" style="--accent:{accent}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="src">{source}</div></div>'
    )


def source_chip(kind: str) -> str:
    mapping = {
        "notebook": "Source: Original executed notebook",
        "local": "Source: Local retraining",
        "verified": "Source: Verified saved model evaluation",
        "mixed": "Sources kept separate — not a unified leaderboard",
    }
    return mapping.get(kind, kind)


def unavailable_box(message: str) -> None:
    st.warning(f"**Unavailable** — {message}")
