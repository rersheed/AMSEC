"""Every dashboard page renders without an exception (headless AppTest)."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from amsec.dashboard.session import PAGES
from amsec.notebook.provenance import ROOT


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.session_state["dashboard_page"] = page
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"


def test_no_presentation_mode():
    src = (ROOT / "app.py").read_text(encoding="utf-8").lower()
    assert "viva" not in src and "presentation" not in src
    assert not any((ROOT / "src" / "amsec").rglob("viva*.py"))
