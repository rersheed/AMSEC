"""Quick consistency checks for viva mode."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.dashboard import viva
from amsec.research import local_metrics_enriched, notebook_final_metrics


def main() -> None:
    slides = viva._load_slides()
    assert len(slides) == 21, len(slides)
    nb = notebook_final_metrics()
    loc = local_metrics_enriched()
    assert abs(nb["accuracy"] - 0.97789) < 1e-9
    assert loc is not None
    assert abs(loc["accuracy"] - 0.9662666666666667) < 1e-9
    pct = f"{nb['accuracy'] * 100:.3f}%"
    assert pct == "97.789%", pct
    payload = viva._export_html(slides)
    assert b"97.789%" in payload
    assert slides[17]["id"] == "live_demo"
    print("slides", len(slides))
    print("notebook_pct", pct)
    print("local_acc", round(loc["accuracy"], 5))
    print("export_bytes", len(payload))
    print("ok")


if __name__ == "__main__":
    main()
