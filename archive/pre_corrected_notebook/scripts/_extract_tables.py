"""Pull printed result tables from notebook outputs into JSON."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(
    r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild\others files that can help"
    r"\Final phd-Experiment-android-malware.ipynb"
)
OUT = Path(r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild\docs\extracted\tables")
OUT.mkdir(parents=True, exist_ok=True)
nb = json.loads(NB.read_text(encoding="utf-8"))


def text_of(output) -> str:
    if output.get("output_type") == "stream":
        value = output.get("text", "")
        return "".join(value) if isinstance(value, list) else str(value)
    data = output.get("data") or {}
    chunks = []
    for key in ("text/plain", "text/html"):
        if key in data:
            value = data[key]
            chunks.append(f"\n## {key}\n" + ("".join(value) if isinstance(value, list) else str(value)))
    return "\n".join(chunks)


for index in (2, 4, 22):
    parts = []
    for output in nb["cells"][index].get("outputs", []):
        blob = text_of(output)
        if any(token in blob for token in ("F1-Score", "Rank", "Sampling", "Feature_", "t_")):
            parts.append(blob)
    dest = OUT / f"cell_{index}_tables.txt"
    dest.write_text("\n\n====\n\n".join(parts), encoding="utf-8")
    print(index, dest.stat().st_size)
