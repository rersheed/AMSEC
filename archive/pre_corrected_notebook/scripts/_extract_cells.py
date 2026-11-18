"""Extract key notebook cells and metric lines for the source audit."""
from __future__ import annotations

import json
import re
from pathlib import Path

NB = Path(
    r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild\others files that can help"
    r"\Final phd-Experiment-android-malware.ipynb"
)
OUT = Path(r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild\docs\extracted")
OUT.mkdir(parents=True, exist_ok=True)

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]

METRIC = re.compile(
    r"(Accuracy|Precision|Recall|F1|AUC|MCC|Balanced|FPR|FNR|Specificity|"
    r"Parameters|FLOPs|BEST|best|SMOTE-ENN|saved|Selected|shape|threshold|"
    r"features|Logistic|stack)",
    re.I,
)


def src(cell) -> str:
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else str(value)


def streams(cell) -> str:
    chunks = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text = output.get("text", [])
            chunks.append("".join(text) if isinstance(text, list) else str(text))
        elif output.get("output_type") == "error":
            chunks.append("ERROR " + str(output.get("ename")) + ": " + str(output.get("evalue")))
    return "\n".join(chunks)


for index in (2, 4, 22):
    cell = cells[index]
    (OUT / f"cell_{index}_source.py").write_text(src(cell), encoding="utf-8")
    text = streams(cell)
    (OUT / f"cell_{index}_stream.txt").write_text(text, encoding="utf-8")
    kept = [line for line in text.splitlines() if METRIC.search(line)]
    (OUT / f"cell_{index}_metrics.txt").write_text("\n".join(kept), encoding="utf-8")
    print(index, "source", len(src(cell)), "stream", len(text), "metric_lines", len(kept))
