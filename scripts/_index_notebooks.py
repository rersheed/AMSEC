"""Index notebook cells for the source audit. Temporary inspection helper."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild\others files that can help")
OUT = Path(r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild\docs")
OUT.mkdir(exist_ok=True)

NAMES = [
    "Final phd-Experiment-android-malware.ipynb",
    "notebook48605fb318 (3).ipynb",
]


def flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def main() -> None:
    for name in NAMES:
        nb = json.loads((BASE / name).read_text(encoding="utf-8"))
        cells = nb.get("cells", [])
        lines = [
            f"# {name}",
            f"nbformat={nb.get('nbformat')} cells={len(cells)}",
            "",
        ]
        for i, cell in enumerate(cells):
            src = flatten(cell.get("source"))
            head = " | ".join(src.strip().splitlines())[:400]
            outs = cell.get("outputs", [])
            bits = []
            for output in outs[:8]:
                kind = output.get("output_type")
                if kind == "stream":
                    text = flatten(output.get("text"))[:600].replace("\n", " | ")
                    if text.strip():
                        bits.append("STREAM:" + text)
                elif kind in ("execute_result", "display_data"):
                    data = output.get("data", {})
                    if "text/plain" in data:
                        text = flatten(data["text/plain"])[:400].replace("\n", " | ")
                        bits.append("RES:" + text)
                elif kind == "error":
                    bits.append(
                        "ERR:"
                        + str(output.get("ename"))
                        + " "
                        + str(output.get("evalue"))[:240]
                    )
            lines.append(
                f"--- CELL {i} type={cell.get('cell_type')} "
                f"exec={cell.get('execution_count')} src_len={len(src)} n_out={len(outs)}"
            )
            lines.append(head)
            for bit in bits[:4]:
                lines.append("  " + bit[:700])
            lines.append("")
        dest = OUT / (name.split(".")[0][:24].replace(" ", "_") + "_index.txt")
        dest.write_text("\n".join(lines), encoding="utf-8")
        print(dest.name, dest.stat().st_size)


if __name__ == "__main__":
    main()
