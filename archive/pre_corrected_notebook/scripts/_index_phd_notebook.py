"""Index Final PhD notebook cells for the experiment audit."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "others files that can help" / "Final phd-Experiment-android-malware.ipynb"
OUT = ROOT / "docs" / "extracted" / "phd_cell_index.json"


def cell_text(cell: dict) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append("".join(out.get("text", [])))
        elif out.get("output_type") in ("execute_result", "display_data"):
            plain = out.get("data", {}).get("text/plain")
            if plain:
                chunks.append("".join(plain) if isinstance(plain, list) else plain)
    return "\n".join(chunks)


def main() -> None:
    data = json.loads(NB.read_text(encoding="utf-8"))
    rows = []
    for i, cell in enumerate(data["cells"]):
        src = "".join(cell.get("source", []))
        text = cell_text(cell)
        rows.append(
            {
                "index": i,
                "type": cell.get("cell_type"),
                "src_len": len(src),
                "out_len": len(text),
                "src_head": src[:300].replace("\n", " | "),
                "out_head": text[:500].replace("\n", " | "),
                "mentions": sorted(
                    set(
                        re.findall(
                            r"SMOTE|ADASYN|MobileNet|Attention|BiLSTM|Genetic|mutual_info|"
                            r"NF-UQ|dynamic_api|feature_vectors|Ensemble|Stacking|LRP|"
                            r"FLOP|parameter|train_test_split|nrows|SMOTEENN|SMOTETomek",
                            src,
                            flags=re.I,
                        )
                    )
                ),
            }
        )
        src_path = ROOT / "docs" / "extracted" / f"phd_cell_{i}_source.py"
        out_path = ROOT / "docs" / "extracted" / f"phd_cell_{i}_stream.txt"
        if cell.get("cell_type") == "code":
            src_path.write_text(src, encoding="utf-8")
            out_path.write_text(text, encoding="utf-8")
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"cells={len(rows)} wrote {OUT}")
    for row in rows:
        print(
            f"cell {row['index']:02d} {row['type']:8s} "
            f"src={row['src_len']:6d} out={row['out_len']:6d} "
            f"mentions={','.join(row['mentions'][:8])}"
        )


if __name__ == "__main__":
    main()
