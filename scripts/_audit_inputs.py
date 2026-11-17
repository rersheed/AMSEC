"""Schema and artifact audit. Writes docs/extracted/input_audit.json."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\KASU-PC\projects\AMSEC_Streamlit_Rebuild")
DATA = ROOT / "others files that can help"
UP = ROOT / "upstream" / "Models"
OUT = ROOT / "docs" / "extracted" / "input_audit.json"


def sha256(path: Path, chunk=1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def audit_csv(path: Path) -> dict:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    # Label-like columns only for full counts; numeric profile from a sample.
    sample = pd.read_csv(path, nrows=5000)
    info = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "n_columns": len(columns),
        "columns": columns,
        "sample_rows": int(len(sample)),
        "sample_nulls": {c: int(sample[c].isna().sum()) for c in columns if int(sample[c].isna().sum())},
        "dtypes_sample": {c: str(sample[c].dtype) for c in columns},
    }
    label = next((c for c in columns if c.lower() in {"label", "malware", "class", "attack"}), None)
    info["label_column"] = label
    if label:
        counts = pd.read_csv(path, usecols=[label])[label].value_counts(dropna=False)
        info["rows"] = int(counts.sum())
        info["label_counts"] = {str(k): int(v) for k, v in counts.items()}
    return info


def audit_pickle(path: Path) -> dict:
    item = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        item["type"] = type(obj).__name__
        item["module"] = type(obj).__module__
        if hasattr(obj, "classes_"):
            item["classes_"] = [str(x) for x in list(obj.classes_)]
        if hasattr(obj, "n_features_in_"):
            item["n_features_in_"] = int(obj.n_features_in_)
        if hasattr(obj, "feature_names_in_"):
            item["feature_names_in_"] = [str(x) for x in list(obj.feature_names_in_)]
        if isinstance(obj, (list, tuple)):
            item["length"] = len(obj)
            item["preview"] = [str(x) for x in list(obj)[:30]]
        elif isinstance(obj, dict):
            item["keys"] = [str(k) for k in list(obj)[:40]]
        else:
            item["repr"] = repr(obj)[:500]
    except Exception as exc:
        item["load_error"] = f"{type(exc).__name__}: {exc}"
    return item


def main() -> None:
    report = {
        "datasets": [audit_csv(p) for p in sorted(DATA.glob("*.csv"))],
        "upstream_pickles": [audit_pickle(p) for p in sorted(UP.glob("*.pkl"))],
        "upstream_other": [
            {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(UP.iterdir())
            if p.suffix.lower() in {".h5", ".json"}
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
