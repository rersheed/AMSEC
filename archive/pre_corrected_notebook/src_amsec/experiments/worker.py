"""Subprocess worker for Streamlit Laboratory (non-blocking UI)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from amsec.experiments.configuration import ExperimentConfig
from amsec.experiments.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMSEC experiment worker")
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))

    def progress(payload: dict) -> None:
        # progress.json is written by runners; also echo for log capture
        print(json.dumps({"progress": payload}), flush=True)

    result = run_experiment(config, progress=progress)
    print(json.dumps({"result": {"run_id": result["run_id"], "status": result["status"]}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
