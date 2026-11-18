"""CLI: launch a single AMSEC experiment (blocking)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.experiments.configuration import SMOKE_STACKING_CONFIG, ExperimentConfig, NOTEBOOK_DEFAULT_STACKING
from amsec.experiments.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny leakage-controlled smoke run")
    parser.add_argument("--config", type=str, default="", help="JSON config path")
    parser.add_argument("--protocol", choices=["notebook_reproduction", "leakage_controlled"], default="")
    parser.add_argument("--sampling", default="")
    parser.add_argument("--nrows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    elif args.smoke:
        config = SMOKE_STACKING_CONFIG
    else:
        config = NOTEBOOK_DEFAULT_STACKING
    if args.protocol:
        config.protocol = args.protocol
    if args.sampling:
        config.sampling_method = args.sampling
    if args.nrows:
        config.nrows = args.nrows
    if args.epochs:
        config.epochs = args.epochs

    result = run_experiment(config, progress=lambda p: print(p, flush=True))
    print(json.dumps({"run_id": result["run_id"], "status": result["status"], "accuracy": result["metrics"].get("accuracy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
