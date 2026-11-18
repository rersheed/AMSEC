"""Run an executable notebook experiment as an auditable run.

Examples
  python scripts/run_notebook_experiment.py --experiment cell22_dual_dataset
  python scripts/run_notebook_experiment.py --experiment cell04_stacking9            # full reproduction
  python scripts/run_notebook_experiment.py --experiment cell04_stacking9 --select 3   # partial (SMOTE-ENN entry)
  python scripts/run_notebook_experiment.py --experiment cell02_lightweight16 --smoke
  python scripts/run_notebook_experiment.py --run-dir experiment_runs/run_0003        # worker mode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amsec.lab.runs import EXECUTABLE, SMOKE_OVERRIDES, create_run, execute_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=sorted(EXECUTABLE))
    ap.add_argument("--select", default="", help="Comma list: entry numbers (cells 2/4) or method keys (cell 22)")
    ap.add_argument("--smoke", action="store_true", help="Reduced plumbing check (cells 2/4 only)")
    ap.add_argument("--label", default="")
    ap.add_argument("--run-dir", default="", help="Execute an existing queued run (worker mode)")
    args = ap.parse_args()
    if args.run_dir:
        path = Path(args.run_dir)
    else:
        if not args.experiment:
            ap.error("--experiment or --run-dir is required")
        selection = [s.strip() for s in args.select.split(",") if s.strip()] or None
        overrides = SMOKE_OVERRIDES.get(args.experiment) if args.smoke else None
        path = create_run(args.experiment, selection, overrides, args.label)
        print(f"created {path.name}", flush=True)
    summary = execute_run(path)
    print(json.dumps({"run": path.name, **summary}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
