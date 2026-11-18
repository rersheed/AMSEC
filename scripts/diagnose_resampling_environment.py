"""Diagnose resampling-count reproducibility (cells 2/4) across library versions.

Runs the notebook's own ``apply_sampling_method`` (extracted from cell 4 by AST)
on the exact phase-3 matrix of a run (saved as .npy) and prints the post-sampling
row counts, which the notebook printed for every entry.

Usage: python scripts/diagnose_resampling_environment.py <notebook.ipynb> <X.npy> <y.npy>
"""
from __future__ import annotations

import ast
import json
import sys

import numpy as np

NOTEBOOK_COUNTS = {"SMOTE": 201172, "ADASYN": 202084, "SMOTE-ENN": 191389, "SMOTE-Tomek": 200606, "SMOTE-NCR": 201172,
                   "ADASYN-ENN": 176601, "ADASYN-Tomek": 201140, "ADASYN-NCR": 201172}


def main() -> None:
    import imblearn
    import sklearn
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, TomekLinks

    nb = json.load(open(sys.argv[1], encoding="utf-8"))
    src = "".join(nb["cells"][4]["source"])
    node = next(n for n in ast.parse(src).body if getattr(n, "name", "") == "apply_sampling_method")
    ns = dict(SMOTE=SMOTE, ADASYN=ADASYN, SMOTEENN=SMOTEENN, SMOTETomek=SMOTETomek, NearMiss=NearMiss,
              EditedNearestNeighbours=EditedNearestNeighbours, TomekLinks=TomekLinks)
    exec(compile(ast.Module([node], []), "<cell 4>", "exec"), ns)
    X = np.load(sys.argv[2]); y = np.load(sys.argv[3])
    print({"sklearn": sklearn.__version__, "imblearn": imblearn.__version__, "numpy": np.__version__}, flush=True)
    for method, expected in NOTEBOOK_COUNTS.items():
        _, yb = ns["apply_sampling_method"](X, y, method)
        print(f"{method:13s} notebook={expected:7d} here={len(yb):7d} {'MATCH' if len(yb) == expected else 'DIFF'}", flush=True)


if __name__ == "__main__":
    main()
