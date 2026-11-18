"""Diagnose resampling-count reproducibility (cells 2/4) across library versions.

Runs the notebook's own ``apply_sampling_method`` (extracted from cell 4 by AST)
on the exact phase-3 matrix of a run (saved as .npy) and prints the post-sampling
row counts, which the notebook printed for every entry.

Usage: python scripts/diagnose_resampling_environment.py <notebook.ipynb> <X.npy> <y.npy>
       python scripts/diagnose_resampling_environment.py <notebook.ipynb> --rebuild <nids.csv> <comma-separated post-VT indices>
--rebuild recomputes the phase-3 matrix in *this* environment (optimize_memory → VT → GA columns →
RobustScaler → Yeo-Johnson → nan_to_num) and prints the PowerTransformer lambdas, to test SciPy/sklearn effects.
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
    import scipy
    print({"sklearn": sklearn.__version__, "imblearn": imblearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}, flush=True)
    if sys.argv[2] == "--rebuild":
        import pandas as pd
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import PowerTransformer, RobustScaler

        opt = next(n for n in ast.parse(src).body if getattr(n, "name", "") == "optimize_memory")
        ons = {"np": np}
        exec(compile(ast.Module([opt], []), "<cell 4>", "exec"), ons)
        raw = pd.read_csv(sys.argv[3], nrows=150000, low_memory=False)
        y = (raw["Label"].to_numpy() > 0).astype(int)
        df = ons["optimize_memory"](raw.drop(columns=["Label"]).copy())
        X = VarianceThreshold(threshold=1e-5).fit_transform(df.select_dtypes(include=[np.number]).fillna(0).values)
        X = X[:, [int(i) for i in sys.argv[4].split(",")]]
        X = RobustScaler().fit_transform(X)
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        X = pt.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
        print("lambdas:", [round(float(v), 10) for v in pt.lambdas_], "unique rows:", len(np.unique(X, axis=0)), flush=True)
    else:
        X = np.load(sys.argv[2]); y = np.load(sys.argv[3])
    for method, expected in NOTEBOOK_COUNTS.items():
        _, yb = ns["apply_sampling_method"](X, y, method)
        print(f"{method:13s} notebook={expected:7d} here={len(yb):7d} {'MATCH' if len(yb) == expected else 'DIFF'}", flush=True)


if __name__ == "__main__":
    main()
