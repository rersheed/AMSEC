"""Diagnose GA reproducibility across library versions (no TensorFlow needed).

Replays cells 2/4 phases 1-2 with the notebook's own ``optimize_memory`` and
``GeneticFeatureSelector`` (extracted from the notebook by AST and exec'd), then
prints the GA checkpoints that the notebook printed (every 5 generations).
The notebook (cells 2 and 4, identical): Gen 5 0.9624/13, Gen 10 0.9683/9,
Gen 20 0.9715/5, final 5 of 41.

Usage: python scripts/diagnose_ga_environment.py <notebook.ipynb> <nids.csv> [generations]
Run it under different scikit-learn / numpy / pandas versions to locate the drift.
"""
from __future__ import annotations

import ast
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def notebook_defs(nb_path: str, names: list[str]) -> dict:
    nb = json.load(open(nb_path, encoding="utf-8"))
    src = "".join(nb["cells"][4]["source"])
    tree = ast.parse(src)
    ns = {"np": np, "pd": pd, "RandomForestClassifier": RandomForestClassifier, "train_test_split": train_test_split,
          "accuracy_score": accuracy_score, "f1_score": f1_score}
    for node in tree.body:
        if getattr(node, "name", None) in names:
            exec(compile(ast.Module([node], []), "<notebook cell 4>", "exec"), ns)
    return ns


def main() -> None:
    nb_path, csv = sys.argv[1], sys.argv[2]
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    ns = notebook_defs(nb_path, ["optimize_memory", "GeneticFeatureSelector"])
    print({"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "python": sys.version.split()[0]}, flush=True)
    np.random.seed(42); random.seed(42)  # set_seed(42) minus tf
    sample = pd.read_csv(csv, nrows=1000)
    label_col = next(c for c in sample.columns if c.lower() in ["label", "attack", "class"])
    keep = [c for c in sample.columns if c != label_col]
    chunks, total = [], 0
    for ch in pd.read_csv(csv, usecols=keep, chunksize=50000):
        chunks.append(ch); total += len(ch)
        if total >= 150000:
            break
    df = ns["optimize_memory"](pd.concat(chunks, ignore_index=True))
    y = (pd.read_csv(csv, usecols=[label_col], nrows=150000)[label_col].to_numpy() > 0).astype(int)
    X = VarianceThreshold(threshold=1e-5).fit_transform(df.select_dtypes(include=[np.number]).fillna(0).values)
    idx = np.random.choice(X.shape[0], 20000, replace=False)
    print("sample idx head:", idx[:5].tolist(), "X dtype:", X.dtype, "n_features:", X.shape[1], flush=True)
    Xs = RobustScaler().fit_transform(X[idx])
    mi = SelectKBest(mutual_info_classif, k=min(100, Xs.shape[1]))
    Xm = mi.fit_transform(Xs, y[idx])
    probe = np.random.get_state()[1][:3].tolist()
    print("global RNG after MI (first state words):", probe, "pos:", np.random.get_state()[2], flush=True)
    t = time.time()
    ga = ns["GeneticFeatureSelector"](Xm, y[idx], population_size=40, generations=gens, mutation_rate=0.12, crossover_rate=0.9)
    sel, hist = ga.run()
    print("GA history:", [round(h, 4) for h in hist], "selected:", sel.tolist(), f"({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
