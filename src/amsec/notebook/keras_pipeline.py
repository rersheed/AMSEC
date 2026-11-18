"""Cells 2 and 4 — procedural phases refactored into a callable pipeline.

Class/function definitions come verbatim from the notebook (generated
modules ``keras_common``, ``cell02_models``, ``cell04_models``). This file
reproduces the cells' *top-level script* in the same order, so the global
NumPy RNG stream (seeded once by ``set_seed(42)``) is consumed exactly as in
the notebook when the full configuration list is run.

Phases (both cells)
  1  load first 150,000 NF-UQ-NIDS rows in 50,000-row chunks; optimize_memory
  2  VarianceThreshold(1e-5) → 20,000-row GA sample → RobustScaler →
     SelectKBest(MI, k=min(100,n)) → GA(pop 40, gen 30)
  3  RobustScaler + Yeo-Johnson PowerTransformer on the *full* selected matrix;
     per configuration: resample full matrix → cap 180,000 → 85/15 split →
     88/12 train/val split → train (30 epochs, batch 128) → evaluate
  4  rank by weighted F1 (``rank(method='min')``)
  7  best configuration
  8  retrain best model on an un-resampled 80/20 split (test used as
     validation) → API evaluation (skipped in the historical run: API file absent)
  9  LRP (|x · ∂y/∂x|) on 3 random test rows of the phase-8 split

Deviations are only allowed through explicit parameters and are recorded in
``deviations`` of the returned/run metadata (e.g. smoke runs).
"""
from __future__ import annotations

import gc
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

NOTEBOOK_DEFAULTS = {
    "max_rows": 150_000,
    "chunk_size": 50_000,
    "ga_sample": 20_000,
    "ga_population": 40,
    "ga_generations": 30,
    "epochs": 30,
    "batch_size": 128,
    "balanced_cap": 180_000,
    "api_dataset_available": False,  # historical run: "API dataset not available"
}

CELL04_SAMPLING = [
    "SMOTE", "ADASYN", "SMOTE-ENN", "SMOTE-Tomek", "SMOTE-NCR",
    "ADASYN-ENN", "ADASYN-Tomek", "ADASYN-NCR", "SMOTE",  # SMOTE duplicated for 9th config (notebook)
]
CELL02_SAMPLING = ["SMOTE", "ADASYN", "SMOTE-ENN", "SMOTE-Tomek", "SMOTE-NCR", "ADASYN-ENN", "ADASYN-Tomek", "ADASYN-NCR"]
CELL02_MODELS = [{"name": "MobileNetV4", "use_attention": False}, {"name": "MobileNetV4+Attention", "use_attention": True}]

# What each notebook sampler label actually executes (labels can mislead).
SAMPLER_SEMANTICS = {
    "SMOTE": "SMOTE(k_neighbors=5)",
    "ADASYN": "ADASYN(n_neighbors=5)",
    "SMOTE-ENN": "SMOTEENN()",
    "SMOTE-Tomek": "SMOTETomek()",
    "SMOTE-NCR": "SMOTE → NearMiss(version=1, n_neighbors=3)  — not Neighbourhood Cleaning Rule",
    "ADASYN-ENN": "ADASYN → EditedNearestNeighbours(n_neighbors=3)",
    "ADASYN-Tomek": "ADASYN → TomekLinks()",
    "ADASYN-NCR": "ADASYN → NearMiss(version=1, n_neighbors=3)  — not Neighbourhood Cleaning Rule",
}


def config_list(experiment: str) -> list[dict]:
    """Ordered configuration entries exactly as the notebook iterates them."""
    if experiment == "cell04":
        return [{"entry": i + 1, "sampling_method": m, "model_type": "Ensemble_Stacking",
                 "config": f"{m}_GA_Ensemble_Stacking"} for i, m in enumerate(CELL04_SAMPLING)]
    if experiment == "cell02":
        out, n = [], 0
        for m in CELL02_SAMPLING:
            for c in CELL02_MODELS:
                n += 1
                out.append({"entry": n, "sampling_method": m, "model_type": c["name"],
                            "use_attention": c["use_attention"], "config": f"{m} + GA + {c['name']}"})
        return out
    raise ValueError(experiment)


def _history_dict(h) -> dict:
    try:
        return {k: [float(v) for v in vals] for k, vals in h.history.items()}
    except Exception:
        return {}


def run_keras_experiment(
    experiment: str,
    data_dir: Path | str,
    out_dir: Path | str,
    entries: list[int] | None = None,
    overrides: dict | None = None,
    run_post_phases: bool = True,
    log: Callable[[str], None] = print,
    progress: Callable[[dict], None] | None = None,
    save_models: bool = True,
) -> dict:
    """Run cell 2 (``experiment='cell02'``) or cell 4 (``'cell04'``)."""
    import joblib
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PowerTransformer, RobustScaler

    from amsec.notebook import keras_common as kc
    from amsec.notebook.provenance import DATASET_FILES

    if experiment not in ("cell02", "cell04"):
        raise ValueError(experiment)
    params = dict(NOTEBOOK_DEFAULTS)
    deviations = {}
    for k, v in (overrides or {}).items():
        if k not in params:
            raise ValueError(f"Unknown override {k}")
        if v != params[k]:
            deviations[k] = {"notebook": params[k], "used": v}
        params[k] = v
    all_configs = config_list(experiment)
    selected_entries = sorted(entries) if entries else [c["entry"] for c in all_configs]
    if selected_entries != [c["entry"] for c in all_configs]:
        deviations["entries"] = {
            "notebook": "all", "used": selected_entries,
            "note": "Global NumPy RNG is consumed by the 180,000-row cap of skipped entries in the notebook; "
                    "subset runs may therefore draw different random rows.",
        }

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    data_dir = Path(data_dir)
    df1_path = str(data_dir / DATASET_FILES["nids"])
    df2_path = str(data_dir / DATASET_FILES["api"])

    def emit(payload: dict) -> None:
        if progress:
            progress(payload)

    warnings.filterwarnings("ignore")
    kc.set_seed(42)
    t_start = time.time()

    # ---------------- PHASE 1: DATA LOADING (notebook logic) ----------------
    emit({"phase": "load", "message": "Phase 1 — loading NF-UQ-NIDS"})
    log("PHASE 1: DATA LOADING")
    nids_df = None; nids_labels = None
    sample_chunk = pd.read_csv(df1_path, nrows=1000)
    label_col = None
    for col in sample_chunk.columns:
        if col.lower() in ['label', 'attack', 'class']:
            label_col = col
            break
    if label_col is None:
        raise RuntimeError("No label column in NIDS file")
    log(f"  Using '{label_col}' as label column")
    cols_to_keep = [col for col in sample_chunk.columns if col != label_col]
    chunk_size = params["chunk_size"]; max_rows = params["max_rows"]
    chunks = []; total_rows = 0
    for chunk in pd.read_csv(df1_path, usecols=cols_to_keep, chunksize=chunk_size):
        chunks.append(chunk); total_rows += len(chunk)
        if total_rows >= max_rows:
            break
    nids_df = pd.concat(chunks, ignore_index=True)
    nids_df = kc.optimize_memory(nids_df)
    label_chunks = []
    for chunk in pd.read_csv(df1_path, usecols=[label_col], chunksize=chunk_size):
        label_chunks.append(chunk)
        if len(pd.concat(label_chunks, ignore_index=True)) >= max_rows:
            break
    nids_labels = pd.concat(label_chunks, ignore_index=True).values.flatten()[:len(nids_df)]
    nids_labels = (nids_labels > 0).astype(int)
    log(f"✓ Loaded NIDS: {nids_df.shape}")
    log(f"  Label distribution: Normal={np.sum(nids_labels==0)}, Attack={np.sum(nids_labels==1)}")
    column_dtypes = {c: str(t) for c, t in nids_df.dtypes.items()}
    gc.collect()

    api_df = None; api_labels = None
    if params["api_dataset_available"]:
        api_df = pd.read_csv(df2_path)
        api_df = kc.optimize_memory(api_df)
        if 'malware' in api_df.columns:
            api_labels = api_df['malware'].values
            api_df = api_df.drop('malware', axis=1)
        api_labels = (api_labels > 0).astype(int)
        log(f"✓ Loaded API: {api_df.shape}")
    else:
        log("  API dataset treated as unavailable (as in the historical notebook run).")

    # ---------------- PHASE 2: HYBRID GA + MI ----------------
    emit({"phase": "feature_selection", "message": "Phase 2 — VarianceThreshold → MI → GA"})
    numeric_frame = nids_df.select_dtypes(include=[np.number])
    numeric_names = [str(c) for c in numeric_frame.columns]
    X_nids = numeric_frame.fillna(0).values
    y_nids = nids_labels
    vt = VarianceThreshold(threshold=1e-5)
    X_nids = vt.fit_transform(X_nids)
    vt_names = [n for n, keep in zip(numeric_names, vt.get_support()) if keep]
    log(f"  After variance threshold: {X_nids.shape[1]} features")
    if X_nids.shape[0] > params["ga_sample"]:
        sample_idx = np.random.choice(X_nids.shape[0], params["ga_sample"], replace=False)
        X_ga_sample = X_nids[sample_idx]; y_ga_sample = y_nids[sample_idx]
    else:
        X_ga_sample = X_nids; y_ga_sample = y_nids
    scaler_ga = RobustScaler()
    X_ga_scaled = scaler_ga.fit_transform(X_ga_sample)
    k_mi = min(100, X_ga_scaled.shape[1])
    mi_selector = SelectKBest(mutual_info_classif, k=k_mi)
    X_ga_mi = mi_selector.fit_transform(X_ga_scaled, y_ga_sample)
    mi_mask = mi_selector.get_support()
    ga_selector = kc.GeneticFeatureSelector(
        X_ga_mi, y_ga_sample,
        population_size=params["ga_population"], generations=params["ga_generations"],
        mutation_rate=0.12, crossover_rate=0.9,
    )
    t_ga = time.time()
    selected_rel, fitness_history = ga_selector.run()
    ga_seconds = time.time() - t_ga
    mi_indices = np.where(mi_mask)[0]
    selected_features_ga = mi_indices[selected_rel]
    X_nids_selected = X_nids[:, selected_features_ga]
    selected_names = [vt_names[i] for i in selected_features_ga]
    log(f"[2.3] Final selected features: {len(selected_features_ga)} → {selected_names}")
    mi_scores = getattr(mi_selector, "scores_", None)
    pd.DataFrame({"generation": np.arange(1, len(fitness_history) + 1), "best_fitness": fitness_history}).to_csv(out / "ga_fitness.csv", index=False)
    feature_record = {
        "numeric_columns": numeric_names, "after_variance_threshold": vt_names,
        "mi_k": int(k_mi), "mi_scores": {vt_names[i]: float(s) for i, s in enumerate(mi_scores)} if mi_scores is not None else {},
        "selected_indices_post_vt": [int(i) for i in selected_features_ga], "selected_features": selected_names,
        "ga_seconds": ga_seconds,
    }
    (out / "selected_features.json").write_text(json.dumps(feature_record, indent=2), encoding="utf-8")

    # ---------------- PHASE 3: CONFIGURATIONS ----------------
    scaler_nids = RobustScaler()
    X_nids_scaled = scaler_nids.fit_transform(X_nids_selected)
    pt = None
    try:
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        X_nids_scaled = pt.fit_transform(X_nids_scaled)
    except Exception:
        pt = None
    X_nids_scaled = np.nan_to_num(X_nids_scaled, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)
    joblib.dump({
        "variance_threshold": vt, "selected_indices_post_vt": [int(i) for i in selected_features_ga],
        "numeric_columns": numeric_names, "selected_features": selected_names, "column_dtypes": column_dtypes,
        "robust_scaler": scaler_nids, "power_transformer": pt,
        "nan_to_num": {"nan": 0.0, "posinf": 5.0, "neginf": -5.0},
    }, out / "models" / "preprocessing.joblib")

    if experiment == "cell04":
        from amsec.notebook.cell04_models import EnsembleStackingModel
    else:
        from amsec.notebook.cell02_models import MobileNetV4Model

    all_results: list[dict] = []
    failures: dict = {}
    for cfg in all_configs:
        if cfg["entry"] not in selected_entries:
            continue
        method = cfg["sampling_method"]
        tag = f"entry{cfg['entry']:02d}"
        log(f"\nConfig {cfg['entry']}/{len(all_configs)}: {cfg['config']}")
        emit({"phase": "config", "entry": cfg["entry"], "total": len(all_configs), "config": cfg["config"]})
        try:
            X_balanced, y_balanced = kc.apply_sampling_method(X_nids_scaled, y_nids, method)
            n_after_sampling = int(len(y_balanced))
            counts_after_sampling = {"0": int(np.sum(y_balanced == 0)), "1": int(np.sum(y_balanced == 1))}
            log(f"  After {method}: {len(y_balanced)} samples  Normal={np.sum(y_balanced==0)}, Attack={np.sum(y_balanced==1)}")
            if len(y_balanced) > params["balanced_cap"]:
                idx = np.random.choice(len(y_balanced), params["balanced_cap"], replace=False)
                X_balanced = X_balanced[idx]; y_balanced = y_balanced[idx]
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.12, random_state=42, stratify=y_train)
            log(f"  Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

            if experiment == "cell04":
                model = EnsembleStackingModel(input_dim=X_train.shape[1], num_classes=2)
                model.build_models()
                history = model.train(X_train, y_train, X_val, y_val, epochs=params["epochs"], batch_size=params["batch_size"])
                results = model.evaluate(X_test, y_test)
                hist_record = {"mobilenetv4": _history_dict(history["mobile_history"]),
                               "lightweight": _history_dict(history["light_history"])}
            else:
                model = MobileNetV4Model(input_dim=X_train.shape[1], num_classes=2, use_attention=cfg["use_attention"])
                model.build_model()
                history = model.train(X_train, y_train, X_val, y_val, epochs=params["epochs"], batch_size=params["batch_size"])
                results = model.evaluate(X_test, y_test)
                hist_record = {cfg["model_type"]: _history_dict(history)}

            row = {
                "entry": cfg["entry"], "config": cfg["config"], "sampling_method": method, "model_type": cfg["model_type"],
                "accuracy": float(results['accuracy']), "precision": float(results['precision']),
                "recall": float(results['recall']), "f1_score": float(results['f1_score']),
                "balanced_accuracy": float(results['balanced_accuracy']), "mcc": float(results['mcc']),
                "roc_auc": float(results['roc_auc']), "parameters": int(results['n_params']), "flops": int(results['flops']),
                "training_time": float(results['training_time']), "inference_time": float(results['inference_time']),
                "train_samples": int(len(y_train)), "val_samples": int(len(y_val)), "test_samples": int(len(y_test)),
                "samples_after_sampling": n_after_sampling, "class_counts_after_sampling": counts_after_sampling,
                "confusion_matrix": np.asarray(results['confusion_matrix']).tolist(),
            }
            all_results.append(row)
            (out / f"history_{tag}.json").write_text(json.dumps(hist_record), encoding="utf-8")
            pd.DataFrame({"y_true": y_test.astype(int), "proba_attack": np.asarray(results["probabilities"])[:, 1]}).to_csv(
                out / f"test_predictions_{tag}.csv", index=False)
            log(f"  ✓ {cfg['config']}  Accuracy: {row['accuracy']:.5f}, F1: {row['f1_score']:.5f}, ROC-AUC: {row['roc_auc']:.5f}")
            if save_models:
                mdir = out / "models" / tag; mdir.mkdir(parents=True, exist_ok=True)
                if experiment == "cell04":
                    model.mobilenetv4.model.save(mdir / "mobilenetv4.keras")
                    model.lightweight.model.save(mdir / "lightweight.keras")
                    joblib.dump(model.meta_learner, mdir / "meta_learner.joblib")
                else:
                    model.model.save(mdir / "model.keras")
            del model
            gc.collect()
        except Exception as e:  # notebook: print and continue
            failures[cfg["config"] + f" (entry {cfg['entry']})"] = repr(e)
            log(f"  Error: {e}\n{traceback.format_exc()}")
            continue
        _write_results(out, all_results)

    results_df = _write_results(out, all_results)

    # ---------------- PHASES 7–9 ----------------
    post = {}
    if run_post_phases and all_results:
        emit({"phase": "post", "message": "Phases 7–9 — best model, retrain, LRP"})
        best_result = max(all_results, key=lambda x: x['f1_score'])
        post["best_result"] = {k: best_result[k] for k in ("entry", "config", "sampling_method", "model_type", "f1_score", "accuracy")}
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_nids_scaled, y_nids, test_size=0.2, random_state=42, stratify=y_nids)
        if experiment == "cell04":
            best_obj = EnsembleStackingModel(input_dim=X_nids_scaled.shape[1], num_classes=2)
            best_obj.build_models()
            best_obj.train(X_train_full, y_train_full, X_test_full, y_test_full, epochs=params["epochs"], batch_size=params["batch_size"])
            api_results = kc_process_api(experiment, api_df, api_labels, best_obj, scaler_nids, selected_features_ga, out)
            lrp_model = best_obj.mobilenetv4.model
            predict = best_obj.predict_proba
        else:
            best_obj = MobileNetV4Model(input_dim=X_nids_scaled.shape[1], num_classes=2,
                                        use_attention='Attention' in best_result['model_type'])
            best_obj.build_model()
            best_obj.train(X_train_full, y_train_full, X_test_full, y_test_full, epochs=params["epochs"], batch_size=params["batch_size"])
            api_results = kc_process_api(experiment, api_df, api_labels, best_obj.model, scaler_nids, selected_features_ga, out)
            lrp_model = best_obj.model
            predict = lambda x: best_obj.model.predict(x, verbose=0)  # noqa: E731
        post["api_evaluation"] = api_results
        lrp_explainer = kc.LRPExplainer(lrp_model)
        sample_indices = np.random.choice(len(X_test_full), min(3, len(X_test_full)), replace=False)
        explanations = []
        for idx in sample_indices:
            X_sample = X_test_full[idx:idx+1]
            pred_proba = predict(X_sample); pred_class = int(np.argmax(pred_proba))
            lrp_result = lrp_explainer.explain(X_sample, target_class=pred_class)
            explanations.append({
                "sample_idx": int(idx), "true_label": int(y_test_full[idx]), "predicted_label": pred_class,
                "prediction_proba": float(np.asarray(pred_proba)[0][pred_class]),
                "relevance_scores": [float(v) for v in lrp_result["relevance_scores"]],
                "feature_names": selected_names,
            })
            log(f"  Sample {idx}: True={y_test_full[idx]}, Predicted={pred_class}, Confidence={np.asarray(pred_proba)[0][pred_class]:.4f}")
        post["lrp_explanations"] = explanations
        (out / "lrp_explanations.json").write_text(json.dumps(explanations, indent=2), encoding="utf-8")

    meta = {
        "experiment": experiment, "parameters": params, "deviations": deviations, "entries_run": selected_entries,
        "nids_shape": list(nids_df.shape), "label_counts": {"normal": int(np.sum(nids_labels == 0)), "attack": int(np.sum(nids_labels == 1))},
        "features": feature_record, "failures": failures, "post_phases": {k: v for k, v in post.items() if k != "lrp_explanations"},
        "wall_seconds": time.time() - t_start,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return {"results": results_df, "metadata": meta, "out_dir": str(out)}


def kc_process_api(experiment, api_df, api_labels, model, scaler, selected_features, out: Path):
    """Phase 8 API evaluation via the notebook's own ``process_api_dataset``.

    Only file-system side effects are redirected: the notebook saves figures to
    ``/kaggle/working/plots``; here ``plt.savefig``/``plt.show`` write into the run
    folder instead. The computation is untouched.
    """
    if api_df is None or api_labels is None:
        print("API dataset not available")
        return None
    import importlib

    import matplotlib.pyplot as plt

    mod = importlib.import_module(f"amsec.notebook.{experiment}_api")
    original_savefig, original_show = plt.savefig, plt.show
    plots = out / "plots"; plots.mkdir(exist_ok=True)
    plt.savefig = lambda path, *a, **k: original_savefig(plots / Path(str(path)).name, *a, **k)  # type: ignore[assignment]
    plt.show = lambda *a, **k: plt.close("all")  # type: ignore[assignment]
    try:
        result = mod.process_api_dataset(api_df, api_labels, model, scaler, selected_features)
    finally:
        plt.savefig, plt.show = original_savefig, original_show
    return result


def _write_results(out: Path, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df):
        df["rank"] = df["f1_score"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    flat = df.drop(columns=[c for c in ("confusion_matrix", "class_counts_after_sampling") if c in df.columns])
    flat.to_csv(out / "results.csv", index=False)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return df
