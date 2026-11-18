"""Leakage-controlled validation protocol (scientifically separate from notebook reproduction)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from amsec.artifacts.storage import RunStorage
from amsec.data.feature_engineering import subsample_rows
from amsec.data.loaders import DatasetPaths, load_nids_dataset
from amsec.data.preprocessing import notebook_scale_matrix, variance_filter
from amsec.evaluation.curves import confusion_frames, roc_pr_frames
from amsec.evaluation.metrics import classification_metrics
from amsec.experiments.configuration import ExperimentConfig
from amsec.feature_selection.genetic_algorithm import GeneticFeatureSelector
from amsec.feature_selection.mutual_information import mutual_info_prefilter
from amsec.notebook_models.ensemble_stacking import EnsembleStackingTorch
from amsec.resampling.strategies import apply_sampling_method


def run_leakage_controlled_stacking(
    config: ExperimentConfig,
    storage: RunStorage,
    progress: Callable | None = None,
) -> dict:
    def emit(payload: dict) -> None:
        storage.save_progress(payload)
        if progress:
            progress(payload)

    emit({"phase": "load", "message": "Loading NIDS dataset"})
    bundle = load_nids_dataset(DatasetPaths(), nrows=config.nrows)
    x_raw = bundle["X"]
    y = bundle["y"]
    feature_names = bundle["feature_names"]
    dataset_meta = bundle["meta"]

    # 1. Split original observations first
    emit({"phase": "split", "message": "Hold-out split before fit"})
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.15, random_state=config.random_seed, stratify=y
    )
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_raw, y_train, test_size=0.12, random_state=config.random_seed, stratify=y_train
    )

    # 2. Variance / selection / scale on training only
    emit({"phase": "variance", "message": "Variance threshold on train"})
    x_train_f, vt = variance_filter(x_train_raw)
    x_val_f = vt.transform(x_val_raw)
    x_test_f = vt.transform(x_test_raw)
    kept_names = [feature_names[i] for i, keep in enumerate(vt.get_support()) if keep]

    x_ga, y_ga = subsample_rows(x_train_f, y_train, config.ga_sample_rows, seed=config.random_seed)
    x_ga_scaled, scaler_ga, _ = notebook_scale_matrix(x_ga, fit=True, use_power=False)

    emit({"phase": "mi", "message": "MI on training fold only"})
    x_mi, mi_indices, mi_selector = mutual_info_prefilter(x_ga_scaled, y_ga)
    # Map MI mask onto full train matrix using a fresh SelectKBest fit on train
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    k_eff = min(100, x_train_f.shape[1])

    def _mi(xx, yy):
        return mutual_info_classif(xx, yy, random_state=config.random_seed)

    # Fit MI on scaled train
    x_train_scaled_tmp, scaler_tmp, _ = notebook_scale_matrix(x_train_f, fit=True, use_power=False)
    mi_full = SelectKBest(_mi, k=k_eff)
    mi_full.fit(x_train_scaled_tmp, y_train)
    mi_indices = np.where(mi_full.get_support())[0]
    x_train_mi = x_train_scaled_tmp[:, mi_indices]
    # GA on train MI features (subsample)
    x_ga2, y_ga2 = subsample_rows(x_train_mi, y_train, min(config.ga_sample_rows, len(y_train)), seed=config.random_seed)

    emit({"phase": "ga", "message": "GA on training fold only"})
    ga = GeneticFeatureSelector(
        x_ga2,
        y_ga2,
        population_size=config.ga_population,
        generations=config.ga_generations,
        random_state=config.random_seed,
        n_estimators=25 if config.smoke else 100,
        progress_callback=emit,
    )
    selected_rel, fitness_history = ga.run()
    selected_indices = [int(mi_indices[i]) for i in selected_rel]
    selected_names = [kept_names[i] for i in selected_indices]
    storage.save_selected_features(selected_names, selected_indices)
    storage.write_csv(
        "ga_fitness.csv",
        __import__("pandas").DataFrame({"generation": list(range(1, len(fitness_history) + 1)), "best_fitness": fitness_history}),
    )

    x_train_sel = x_train_f[:, selected_indices]
    x_val_sel = x_val_f[:, selected_indices]
    x_test_sel = x_test_f[:, selected_indices]

    emit({"phase": "scale", "message": "Fit scaler on train only"})
    x_train_s, scaler, power = notebook_scale_matrix(x_train_sel, fit=True, use_power=True)
    x_val_s, _, _ = notebook_scale_matrix(x_val_sel, fit=False, scaler=scaler, power=power, use_power=power is not None)
    x_test_s, _, _ = notebook_scale_matrix(x_test_sel, fit=False, scaler=scaler, power=power, use_power=power is not None)

    emit({"phase": "resample", "message": f"Resample train only: {config.sampling_method}"})
    x_train_b, y_train_b = apply_sampling_method(x_train_s, y_train, config.sampling_method, random_state=config.random_seed)
    if len(y_train_b) > config.max_balanced_rows:
        rng = np.random.RandomState(config.random_seed)
        idx = rng.choice(len(y_train_b), config.max_balanced_rows, replace=False)
        x_train_b, y_train_b = x_train_b[idx], y_train_b[idx]

    emit({"phase": "train", "message": "Training ensemble", "train": len(y_train_b), "val": len(y_val), "test": len(y_test)})
    ensemble = EnsembleStackingTorch(input_dim=x_train_b.shape[1])
    # Meta-learner: use validation base predictions (less leaky than train in-sample)
    history = ensemble.train(
        x_train_b,
        y_train_b,
        x_val_s,
        y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        progress=emit,
        meta_on_train=False,
    )
    storage.save_history(history)
    storage.save_models(ensemble)

    y_proba = ensemble.predict_proba(x_test_s)
    y_pred = y_proba.argmax(axis=1)
    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics.update(
        {
            "parameters": ensemble.params,
            "flops": ensemble.flops,
            "training_seconds": ensemble.training_time,
            "backend": ensemble.backend,
            "protocol": "leakage_controlled",
            "sampling_method": config.sampling_method,
            "n_features_selected": len(selected_names),
            "selected_features": selected_names,
            "split_sizes": {"train": int(len(y_train_b)), "val": int(len(y_val)), "test": int(len(y_test))},
            "train_class_counts_after_resample": {"0": int((y_train_b == 0).sum()), "1": int((y_train_b == 1).sum())},
            "leakage_controls": [
                "split_before_fit",
                "scaler_train_only",
                "mi_ga_train_only",
                "resample_train_only",
                "meta_learner_on_validation_predictions",
                "evaluate_once_on_held_out_test",
            ],
        }
    )
    storage.save_metrics(metrics)
    storage.save_predictions(y_test, y_pred, y_proba)
    raw_cm, norm_cm = confusion_frames(metrics["confusion_matrix"])
    storage.write_csv("confusion_matrix.csv", raw_cm.reset_index().rename(columns={"index": "row"}))
    storage.write_csv("confusion_matrix_normalized.csv", norm_cm.reset_index().rename(columns={"index": "row"}))
    roc, pr, curve_summary = roc_pr_frames(y_test, y_proba[:, 1])
    storage.write_csv("roc_curve.csv", roc)
    storage.write_csv("precision_recall_curve.csv", pr)
    storage.write_json("complexity.json", {"parameters": ensemble.params, "flops": ensemble.flops, "flops_rule": "2 * parameters"})
    storage.write_json(
        "model_manifest.json",
        {
            "backend": ensemble.backend,
            "input_dim": int(x_train_b.shape[1]),
            "selected_features": selected_names,
            "protocol": "leakage_controlled",
        },
    )
    storage.save_metadata({"dataset": dataset_meta, "protocol": "leakage_controlled", "curve_summary": curve_summary})
    emit({"phase": "done", "message": "completed", "metrics": {"accuracy": metrics["accuracy"], "f1": metrics["f1"]}})
    return metrics
