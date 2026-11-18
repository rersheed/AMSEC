"""Original notebook procedure for ensemble stacking (cell 4 order).

WARNING: This protocol intentionally preserves leakage present in the notebook:
scaler/MI/GA/resampling before split, and meta-learner fit on in-sample predictions.
"""
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


def run_notebook_stacking(
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
    x = bundle["X"]
    y = bundle["y"]
    feature_names = bundle["feature_names"]
    dataset_meta = bundle["meta"]

    emit({"phase": "variance", "message": "Variance threshold"})
    x, vt = variance_filter(x)
    kept_names = [feature_names[i] for i, keep in enumerate(vt.get_support()) if keep]

    # GA subsample (notebook)
    x_ga, y_ga = subsample_rows(x, y, config.ga_sample_rows, seed=config.random_seed)
    x_ga_scaled, _, _ = notebook_scale_matrix(x_ga, fit=True, use_power=False)

    emit({"phase": "mi", "message": "Mutual information prefilter"})
    x_mi, mi_indices, _ = mutual_info_prefilter(x_ga_scaled, y_ga)
    mi_names = [kept_names[i] for i in mi_indices]

    emit({"phase": "ga", "message": "Genetic algorithm", "generations": config.ga_generations})
    ga = GeneticFeatureSelector(
        x_mi,
        y_ga,
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

    x_selected = x[:, selected_indices]
    # Leakage: fit scaler on full matrix before split (notebook)
    emit({"phase": "scale", "message": "Scaling full matrix (notebook order)"})
    x_scaled, scaler, power = notebook_scale_matrix(x_selected, fit=True, use_power=True)

    emit({"phase": "resample", "message": f"Applying {config.sampling_method} before split"})
    x_bal, y_bal = apply_sampling_method(x_scaled, y, config.sampling_method, random_state=config.random_seed)
    if len(y_bal) > config.max_balanced_rows:
        rng = np.random.RandomState(config.random_seed)
        idx = rng.choice(len(y_bal), config.max_balanced_rows, replace=False)
        x_bal, y_bal = x_bal[idx], y_bal[idx]

    x_train, x_test, y_train, y_test = train_test_split(
        x_bal, y_bal, test_size=0.15, random_state=config.random_seed, stratify=y_bal
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.12, random_state=config.random_seed, stratify=y_train
    )

    emit({"phase": "train", "message": "Training ensemble stacking", "train": len(y_train), "val": len(y_val), "test": len(y_test)})
    ensemble = EnsembleStackingTorch(input_dim=x_train.shape[1])
    history = ensemble.train(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        progress=emit,
        meta_on_train=True,
    )
    storage.save_history(history)
    storage.save_models(ensemble)

    y_proba = ensemble.predict_proba(x_test)
    y_pred = y_proba.argmax(axis=1)
    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics.update(
        {
            "parameters": ensemble.params,
            "flops": ensemble.flops,
            "training_seconds": ensemble.training_time,
            "backend": ensemble.backend,
            "protocol": "notebook_reproduction",
            "sampling_method": config.sampling_method,
            "n_features_selected": len(selected_names),
            "selected_features": selected_names,
            "split_sizes": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
            "balanced_class_counts": {"0": int((y_bal == 0).sum()), "1": int((y_bal == 1).sum())},
            "leakage_preserved": [
                "scaler_fit_before_split",
                "mi_ga_before_split",
                "resample_before_split",
                "meta_learner_on_train_predictions",
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
    storage.write_json(
        "complexity.json",
        {"parameters": ensemble.params, "flops": ensemble.flops, "flops_rule": "2 * parameters"},
    )
    storage.write_json(
        "model_manifest.json",
        {
            "backend": ensemble.backend,
            "input_dim": int(x_train.shape[1]),
            "selected_features": selected_names,
            "scaler": "RobustScaler+PowerTransformer",
            "meta_learner": "LogisticRegression",
        },
    )
    storage.save_metadata(
        {
            "dataset": dataset_meta,
            "protocol": "notebook_reproduction",
            "curve_summary": curve_summary,
            "mi_prefilter_features": mi_names,
            "notebook_source": "Final phd-Experiment-android-malware.ipynb#cell4",
            "keras_available": __import__("amsec.notebook_models.keras_builders", fromlist=["keras_available"]).keras_available(),
        }
    )
    emit({"phase": "done", "message": "completed", "metrics": {"accuracy": metrics["accuracy"], "f1": metrics["f1"]}})
    return metrics
