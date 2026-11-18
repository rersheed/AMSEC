"""Unified experiment entrypoint used by the Laboratory and CLI worker."""
from __future__ import annotations

from typing import Callable

from amsec.artifacts.registry import ExperimentRegistry
from amsec.artifacts.storage import RunStorage
from amsec.experiments.configuration import ExperimentConfig
from amsec.experiments.leakage_controlled import run_leakage_controlled_stacking
from amsec.experiments.notebook_reproduction import run_notebook_stacking


def run_experiment(config: ExperimentConfig, progress: Callable | None = None) -> dict:
    registry = ExperimentRegistry()
    run_id = config.run_id or registry.next_run_id()
    config.run_id = run_id
    storage = RunStorage(run_id)
    storage.save_configuration(config.to_dict())
    storage.mark_status("running")
    registry.register(
        {
            "run_id": run_id,
            "status": "running",
            "family": config.family,
            "protocol": config.protocol,
            "sampling_method": config.sampling_method,
            "model_variant": config.model_variant,
            "smoke": config.smoke,
        }
    )
    try:
        if config.family == "ensemble_stacking" and config.protocol == "notebook_reproduction":
            metrics = run_notebook_stacking(config, storage, progress=progress)
        elif config.family == "ensemble_stacking" and config.protocol == "leakage_controlled":
            metrics = run_leakage_controlled_stacking(config, storage, progress=progress)
        elif config.family == "mobilenet_ablation":
            # Reuse stacking path with single-tower training via ensemble for now is wrong.
            # Train MobileNet-only by using ensemble_stacking until dedicated ablation runner lands;
            # mark explicitly.
            raise NotImplementedError(
                "MobileNet ablation runner: use ensemble_stacking single configs for now, "
                "or extend experiments/mobilenet_ablation.py. Full 16-config matrix is deliberate batch-only."
            )
        elif config.family == "cnn_bilstm":
            raise NotImplementedError(
                "CNN–BiLSTM runner is scaffolded in notebook_models.cnn_bilstm; "
                "wire a dedicated runner before claiming cell-22 reproduction."
            )
        else:
            raise ValueError(f"Unsupported family/protocol: {config.family}/{config.protocol}")

        storage.mark_status("completed", accuracy=metrics.get("accuracy"), f1=metrics.get("f1"))
        registry.register(
            {
                "run_id": run_id,
                "status": "completed",
                "family": config.family,
                "protocol": config.protocol,
                "sampling_method": config.sampling_method,
                "model_variant": config.model_variant,
                "smoke": config.smoke,
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
            }
        )
        return {"run_id": run_id, "status": "completed", "metrics": metrics}
    except Exception as exc:
        storage.mark_status("failed", error=str(exc))
        registry.register(
            {
                "run_id": run_id,
                "status": "failed",
                "family": config.family,
                "protocol": config.protocol,
                "sampling_method": config.sampling_method,
                "error": str(exc),
            }
        )
        raise
