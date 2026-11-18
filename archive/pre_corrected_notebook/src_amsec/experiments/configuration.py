"""Experiment configuration objects for the Laboratory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    family: str = "ensemble_stacking"  # mobilenet_ablation | ensemble_stacking | cnn_bilstm
    protocol: str = "notebook_reproduction"  # notebook_reproduction | leakage_controlled
    sampling_method: str = "SMOTE-ENN"
    model_variant: str = "ensemble"  # mobilenet | mobilenet_attention | ensemble | cnn_bilstm
    dataset: str = "nids"  # nids | api
    nrows: int | None = 150_000
    epochs: int = 30
    batch_size: int = 128
    random_seed: int = 42
    ga_population: int = 40
    ga_generations: int = 30
    ga_sample_rows: int = 20_000
    max_balanced_rows: int = 180_000
    smoke: bool = False
    run_id: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


SMOKE_STACKING_CONFIG = ExperimentConfig(
    family="ensemble_stacking",
    protocol="leakage_controlled",
    sampling_method="SMOTE",
    model_variant="ensemble",
    dataset="nids",
    nrows=4_000,
    epochs=1,
    batch_size=64,
    random_seed=42,
    ga_population=4,
    ga_generations=2,
    ga_sample_rows=2_000,
    max_balanced_rows=6_000,
    smoke=True,
    notes="Local smoke profile — not a full reproduction.",
)

NOTEBOOK_DEFAULT_STACKING = ExperimentConfig(
    family="ensemble_stacking",
    protocol="notebook_reproduction",
    sampling_method="SMOTE-ENN",
    model_variant="ensemble",
    dataset="nids",
    nrows=150_000,
    epochs=30,
    batch_size=128,
    random_seed=42,
    ga_population=40,
    ga_generations=30,
    ga_sample_rows=20_000,
    max_balanced_rows=180_000,
    smoke=False,
    notes="Cell-4 notebook hyperparameters.",
)
