"""Configuration and lineage metadata for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Stable configuration recorded with every training run."""

    experiment_name: str = "pulseops-discharge-forecast"
    model_type: str = "extra_trees"
    feature_set: str = "all_context"
    random_seed: int = 42
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
