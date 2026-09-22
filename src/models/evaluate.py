"""Evaluate forecasting predictions and persist reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.metrics import regression_metrics


def evaluate_predictions(
    actual: pd.Series,
    predicted: pd.Series,
    output_path: Path | None = None,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and WAPE, optionally writing JSON metrics."""
    metrics = regression_metrics(actual, predicted)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
