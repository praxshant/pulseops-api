"""Regression metrics for discharge forecasting."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE, and WAPE on aligned non-null observations."""
    values = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if values.empty:
        raise ValueError("Cannot calculate metrics without valid observations")
    errors = values["actual"] - values["predicted"]
    denominator = values["actual"].abs().sum()
    return {
        "mae": float(mean_absolute_error(values["actual"], values["predicted"])),
        "rmse": float(mean_squared_error(values["actual"], values["predicted"]) ** 0.5),
        "wape": float(errors.abs().sum() / denominator) if denominator else 0.0,
        "rows": int(len(values)),
    }
