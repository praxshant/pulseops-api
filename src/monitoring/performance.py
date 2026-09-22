"""Rolling prediction performance against actuals."""

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class PerformanceReport(BaseModel):
    window_start: str
    window_end: str
    mae: float | None
    rmse: float | None
    wape: float | None
    n_predictions: int
    n_actuals_matched: int


def compute_performance_report(
    inference_log_path: Path, actuals_path: Path, window_days: int = 7
) -> PerformanceReport:
    """Compute rolling MAE, RMSE, and WAPE by joining predictions to actuals."""
    if not inference_log_path.exists() or not actuals_path.exists():
        return PerformanceReport(
            window_start="",
            window_end="",
            mae=None,
            rmse=None,
            wape=None,
            n_predictions=0,
            n_actuals_matched=0
        )

    # Read inference log
    records = []
    with open(inference_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        return PerformanceReport(
            window_start="",
            window_end="",
            mae=None,
            rmse=None,
            wape=None,
            n_predictions=0,
            n_actuals_matched=0
        )

    preds = pd.DataFrame(records)
    preds["forecast_date"] = pd.to_datetime(preds["forecast_date"])
    
    # Filter to window
    end_date = preds["forecast_date"].max()
    start_date = end_date - pd.Timedelta(days=window_days)
    preds = preds[preds["forecast_date"] > start_date]
    
    if preds.empty:
        return PerformanceReport(
            window_start=str(start_date.date()), window_end=str(end_date.date()), 
            mae=None, rmse=None, wape=None, n_predictions=0, n_actuals_matched=0
        )

    # Load actuals
    actuals = pd.read_parquet(actuals_path)
    actuals["date"] = pd.to_datetime(actuals["date"])
    
    # Merge
    merged = preds.merge(
        actuals, 
        left_on=["org_code", "forecast_date"], 
        right_on=["Org Code", "date"], 
        how="inner"
    )
    
    if merged.empty:
        return PerformanceReport(
            window_start=str(start_date.date()), window_end=str(end_date.date()), 
            mae=None, rmse=None, wape=None, n_predictions=len(preds), n_actuals_matched=0
        )
    
    # In PulseOps target is usually 'target_next_day', but we use the actuals file
    # We should match the prediction with the actual attended volume for that date.
    if "target_next_day" in merged.columns:
        actual_col = "target_next_day"
    elif "Total Attendances" in merged.columns:
        actual_col = "Total Attendances"
    else:
        return PerformanceReport(
            window_start=str(start_date.date()), window_end=str(end_date.date()), 
            mae=None, rmse=None, wape=None, n_predictions=len(preds), n_actuals_matched=0
        )
    
    merged = merged.dropna(subset=[actual_col, "prediction"])
    
    if merged.empty:
        return PerformanceReport(
            window_start=str(start_date.date()), window_end=str(end_date.date()), 
            mae=None, rmse=None, wape=None, n_predictions=len(preds), n_actuals_matched=0
        )

    mae = float((merged["prediction"] - merged[actual_col]).abs().mean())
    rmse = float(((merged["prediction"] - merged[actual_col]) ** 2).mean() ** 0.5)
    sum_actual = float(merged[actual_col].sum())
    
    if sum_actual > 0:
        wape = float((merged["prediction"] - merged[actual_col]).abs().sum() / sum_actual)
    else:
        wape = 0.0
    
    return PerformanceReport(
        window_start=str(start_date.date()),
        window_end=str(end_date.date()),
        mae=mae,
        rmse=rmse,
        wape=wape,
        n_predictions=len(preds),
        n_actuals_matched=len(merged)
    )
