"""Deterministic forecasting baselines."""

from __future__ import annotations

import pandas as pd


def previous_day_baseline(frame: pd.DataFrame) -> pd.Series:
    """Predict tomorrow with the current day's discharge count."""
    return frame["daily_discharges"].rename("prediction")


def previous_weekday_baseline(frame: pd.DataFrame) -> pd.Series:
    """Predict tomorrow with the same organisation's value seven days earlier."""
    source = frame[["Org Code", "date", "daily_discharges"]].copy()
    source["date"] = source["date"] + pd.Timedelta(days=7)
    source = source.rename(columns={"daily_discharges": "prediction"})
    predictions = frame[["Org Code", "date"]].merge(
        source,
        on=["Org Code", "date"],
        how="left",
        validate="one_to_one",
    )
    return predictions["prediction"].rename("prediction")
