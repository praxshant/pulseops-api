"""Chronological train, validation, and test splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Inclusive date boundaries for a forecasting evaluation."""

    train_start: str = "2026-04-01"
    train_end: str = "2026-06-30"
    validation_start: str = "2026-07-01"
    validation_end: str = "2026-07-31"
    test_start: str = "2026-08-01"
    test_end: str = "2026-08-31"

    def as_dict(self) -> dict[str, str]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def chronological_split(
    frame: pd.DataFrame,
    config: TemporalSplitConfig,
) -> dict[str, pd.DataFrame]:
    """Split rows by date without randomization or overlap."""
    dates = pd.to_datetime(frame["date"], errors="coerce")
    boundaries = {
        "train": (config.train_start, config.train_end),
        "validation": (config.validation_start, config.validation_end),
        "test": (config.test_start, config.test_end),
    }
    splits: dict[str, pd.DataFrame] = {}
    for name, (start, end) in boundaries.items():
        mask = dates.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        splits[name] = (
            frame.loc[mask]
            .copy()
            .sort_values(["Org Code", "date"])
            .reset_index(drop=True)
        )
    if any(split.empty for split in splits.values()):
        raise ValueError("Temporal split produced an empty partition")
    if set(splits["train"]["date"]).intersection(splits["validation"]["date"]):
        raise ValueError("Train and validation dates overlap")
    if set(splits["validation"]["date"]).intersection(splits["test"]["date"]):
        raise ValueError("Validation and test dates overlap")
    return splits
