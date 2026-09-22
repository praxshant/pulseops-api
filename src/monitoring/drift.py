"""Advanced ML monitoring with drift detection and performance tracking."""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel


class DriftStatus(str, Enum):
    STABLE = "STABLE"
    WARNING = "WARNING"
    ALERT = "ALERT"


class DriftReport(BaseModel):
    timestamp: str
    feature: str
    psi: float
    mean_shift: float
    status: DriftStatus


def mean_shift(reference: pd.Series, current: pd.Series) -> float:
    """Return the absolute difference between reference and current means."""
    if reference.empty or current.empty:
        raise ValueError("Both reference and current series must be non-empty")
    return float(abs(reference.mean() - current.mean()))


def population_stability_index(
    reference: pd.Series, current: pd.Series, num_buckets: int = 10
) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions."""
    if reference.empty or current.empty:
        raise ValueError("Both reference and current series must be non-empty")
    if reference.nunique() == 1 and current.nunique() == 1 and reference.iloc[0] == current.iloc[0]:
        return 0.0

    # Create buckets based on reference distribution
    breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, num_buckets + 1)))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)

    ref_percents = ref_counts / len(reference)
    curr_percents = curr_counts / len(current)

    # Avoid zero division
    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)

    psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
    return float(psi)


def detect_drift(
    psi_value: float,
    warning_threshold: float = 0.1,
    alert_threshold: float = 0.2
) -> DriftStatus:
    if psi_value >= alert_threshold:
        return DriftStatus.ALERT
    elif psi_value >= warning_threshold:
        return DriftStatus.WARNING
    return DriftStatus.STABLE
