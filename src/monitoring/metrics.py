"""Prometheus metrics exposed by the service."""

from prometheus_client import REGISTRY, Counter, Gauge, Histogram


def _counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    try:
        return Counter(name, doc, labels or [])
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


def _histogram(name: str, doc: str) -> Histogram:
    try:
        return Histogram(name, doc)
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


def _gauge(name: str, doc: str) -> Gauge:
    try:
        return Gauge(name, doc)
    except ValueError:
        return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]


PREDICTION_COUNT = _counter(
    "pulseops_predictions_total",
    "Total number of predictions served",
    ["status"],
)

PREDICTION_LATENCY = _histogram(
    "pulseops_prediction_latency_seconds",
    "Prediction latency in seconds",
)

DRIFT_STATUS = _gauge(
    "pulseops_drift_status",
    "Drift status (0=STABLE, 1=WARNING, 2=ALERT)",
)

ROLLING_MAE = _gauge(
    "pulseops_rolling_mae",
    "Rolling Mean Absolute Error",
)

ROLLING_WAPE = _gauge(
    "pulseops_rolling_wape",
    "Rolling Weighted Absolute Percentage Error",
)

DATA_QUALITY_FAILURES = _counter(
    "pulseops_data_quality_failures_total",
    "Total number of data quality check failures",
)
