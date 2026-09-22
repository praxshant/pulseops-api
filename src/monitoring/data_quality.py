"""Pre-inference data quality checks."""

from datetime import datetime, timezone


def check_feature_completeness(features: dict, required_features: list[str]) -> list[str]:
    """Return a list of missing feature names."""
    return [f for f in required_features if f not in features or features[f] is None]


def check_feature_ranges(features: dict) -> list[str]:
    """Return a list of features with out-of-range values."""
    errors = []
    # Arbitrary domain checks for example
    for k, v in features.items():
        if isinstance(v, (int, float)) and v < 0:
            if "lag" in k or "rolling" in k or k == "previous_month_total_attendances":
                errors.append(f"{k} cannot be negative")
    return errors


def check_data_staleness(forecast_date: str, tolerance_days: int = 7) -> bool:
    """Return True if the forecast date is too far in the past."""
    try:
        dt = datetime.strptime(forecast_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt).days > tolerance_days
    except ValueError:
        return False
