"""Load, validate, and fingerprint the canonical forecasting dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "Org Code",
    "date",
    "daily_discharges",
    "target_next_day",
}


def schema_hash(frame: pd.DataFrame) -> str:
    """Hash ordered column names and dtypes for schema lineage."""
    schema = [(column, str(frame[column].dtype)) for column in frame.columns]
    payload = json.dumps(schema, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a dataset file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_forecasting_dataset(frame: pd.DataFrame) -> None:
    """Enforce the canonical organisation-day dataset contract."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing forecasting columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Forecasting dataset is empty")
    if frame["Org Code"].isna().any():
        raise ValueError("Forecasting dataset contains missing organisation identifiers")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Forecasting dataset contains invalid dates")
    if frame.duplicated(["Org Code", "date"]).any():
        raise ValueError("Forecasting dataset contains duplicate organisation/date keys")
    values = pd.to_numeric(frame["daily_discharges"], errors="coerce")
    if values.isna().any() or (values < 0).any():
        raise ValueError("daily_discharges must be numeric and non-negative")


def load_forecasting_dataset(path: Path) -> pd.DataFrame:
    """Read and validate a canonical Parquet dataset."""
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    validate_forecasting_dataset(frame)
    return frame.sort_values(["Org Code", "date"]).reset_index(drop=True)


def dataset_manifest(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Return reproducibility metadata for a loaded dataset."""
    return {
        "dataset_path": str(path),
        "dataset_hash": file_sha256(path),
        "schema_hash": schema_hash(frame),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "date_min": frame["date"].min().date().isoformat(),
        "date_max": frame["date"].max().date().isoformat(),
        "feature_columns": list(frame.columns),
    }
