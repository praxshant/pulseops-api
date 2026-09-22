"""Structured inference logging for prediction observability."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_inference_event(
    log_dir: Path,
    org_code: str,
    forecast_date: str,
    prediction: float,
    model_version: str,
    run_id: str,
    latency_ms: float,
    status: str = "ok",
    features: dict[str, float] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append one structured inference record and return it."""
    record: dict[str, Any] = {
        "prediction_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "org_code": org_code,
        "forecast_date": forecast_date,
        "prediction": prediction,
        "model_version": model_version,
        "run_id": run_id,
        "latency_ms": round(latency_ms, 3),
        "status": status,
        **extra,
    }
    if features is not None:
        record["features"] = features
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "inference.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
    return record
