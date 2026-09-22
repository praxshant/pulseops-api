"""FastAPI application for real model serving with inference observability."""

from __future__ import annotations

import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from api.inference_log import log_inference_event
from monitoring.data_quality import (
    check_data_staleness,
    check_feature_completeness,
    check_feature_ranges,
)
from monitoring.drift import DriftReport, DriftStatus, detect_drift, population_stability_index
from monitoring.metrics import (
    DATA_QUALITY_FAILURES,
    DRIFT_STATUS,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
)

app = FastAPI(title="PulseOps API", version="0.3.0")

# --------------------------------------------------------------------------- #
# Model state                                                                   #
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, Any] = {}
_INFERENCE_LOG_DIR = Path("artifacts/inference")
_ACTUALS_PATH = Path("data/processed/forecasting_dataset.parquet")


def _release_freed_memory() -> None:
    """Return memory freed after deserialization back to the OS (glibc only).

    joblib rebuilding the tree ensemble briefly allocates far more than the
    model's steady-state footprint. On glibc that freed memory is retained by
    the allocator, keeping RSS near the load spike; malloc_trim hands it back
    so the container stays well under tight memory limits. No-op off glibc.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _load_model() -> tuple[Any, str, str]:
    """Load the latest registered model from disk or return cached instance."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["version"], _MODEL_CACHE["run_id"]

    import glob
    import json

    runs_root = Path("artifacts/runs")
    run_jsons = sorted(glob.glob(str(runs_root / "*/run.json")), reverse=True)

    for run_json_path in run_jsons:
        meta = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
        gate = meta.get("quality_gate", {})
        if gate.get("passed") is True:
            # Resolve model artifact relative to the run.json directory
            model_path = Path(run_json_path).parent / "model.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                _release_freed_memory()
                run_id = meta["run_id"]
                version = run_id[:8]
                _MODEL_CACHE.update({"model": model, "version": version, "run_id": run_id})
                return model, version, run_id

    raise RuntimeError(
        "No quality-gate-passing model found in artifacts/runs/. "
        "Run python -m models.train first."
    )


@app.on_event("startup")
def _preload_model() -> None:
    """Load the model once at boot so the memory spike happens before traffic.

    If the container survives startup it has proven it fits in memory; a
    request-time load could otherwise spike an already-busy process over the
    limit. Failures here surface immediately in the deploy logs.
    """
    try:
        _load_model()
    except RuntimeError:
        # No model yet — surfaced per-request as a 503; don't block startup.
        pass


# --------------------------------------------------------------------------- #
# Feature columns (must match training)                                         #
# --------------------------------------------------------------------------- #
REQUIRED_FEATURES = [
    "day_of_week",
    "is_weekend",
    "lag_1_discharge",
    "lag_7_discharge",
    "lag_14_discharge",
    "lag_28_discharge",
    "rolling_7d_discharge",
    "rolling_14d_discharge",
    "rolling_28d_discharge",
    "previous_month_total_attendances",
    "previous_month_emergency_admissions",
    "previous_month_over_4_hours",
    "ae_context_available",
    "occupied_beds",
    "available_beds",
    "occupancy_rate",
    "bed_context_available",
]


# --------------------------------------------------------------------------- #
# Request / response schemas                                                    #
# --------------------------------------------------------------------------- #
class PredictionRequest(BaseModel):
    org_code: str = Field(..., description="NHS organisation code")
    forecast_date: str = Field(..., description="Date to forecast (YYYY-MM-DD)")
    features: dict[str, float] = Field(
        ..., description="Feature values keyed by feature name"
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted next-day discharge count")
    org_code: str
    forecast_date: str
    model_version: str
    run_id: str
    prediction_id: str
    latency_ms: float


# --------------------------------------------------------------------------- #
# Endpoints                                                                     #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    """Report service health for container orchestration."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift")
def drift() -> dict[str, Any]:
    """Compute PSI-based feature drift: training reference vs recent inference inputs."""
    runs_root = Path("artifacts/runs")
    run_jsons = sorted(glob.glob(str(runs_root / "*/run.json")), reverse=True)

    # Find the latest quality-gate-passing run
    best_run: dict[str, Any] | None = None
    for rj in run_jsons:
        meta = json.loads(Path(rj).read_text(encoding="utf-8"))
        if meta.get("quality_gate", {}).get("passed") is True:
            best_run = meta
            break

    if best_run is None:
        return {"status": "no_model", "reports": []}

    # Load training reference data
    dataset_path = Path(best_run["dataset"]["dataset_path"].replace("\\", "/"))
    if not dataset_path.exists():
        return {"status": "reference_data_unavailable", "reports": []}

    reference_df = pd.read_parquet(dataset_path)
    feature_cols = best_run["features"]

    # Load recent inference feature values from log
    inference_log = _INFERENCE_LOG_DIR / "inference.jsonl"
    current_rows: list[dict[str, float]] = []
    if inference_log.exists():
        with inference_log.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "features" in rec and isinstance(rec["features"], dict):
                        current_rows.append(rec["features"])

    if not current_rows:
        return {
            "status": "no_inference_data",
            "message": (
                "No inference records with feature values yet. Make some /predict calls first."
            ),
            "reports": [],
        }

    current_df = pd.DataFrame(current_rows)

    # Compute PSI for each feature
    reports: list[dict[str, Any]] = []
    overall_status = DriftStatus.STABLE
    now_ts = datetime.now(timezone.utc).isoformat()

    for feat in feature_cols:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref_series = reference_df[feat].dropna()
        cur_series = current_df[feat].dropna()
        if ref_series.empty or cur_series.empty:
            continue
        psi = population_stability_index(ref_series, cur_series)
        shift = float(abs(ref_series.mean() - cur_series.mean()))
        status = detect_drift(psi)
        if status == DriftStatus.ALERT:
            overall_status = DriftStatus.ALERT
        elif status == DriftStatus.WARNING and overall_status == DriftStatus.STABLE:
            overall_status = DriftStatus.WARNING
        reports.append(DriftReport(
            timestamp=now_ts,
            feature=feat,
            psi=round(psi, 6),
            mean_shift=round(shift, 4),
            status=status,
        ).model_dump())

    # Update Prometheus drift gauge
    drift_map = {DriftStatus.STABLE: 0, DriftStatus.WARNING: 1, DriftStatus.ALERT: 2}
    DRIFT_STATUS.set(drift_map[overall_status])

    return {
        "status": overall_status,
        "run_id": best_run["run_id"],
        "n_inference_samples": len(current_rows),
        "n_reference_samples": len(reference_df),
        "reports": reports,
    }


@app.get("/performance")
def performance() -> dict[str, Any]:
    """Compute and return rolling MAE/WAPE from inference log."""
    from monitoring.metrics import ROLLING_MAE, ROLLING_WAPE
    from monitoring.performance import compute_performance_report
    
    report = compute_performance_report(_INFERENCE_LOG_DIR / "inference.jsonl", _ACTUALS_PATH)
    if report.mae is not None:
        ROLLING_MAE.set(report.mae)
    if report.wape is not None:
        ROLLING_WAPE.set(report.wape)
        
    return report.model_dump()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Load the production model and return a real prediction."""
    t0 = time.perf_counter()

    # 1. Data Quality Checks
    missing = check_feature_completeness(request.features, REQUIRED_FEATURES)
    range_errors = check_feature_ranges(request.features)
    
    if missing or range_errors:
        DATA_QUALITY_FAILURES.inc()
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=422,
            detail={"missing": missing, "range_errors": range_errors},
        )
        
    is_stale = check_data_staleness(request.forecast_date)

    try:
        model, version, run_id = _load_model()
    except RuntimeError as exc:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    feature_row = pd.DataFrame([{f: request.features[f] for f in REQUIRED_FEATURES}])
    prediction_value = float(model.predict(feature_row)[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    # Prometheus update (latency takes seconds, we have ms)
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    PREDICTION_COUNT.labels(status="ok").inc()

    record = log_inference_event(
        log_dir=_INFERENCE_LOG_DIR,
        org_code=request.org_code,
        forecast_date=request.forecast_date,
        prediction=prediction_value,
        model_version=version,
        run_id=run_id,
        latency_ms=latency_ms,
        features=request.features,
        stale_data=is_stale,
    )

    return PredictionResponse(
        prediction=prediction_value,
        org_code=request.org_code,
        forecast_date=request.forecast_date,
        model_version=version,
        run_id=run_id,
        prediction_id=record["prediction_id"],
        latency_ms=round(latency_ms, 3),
    )
