"""Reproducible baseline and candidate training runner."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.dataset import dataset_manifest, load_forecasting_dataset
from models.baselines import previous_day_baseline, previous_weekday_baseline
from models.metrics import regression_metrics
from models.run_config import RunConfig
from models.split import TemporalSplitConfig, chronological_split
from models.tracking import RunTracker

FEATURE_SETS: dict[str, list[str]] = {
    "calendar": ["day_of_week", "is_weekend"],
    "calendar_lags": [
        "day_of_week", "is_weekend", "lag_1_discharge", "lag_7_discharge",
        "lag_14_discharge", "lag_28_discharge",
    ],
    "calendar_lags_rolling": [
        "day_of_week", "is_weekend", "lag_1_discharge", "lag_7_discharge",
        "lag_14_discharge", "lag_28_discharge", "rolling_7d_discharge",
        "rolling_14d_discharge", "rolling_28d_discharge",
    ],
    "all_context": [
        "day_of_week", "is_weekend", "lag_1_discharge", "lag_7_discharge",
        "lag_14_discharge", "lag_28_discharge", "rolling_7d_discharge",
        "rolling_14d_discharge", "rolling_28d_discharge",
        "previous_month_total_attendances", "previous_month_emergency_admissions",
        "previous_month_over_4_hours", "ae_context_available", "occupied_beds",
        "available_beds", "occupancy_rate", "bed_context_available",
    ],
}


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, branch, dirty = "unknown", "unknown", True
    return {"git_commit": commit, "git_branch": branch, "dirty_worktree": dirty}


def _environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _model(model_type: str, seed: int) -> object:
    if model_type == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _score_split(frame: pd.DataFrame, prediction: pd.Series) -> dict[str, float]:
    return regression_metrics(frame["target_next_day"], prediction)


def run_training(
    dataset_path: Path,
    run_config: RunConfig | None = None,
    split_config: TemporalSplitConfig | None = None,
    artifact_root: Path = Path("artifacts/runs"),
    register_model: bool = False,
) -> dict[str, Any]:
    """Train a candidate model and record complete local lineage metadata."""
    import dataclasses
    config = run_config or RunConfig()
    split = split_config or TemporalSplitConfig()
    tracker = RunTracker(artifact_root)
    
    try:
        tracker.log("dataset_load_started", dataset_path=str(dataset_path))
        frame = load_forecasting_dataset(dataset_path)
        tracker.log("dataset_loaded", rows=len(frame), columns=len(frame.columns))
        tracker.log(
            "dataset_validated",
            date_min=str(frame["date"].min().date()),
            date_max=str(frame["date"].max().date()),
        )
        splits = chronological_split(frame, split)
        tracker.log(
            "split_created",
            split=split.as_dict(),
            rows={name: len(value) for name, value in splits.items()},
        )
        started_at = tracker.started_at
        run_id = tracker.run_id
        run_dir = tracker.run_dir
        feature_columns = FEATURE_SETS[config.feature_set]
        missing_features = set(feature_columns).difference(frame.columns)
        if missing_features:
            raise ValueError(f"Missing selected features: {sorted(missing_features)}")
        tracker.log(
            "features_selected",
            feature_set=config.feature_set,
            feature_count=len(feature_columns),
            features=feature_columns,
        )

        model = _model(config.model_type, config.random_seed)
        
        if not config.hyperparameters:
            params = model.get_params() if hasattr(model, "get_params") else {}
            config = dataclasses.replace(config, hyperparameters=params)

        tracker.log(
            "model_initialized",
            model_type=config.model_type,
            random_seed=config.random_seed,
            hyperparameters=config.hyperparameters,
        )
        
        train = splits["train"].dropna(subset=["target_next_day"])
        training_started_at = datetime.now(timezone.utc)
        tracker.log("training_started", rows=len(train))
        model.fit(train[feature_columns], train["target_next_day"])
        tracker.log(
            "training_completed",
            duration_seconds=(datetime.now(timezone.utc) - training_started_at).total_seconds(),
        )

        metrics: dict[str, dict[str, dict[str, float]]] = {}
        for split_name, split_frame in splits.items():
            tracker.log(f"{split_name}_evaluation_started", rows=len(split_frame))
            usable = split_frame.dropna(subset=["target_next_day"])
            candidate_prediction = pd.Series(
                model.predict(usable[feature_columns]), index=usable.index
            )
            predictions = usable[["Org Code", "date", "target_next_day"]].assign(
                prediction=candidate_prediction.to_numpy(),
                previous_day=previous_day_baseline(usable).to_numpy(),
                previous_weekday=previous_weekday_baseline(usable).to_numpy(),
            )
            predictions.to_parquet(run_dir / f"{split_name}_predictions.parquet", index=False)
            metrics[split_name] = {
                "candidate": _score_split(usable, candidate_prediction),
                "previous_day": _score_split(usable, predictions["previous_day"]),
                "previous_weekday": _score_split(usable, predictions["previous_weekday"]),
            }
            tracker.log(f"{split_name}_evaluation_completed", metrics=metrics[split_name])

        from models.quality_gate import passes_quality_gate
        gate_passed, gate_reason = passes_quality_gate(metrics, split_name="validation")
        tracker.log("quality_gate_evaluated", passed=gate_passed, reason=gate_reason)

        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path)
        tracker.log(
            "artifacts_saved",
            model_artifact=str(model_path),
            prediction_artifacts=len(splits),
        )
        finished_at = datetime.now(timezone.utc)
        result = {
            "run_id": run_id,
            "run_started_at": started_at.isoformat(),
            "run_finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "config": config.as_dict(),
            "split": split.as_dict(),
            "features": feature_columns,
            "dataset": dataset_manifest(frame, dataset_path),
            "code": _git_metadata(),
            "environment": _environment_metadata(),
            "rows_seen": len(train),
            "metrics": metrics,
            "quality_gate": {"passed": gate_passed, "reason": gate_reason},
            "is_registered": register_model,
            "model_artifact": str(model_path),
            "events_artifact": str(tracker.events_path),
        }
        (run_dir / "run.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        
        import mlflow
        from config import settings
        from dotenv import load_dotenv
        load_dotenv()
        
        if settings.dagshub_repo_owner and settings.dagshub_repo_name:
            import dagshub
            dagshub.init(repo_owner=settings.dagshub_repo_owner, repo_name=settings.dagshub_repo_name, mlflow=True)
        else:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
        mlflow.set_experiment(config.experiment_name)
        
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({
                "model_type": config.model_type,
                "feature_set": config.feature_set,
                "random_seed": config.random_seed,
            })
            if config.hyperparameters:
                mlflow.log_params(config.hyperparameters)
            mlflow.log_params({"split_" + k: v for k, v in split.as_dict().items()})
            
            for split_name, split_metrics in metrics.items():
                if "candidate" in split_metrics:
                    for metric_name, value in split_metrics["candidate"].items():
                        mlflow.log_metric(f"{split_name}_{metric_name}", value)
            
            mlflow.set_tags({
                "git_commit": result["code"]["git_commit"],
                "dataset_hash": result["dataset"]["dataset_hash"],
                "schema_hash": result["dataset"]["schema_hash"],
                "experiment_name": config.experiment_name,
                "environment": settings.environment,
                "quality_gate_passed": str(gate_passed),
                "quality_gate_reason": gate_reason,
            })
            
            mlflow.log_artifacts(str(run_dir))
            
            if gate_passed and register_model:
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=settings.model_name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

        tracker.finish(duration_seconds=result["duration_seconds"], metrics=metrics)
        return result
    except Exception as e:
        tracker.fail(error_message=str(e), exception_type=type(e).__name__)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PulseOps discharge forecasting"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/forecasting_dataset.parquet"),
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "hist_gradient_boosting", "extra_trees", "random_forest"],
        default="extra_trees",
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="all_context",
    )
    args = parser.parse_args()
    result = run_training(
        args.dataset,
        RunConfig(model_type=args.model_type, feature_set=args.feature_set),
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
