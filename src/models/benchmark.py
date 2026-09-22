"""Run a validation-selected algorithm bake-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.run_config import RunConfig
from models.train import run_training

MODEL_TYPES = ("ridge", "hist_gradient_boosting", "extra_trees", "random_forest")
FEATURE_SET = "all_context"


def run_benchmark(
    dataset_path: Path,
    output_path: Path = Path("artifacts/model_benchmark.json"),
) -> list[dict[str, object]]:
    """Train each candidate on the same split and save comparable results."""
    results = []
    for model_type in MODEL_TYPES:
        result = run_training(
            dataset_path,
            run_config=RunConfig(model_type=model_type, feature_set=FEATURE_SET),
            register_model=False,
        )
        results.append(
            {
                "run_id": result["run_id"],
                "model_type": model_type,
                "feature_set": FEATURE_SET,
                "validation": result["metrics"]["validation"]["candidate"],
                "test": result["metrics"]["test"]["candidate"],
                "model_artifact": result["model_artifact"],
            }
        )
    results.sort(key=lambda item: item["validation"]["mae"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PulseOps forecasting algorithms")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/forecasting_dataset.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model_benchmark.json"),
    )
    args = parser.parse_args()
    results = run_benchmark(args.dataset, args.output)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
