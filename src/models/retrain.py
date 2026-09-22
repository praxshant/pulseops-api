"""Automated retraining pipeline orchestrator."""

import json
from pathlib import Path

from pydantic import BaseModel

from models.quality_gate import is_improvement
from models.registry import get_champion_validation_mae, get_production_champion
from models.run_config import RunConfig
from models.train import run_training


class RetrainResult(BaseModel):
    candidate_run_id: str
    champion_run_id: str | None
    gate_passed: bool
    champion_replaced: bool
    improvement_pct: float | None
    reason: str


def retrain(
    dataset_path: Path,
    model_type: str = "extra_trees",
    feature_set: str = "all_context",
    register_if_better: bool = True,
) -> RetrainResult:
    """Execute retraining pipeline and potentially update production champion."""
    
    # 1. Candidate training without registering
    candidate_res = run_training(
        dataset_path,
        run_config=RunConfig(model_type=model_type, feature_set=feature_set),
        register_model=False,
    )
    
    candidate_id = candidate_res["run_id"]
    gate = candidate_res.get("quality_gate", {})
    gate_passed = gate.get("passed", False)
    
    if not gate_passed:
        result = RetrainResult(
            candidate_run_id=candidate_id,
            champion_run_id=None,
            gate_passed=False,
            champion_replaced=False,
            improvement_pct=None,
            reason=f"Candidate failed quality gate: {gate.get('reason')}"
        )
        _write_report(result)
        return result
        
    candidate_mae = candidate_res["metrics"]["validation"]["candidate"]["mae"]
    
    # 2. Get current champion
    champ_run = get_production_champion()
    
    if not champ_run:
        # No champion exists, auto-register this one
        if register_if_better:
            run_training(
                dataset_path,
                run_config=RunConfig(model_type=model_type, feature_set=feature_set),
                register_model=True,
            )
        result = RetrainResult(
            candidate_run_id=candidate_id,
            champion_run_id=None,
            gate_passed=True,
            champion_replaced=True,
            improvement_pct=None,
            reason="No existing champion. Candidate auto-promoted."
        )
        _write_report(result)
        return result

    champ_id = champ_run["run_id"]
    champ_mae = get_champion_validation_mae()
    if champ_mae is None:
        champ_mae = float("inf")
    
    # 3. Compare with champion
    better = is_improvement(candidate_mae, champ_mae)
    improvement = (champ_mae - candidate_mae) / champ_mae if champ_mae > 0 else 0.0
    
    if better and register_if_better:
        run_training(
            dataset_path,
            run_config=RunConfig(model_type=model_type, feature_set=feature_set),
            register_model=True,
        )
        reason = (
            f"Candidate (MAE {candidate_mae:.3f}) beat champion "
            f"(MAE {champ_mae:.3f}). Replaced."
        )
        replaced = True
    else:
        reason = (
            f"Candidate (MAE {candidate_mae:.3f}) did not beat "
            f"champion (MAE {champ_mae:.3f}) by >= 1%."
        )
        replaced = False
        
    result = RetrainResult(
        candidate_run_id=candidate_id,
        champion_run_id=champ_id,
        gate_passed=True,
        champion_replaced=replaced,
        improvement_pct=improvement,
        reason=reason
    )
    _write_report(result)
    return result


def _write_report(res: RetrainResult) -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrain_report.json"
    out_path.write_text(json.dumps(res.model_dump(), indent=2))
