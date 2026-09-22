"""Helper methods for interacting with the local model registry via artifacts."""

import json
from pathlib import Path


def get_production_champion(artifact_root: Path = Path("artifacts/runs")) -> dict | None:
    """Read the latest chronologically registered run.json."""
    if not artifact_root.exists():
        return None

    runs = []
    for run_dir in artifact_root.iterdir():
        if not run_dir.is_dir():
            continue
            
        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue
            
        try:
            meta = json.loads(run_file.read_text(encoding="utf-8"))
            if meta.get("is_registered") is True:
                runs.append(meta)
        except Exception:
            continue

    runs.sort(key=lambda run: run.get("run_started_at", ""), reverse=True)
    return runs[0] if runs else None


def get_champion_validation_mae(artifact_root: Path = Path("artifacts/runs")) -> float | None:
    """Return the validation candidate MAE for the current production champion."""
    champ = get_production_champion(artifact_root)
    if not champ:
        return None
    try:
        return champ["metrics"]["validation"]["candidate"]["mae"]
    except KeyError:
        return None
