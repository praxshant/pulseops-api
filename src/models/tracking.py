"""Structured local lineage events for training runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class RunTracker:
    """Write timestamped, append-only lifecycle events for one training run."""

    def __init__(self, artifact_root: Path) -> None:
        self.run_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self.sequence = 0
        self.run_dir = artifact_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.log("run_created")

    def log(self, event: str, **details: Any) -> dict[str, Any]:
        """Append one structured event and return the serialized record."""
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        return record

    def finish(self, **details: Any) -> dict[str, Any]:
        """Record the terminal run event."""
        return self.log("run_completed", **details)

    def fail(self, error_message: str, **details: Any) -> dict[str, Any]:
        """Record a terminal failure event."""
        return self.log("run_failed", error_message=error_message, **details)
