"""
File-backed storage for scenario job metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.backtest_engine.services.paths import get_results_dir

from .scenario_job_models import ScenarioJobMetadata


class ScenarioJobStore:
    """File-backed metadata store for queued and completed scenario jobs."""

    def __init__(self, results_dir: Optional[str] = None) -> None:
        self.results_root = Path(results_dir) if results_dir is not None else get_results_dir()
        self.jobs_dir = self.results_root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        """Returns the metadata file path for one job identifier."""
        return self.jobs_dir / f"{job_id}.json"

    def save(self, metadata: ScenarioJobMetadata) -> ScenarioJobMetadata:
        """Persists one job metadata record."""
        path = self._job_path(metadata.job_id)
        path.write_text(json.dumps(metadata.to_public_dict(), indent=2), encoding="utf-8")
        return metadata

    def get(self, job_id: str) -> Optional[ScenarioJobMetadata]:
        """Loads one job metadata record by identifier."""
        path = self._job_path(job_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        raw.pop("progress_pct", None)
        return ScenarioJobMetadata(**raw)

    def list(self, limit: int = 20) -> List[ScenarioJobMetadata]:
        """Lists recent jobs newest-first."""
        records: List[ScenarioJobMetadata] = []
        for path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            job = self.get(path.stem)
            if job is not None:
                records.append(job)
            if len(records) >= limit:
                break
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records
