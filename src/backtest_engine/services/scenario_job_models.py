"""
Scenario job domain models and queue configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from src.backtest_engine.analytics.scenario_engine import ArtifactFamily, JobType


ScenarioJobStatus = Literal["queued", "running", "completed", "failed", "timeout"]
FINAL_SCENARIO_JOB_STATES = {"completed", "failed", "timeout"}
SUPPORTED_QUEUE_JOB_TYPES: tuple[JobType, ...] = (JobType.STRESS_RERUN,)


@dataclass(frozen=True)
class TerminalQueueConfig:
    """Execution policy for terminal-driven async scenario jobs."""

    redis_url: Optional[str]
    queue_name: str
    timeout_seconds: int
    max_retries: int
    sse_max_updates_per_second: float
    worker_start_grace_seconds: float = 2.0
    worker_stop_timeout_seconds: float = 2.0


@dataclass
class ScenarioJobMetadata:
    """Persistent metadata for one queued or completed scenario rerun."""

    job_id: str
    status: ScenarioJobStatus
    created_at: str
    baseline_results_dir: str
    baseline_run_id: str
    scenario_type: str
    scenario_params: Dict[str, Any]
    timeout_seconds: int
    max_retries: int
    failure_state: str
    queue_name: str
    job_type: str = JobType.STRESS_RERUN.value
    scenario_family: str = ""
    simulation_family: str = ""
    artifact_family: str = ArtifactFamily.SCENARIOS.value
    progress_stage_id: str = ""
    progress_stage_label: str = ""
    progress_stage_order: int = 0
    progress_stage_count: int = 0
    input_contract_version: str = ""
    seed: Optional[int] = None
    scenario_spec: Dict[str, Any] = field(default_factory=dict)
    progress_current: int = 0
    progress_total: int = 0
    progress_message: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: Optional[float] = None
    output_artifact_path: str = ""
    artifact_paths: List[str] = field(default_factory=list)
    rq_job_id: str = ""
    last_error: str = ""

    def __post_init__(self) -> None:
        """Backfills compatibility fields when loading older job metadata."""
        if not self.job_type:
            self.job_type = self.scenario_type or JobType.STRESS_RERUN.value
        if not self.scenario_type:
            self.scenario_type = self.job_type
        if not self.artifact_family:
            self.artifact_family = ArtifactFamily.SCENARIOS.value
        if not self.progress_stage_count and self.progress_total:
            self.progress_stage_count = int(self.progress_total)
        if not self.progress_stage_order and self.progress_current:
            self.progress_stage_order = int(self.progress_current)

    def to_public_dict(self) -> Dict[str, Any]:
        """Returns JSON-safe metadata for UI responses and SSE events."""
        data = asdict(self)
        total = max(0, int(self.progress_total))
        current = max(0, int(self.progress_current))
        data["progress_pct"] = round(current / total * 100.0, 1) if total > 0 else 0.0
        return data
