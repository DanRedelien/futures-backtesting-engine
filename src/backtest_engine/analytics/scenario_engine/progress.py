from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

from pydantic import BaseModel, ConfigDict

from .contracts import JobType


class ProgressStageId(str, Enum):
    """Enumerates the canonical worker progress stages for scenario jobs."""

    LOAD_BASELINE = "load_baseline"
    BUILD_SCENARIO_INPUTS = "build_scenario_inputs"
    PREPARE_EXECUTION_MODEL = "prepare_execution_model"
    RUN_BACKTEST_OR_SIMULATION = "run_backtest_or_simulation"
    COMPUTE_POST_METRICS = "compute_post_metrics"
    WRITE_ARTIFACTS = "write_artifacts"
    FINALIZE_METADATA = "finalize_metadata"


class ProgressStage(BaseModel):
    """
    Defines one normalized stage within a scenario worker lifecycle.

    Methodology:
        A shared stage model gives the UI and worker metadata a stable contract
        even when different job families eventually skip or collapse stages.
    """

    model_config = ConfigDict(frozen=True)

    stage_id: ProgressStageId
    stage_label: str
    stage_order: int
    stage_count: int


_STAGE_LABELS: Dict[ProgressStageId, str] = {
    ProgressStageId.LOAD_BASELINE: "Load baseline",
    ProgressStageId.BUILD_SCENARIO_INPUTS: "Build scenario inputs",
    ProgressStageId.PREPARE_EXECUTION_MODEL: "Prepare execution model",
    ProgressStageId.RUN_BACKTEST_OR_SIMULATION: "Run backtest or simulation",
    ProgressStageId.COMPUTE_POST_METRICS: "Compute post metrics",
    ProgressStageId.WRITE_ARTIFACTS: "Write artifacts",
    ProgressStageId.FINALIZE_METADATA: "Finalize metadata",
}

_DEFAULT_STAGE_FLOW: Tuple[ProgressStageId, ...] = (
    ProgressStageId.LOAD_BASELINE,
    ProgressStageId.BUILD_SCENARIO_INPUTS,
    ProgressStageId.PREPARE_EXECUTION_MODEL,
    ProgressStageId.RUN_BACKTEST_OR_SIMULATION,
    ProgressStageId.COMPUTE_POST_METRICS,
    ProgressStageId.WRITE_ARTIFACTS,
    ProgressStageId.FINALIZE_METADATA,
)


def get_progress_stages(job_type: JobType | str) -> Tuple[ProgressStage, ...]:
    """
    Returns the canonical progress stages for one scenario job family.

    Methodology:
        Plan A intentionally reuses one normalized stage flow for all current
        job types so queue metadata widens before workers branch in complexity.
    """

    _ = JobType(job_type)
    stage_count = len(_DEFAULT_STAGE_FLOW)
    return tuple(
        ProgressStage(
            stage_id=stage_id,
            stage_label=_STAGE_LABELS[stage_id],
            stage_order=index,
            stage_count=stage_count,
        )
        for index, stage_id in enumerate(_DEFAULT_STAGE_FLOW, start=1)
    )


def get_progress_stage(job_type: JobType | str, stage_id: ProgressStageId | str) -> ProgressStage:
    """Returns one stage definition by job family and stage identifier."""

    resolved_stage_id = ProgressStageId(stage_id)
    for stage in get_progress_stages(job_type):
        if stage.stage_id == resolved_stage_id:
            return stage
    raise ValueError(f"Unknown progress stage: {stage_id}")


def build_progress_metadata(job_type: JobType | str, stage_id: ProgressStageId | str) -> Dict[str, int | str]:
    """Builds JSON-safe progress fields for one stage transition."""

    stage = get_progress_stage(job_type=job_type, stage_id=stage_id)
    return {
        "progress_current": stage.stage_order,
        "progress_total": stage.stage_count,
        "progress_stage_id": stage.stage_id.value,
        "progress_stage_label": stage.stage_label,
        "progress_stage_order": stage.stage_order,
        "progress_stage_count": stage.stage_count,
    }
