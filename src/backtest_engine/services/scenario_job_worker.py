"""
Scenario job worker entrypoint and metadata mutation helpers.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from src.backtest_engine.services.artifact_service import load_result_bundle_uncached
from src.backtest_engine.analytics.scenario_engine import (
    ProgressStageId,
    ScenarioSpec,
    build_progress_metadata,
)

from .scenario_job_store import ScenarioJobStore


def _utc_now_iso() -> str:
    """Returns the current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _scenario_job_id() -> str:
    """Builds a unique identifier for one async scenario job."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"scenario-job-{timestamp}-{uuid4().hex[:8]}"


def _update_job_metadata(
    store: ScenarioJobStore,
    job_id: str,
    **updates: Any,
) -> Optional[Any]:
    """Loads, mutates, and persists one job record."""
    metadata = store.get(job_id)
    if metadata is None:
        return None
    for key, value in updates.items():
        setattr(metadata, key, value)
    return store.save(metadata)


def _update_job_stage(
    store: ScenarioJobStore,
    job_id: str,
    *,
    job_type: str,
    stage_id: ProgressStageId,
    progress_message: str,
    **updates: Any,
) -> Optional[Any]:
    """Applies one normalized stage transition to the persisted job record."""
    stage_updates = build_progress_metadata(job_type=job_type, stage_id=stage_id)
    stage_updates["progress_message"] = progress_message
    stage_updates.update(updates)
    return _update_job_metadata(store, job_id, **stage_updates)


def run_portfolio_scenario_job(
    *,
    job_id: str,
    baseline_results_dir: str,
    scenario_spec_payload: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """
    Executes one queued scenario rerun inside an RQ worker process.

    Methodology:
        Workers always load the baseline bundle afresh from persisted artifacts,
        update file-backed metadata before and after the expensive subprocess
        step, and write final scenario outputs back to Parquet artifacts. Redis
        holds queue state, while durable metadata remains in `results/jobs/`.
    """
    from src.backtest_engine.services.scenario_runner_service import run_portfolio_scenario

    store = ScenarioJobStore(results_dir=baseline_results_dir)
    scenario_spec = ScenarioSpec.model_validate(scenario_spec_payload)
    started_at = _utc_now_iso()
    _update_job_stage(
        store,
        job_id,
        job_type=scenario_spec.job_type.value,
        stage_id=ProgressStageId.LOAD_BASELINE,
        progress_message="Loading baseline artifacts.",
        status="running",
        started_at=started_at,
    )

    try:
        bundle = load_result_bundle_uncached(results_dir=baseline_results_dir)
        if bundle is None or bundle.run_type != "portfolio":
            raise ValueError("Baseline portfolio artifacts are unavailable for scenario rerun.")
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.BUILD_SCENARIO_INPUTS,
            progress_message="Validating scenario contract.",
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.PREPARE_EXECUTION_MODEL,
            progress_message="Preparing execution overrides.",
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.RUN_BACKTEST_OR_SIMULATION,
            progress_message="Running child portfolio backtest.",
        )
        scenario_root = run_portfolio_scenario(
            bundle=bundle,
            scenario_spec=scenario_spec,
            timeout_seconds=timeout_seconds,
        )

        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        duration_seconds = (completed_dt - started_dt).total_seconds()
        artifact_path = str((scenario_root / "portfolio").resolve())
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.COMPUTE_POST_METRICS,
            progress_message="Collecting scenario output metadata.",
            output_artifact_path=artifact_path,
            artifact_paths=[artifact_path],
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.WRITE_ARTIFACTS,
            progress_message="Writing final scenario manifests.",
            output_artifact_path=artifact_path,
            artifact_paths=[artifact_path],
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.FINALIZE_METADATA,
            progress_message="Scenario artifacts completed.",
            status="completed",
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            output_artifact_path=artifact_path,
            artifact_paths=[artifact_path],
        )
        return {"job_id": job_id, "output_artifact_path": artifact_path}
    except subprocess.TimeoutExpired as exc:
        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        _update_job_metadata(
            store,
            job_id,
            status="timeout",
            completed_at=completed_at,
            duration_seconds=(completed_dt - started_dt).total_seconds(),
            progress_message="Scenario rerun timed out.",
            last_error=str(exc),
        )
        raise
    except Exception as exc:
        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        _update_job_metadata(
            store,
            job_id,
            status="failed",
            completed_at=completed_at,
            duration_seconds=(completed_dt - started_dt).total_seconds(),
            progress_message="Scenario rerun failed.",
            last_error=str(exc),
        )
        raise
