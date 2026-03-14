from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.jobs import (
    FINAL_SCENARIO_JOB_STATES,
    ScenarioJobMetadata,
    ScenarioJobService,
)
from src.backtest_engine.analytics.terminal_ui.service import TerminalRuntimeContext


def _read_simulation_backlog(todo_path: Path) -> list[str]:
    """Reads the reserved simulation backlog section from TODO.md."""
    if not todo_path.exists():
        return []

    lines = todo_path.read_text(encoding="utf-8").splitlines()
    collecting = False
    items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "## Simulation Analysis Backlog":
            collecting = True
            continue
        if collecting and stripped.startswith("## "):
            break
        if collecting and stripped.startswith("- [ ] "):
            items.append(stripped[6:].strip())
        elif collecting and stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _select_active_job(
    jobs: list[ScenarioJobMetadata],
    selected_job_id: Optional[str],
) -> Optional[ScenarioJobMetadata]:
    """Chooses the active job card shown in the operations panel."""
    if selected_job_id:
        for job in jobs:
            if job.job_id == selected_job_id:
                return job
    for job in jobs:
        if job.status in {"queued", "running"}:
            return job
    return jobs[0] if jobs else None


def make_operations_context_builder(
    *,
    job_service: ScenarioJobService,
    todo_path: Path,
) -> Callable[..., Dict[str, Any]]:
    """Builds the operations-panel context factory shared by partial and POST routes."""

    def _build_operations_context(
        bundle: Any,
        *,
        selected_job_id: Optional[str] = None,
        queue_message: str = "",
    ) -> Dict[str, Any]:
        jobs = job_service.list_jobs(limit=20)
        active_job = _select_active_job(jobs, selected_job_id)
        compatibility = getattr(bundle, "compatibility", None)
        can_queue_scenario = (
            bundle.run_type == "portfolio"
            and (compatibility is None or compatibility.is_rerunnable)
        )
        if bundle.run_type != "portfolio":
            queue_block_reason = "Async scenario reruns are only available for portfolio artifacts."
        elif compatibility is not None and not compatibility.is_rerunnable:
            queue_block_reason = compatibility.reason or "This artifact is view-only and cannot be rerun."
        else:
            queue_block_reason = ""

        return {
            "queue_status": job_service.queue_status(),
            "jobs": [job.to_public_dict() for job in jobs],
            "active_job": active_job.to_public_dict() if active_job is not None else None,
            "selected_job_id": active_job.job_id if active_job is not None else "",
            "can_queue_scenario": can_queue_scenario,
            "queue_block_reason": queue_block_reason,
            "queue_message": queue_message,
            "simulation_backlog": _read_simulation_backlog(todo_path),
        }

    return _build_operations_context


def register_operations_routes(
    app: FastAPI,
    *,
    runtime: TerminalRuntimeContext,
    templates: Any,
    job_service: ScenarioJobService,
    results_dir: Optional[str],
    load_bundle_for_partial: Callable[[], tuple[Optional[Any], Optional[HTMLResponse]]],
    coerce_float: Callable[[Optional[str], float], float],
    build_operations_context: Callable[..., Dict[str, Any]],
) -> None:
    """Registers job queue, SSE, and operations form routes."""

    @app.post("/partials/queue-scenario", response_class=HTMLResponse)
    async def queue_scenario(request: Request) -> HTMLResponse:
        """Queues one async scenario rerun and re-renders operations."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response

        form = await request.form()
        stress = StressMultipliers(
            volatility=coerce_float(
                str(form.get("stress_volatility", "")),
                runtime.risk_config.stress_defaults.volatility,
            ),
            slippage=coerce_float(
                str(form.get("stress_slippage", "")),
                runtime.risk_config.stress_defaults.slippage,
            ),
            commission=coerce_float(
                str(form.get("stress_commission", "")),
                runtime.risk_config.stress_defaults.commission,
            ),
        )

        if bundle.run_type != "portfolio":
            context = build_operations_context(
                bundle,
                queue_message="Scenario reruns are only available for portfolio artifacts.",
            )
        else:
            compatibility = bundle.compatibility
            if compatibility is not None and not compatibility.is_rerunnable:
                context = build_operations_context(
                    bundle,
                    queue_message=compatibility.reason or "This artifact is view-only and cannot be rerun.",
                )
            else:
                job = job_service.enqueue_portfolio_scenario(
                    bundle=bundle,
                    stress=stress,
                    baseline_results_dir=results_dir,
                )
                queue_message = (
                    "Scenario job queued."
                    if job.status == "queued"
                    else "Scenario job could not be queued because Redis or RQ is unavailable."
                )
                context = build_operations_context(
                    bundle,
                    selected_job_id=job.job_id,
                    queue_message=queue_message,
                )

        return templates.TemplateResponse(
            request,
            "partials/panel_operations.html",
            {
                "request": request,
                **context,
            },
        )

    @app.get("/api/jobs", response_class=JSONResponse)
    def jobs_index() -> JSONResponse:
        """Returns queue policy and recent job metadata for debugging."""
        return JSONResponse(
            {
                "queue": job_service.queue_status(),
                "jobs": [job.to_public_dict() for job in job_service.list_jobs(limit=20)],
            }
        )

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        """Streams throttled SSE updates for one async scenario job."""

        async def _event_stream() -> Any:
            last_payload = ""
            max_rate = max(0.1, float(runtime.queue_config.sse_max_updates_per_second))
            sleep_seconds = max(0.5, 1.0 / max_rate)
            while True:
                job = job_service.get_job(job_id)
                if job is None:
                    payload = json.dumps({"job_id": job_id, "status": "failed", "last_error": "Job not found."})
                    yield f"event: status\ndata: {payload}\n\n"
                    break

                payload = json.dumps(job.to_public_dict())
                if payload != last_payload:
                    yield f"event: status\ndata: {payload}\n\n"
                    last_payload = payload

                if job.status in FINAL_SCENARIO_JOB_STATES:
                    break
                await asyncio.sleep(sleep_seconds)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
