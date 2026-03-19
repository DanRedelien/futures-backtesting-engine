from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.backtest_engine.analytics.dashboard.core.scenario_runner import get_baseline_run_id
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.jobs import (
    FINAL_SCENARIO_JOB_STATES,
    ScenarioJobMetadata,
    ScenarioJobService,
)
from src.backtest_engine.analytics.terminal_ui.service import TerminalRuntimeContext

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


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


def _split_jobs(jobs: list[ScenarioJobMetadata]) -> tuple[list[ScenarioJobMetadata], list[ScenarioJobMetadata]]:
    """Splits jobs into active and recent-finalized collections for panel rendering."""

    active_jobs = [job for job in jobs if job.status in {"queued", "running"}]
    recent_jobs = [job for job in jobs if job.status not in {"queued", "running"}]
    return active_jobs, recent_jobs


def _available_launch_families() -> list[dict[str, str]]:
    """Returns the currently queueable scenario families for the Stress Testing tab."""

    return [
        {
            "value": "execution_shock",
            "label": "Execution Shock",
            "description": "Available now. Re-runs the strategy with changed volatility, spread, and commission assumptions.",
        }
    ]


def _future_launch_families() -> list[dict[str, str]]:
    """Returns future scenario families not yet available in the public queue surface."""

    return [
        {
            "label": "Market Replay",
            "description": "Reserved for historical replay windows after the replay-family launch path is wired.",
        },
        {
            "label": "Tail Event Rerun",
            "description": "Reserved for explicit tail-event injections after the dedicated execution path exists.",
        },
        {
            "label": "Simulation Analysis",
            "description": "Simulation families remain reserved until later roadmap phases.",
        },
    ]


def _render_jobs_panel_template(
    templates: Any,
    request: Request,
    *,
    panel_name: str,
    context: Dict[str, Any],
) -> HTMLResponse:
    """Renders either the stress-testing or operations panel with shared job context."""

    template_name = (
        "partials/panel_stress_testing.html"
        if panel_name == "stress-testing"
        else "partials/panel_operations.html"
    )
    return templates.TemplateResponse(
        request,
        template_name,
        {
            "request": request,
            **context,
        },
    )


def make_operations_context_builder(
    *,
    job_service: ScenarioJobService,
    todo_path: Path,
) -> Callable[..., Dict[str, Any]]:
    """Builds the operations-panel context factory shared by partial and POST routes."""

    def _build_operations_context(
        bundle: Any,
        *,
        launch_stress: Optional[StressMultipliers] = None,
        queue_message: str = "",
    ) -> Dict[str, Any]:
        jobs = job_service.list_jobs(limit=20)
        active_jobs, recent_jobs = _split_jobs(jobs)
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

        resolved_stress = launch_stress or StressMultipliers(
            volatility=1.0,
            slippage=1.0,
            commission=1.0,
        )
        return {
            "queue_status": job_service.queue_status(),
            "jobs": [job.to_public_dict() for job in jobs],
            "active_jobs": [job.to_public_dict() for job in active_jobs],
            "recent_jobs": [job.to_public_dict() for job in recent_jobs[:10]],
            "can_queue_scenario": can_queue_scenario,
            "queue_block_reason": queue_block_reason,
            "queue_message": queue_message,
            "baseline_run_id": get_baseline_run_id(bundle) if bundle.run_type == "portfolio" else "",
            "launch_stress": {
                "volatility": float(resolved_stress.volatility),
                "slippage": float(resolved_stress.slippage),
                "commission": float(resolved_stress.commission),
            },
            "available_launch_families": _available_launch_families(),
            "future_launch_families": _future_launch_families(),
            "install_state": {"status": "idle", "output": "", "error": "", "started_at": ""},
            "risk_vs_stress_notice": (
                "Risk stays approximation-only. Stress Testing queues a child backtest and writes saved scenario artifacts."
            ),
            "operations_notice": (
                "Launch new reruns from Stress Testing. Operations remains the diagnostic view for queue state and job history."
            ),
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

    install_state: Dict[str, Any] = {"status": "idle", "output": "", "error": "", "started_at": ""}
    install_lock = threading.Lock()

    def _build_stress_context(
        bundle: Any,
        *,
        launch_stress: Optional[StressMultipliers] = None,
        queue_message: str = "",
    ) -> Dict[str, Any]:
        """Builds the stress panel context, injecting the current install state."""
        ctx = build_operations_context(bundle, launch_stress=launch_stress, queue_message=queue_message)
        with install_lock:
            ctx["install_state"] = dict(install_state)
        return ctx

    async def _resolve_launch_stress(request: Request) -> StressMultipliers:
        """Reads launch stress controls from one form submission."""
        form = await request.form()
        return StressMultipliers(
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

    @app.get("/partials/stress-testing/status", response_class=HTMLResponse)
    async def stress_testing_status(request: Request) -> HTMLResponse:
        """Re-renders the Stress Testing panel for polling-based auto-refresh."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response
        context = _build_stress_context(bundle)
        return _render_jobs_panel_template(templates, request, panel_name="stress-testing", context=context)

    @app.post("/partials/deps/install", response_class=HTMLResponse)
    async def deps_install(request: Request) -> HTMLResponse:
        """Runs pip install in a background thread and re-renders the Stress Testing panel."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response

        with install_lock:
            if install_state["status"] == "running":
                context = _build_stress_context(bundle, queue_message="Package installation is already in progress.")
                return _render_jobs_panel_template(templates, request, panel_name="stress-testing", context=context)
            install_state["status"] = "running"
            install_state["output"] = ""
            install_state["error"] = ""
            install_state["started_at"] = datetime.now(timezone.utc).isoformat()

        def _run_install() -> None:
            requirements_path = str(_PROJECT_ROOT / "requirements.txt")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", requirements_path],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                with install_lock:
                    if result.returncode == 0:
                        install_state["status"] = "success"
                        install_state["output"] = (result.stdout or "")[-800:].strip()
                    else:
                        install_state["status"] = "failed"
                        install_state["error"] = (result.stderr or result.stdout or "")[-800:].strip()
            except Exception as exc:
                with install_lock:
                    install_state["status"] = "failed"
                    install_state["error"] = str(exc)

        threading.Thread(target=_run_install, daemon=True).start()
        context = _build_stress_context(bundle, queue_message="Installing packages. This may take a moment.")
        return _render_jobs_panel_template(templates, request, panel_name="stress-testing", context=context)

    @app.post("/partials/redis/start", response_class=HTMLResponse)
    async def redis_start(request: Request) -> HTMLResponse:
        """Starts the managed local redis-server and re-renders the Stress Testing panel."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response
        queue_message = ""
        try:
            snap = job_service.start_managed_redis()
            state = str(snap.get("state", "stopped"))
            if state == "live":
                queue_message = "Redis is live. Start the local worker to begin stress testing."
            elif state == "starting":
                queue_message = "Redis is starting. The panel will refresh automatically."
            else:
                queue_message = snap.get("last_error", "") or "Redis could not start."
        except RuntimeError as exc:
            queue_message = str(exc)
        context = _build_stress_context(bundle, queue_message=queue_message)
        return _render_jobs_panel_template(templates, request, panel_name="stress-testing", context=context)

    @app.post("/partials/redis/stop", response_class=HTMLResponse)
    async def redis_stop(request: Request) -> HTMLResponse:
        """Stops the managed local redis-server and re-renders the Stress Testing panel."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response
        queue_message = ""
        try:
            job_service.stop_managed_redis()
            queue_message = "Redis stopped."
        except RuntimeError as exc:
            queue_message = str(exc)
        context = _build_stress_context(bundle, queue_message=queue_message)
        return _render_jobs_panel_template(templates, request, panel_name="stress-testing", context=context)

    @app.post("/partials/worker/start", response_class=HTMLResponse)
    async def start_worker(request: Request) -> HTMLResponse:
        """Starts the managed local worker and re-renders the Stress Testing panel."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response

        stress = await _resolve_launch_stress(request)
        try:
            worker_snapshot = job_service.start_managed_worker()
            state = str(worker_snapshot.get("state", "stopped"))
            queue_message = (
                "Local worker is starting. The panel will refresh automatically."
                if state == "starting"
                else "Local worker is running. You can queue a stress test now."
            )
        except RuntimeError as exc:
            queue_message = str(exc)
        context = _build_stress_context(
            bundle,
            launch_stress=stress,
            queue_message=queue_message,
        )
        return _render_jobs_panel_template(
            templates,
            request,
            panel_name="stress-testing",
            context=context,
        )

    @app.post("/partials/queue-scenario", response_class=HTMLResponse)
    async def queue_scenario(request: Request) -> HTMLResponse:
        """Queues one async scenario rerun and re-renders the Stress Testing panel."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response

        form = await request.form()
        launch_family = str(form.get("launch_family", "execution_shock")).strip() or "execution_shock"
        stress = await _resolve_launch_stress(request)
        queue_status = job_service.queue_status()

        if bundle.run_type != "portfolio":
            context = _build_stress_context(
                bundle,
                launch_stress=stress,
                queue_message="Scenario reruns are only available for portfolio artifacts.",
            )
        else:
            compatibility = bundle.compatibility
            if compatibility is not None and not compatibility.is_rerunnable:
                context = _build_stress_context(
                    bundle,
                    launch_stress=stress,
                    queue_message=compatibility.reason or "This artifact is view-only and cannot be rerun.",
                )
            elif not bool(queue_status.get("ready_to_queue")):
                context = _build_stress_context(
                    bundle,
                    launch_stress=stress,
                    queue_message=str(
                        queue_status.get(
                            "readiness_message",
                            "Stress Testing is not ready to queue a rerun yet.",
                        )
                    ),
                )
            elif launch_family != "execution_shock":
                context = _build_stress_context(
                    bundle,
                    launch_stress=stress,
                    queue_message=f"`{launch_family}` is not available yet. Only execution-shock reruns can be queued.",
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
                context = _build_stress_context(
                    bundle,
                    launch_stress=stress,
                    queue_message=queue_message,
                )

        return _render_jobs_panel_template(
            templates,
            request,
            panel_name="stress-testing",
            context=context,
        )

    @app.post("/api/jobs/{job_id}/cancel", response_class=HTMLResponse)
    async def cancel_job(request: Request, job_id: str) -> HTMLResponse:
        """Cancels a queued or running job and re-renders the Stress Testing panel."""
        job_service.cancel_job(job_id)
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response
        context = _build_stress_context(bundle)
        return _render_jobs_panel_template(templates, request, panel_name="stress-testing", context=context)

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
