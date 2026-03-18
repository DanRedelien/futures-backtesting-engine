from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.jobs import ScenarioJobService
from src.backtest_engine.analytics.terminal_ui.routes_charts import (
    register_chart_routes,
)
from src.backtest_engine.analytics.terminal_ui.routes_operations import (
    make_operations_context_builder,
    register_operations_routes,
)
from src.backtest_engine.analytics.terminal_ui.routes_partials import (
    register_partial_routes,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    inspect_terminal_bundle,
    load_terminal_bundle,
    load_terminal_runtime_context,
)
from src.backtest_engine.analytics.terminal_ui.table_builders import (
    build_shell_context,
)


_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TODO_PATH = _PROJECT_ROOT / "TODO.md"


def _build_static_asset_version() -> str:
    """Builds a cache-busting token from current static asset mtimes."""
    static_files = (
        _STATIC_DIR / "terminal.css",
        _STATIC_DIR / "terminal.js",
        _STATIC_DIR / "charts.js",
        _STATIC_DIR / "operations.js",
    )
    existing_files = [path for path in static_files if path.exists()]
    if not existing_files:
        return "1"
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in existing_files)
    return str(latest_mtime_ns)


def _coerce_float(value: Optional[str], fallback: float) -> float:
    """Parses a query parameter into float while preserving safe defaults."""
    if value in (None, ""):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value: Optional[str], fallback: int) -> int:
    """Parses a query parameter into int while preserving safe defaults."""
    if value in (None, ""):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _build_stress_from_query(request: Request, defaults: StressMultipliers) -> StressMultipliers:
    """Builds stress multipliers from sidebar query parameters."""
    return StressMultipliers(
        volatility=_coerce_float(request.query_params.get("stress_volatility"), defaults.volatility),
        slippage=_coerce_float(request.query_params.get("stress_slippage"), defaults.slippage),
        commission=_coerce_float(request.query_params.get("stress_commission"), defaults.commission),
    )


def _render_bundle_error(
    request: Request,
    templates: Jinja2Templates,
    *,
    title: str,
    message: str,
) -> HTMLResponse:
    """Renders a bundle-state error into the full dashboard shell."""
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "page_title": "Quant Terminal",
            "static_asset_version": _build_static_asset_version(),
            "error_title": title,
            "error_message": message,
        },
    )


def _render_fragment_error(*, title: str, message: str) -> HTMLResponse:
    """Renders a compact inline error for HTMX partial swap targets."""
    html = (
        '<div class="terminal-fragment-error">'
        f'<span class="terminal-fragment-error__label">{title}</span>'
        f'<span class="terminal-fragment-error__text"> — {message}</span>'
        "</div>"
    )
    return HTMLResponse(content=html)


def create_terminal_dashboard_app(results_dir: Optional[str] = None) -> FastAPI:
    """
    Creates the FastAPI terminal dashboard.

    Methodology:
        The factory accepts an optional results root so tests can mount the same
        app against temporary artifact bundles without mutating global state.
    """
    runtime = load_terminal_runtime_context()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app = FastAPI(title="Quant Terminal Dashboard", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    job_service = ScenarioJobService(results_dir=results_dir, config=runtime.queue_config)
    build_operations_context = make_operations_context_builder(
        job_service=job_service,
        todo_path=_TODO_PATH,
    )

    def _load_bundle_or_error(request: Request) -> tuple[Optional[Any], Optional[HTMLResponse]]:
        """Returns bundle or a full-shell error response."""
        status = inspect_terminal_bundle(results_dir=results_dir)
        bundle = load_terminal_bundle(results_dir=results_dir)
        if bundle is not None:
            return bundle, None

        if status.state == "incomplete":
            return None, _render_bundle_error(
                request,
                templates,
                title="Incomplete Artifacts",
                message=status.reason or "Result artifacts are incomplete and cannot be loaded.",
            )

        return None, _render_bundle_error(
            request,
            templates,
            title="No Artifacts Found",
            message="Run a backtest first, then reload the terminal dashboard.",
        )

    def _load_bundle_for_partial() -> tuple[Optional[Any], Optional[HTMLResponse]]:
        """Returns bundle or a compact fragment error for HTMX partial endpoints."""
        status = inspect_terminal_bundle(results_dir=results_dir)
        bundle = load_terminal_bundle(results_dir=results_dir)
        if bundle is not None:
            return bundle, None

        if status.state == "incomplete":
            return None, _render_fragment_error(
                title="Incomplete Artifacts",
                message=status.reason or "Result artifacts are incomplete.",
            )

        return None, _render_fragment_error(
            title="No Artifacts",
            message="Run a backtest first, then reload.",
        )

    @app.get("/health", response_class=JSONResponse)
    def health() -> Dict[str, str]:
        """Returns a lightweight readiness payload for local launches."""
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        """Renders the fixed terminal shell for the active artifact bundle."""
        bundle, error_response = _load_bundle_or_error(request)
        if error_response is not None:
            return error_response
        shell = build_shell_context(bundle, runtime)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "request": request,
                "page_title": "Quant Terminal",
                "static_asset_version": _build_static_asset_version(),
                "shell": shell,
                "loading_words": list(runtime.loading_words),
                "loading_word_interval_ms": runtime.loading_word_interval_ms,
                "loading_eta_per_request_seconds": runtime.loading_eta_per_request_seconds,
            },
        )

    register_partial_routes(
        app,
        templates=templates,
        runtime=runtime,
        load_bundle_for_partial=_load_bundle_for_partial,
        build_stress_from_query=_build_stress_from_query,
        coerce_int=_coerce_int,
        build_operations_context=build_operations_context,
    )
    register_operations_routes(
        app,
        runtime=runtime,
        templates=templates,
        job_service=job_service,
        results_dir=results_dir,
        load_bundle_for_partial=_load_bundle_for_partial,
        coerce_float=_coerce_float,
        build_operations_context=build_operations_context,
    )
    register_chart_routes(
        app,
        runtime=runtime,
        results_dir=results_dir,
        build_stress_from_query=_build_stress_from_query,
    )
    return app


app = create_terminal_dashboard_app()
