from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.chart_builders import (
    build_pnl_distribution_payload,
)
from src.backtest_engine.analytics.terminal_ui.constants import (
    DEFAULT_BOTTOM_TAB,
    DEFAULT_CORRELATION_HORIZON,
)
from src.backtest_engine.analytics.terminal_ui.risk_builders import (
    build_risk_panel_context,
)
from src.backtest_engine.analytics.terminal_ui.service import TerminalRuntimeContext
from src.backtest_engine.analytics.terminal_ui.table_builders import (
    build_decomposition_table,
    build_exit_detail_table,
    build_exit_summary_table,
    build_shell_context,
    build_strategy_stats_table,
    build_top_ribbon_metrics,
)


def register_partial_routes(
    app: FastAPI,
    *,
    templates: Any,
    runtime: TerminalRuntimeContext,
    load_bundle_for_partial: Callable[[], tuple[Optional[Any], Optional[HTMLResponse]]],
    build_stress_from_query: Callable[[Request, StressMultipliers], StressMultipliers],
    coerce_int: Callable[[Optional[str], int], int],
    build_operations_context: Callable[..., Dict[str, Any]],
) -> None:
    """Registers HTMX partial routes for the terminal shell."""

    @app.get("/partials/top-ribbon", response_class=HTMLResponse)
    def top_ribbon(request: Request) -> HTMLResponse:
        """Renders the above-the-fold metric ribbon."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response
        metrics = build_top_ribbon_metrics(bundle, runtime)
        return templates.TemplateResponse(
            request,
            "partials/top_ribbon.html",
            {"request": request, "metrics": metrics},
        )

    @app.get("/partials/main-stage", response_class=HTMLResponse)
    def main_stage(request: Request) -> HTMLResponse:
        """Renders the main chart and terminal report shell."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response
        shell = build_shell_context(bundle, runtime)
        return templates.TemplateResponse(
            request,
            "partials/main_stage.html",
            {"request": request, "shell": shell},
        )

    @app.get("/partials/bottom-panel", response_class=HTMLResponse)
    def bottom_panel(request: Request) -> HTMLResponse:
        """Renders the active lower analysis panel."""
        bundle, error_response = load_bundle_for_partial()
        if error_response is not None:
            return error_response

        tab = request.query_params.get("tab", DEFAULT_BOTTOM_TAB)
        risk_scope = request.query_params.get(
            "risk_scope",
            "portfolio" if bundle.run_type == "portfolio" else "single",
        )
        exit_strategy = request.query_params.get("exit_strategy", "__all__")
        correlation_horizon = request.query_params.get(
            "correlation_horizon",
            DEFAULT_CORRELATION_HORIZON,
        )
        stress = build_stress_from_query(request, runtime.risk_config.stress_defaults)
        page = coerce_int(request.query_params.get("page"), 1)

        if tab == "strategy-stats":
            frame = build_strategy_stats_table(bundle)
            return templates.TemplateResponse(
                request,
                "partials/panel_strategy_stats.html",
                {
                    "request": request,
                    "columns": frame.columns.tolist(),
                    "rows": frame.to_dict("records"),
                },
            )

        if tab == "pnl-distribution":
            payload = build_pnl_distribution_payload(bundle)
            return templates.TemplateResponse(
                request,
                "partials/panel_pnl_distribution.html",
                {
                    "request": request,
                    "summary": payload.get("summary", {}),
                },
            )

        if tab == "decomposition":
            frame = build_decomposition_table(bundle, runtime)
            return templates.TemplateResponse(
                request,
                "partials/panel_decomposition.html",
                {
                    "request": request,
                    "columns": frame.columns.tolist(),
                    "rows": frame.to_dict("records"),
                    "is_available": not frame.empty,
                },
            )

        if tab == "correlations":
            return templates.TemplateResponse(
                request,
                "partials/panel_correlations.html",
                {
                    "request": request,
                    "horizon": correlation_horizon,
                    "is_portfolio": bundle.run_type == "portfolio",
                },
            )

        if tab == "risk":
            context = build_risk_panel_context(
                bundle,
                runtime,
                risk_scope=risk_scope,
                stress=stress,
            )
            return templates.TemplateResponse(
                request,
                "partials/panel_risk.html",
                {
                    "request": request,
                    "active_risk_scope": risk_scope,
                    "stress_volatility": stress.volatility,
                    "stress_slippage": stress.slippage,
                    "stress_commission": stress.commission,
                    **context,
                },
            )

        if tab == "exit-analysis":
            summary = build_exit_summary_table(bundle)
            detail_frame, total_rows = build_exit_detail_table(
                bundle,
                strategy_name=exit_strategy,
                page=page,
                page_size=runtime.trade_page_size,
            )
            total_pages = max(1, (total_rows + runtime.trade_page_size - 1) // runtime.trade_page_size)
            return templates.TemplateResponse(
                request,
                "partials/panel_exit_analysis.html",
                {
                    "request": request,
                    "summary_columns": summary.columns.tolist(),
                    "summary_rows": summary.to_dict("records"),
                    "detail_columns": detail_frame.columns.tolist(),
                    "detail_rows": detail_frame.to_dict("records"),
                    "exit_strategy": exit_strategy,
                    "page": page,
                    "total_pages": total_pages,
                    "has_prev": page > 1,
                    "has_next": page < total_pages,
                },
            )

        if tab == "operations":
            context = build_operations_context(
                bundle,
                selected_job_id=request.query_params.get("selected_job_id"),
            )
            return templates.TemplateResponse(
                request,
                "partials/panel_operations.html",
                {
                    "request": request,
                    **context,
                },
            )

        return templates.TemplateResponse(
            request,
            "partials/panel_terminal_report.html",
            {"request": request, "report_text": bundle.report or "No report available."},
        )
