from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import pandas as pd

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.core.transforms import (
    compute_exit_summary,
    compute_strategy_decomp,
    compute_strategy_stats,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.constants import (
    BASE_BOTTOM_TABS,
    DEFAULT_BOTTOM_TAB,
    DEFAULT_CORRELATION_HORIZON,
    PORTFOLIO_ONLY_BOTTOM_TABS,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    _build_risk_profile_for_scope,
    _format_currency,
    _format_pct,
    _format_ratio,
    TerminalShellContext,
)

if TYPE_CHECKING:
    from src.backtest_engine.analytics.terminal_ui.service import TerminalRuntimeContext


def build_shell_context(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> TerminalShellContext:
    """Builds the portfolio-first shell metadata used by the Jinja templates."""
    is_portfolio = bundle.run_type == "portfolio"
    tabs: List[Dict[str, str]] = list(BASE_BOTTOM_TABS)
    hidden_panels: List[str] = []

    if is_portfolio:
        tabs[1:1] = list(PORTFOLIO_ONLY_BOTTOM_TABS)
    else:
        hidden_panels.extend(["decomposition", "correlations"])

    risk_scope_options: List[Dict[str, str]] = []
    if is_portfolio:
        risk_scope_options.append({"value": "portfolio", "label": "Portfolio"})
        for strategy_name in (bundle.slots or {}).values():
            risk_scope_options.append({"value": strategy_name, "label": strategy_name})
    else:
        risk_scope_options.append({"value": "single", "label": "Single Asset"})

    exit_strategy_options: List[Dict[str, str]] = []
    if is_portfolio and bundle.slots:
        exit_strategy_options.append({"value": "__all__", "label": "All Strategies"})
        for strategy_name in (bundle.slots or {}).values():
            exit_strategy_options.append({"value": strategy_name, "label": strategy_name})
    else:
        exit_strategy_options.append({"value": "__all__", "label": "Single Asset"})

    artifact_metadata = bundle.artifact_metadata
    scenario_notice = (
        "Queue async scenario reruns from Operations. Simulation Analysis stays backlog-only."
        if is_portfolio
        else "Single-asset mode reuses the same shell and hides portfolio-only panels."
    )
    report_preview = (bundle.report or "").strip()
    preview_text = report_preview if report_preview else "No report available."

    return TerminalShellContext(
        mode=bundle.run_type,
        mode_label="Portfolio" if is_portfolio else "Single Asset",
        artifact_id=artifact_metadata.artifact_id if artifact_metadata is not None else "unknown",
        artifact_created_at=(
            artifact_metadata.artifact_created_at if artifact_metadata is not None else ""
        ),
        engine_version=artifact_metadata.engine_version if artifact_metadata is not None else "unknown",
        schema_version=artifact_metadata.schema_version if artifact_metadata is not None else "unknown",
        tabs=tuple(tabs),
        default_tab=DEFAULT_BOTTOM_TAB,
        risk_scope_options=tuple(risk_scope_options),
        default_risk_scope=risk_scope_options[0]["value"],
        exit_strategy_options=tuple(exit_strategy_options),
        default_exit_strategy=exit_strategy_options[0]["value"],
        hidden_panels=tuple(hidden_panels),
        default_correlation_horizon=DEFAULT_CORRELATION_HORIZON,
        stress_defaults=runtime.risk_config.stress_defaults,
        stress_bounds={
            "min": runtime.risk_config.stress_slider_min,
            "max": runtime.risk_config.stress_slider_max,
            "step": runtime.risk_config.stress_slider_step,
        },
        report_preview=preview_text,
        scenario_notice=scenario_notice,
    )


def build_top_ribbon_metrics(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> List[Dict[str, str]]:
    """Builds the above-the-fold metric ribbon from canonical metric sources."""
    base_profile = _build_risk_profile_for_scope(
        bundle=bundle,
        runtime=runtime,
        risk_scope="portfolio" if bundle.run_type == "portfolio" else "single",
        stress=StressMultipliers(volatility=1.0, slippage=1.0, commission=1.0),
    )
    metrics = bundle.metrics or {}

    def _metric(label: str, value: str) -> Dict[str, str]:
        return {"label": label, "value": value}

    return [
        _metric("Total Return", _format_pct(float(metrics.get("Total Return", float("nan"))) * 100.0)),
        _metric("CAGR", _format_pct(float(metrics.get("CAGR", float("nan"))) * 100.0)),
        _metric("Total PnL", _format_currency(float(base_profile.summary.get("total_pnl", float("nan"))))),
        _metric("Sharpe", _format_ratio(float(base_profile.summary.get("sharpe", float("nan"))))),
        _metric("Max DD", _format_pct(float(base_profile.summary.get("max_drawdown_pct", float("nan"))))),
        _metric("VaR 95", _format_currency(float(base_profile.summary.get("var_primary", float("nan"))))),
        _metric("Win Rate", _format_pct(float(metrics.get("Win Rate", float("nan"))) * 100.0)),
        _metric("Trades", f"{int(metrics.get('Total Trades', 0)):,}"),
    ]


def build_strategy_stats_table(bundle: ResultBundle) -> pd.DataFrame:
    """Builds the canonical Strategy Stats table for the active bundle."""
    slots = bundle.slots if bundle.run_type == "portfolio" else {"single": "Single Asset"}
    return compute_strategy_stats(bundle.trades, slots)


def build_decomposition_table(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> pd.DataFrame:
    """Builds the strategy decomposition table for portfolio mode."""
    if bundle.run_type != "portfolio":
        return pd.DataFrame()
    return compute_strategy_decomp(
        trades_df=bundle.trades,
        history=bundle.history,
        slots=bundle.slots or {},
        tail_confidence=runtime.risk_config.var_confidence_primary,
    )


def build_exit_summary_table(bundle: ResultBundle) -> pd.DataFrame:
    """Builds the exit summary table for the active bundle."""
    slots = bundle.slots if bundle.run_type == "portfolio" else {"single": "Single Asset"}
    return compute_exit_summary(bundle.trades, slots)


def build_exit_detail_table(
    bundle: ResultBundle,
    strategy_name: str,
    *,
    page: int,
    page_size: int,
) -> Tuple[pd.DataFrame, int]:
    """Builds a paginated trade-detail table for exit-analysis drilldowns."""
    trades = bundle.trades.copy() if bundle.trades is not None else pd.DataFrame()
    if trades.empty:
        return pd.DataFrame(), 0

    if bundle.run_type == "portfolio" and strategy_name not in {"", "__all__"} and "strategy" in trades.columns:
        trades = trades[trades["strategy"] == strategy_name].copy()

    columns = [
        column_name
        for column_name in (
            "strategy",
            "symbol",
            "direction",
            "entry_time",
            "exit_time",
            "pnl",
            "mfe",
            "mae",
            "pnl_decay_60m",
            "exit_reason",
        )
        if column_name in trades.columns
    ]
    projected = trades[columns].copy() if columns else trades.copy()
    total_rows = len(projected)
    if total_rows == 0:
        return projected, 0

    safe_page = max(1, page)
    start = (safe_page - 1) * page_size
    end = start + page_size
    return projected.iloc[start:end].reset_index(drop=True), total_rows
