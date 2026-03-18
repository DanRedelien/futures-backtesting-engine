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
    DECOMPOSITION_SORT_COLUMN,
    DEFAULT_BOTTOM_TAB,
    DEFAULT_CORRELATION_HORIZON,
    PORTFOLIO_ONLY_BOTTOM_TABS,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    _build_risk_profile_for_scope,
    _format_currency,
    _format_p_value,
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


def _format_hold_time(total_minutes: float) -> str:
    """Formats a hold-time duration (in minutes) as a compact human-readable string."""
    if pd.isna(total_minutes) or total_minutes < 0:
        return "N/A"
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h {minutes}m"


def _compute_hold_time_minutes(trades: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Derives max, min, and average hold time in minutes from a trades frame.

    Returns (max_minutes, min_minutes, avg_minutes); all NaN when unavailable.
    """
    nan = float("nan")
    if trades is None or trades.empty:
        return nan, nan, nan
    if "entry_time" not in trades.columns or "exit_time" not in trades.columns:
        return nan, nan, nan
    durations = (
        pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
    ).dt.total_seconds() / 60.0
    durations = durations.dropna()
    if durations.empty:
        return nan, nan, nan
    return float(durations.max()), float(durations.min()), float(durations.mean())


def build_top_ribbon_metrics(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> List[Dict[str, str]]:
    """
    Builds three rows of metric cards covering the full terminal report output.

    Ordering methodology:
    1) Headline outcome and drawdown come first.
    2) Closely related quality pairs remain adjacent (Sharpe + Deflated Sharpe,
       T-Stat + P-Value).
    3) Trade-quality and execution diagnostics remain in later positions.
    """
    base_profile = _build_risk_profile_for_scope(
        bundle=bundle,
        runtime=runtime,
        risk_scope="portfolio" if bundle.run_type == "portfolio" else "single",
        stress=StressMultipliers(volatility=1.0, slippage=1.0, commission=1.0),
    )
    metrics = bundle.metrics or {}
    max_hold, min_hold, avg_hold = _compute_hold_time_minutes(bundle.trades)

    def _metric(label: str, value: str) -> Dict[str, str]:
        return {"label": label, "value": value}

    nan = float("nan")
    return [
        # Row 1 — outcome and primary risk context
        _metric("Total Return", _format_pct(float(metrics.get("Total Return", nan)) * 100.0)),
        _metric("CAGR", _format_pct(float(metrics.get("CAGR", nan)) * 100.0)),
        _metric("Total PnL", _format_currency(float(base_profile.summary.get("total_pnl", nan)))),
        _metric("Max DD", _format_pct(float(metrics.get("Max Drawdown", nan)) * 100.0)),
        _metric("Volatility", _format_pct(float(metrics.get("Volatility", nan)) * 100.0)),
        _metric("VaR 95", _format_currency(float(base_profile.summary.get("var_primary", nan)))),
        # Keep this pair adjacent for quick quality validation.
        _metric("Sharpe", _format_ratio(float(metrics.get("Sharpe Ratio", nan)))),
        _metric("Defl. Sharpe", _format_ratio(float(metrics.get("Deflated Sharpe Ratio", nan)))),
        # Row 2 — risk-adjusted and trade-quality diagnostics
        _metric("Sortino", _format_ratio(float(metrics.get("Sortino Ratio", nan)))),
        _metric("Calmar", _format_ratio(float(metrics.get("Calmar Ratio", nan)))),
        _metric("Profit Factor", _format_ratio(float(metrics.get("Profit Factor", nan)))),
        _metric("Win Rate", _format_pct(float(metrics.get("Win Rate", nan)) * 100.0)),
        _metric("Avg Trade", _format_currency(float(metrics.get("Avg Trade", nan)))),
        _metric("Avg Win", _format_currency(float(metrics.get("Avg Win", nan)))),
        _metric("Avg Loss", _format_currency(float(metrics.get("Avg Loss", nan)))),
        _metric("Trades", f"{int(metrics.get('Total Trades', 0)):,}"),
        # Row 3 — statistical significance and execution stats
        # Keep this pair adjacent for significance interpretation.
        _metric("T-Stat", _format_ratio(float(metrics.get("T-Statistic", nan)))),
        _metric("P-Value", _format_p_value(float(metrics.get("P-Value", nan)))),
        _metric("Max Hold", _format_hold_time(max_hold)),
        _metric("Avg Hold", _format_hold_time(avg_hold)),
        _metric("Min Hold", _format_hold_time(min_hold)),
    ]


def build_strategy_stats_table(bundle: ResultBundle) -> pd.DataFrame:
    """Builds the canonical Strategy Stats table for the active bundle."""
    slots = bundle.slots if bundle.run_type == "portfolio" else {"single": "Single Asset"}
    return compute_strategy_stats(bundle.trades, slots)


def build_decomposition_table(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    sort_by: str = DECOMPOSITION_SORT_COLUMN,
) -> pd.DataFrame:
    """Builds the strategy decomposition table for portfolio mode."""
    if bundle.run_type != "portfolio":
        return pd.DataFrame()
    table = compute_strategy_decomp(
        trades_df=bundle.trades,
        history=bundle.history,
        slots=bundle.slots or {},
        tail_confidence=runtime.risk_config.var_confidence_primary,
    )
    if table.empty:
        return table

    resolved_sort_by = sort_by if sort_by in table.columns else (
        DECOMPOSITION_SORT_COLUMN if DECOMPOSITION_SORT_COLUMN in table.columns else table.columns[0]
    )
    numeric_sort = pd.to_numeric(table[resolved_sort_by], errors="coerce")
    if numeric_sort.notna().any():
        sortable = table.assign(__sort_value=numeric_sort)
        sorted_table = sortable.sort_values(
            by="__sort_value",
            ascending=False,
            na_position="last",
            kind="mergesort",
        ).drop(columns=["__sort_value"])
        return sorted_table.reset_index(drop=True)

    return table.sort_values(
        by=resolved_sort_by,
        ascending=True,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


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
    trade_log_numeric_columns = ("pnl", "mfe", "mae", "pnl_decay_60m")
    for column_name in trade_log_numeric_columns:
        if column_name not in projected.columns:
            continue
        numeric = pd.to_numeric(projected[column_name], errors="coerce")
        projected[column_name] = numeric.map(
            lambda value: f"{float(value):.2f}" if pd.notna(value) else "N/A"
        )
    total_rows = len(projected)
    if total_rows == 0:
        return projected, 0

    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    safe_page = max(1, min(page, total_pages))
    start = (safe_page - 1) * page_size
    end = start + page_size
    return projected.iloc[start:end].reset_index(drop=True), total_rows
