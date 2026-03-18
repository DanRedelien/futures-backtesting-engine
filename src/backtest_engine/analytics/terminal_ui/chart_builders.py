from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_bar_pnl_matrix,
    compute_exposure_correlation,
    compute_pnl_dist_stats,
    compute_rolling_sharpe,
    compute_strategy_correlation,
)
from src.backtest_engine.analytics.dashboard.core.transforms.pnl import (
    build_strategy_equity_curve,
    derive_daily_pnl_from_equity,
)
from src.backtest_engine.analytics.terminal_ui.constants import (
    DECOMPOSITION_PNL_CONTRIB_COLUMN,
    DECOMPOSITION_RISK_COLUMN,
    DECOMPOSITION_SORT_COLUMN,
    LABEL_BENCHMARK,
    LABEL_CVAR_95,
    LABEL_DRAWDOWN_PCT,
    LABEL_LONG,
    LABEL_MEAN,
    LABEL_PORTFOLIO_TOTAL,
    LABEL_SHORT,
    LABEL_STRATEGY,
    LABEL_VAR_95,
    LABEL_VAR_99,
    LABEL_ZERO_THRESHOLD,
    PNL_DIST_BASE_BINS_CAP,
    PNL_DIST_BASE_BINS_FLOOR,
    PNL_DIST_DETAILED_BINS_CAP,
    PNL_DIST_DETAILED_BINS_FLOOR,
    PNL_DIST_DETAILED_MULTIPLIER,
    PNL_DIST_FD_WIDTH_FACTOR,
    PNL_DIST_SAMPLE_BIN_CAP,
    PNL_DIST_SAMPLE_BIN_FLOOR,
    TITLE_EQUITY_CURVE,
    TITLE_EXPOSURE_CORRELATION,
    TITLE_PNL_DISTRIBUTION,
    TITLE_ROLLING_SHARPE,
    TITLE_STRATEGY_CORRELATION,
    TITLE_STRATEGY_DECOMPOSITION,
    Y_AXIS_CUMULATIVE_PNL,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    _cache_payload,
    _points_from_series,
    _resolve_slot_id_for_risk_scope,
)
from src.backtest_engine.analytics.terminal_ui.table_builders import (
    build_decomposition_table,
)

if TYPE_CHECKING:
    from src.backtest_engine.analytics.terminal_ui.service import TerminalRuntimeContext


def build_equity_chart_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> Dict[str, Any]:
    """Builds the payload for the primary TradingView equity chart."""
    # Main Equity must expose the complete backtest history for portfolio review.
    full_history_points = None
    initial_capital = float(bundle.history["total_value"].iloc[0])
    series: List[Dict[str, Any]] = []

    if bundle.benchmark is not None and not bundle.benchmark.empty and "close" in bundle.benchmark.columns:
        benchmark = bundle.benchmark["close"]
        common_index = bundle.history.index.intersection(benchmark.index)
        if len(common_index) > 1:
            benchmark_pnl = (
                benchmark.loc[common_index] / float(benchmark.loc[common_index].iloc[0]) - 1.0
            ) * initial_capital
            series.append(
                {
                    "name": LABEL_BENCHMARK,
                    "color": runtime.benchmark_color,
                    "lineWidth": 2,
                    "style": 1,
                    "points": _points_from_series(benchmark_pnl, full_history_points),
                }
            )

    if bundle.run_type == "portfolio":
        for index, (slot_id, strategy_name) in enumerate((bundle.slots or {}).items()):
            column_name = f"slot_{slot_id}_pnl"
            if column_name not in bundle.history.columns:
                continue
            display_name = strategy_name.replace("Strategy", "").rstrip()
            series.append(
                {
                    "name": display_name,
                    "color": runtime.strategy_colors[index % len(runtime.strategy_colors)],
                    "lineWidth": 2,
                    "points": _points_from_series(bundle.history[column_name], full_history_points),
                }
            )
        total_pnl = bundle.history["total_value"] - initial_capital
        series.append(
            {
                "name": LABEL_PORTFOLIO_TOTAL,
                "color": runtime.portfolio_total_color,
                "lineWidth": 3,
                "points": _points_from_series(total_pnl, full_history_points),
            }
        )
    else:
        if bundle.trades is not None and not bundle.trades.empty and "exit_time" in bundle.trades.columns:
            for direction, color, label in (
                ("LONG", runtime.long_color, LABEL_LONG),
                ("SHORT", runtime.short_color, LABEL_SHORT),
            ):
                sub = bundle.trades[bundle.trades["direction"] == direction].copy()
                if sub.empty:
                    continue
                pnl_series = sub.set_index("exit_time")["pnl"].sort_index().groupby(level=0).sum()
                full_index = bundle.history.index.union(pnl_series.index)
                cumulative = (
                    pnl_series.reindex(full_index, fill_value=0.0)
                    .reindex(bundle.history.index, fill_value=0.0)
                    .cumsum()
                )
                series.append(
                    {
                        "name": label,
                        "color": color,
                        "lineWidth": 2,
                        "points": _points_from_series(cumulative, full_history_points),
                    }
                )
        total_pnl = bundle.history["total_value"] - initial_capital
        series.append(
            {
                "name": LABEL_STRATEGY,
                "color": runtime.portfolio_total_color,
                "lineWidth": 3,
                "points": _points_from_series(total_pnl, full_history_points),
            }
        )

    total_equity = bundle.history["total_value"]
    running_max = total_equity.cummax()
    drawdown_pct = (total_equity - running_max) / running_max.replace(0, float("nan")) * 100.0
    series.append(
        {
            "name": LABEL_DRAWDOWN_PCT,
            "color": runtime.drawdown_color,
            "lineWidth": 1,
            "priceScaleId": "drawdown",
            "points": _points_from_series(drawdown_pct.fillna(0.0), full_history_points),
        }
    )

    return {
        "title": TITLE_EQUITY_CURVE,
        "yAxisLabel": Y_AXIS_CUMULATIVE_PNL,
        "series": series,
    }


def build_rolling_sharpe_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> Dict[str, Any]:
    """Builds the payload for the portfolio rolling-Sharpe mini-chart."""
    if bundle.run_type != "portfolio":
        return {"title": TITLE_ROLLING_SHARPE, "series": []}

    return _cache_payload(
        runtime,
        bundle,
        metric_name="rolling_sharpe",
        parameters={
            "window_days": runtime.rolling_sharpe_window_days,
            "risk_free_rate": runtime.risk_free_rate,
        },
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: {
            "title": TITLE_ROLLING_SHARPE,
            "series": [
                {
                    "name": TITLE_ROLLING_SHARPE,
                    "color": runtime.rolling_sharpe_color,
                    "points": _points_from_series(
                        compute_rolling_sharpe(
                            history=bundle.history,
                            window_days=runtime.rolling_sharpe_window_days,
                            risk_free_rate=runtime.risk_free_rate,
                        ),
                        runtime.max_chart_points,
                    ),
                }
            ],
            "thresholds": [{"value": 0.0, "label": LABEL_ZERO_THRESHOLD}],
        },
    )


def _resolve_distribution_equity_for_scope(
    bundle: ResultBundle,
    risk_scope: str,
) -> pd.Series:
    """Resolves the equity curve used by PnL-distribution for one selected scope."""
    portfolio_equity = bundle.history["total_value"]
    if bundle.run_type != "portfolio":
        return portfolio_equity

    normalized_scope = (risk_scope or "portfolio").strip()
    if normalized_scope in {"", "portfolio", "single"}:
        return portfolio_equity

    slot_id = _resolve_slot_id_for_risk_scope(bundle.slots or {}, normalized_scope)
    if slot_id is None:
        return portfolio_equity

    slot_weights = bundle.slot_weights or {}
    slot_weight_value = slot_weights.get(slot_id)
    if slot_weight_value is None:
        slot_weight_value = slot_weights.get(int(slot_id)) if str(slot_id).isdigit() else None
    slot_weight = float(slot_weight_value) if slot_weight_value is not None else None

    strategy_equity = build_strategy_equity_curve(
        history=bundle.history,
        slot_id=str(slot_id),
        slot_weight=slot_weight,
        slot_count=len(bundle.slots or {}),
    )
    if strategy_equity.empty:
        return portfolio_equity
    return strategy_equity


def build_pnl_distribution_payload(
    bundle: ResultBundle,
    *,
    risk_scope: str = "portfolio",
) -> Dict[str, Any]:
    """Builds the ECharts histogram payload for daily PnL distribution."""
    equity = _resolve_distribution_equity_for_scope(bundle, risk_scope=risk_scope)
    daily_pnl = derive_daily_pnl_from_equity(equity)
    clean = daily_pnl.dropna().astype(float)
    if clean.empty:
        return {"title": TITLE_PNL_DISTRIBUTION, "bins": [], "markers": []}

    sample_count = int(len(clean))
    base_bins = min(PNL_DIST_BASE_BINS_CAP, max(PNL_DIST_BASE_BINS_FLOOR, int(np.sqrt(sample_count))))
    detailed_bins = min(
        PNL_DIST_DETAILED_BINS_CAP,
        max(PNL_DIST_DETAILED_BINS_FLOOR, base_bins * PNL_DIST_DETAILED_MULTIPLIER),
    )
    sturges_bins = int(np.ceil(np.log2(sample_count) + 1))

    q75, q25 = np.percentile(clean, [75, 25])
    iqr = float(q75 - q25)
    fd_bins = 0
    if iqr > 0:
        bin_width = PNL_DIST_FD_WIDTH_FACTOR * iqr / np.cbrt(sample_count)
        data_span = float(clean.max() - clean.min())
        if bin_width > 0 and data_span > 0:
            fd_bins = int(np.ceil(data_span / bin_width))

    max_bins_by_sample = max(PNL_DIST_SAMPLE_BIN_FLOOR, min(PNL_DIST_SAMPLE_BIN_CAP, sample_count))
    resolved_bins = max(sturges_bins, fd_bins, detailed_bins)
    histogram, edges = np.histogram(clean, bins=min(max_bins_by_sample, resolved_bins))
    stats = compute_pnl_dist_stats(clean)
    bins = []
    for idx, count in enumerate(histogram.tolist()):
        center = float((edges[idx] + edges[idx + 1]) / 2.0)
        bins.append({"label": f"{center:.0f}", "value": int(count), "center": center})

    summary = {
        key: float(value) if pd.notna(value) and np.isfinite(float(value)) else None
        for key, value in stats.items()
    }

    return {
        "title": TITLE_PNL_DISTRIBUTION,
        "bins": bins,
        "markers": [
            {"label": LABEL_VAR_95, "value": -float(summary["var_95"]) if summary["var_95"] is not None else None},
            {"label": LABEL_CVAR_95, "value": -float(summary["cvar_95"]) if summary["cvar_95"] is not None else None},
            {"label": LABEL_VAR_99, "value": -float(summary["var_99"]) if summary["var_99"] is not None else None},
            {"label": LABEL_MEAN, "value": float(summary["mean"]) if summary["mean"] is not None else None},
        ],
        "summary": summary,
    }


def build_decomposition_chart_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    sort_by: str = DECOMPOSITION_SORT_COLUMN,
) -> Dict[str, Any]:
    """Builds a compact bar-chart payload from the decomposition table."""
    table = build_decomposition_table(bundle=bundle, runtime=runtime, sort_by=sort_by)
    if table.empty:
        return {"title": TITLE_STRATEGY_DECOMPOSITION, "categories": [], "series": []}

    pnl_contrib_series = table[DECOMPOSITION_PNL_CONTRIB_COLUMN].fillna(0.0).astype(float)
    risk_series = table[DECOMPOSITION_RISK_COLUMN].fillna(0.0).astype(float)
    if float(risk_series.abs().max()) <= 1.0:
        risk_series = risk_series * 100.0

    return {
        "title": TITLE_STRATEGY_DECOMPOSITION,
        "yAxisFormat": "percent",
        "showAllCategoryLabels": True,
        "categories": table["Strategy"].tolist(),
        "series": [
            {
                "name": DECOMPOSITION_PNL_CONTRIB_COLUMN,
                "values": [float(value) for value in pnl_contrib_series],
                "yAxisIndex": 0,
            },
            {
                "name": DECOMPOSITION_RISK_COLUMN,
                "values": [float(value) for value in risk_series],
                "yAxisIndex": 0,
            },
        ],
    }


def _build_heatmap_payload(
    matrix: pd.DataFrame,
    title: str,
    *,
    dropped_labels: Optional[Sequence[str]] = None,
    empty_reason: str = "",
) -> Dict[str, Any]:
    """Converts a correlation matrix into an ECharts-ready heatmap payload."""
    if matrix.empty:
        return {
            "title": title,
            "xLabels": [],
            "yLabels": [],
            "values": [],
            "droppedLabels": list(dropped_labels or []),
            "emptyReason": empty_reason,
        }
    values = []
    for y_index, row_name in enumerate(matrix.index.tolist()):
        for x_index, col_name in enumerate(matrix.columns.tolist()):
            values.append([x_index, y_index, float(matrix.loc[row_name, col_name])])
    return {
        "title": title,
        "xLabels": matrix.columns.tolist(),
        "yLabels": matrix.index.tolist(),
        "values": values,
        "droppedLabels": list(dropped_labels or []),
        "emptyReason": "",
    }


def build_strategy_correlation_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    horizon: str,
) -> Dict[str, Any]:
    """Builds the strategy-correlation heatmap payload."""
    if bundle.run_type != "portfolio":
        return _build_heatmap_payload(
            pd.DataFrame(),
            TITLE_STRATEGY_CORRELATION,
            empty_reason="Correlation analysis requires portfolio mode.",
        )

    bar_pnl = build_bar_pnl_matrix(bundle.history, bundle.slots or {})
    if bar_pnl.empty or bar_pnl.shape[1] < 2:
        return _build_heatmap_payload(
            pd.DataFrame(),
            TITLE_STRATEGY_CORRELATION,
            empty_reason="Need at least 2 active strategies with incremental PnL history.",
        )

    def _compute_payload() -> Dict[str, Any]:
        matrix = compute_strategy_correlation(bar_pnl, horizon=horizon)
        if matrix.empty:
            return _build_heatmap_payload(
                pd.DataFrame(),
                f"{TITLE_STRATEGY_CORRELATION} ({horizon})",
                empty_reason=(
                    "Too few observations for this horizon. "
                    "Try 1D or run a longer backtest."
                ),
            )
        return _build_heatmap_payload(
            matrix,
            f"{TITLE_STRATEGY_CORRELATION} ({horizon})",
        )

    return _cache_payload(
        runtime,
        bundle,
        metric_name="strategy_correlation",
        parameters={"horizon": horizon},
        ttl_seconds=runtime.cache_service.policy.correlation_ttl_seconds,
        compute_fn=_compute_payload,
    )


def build_exposure_correlation_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    horizon: str,
) -> Dict[str, Any]:
    """Builds the exposure-correlation heatmap payload."""
    if bundle.exposure is None or bundle.exposure.empty:
        return _build_heatmap_payload(
            pd.DataFrame(),
            TITLE_EXPOSURE_CORRELATION,
            empty_reason="No exposure artifact found for this run.",
        )

    exposure_columns = [c for c in bundle.exposure.columns if c.endswith("_notional")]
    symbol_set = set()
    for column_name in exposure_columns:
        parts = column_name.split("_")
        if len(parts) >= 4 and parts[0] == "slot" and parts[-1] == "notional":
            symbol_set.add("_".join(parts[2:-1]))
    if len(symbol_set) < 2:
        symbols = sorted(symbol_set)
        symbol_hint = ", ".join(symbols) if symbols else "none"
        return _build_heatmap_payload(
            pd.DataFrame(),
            TITLE_EXPOSURE_CORRELATION,
            empty_reason=(
                "Exposure correlation needs at least 2 distinct instruments. "
                f"Detected {len(symbols)} ({symbol_hint})."
            ),
        )

    def _compute_payload() -> Dict[str, Any]:
        matrix, dropped = compute_exposure_correlation(bundle.exposure, horizon=horizon)
        empty_reason = ""
        if matrix.empty:
            if dropped and len(dropped) >= len(symbol_set):
                empty_reason = (
                    "All symbols were dropped due to low variance or insufficient samples "
                    "for this horizon."
                )
            else:
                empty_reason = (
                    "Too few valid exposure observations for this horizon. "
                    "Try 1D or a longer sample."
                )
        return _build_heatmap_payload(
            matrix,
            f"{TITLE_EXPOSURE_CORRELATION} ({horizon})",
            dropped_labels=dropped,
            empty_reason=empty_reason,
        )

    return _cache_payload(
        runtime,
        bundle,
        metric_name="exposure_correlation",
        parameters={"horizon": horizon},
        ttl_seconds=runtime.cache_service.policy.correlation_ttl_seconds,
        compute_fn=_compute_payload,
    )

