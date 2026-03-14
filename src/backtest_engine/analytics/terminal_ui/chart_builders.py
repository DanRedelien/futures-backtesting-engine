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
    derive_daily_pnl_from_equity,
)
from src.backtest_engine.analytics.terminal_ui.constants import (
    LABEL_BENCHMARK,
    LABEL_CVAR_95,
    LABEL_DRAWDOWN_PCT,
    LABEL_LONG,
    LABEL_PORTFOLIO_TOTAL,
    LABEL_SHORT,
    LABEL_STRATEGY,
    LABEL_VAR_95,
    LABEL_VAR_99,
    LABEL_ZERO_THRESHOLD,
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
                    "points": _points_from_series(benchmark_pnl, runtime.max_chart_points),
                }
            )

    if bundle.run_type == "portfolio":
        for index, (slot_id, strategy_name) in enumerate((bundle.slots or {}).items()):
            column_name = f"slot_{slot_id}_pnl"
            if column_name not in bundle.history.columns:
                continue
            series.append(
                {
                    "name": strategy_name,
                    "color": runtime.strategy_colors[index % len(runtime.strategy_colors)],
                    "lineWidth": 2,
                    "points": _points_from_series(bundle.history[column_name], runtime.max_chart_points),
                }
            )
        total_pnl = bundle.history["total_value"] - initial_capital
        series.append(
            {
                "name": LABEL_PORTFOLIO_TOTAL,
                "color": runtime.portfolio_total_color,
                "lineWidth": 3,
                "points": _points_from_series(total_pnl, runtime.max_chart_points),
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
                        "points": _points_from_series(cumulative, runtime.max_chart_points),
                    }
                )
        total_pnl = bundle.history["total_value"] - initial_capital
        series.append(
            {
                "name": LABEL_STRATEGY,
                "color": runtime.portfolio_total_color,
                "lineWidth": 3,
                "points": _points_from_series(total_pnl, runtime.max_chart_points),
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
            "points": _points_from_series(drawdown_pct.fillna(0.0), runtime.max_chart_points),
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


def build_pnl_distribution_payload(
    bundle: ResultBundle,
) -> Dict[str, Any]:
    """Builds the ECharts histogram payload for daily PnL distribution."""
    daily_pnl = derive_daily_pnl_from_equity(bundle.history["total_value"])
    clean = daily_pnl.dropna().astype(float)
    if clean.empty:
        return {"title": TITLE_PNL_DISTRIBUTION, "bins": [], "markers": []}

    histogram, edges = np.histogram(clean, bins=min(40, max(10, int(np.sqrt(len(clean))))))
    stats = compute_pnl_dist_stats(clean)
    bins = []
    for idx, count in enumerate(histogram.tolist()):
        center = float((edges[idx] + edges[idx + 1]) / 2.0)
        bins.append({"label": f"{center:.0f}", "value": int(count), "center": center})

    return {
        "title": TITLE_PNL_DISTRIBUTION,
        "bins": bins,
        "markers": [
            {"label": LABEL_VAR_95, "value": -float(stats["var_95"])},
            {"label": LABEL_CVAR_95, "value": -float(stats["cvar_95"]) if not pd.isna(stats["cvar_95"]) else None},
            {"label": LABEL_VAR_99, "value": -float(stats["var_99"])},
        ],
        "summary": stats,
    }


def build_decomposition_chart_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
) -> Dict[str, Any]:
    """Builds a compact bar-chart payload from the decomposition table."""
    table = build_decomposition_table(bundle=bundle, runtime=runtime)
    if table.empty:
        return {"title": TITLE_STRATEGY_DECOMPOSITION, "categories": [], "series": []}
    return {
        "title": TITLE_STRATEGY_DECOMPOSITION,
        "categories": table["Strategy"].tolist(),
        "series": [
            {
                "name": "Closed PnL ($)",
                "values": [float(value) for value in table["Closed PnL ($)"].fillna(0.0)],
                "yAxisIndex": 0,
            },
            {
                "name": "Risk Contrib (%)",
                "values": [float(value) for value in table["Risk Contrib (%)"].fillna(0.0)],
                "yAxisIndex": 1,
            },
        ],
    }


def _build_heatmap_payload(
    matrix: pd.DataFrame,
    title: str,
    *,
    dropped_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Converts a correlation matrix into an ECharts-ready heatmap payload."""
    if matrix.empty:
        return {
            "title": title,
            "xLabels": [],
            "yLabels": [],
            "values": [],
            "droppedLabels": list(dropped_labels or []),
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
    }


def build_strategy_correlation_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    horizon: str,
) -> Dict[str, Any]:
    """Builds the strategy-correlation heatmap payload."""
    if bundle.run_type != "portfolio":
        return _build_heatmap_payload(pd.DataFrame(), TITLE_STRATEGY_CORRELATION)
    return _cache_payload(
        runtime,
        bundle,
        metric_name="strategy_correlation",
        parameters={"horizon": horizon},
        ttl_seconds=runtime.cache_service.policy.correlation_ttl_seconds,
        compute_fn=lambda: _build_heatmap_payload(
            compute_strategy_correlation(
                build_bar_pnl_matrix(bundle.history, bundle.slots or {}),
                horizon=horizon,
            ),
            f"{TITLE_STRATEGY_CORRELATION} ({horizon})",
        ),
    )


def build_exposure_correlation_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    horizon: str,
) -> Dict[str, Any]:
    """Builds the exposure-correlation heatmap payload."""
    if bundle.exposure is None or bundle.exposure.empty:
        return _build_heatmap_payload(pd.DataFrame(), TITLE_EXPOSURE_CORRELATION)

    def _compute_payload() -> Dict[str, Any]:
        matrix, dropped = compute_exposure_correlation(bundle.exposure, horizon=horizon)
        return _build_heatmap_payload(
            matrix,
            f"{TITLE_EXPOSURE_CORRELATION} ({horizon})",
            dropped_labels=dropped,
        )

    return _cache_payload(
        runtime,
        bundle,
        metric_name="exposure_correlation",
        parameters={"horizon": horizon},
        ttl_seconds=runtime.cache_service.policy.correlation_ttl_seconds,
        compute_fn=_compute_payload,
    )
