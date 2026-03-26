"""
exit_chart_builders.py

Pure payload builders for the exit-analysis deep-dive charts.

All functions are stateless transforms: they accept a ResultBundle and an
optional strategy filter, then return a JSON-serialisable dict that the
browser-side chart renderers consume via the /api/charts/exit-* endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest_engine.services.artifact_service import ResultBundle
from src.backtest_engine.settings import BacktestSettings
from src.backtest_engine.runtime.terminal_ui.constants import (
    TITLE_EXIT_HOLDING_TIME,
    TITLE_EXIT_MFE_MAE,
    TITLE_EXIT_PNL_DECAY,
    TITLE_EXIT_REASON,
    TITLE_EXIT_VOL_REGIME,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filter_trades_for_strategy(
    bundle: ResultBundle,
    strategy_name: str,
) -> pd.DataFrame:
    """
    Filters the bundle trades to a single strategy for exit-analysis builders.

    Passes all rows through for portfolio-global (__all__, empty) requests.
    In single-asset mode the strategy filter is always skipped.

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label to filter to, or "__all__" / "" for all.

    Returns:
        DataFrame of relevant trades (copy), or empty DataFrame when unavailable.
    """
    trades = bundle.trades.copy() if bundle.trades is not None else pd.DataFrame()
    if trades.empty:
        return trades
    if (
        bundle.run_type == "portfolio"
        and strategy_name not in {"", "__all__"}
        and "strategy" in trades.columns
    ):
        trades = trades[trades["strategy"] == strategy_name].copy()
    return trades


# ---------------------------------------------------------------------------
# Public payload builders
# ---------------------------------------------------------------------------


def build_exit_mfe_mae_payload(
    bundle: ResultBundle,
    strategy_name: str = "__all__",
) -> Dict[str, Any]:
    """
    Builds a scatter payload for MFE vs MAE trade-level visualisation.

    Methodology:
    Each trade is plotted at (MAE, MFE). Winners are green, losers red.
    The diagonal y = -x represents the break-even boundary. Points far above
    the diagonal indicate the trade endured drawdown pressure but ultimately
    recovered (favorable excursion >> adverse excursion).

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label, or "__all__" / "" for all trades.

    Returns:
        Scatter payload dict with series, axis config, diagonal, and emptyReason.
    """
    trades = _filter_trades_for_strategy(bundle, strategy_name)
    if trades.empty or "mfe" not in trades.columns or "mae" not in trades.columns:
        return {
            "title": TITLE_EXIT_MFE_MAE,
            "series": [],
            "emptyReason": "MFE/MAE columns unavailable for this bundle.",
        }
    subset = trades.dropna(subset=["mfe", "mae", "pnl"])
    if subset.empty:
        return {
            "title": TITLE_EXIT_MFE_MAE,
            "series": [],
            "emptyReason": "No complete MFE / MAE / PnL rows found.",
        }
    wins = subset[subset["pnl"] >= 0]
    losses = subset[subset["pnl"] < 0]
    series: List[Dict[str, Any]] = []
    if not wins.empty:
        series.append(
            {
                "name": "Winners",
                "color": "#22C55E",
                "points": [
                    {"x": float(x), "y": float(y), "pnl": float(p)}
                    for x, y, p in zip(wins["mae"], wins["mfe"], wins["pnl"])
                ],
            }
        )
    if not losses.empty:
        series.append(
            {
                "name": "Losers",
                "color": "#EF4444",
                "points": [
                    {"x": float(x), "y": float(y), "pnl": float(p)}
                    for x, y, p in zip(losses["mae"], losses["mfe"], losses["pnl"])
                ],
            }
        )
    # Break-even boundary: y = -x, i.e. MFE = |MAE|.
    # Anchored at (0, 0) extending to the most adverse MAE in the data so the
    # line does not stretch past the visible scatter cloud.
    min_mae = float(subset["mae"].min())
    diagonal = {
        "x1": 0.0,
        "y1": 0.0,
        "x2": min_mae,
        "y2": float(abs(min_mae)),
    }
    return {
        "title": TITLE_EXIT_MFE_MAE,
        "xAxisLabel": "MAE ($) [Adverse]",
        "yAxisLabel": "MFE ($) [Favorable]",
        "xAxisReversed": True,
        "series": series,
        "diagonal": diagonal,
        "emptyReason": "",
    }


def build_exit_pnl_decay_payload(
    bundle: ResultBundle,
    strategy_name: str = "__all__",
) -> Dict[str, Any]:
    """
    Builds a category-line payload for the PnL decay forward-horizon chart.

    Methodology:
    Plots hypothetical avg PnL if positions were exited exactly at each
    forward horizon (5m, 15m, ... up to the maximum observed hold time).
    A dashed reference line marks the actual avg PnL achieved at true exit.
    Missing horizons are plotted as None (gaps), not zero, to avoid hiding
    the absence of enrichment data.
    Horizon list is capped at the first bucket >= max_hold to match legacy
    exit_decomposition.py semantics and avoid irrelevant trailing gaps.

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label, or "__all__" / "" for all trades.

    Returns:
        Category-line payload dict.
    """
    trades = _filter_trades_for_strategy(bundle, strategy_name)
    if trades.empty:
        return {"title": TITLE_EXIT_PNL_DECAY, "categories": [], "series": [], "thresholds": []}

    all_horizons = [5, 15, 30, 60, 120, 240, 480, 720, 1440]

    # Derive the maximum observed hold time so that short-hold strategies do
    # not show a long tail of irrelevant forward horizons.
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations_min = (
            pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
        ).dt.total_seconds() / 60.0
        valid = durations_min.dropna()
        max_hold = float(valid.max()) if not valid.empty else float("inf")
        if pd.isna(max_hold) or max_hold <= 0:
            max_hold = float("inf")
    else:
        max_hold = float("inf")

    present_horizons: List[int] = []
    for h in all_horizons:
        if f"pnl_decay_{h}m" not in trades.columns:
            continue
        present_horizons.append(h)
        if h >= max_hold:
            break

    if not present_horizons:
        return {
            "title": TITLE_EXIT_PNL_DECAY,
            "categories": [],
            "series": [],
            "thresholds": [],
            "emptyReason": "No pnl_decay_* columns found. Run with exit enrichment enabled.",
        }

    def _fmt_horizon(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes}m"
        if minutes < 1440:
            return f"{minutes // 60}h"
        return f"{minutes // 1440}d"

    categories = ["Entry"] + [_fmt_horizon(h) for h in present_horizons]
    avg_values: List[Optional[float]] = [0.0]
    for h in present_horizons:
        col = f"pnl_decay_{h}m"
        avg = trades[col].mean()
        avg_values.append(float(avg) if pd.notna(avg) else None)

    actual_avg = float(trades["pnl"].mean()) if "pnl" in trades.columns else 0.0
    threshold_color = "#22C55E" if actual_avg >= 0 else "#EF4444"
    thresholds: List[Dict[str, Any]] = [
        {
            "value": actual_avg,
            "label": f"Actual Avg ${actual_avg:,.0f}",
            "legend": "Actual Avg",
            "color": threshold_color,
        }
    ]
    vertical_markers: List[Dict[str, Any]] = []

    if "exit_reason" in trades.columns and "pnl" in trades.columns:
        time_stop_mask = trades["exit_reason"].astype(str).str.contains("TIME_STOP", na=False)
        time_stop_rows = trades[time_stop_mask]
        if (
            not time_stop_rows.empty
            and "entry_time" in time_stop_rows.columns
            and "exit_time" in time_stop_rows.columns
        ):
            hold_minutes = (
                pd.to_datetime(time_stop_rows["exit_time"]) - pd.to_datetime(time_stop_rows["entry_time"])
            ).dt.total_seconds() / 60.0
            hold_minutes = hold_minutes.dropna()
            hold_minutes = hold_minutes[hold_minutes >= 0]
            if not hold_minutes.empty:
                representative_minutes: float
                if hold_minutes.nunique() == 1:
                    representative_minutes = float(hold_minutes.iloc[0])
                    marker_label = f"Time Stop Hold {representative_minutes:.0f}m"
                else:
                    representative_minutes = float(hold_minutes.mean())
                    marker_label = f"Time Stop Avg Hold {representative_minutes:.0f}m"

                closest_horizon = min(
                    present_horizons,
                    key=lambda horizon: abs(float(horizon) - representative_minutes),
                )
                vertical_markers.append(
                    {
                        "category": _fmt_horizon(closest_horizon),
                        "label": marker_label,
                        "legend": "Time Stop Hold",
                        "color": "#F59E0B",
                    }
                )
    return {
        "title": TITLE_EXIT_PNL_DECAY,
        "categories": categories,
        "series": [
            {
                "name": "Hypothetical Decay",
                "color": "#3B82F6",
                "values": avg_values,
            }
        ],
        "thresholds": thresholds,
        "verticalMarkers": vertical_markers,
        "emptyReason": "",
    }


def build_exit_holding_time_payload(
    bundle: ResultBundle,
    strategy_name: str = "__all__",
) -> Dict[str, Any]:
    """
    Builds a bar-chart payload showing avg PnL segmented by holding-time bucket.

    Methodology:
    Holding time is derived from exit_time - entry_time in minutes. Bucket
    boundaries adapt to the observed maximum hold: 4 equal buckets up to the
    max, plus an overflow bucket. Bar colours are green for positive, red for
    negative avg PnL, consistent with legacy exit-decomposition conventions.

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label, or "__all__" / "" for all trades.

    Returns:
        Bar payload dict compatible with the terminal-ui bar renderer.
    """
    trades = _filter_trades_for_strategy(bundle, strategy_name)
    if (
        trades.empty
        or "entry_time" not in trades.columns
        or "exit_time" not in trades.columns
    ):
        return {"title": TITLE_EXIT_HOLDING_TIME, "categories": [], "series": []}

    durations_min = (
        pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
    ).dt.total_seconds() / 60.0
    max_hold = float(durations_min.dropna().max()) if not durations_min.dropna().empty else 60.0
    if pd.isna(max_hold) or max_hold <= 0:
        max_hold = 60.0

    if max_hold <= 60:
        step, unit = 15, "m"
    elif max_hold <= 240:
        step, unit = 60, "m"
    elif max_hold <= 1440:
        step, unit = 360, "h"
    else:
        step, unit = 1440, "d"

    def _fmt(minutes: float) -> str:
        if unit == "m":
            return f"{int(minutes)}m"
        if unit == "h":
            return f"{int(minutes / 60)}h"
        return f"{int(minutes / 1440)}d"

    b1, b2, b3, b4 = step, step * 2, step * 3, step * 4
    bins = [0.0, float(b1), float(b2), float(b3), float(b4), float("inf")]
    labels = [
        f"<{_fmt(b1)}",
        f"{_fmt(b1)}-{_fmt(b2)}",
        f"{_fmt(b2)}-{_fmt(b3)}",
        f"{_fmt(b3)}-{_fmt(b4)}",
        f">{_fmt(b4)}",
    ]
    trades_copy = trades.copy()
    trades_copy["hold_bucket"] = pd.cut(
        durations_min, bins=bins, labels=labels, right=False
    )
    grouped = (
        trades_copy.groupby("hold_bucket", observed=False)
        .agg(avg_pnl=("pnl", "mean"), count=("pnl", "count"))
        .fillna(0)
    )
    avg_pnl_values = [float(v) for v in grouped["avg_pnl"]]
    item_colors = ["#22C55E" if v >= 0 else "#EF4444" for v in avg_pnl_values]
    return {
        "title": TITLE_EXIT_HOLDING_TIME,
        "categories": labels,
        "series": [
            {
                "name": "Avg PnL",
                "values": avg_pnl_values,
                "itemColors": item_colors,
                "yAxisIndex": 0,
            }
        ],
    }


def build_exit_vol_regime_payload(
    bundle: ResultBundle,
    strategy_name: str = "__all__",
) -> Dict[str, Any]:
    """
    Builds a bar-chart payload showing avg PnL per entry-volatility regime.

    Methodology:
    Consumes the pre-enriched entry_volatility column produced by
    enrich_trades_with_exit_analytics() (percentile-ranked rolling close std
    at entry, sampled with pad semantics). Bucket boundaries are read from
    vol_min_pct / vol_max_pct per-trade columns (strategy config) with fallback
    to BacktestSettings defaults. Bucket labels are fixed as Compression /
    Normal / Panic to preserve numerical comparability with the legacy dashboard.

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label, or "__all__" / "" for all trades.

    Returns:
        Bar payload dict, or empty payload with emptyReason when unavailable.
    """
    trades = _filter_trades_for_strategy(bundle, strategy_name)
    if trades.empty or "entry_volatility" not in trades.columns:
        return {
            "title": TITLE_EXIT_VOL_REGIME,
            "categories": [],
            "series": [],
            "emptyReason": "entry_volatility column not found. Run with exit enrichment enabled.",
        }
    subset = trades.dropna(subset=["entry_volatility", "pnl"]).copy()
    if subset.empty:
        return {
            "title": TITLE_EXIT_VOL_REGIME,
            "categories": [],
            "series": [],
            "emptyReason": "No complete entry_volatility / pnl rows found.",
        }

    settings = BacktestSettings()
    if "vol_min_pct" in trades.columns:
        min_series = pd.to_numeric(trades["vol_min_pct"], errors="coerce").dropna()
    else:
        min_series = pd.Series(dtype=float)
    if "vol_max_pct" in trades.columns:
        max_series = pd.to_numeric(trades["vol_max_pct"], errors="coerce").dropna()
    else:
        max_series = pd.Series(dtype=float)

    v_min = float(min_series.median()) if not min_series.empty else settings.vol_min_pct_default
    v_max = float(max_series.median()) if not max_series.empty else settings.vol_max_pct_default
    if v_min >= v_max:
        v_min = float(settings.vol_min_pct_default)
        v_max = float(settings.vol_max_pct_default)

    bucket_labels = ["Compression", "Normal", "Panic"]
    try:
        subset["vol_bucket"] = pd.cut(
            subset["entry_volatility"],
            bins=[0.0, v_min, v_max, 1.0],
            labels=bucket_labels,
            include_lowest=True,
        )
    except Exception:
        return {
            "title": TITLE_EXIT_VOL_REGIME,
            "categories": [],
            "series": [],
            "emptyReason": "Invalid volatility distribution (bucket construction failed).",
        }

    grouped = (
        subset.groupby("vol_bucket", observed=False)
        .agg(avg_pnl=("pnl", "mean"))
        .fillna(0)
    )
    avg_pnl_values = [float(v) for v in grouped["avg_pnl"]]
    item_colors = ["#22C55E" if v >= 0 else "#EF4444" for v in avg_pnl_values]
    return {
        "title": TITLE_EXIT_VOL_REGIME,
        "categories": bucket_labels,
        "series": [
            {
                "name": "Avg PnL",
                "values": avg_pnl_values,
                "itemColors": item_colors,
                "yAxisIndex": 0,
            }
        ],
        "emptyReason": "",
    }


def build_exit_reason_payload(
    bundle: ResultBundle,
    strategy_name: str = "__all__",
) -> Dict[str, Any]:
    """
    Builds a bar-chart payload for total PnL by exit reason.

    Methodology:
    Aggregates trades by exit_reason, summing total PnL. Sorted descending so
    the most impactful exit reason appears first. Bar colours follow the
    green/red win/loss convention used throughout the terminal-ui chart layer.

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label, or "__all__" / "" for all trades.

    Returns:
        Bar payload dict.
    """
    trades = _filter_trades_for_strategy(bundle, strategy_name)
    if trades.empty or "exit_reason" not in trades.columns:
        return {
            "title": TITLE_EXIT_REASON,
            "categories": [],
            "series": [],
            "emptyReason": "exit_reason column not available.",
        }
    grouped = (
        trades.groupby("exit_reason")
        .agg(total_pnl=("pnl", "sum"))
        .sort_values("total_pnl", ascending=False)
    )
    if grouped.empty:
        return {
            "title": TITLE_EXIT_REASON,
            "categories": [],
            "series": [],
            "emptyReason": "No exit reason data found.",
        }
    categories = [str(r) for r in grouped.index.tolist()]
    total_pnl_values = [float(v) for v in grouped["total_pnl"]]
    item_colors = ["#22C55E" if v >= 0 else "#EF4444" for v in total_pnl_values]
    return {
        "title": TITLE_EXIT_REASON,
        "categories": categories,
        "series": [
            {
                "name": "Total PnL",
                "values": total_pnl_values,
                "itemColors": item_colors,
                "yAxisIndex": 0,
            }
        ],
        "emptyReason": "",
    }


def build_exit_reason_breakdown_stats(
    bundle: ResultBundle,
    strategy_name: str = "__all__",
) -> List[Dict[str, str]]:
    """
    Builds the right-panel breakdown stats table for exit-reason analysis.

    Methodology:
    Aggregates count, win rate, avg pnl, and total pnl per exit_reason label.
    All monetary values are pre-formatted for direct Jinja template rendering.
    Sort order matches the exit_reason bar chart (total_pnl descending).

    Args:
        bundle: Active result bundle.
        strategy_name: Strategy label, or "__all__" / "" for all trades.

    Returns:
        List of row dicts with keys: Exit Reason, Count, Win Rate, Avg PnL, Total PnL.
    """
    trades = _filter_trades_for_strategy(bundle, strategy_name)
    if trades.empty or "exit_reason" not in trades.columns or "pnl" not in trades.columns:
        return []
    grouped = trades.groupby("exit_reason").agg(
        count=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        wins=("pnl", lambda x: int((x >= 0).sum())),
    )
    grouped["win_rate"] = grouped["wins"] / grouped["count"]
    grouped = grouped.sort_values("total_pnl", ascending=False).reset_index()
    rows: List[Dict[str, str]] = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "Exit Reason": str(row["exit_reason"]),
                "Count": str(int(row["count"])),
                "Win Rate": f"{float(row['win_rate']) * 100.0:.1f}%",
                "Avg PnL": f"${float(row['avg_pnl']):,.0f}",
                "Total PnL": f"${float(row['total_pnl']):,.0f}",
            }
        )
    return rows
