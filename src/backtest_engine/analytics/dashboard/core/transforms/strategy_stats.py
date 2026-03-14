from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.backtest_engine.analytics.trades import calc_trade_stats


TERMINAL_STRATEGY_STATS_COLUMNS = (
    "Strategy",
    "Total Trades",
    "Win Rate %",
    "Avg Trade ($)",
    "Max Loss ($)",
    "T-Stat",
    "P-Value",
)


def _resolve_strategy_names(slots: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Normalizes the strategy-name map for portfolio and single-asset views."""
    return dict(slots or {"single": "Single Asset"})


def _select_strategy_trades(
    trades_df: pd.DataFrame,
    strategy_name: str,
    strategy_count: int,
) -> pd.DataFrame:
    """
    Selects trade rows for one strategy under the current artifact mode.

    Methodology:
        Portfolio artifacts should carry an explicit `strategy` column, but the
        single-asset path may not. When there is only one logical strategy, the
        full trade table can safely back the Strategy Stats block.

    Args:
        trades_df: Loaded trades artifact.
        strategy_name: Display name of the strategy being rendered.
        strategy_count: Number of logical strategies in the current bundle.

    Returns:
        Strategy-specific trade slice, or an empty frame when attribution is
        impossible.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    # Single-mode contract: the full trade table always backs the one logical
    # row, regardless of whether a `strategy` column is present. A single-run
    # artifact may carry a `strategy` column whose values differ from the
    # "Single Asset" display label, so the column-filter must not run here.
    if strategy_count == 1:
        return trades_df.copy()

    if "strategy" in trades_df.columns:
        return trades_df[trades_df["strategy"] == strategy_name].copy()

    return pd.DataFrame()


def compute_strategy_stats(
    trades_df: pd.DataFrame,
    slots: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """
    Computes the canonical Strategy Stats block for the terminal UI.

    Methodology:
        This block stays intentionally narrow and sources its trade-level
        significance semantics from `analytics.trades.calc_trade_stats()`, which
        already drives the terminal report. This gives the future terminal UI one
        canonical source for `Total Trades`, `Win Rate`, `Avg Trade`, `T-Stat`,
        and `P-Value` while adding `Max Loss` as the only extra per-strategy row
        metric.

    Args:
        trades_df: Trades artifact with at least a `pnl` column.
        slots: Optional `{slot_id: strategy_name}` mapping.

    Returns:
        DataFrame with one row per strategy and the fixed terminal column set.
    """
    strategy_names = _resolve_strategy_names(slots)
    rows = []

    for strategy_name in strategy_names.values():
        strategy_trades = _select_strategy_trades(
            trades_df=trades_df,
            strategy_name=strategy_name,
            strategy_count=len(strategy_names),
        )
        trade_records = strategy_trades.to_dict("records")
        trade_stats = calc_trade_stats(trade_records)
        pnls = (
            strategy_trades["pnl"].astype(float).dropna()
            if "pnl" in strategy_trades.columns
            else pd.Series(dtype=float)
        )
        max_loss = float(pnls.min()) if not pnls.empty else 0.0

        rows.append(
            {
                "Strategy": strategy_name,
                "Total Trades": int(trade_stats["Total Trades"]),
                "Win Rate %": float(trade_stats["Win Rate"]) * 100.0,
                "Avg Trade ($)": float(trade_stats["Avg Trade"]),
                "Max Loss ($)": max_loss,
                "T-Stat": float(trade_stats["T-Statistic"]),
                "P-Value": float(trade_stats["P-Value"]),
            }
        )

    return pd.DataFrame(rows, columns=list(TERMINAL_STRATEGY_STATS_COLUMNS))


def compute_strategy_stats_map(
    trades_df: pd.DataFrame,
    slots: Optional[Dict[str, str]],
) -> Dict[str, Dict[str, float | int]]:
    """
    Converts the Strategy Stats table into a keyed metric map.

    Args:
        trades_df: Trades artifact with strategy-level rows.
        slots: Optional `{slot_id: strategy_name}` mapping.

    Returns:
        Dict keyed by strategy name for lightweight adapter layers.
    """
    stats_frame = compute_strategy_stats(trades_df=trades_df, slots=slots)
    result: Dict[str, Dict[str, float | int]] = {}

    for row in stats_frame.to_dict("records"):
        strategy_name = str(row["Strategy"])
        result[strategy_name] = {
            "trade_count": int(row["Total Trades"]),
            "win_rate_pct": float(row["Win Rate %"]),
            "avg_trade": float(row["Avg Trade ($)"]),
            "max_loss": float(row["Max Loss ($)"]),
            "tstat": float(row["T-Stat"]),
            "pvalue": float(row["P-Value"]),
        }

    return result
