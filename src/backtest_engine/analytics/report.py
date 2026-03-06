"""
src/backtest_engine/analytics/report.py

Text report formatting for the backtest terminal output.

Responsibility: Turn a metrics dict and trade list into a human-readable,
column-aligned ASCII table — identical to the legacy analytics.py output.
No computation here; only presentation logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .trades import extract_pnls


def _fmt(
    value: Any,
    is_pct:   bool = False,
    is_money: bool = False,
    is_int:   bool = False,
) -> str:
    """
    Formats a scalar value as a right-aligned display string.

    Args:
        value: Numeric value to format.
        is_pct:   Format as percentage (e.g. 12.34%).
        is_money: Format as dollars (e.g. $1,234).
        is_int:   Format as integer with thousands separator.

    Returns:
        Formatted string.
    """
    if pd.isna(value) or value is None:
        return "NaN"
    if is_int:
        return f"{int(value):,}"
    if is_pct:
        return f"{value:.2%}"
    if is_money:
        return f"${value:,.0f}"
    return f"{value:.4f}"


def _fmt_td(td: pd.Timedelta) -> str:
    """
    Formats a Timedelta into a human-readable 'Xd Yh Zm' string.

    Args:
        td: Timedelta representing a hold time.

    Returns:
        Formatted string (e.g. '3h 43m' or '1d 2h 5m').
    """
    if pd.isna(td):
        return "N/A"
    total_sec  = int(td.total_seconds())
    days, rem  = divmod(total_sec, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    return f"{hours}h {minutes}m"


def get_full_report_str(
    metrics: Dict[str, float],
    trades: Optional[List[Any]],
) -> str:
    """
    Builds the complete backtest report as a formatted ASCII string.

    Methodology:
        Returns the exact same text that is printed to stdout so that
        (a) the console log and (b) the Streamlit panel remain byte-for-byte
        identical — a single source of truth for the report layout.

    Args:
        metrics: Dict produced by PerformanceMetrics.calculate_metrics().
        trades: Raw trade list used for total PnL and hold-time statistics.

    Returns:
        Fully formatted multi-line report string.
    """
    if not metrics:
        return "No metrics to display."

    lines: List[str] = []

    # --- Hold time stats ---
    hold_times: List[pd.Timedelta] = []
    if trades:
        for t in trades:
            if hasattr(t, "entry_time") and hasattr(t, "exit_time"):
                hold_times.append(t.exit_time - t.entry_time)

    if hold_times:
        avg_hold: pd.Timedelta = sum(hold_times, pd.Timedelta(0)) / len(hold_times)
        max_hold: pd.Timedelta = max(hold_times)
        min_hold: pd.Timedelta = min(hold_times)
    else:
        avg_hold = max_hold = min_hold = pd.Timedelta(0)

    COL_W:   int = 16
    LABEL_W: int = 20
    sep:     str = "-" * (LABEL_W + COL_W + 4)

    lines.append("\n" + sep)
    lines.append(f"{'BACKTEST RESULTS':^{LABEL_W + COL_W + 4}}")
    lines.append(sep)

    # 1. Core Performance
    core_rows: List[Tuple] = [
        ("Total Return",  metrics.get("Total Return"),  dict(is_pct=True)),
        ("CAGR",          metrics.get("CAGR"),          dict(is_pct=True)),
        ("Volatility",    metrics.get("Volatility"),    dict(is_pct=True)),
        ("Sharpe Ratio",  metrics.get("Sharpe Ratio"),  {}),
        ("Sortino Ratio", metrics.get("Sortino Ratio"), {}),
        ("Max Drawdown",  metrics.get("Max Drawdown"),  dict(is_pct=True)),
        ("Calmar Ratio",  metrics.get("Calmar Ratio"),  {}),
    ]
    for label, val, args in core_rows:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")

    lines.append(sep)

    # 2. Trade Statistics
    total_pnl: float = sum(extract_pnls(trades or []))
    trade_rows: List[Tuple] = [
        ("Total Trades",  metrics.get("Total Trades", 0),  dict(is_int=True)),
        ("Win Rate",      metrics.get("Win Rate", 0),      dict(is_pct=True)),
        ("Profit Factor", metrics.get("Profit Factor", 0), {}),
        ("Avg Trade ($)", metrics.get("Avg Trade", 0),     dict(is_money=True)),
        ("Total PnL ($)", total_pnl,                       dict(is_money=True)),
        ("Avg Win ($)",   metrics.get("Avg Win", 0),       dict(is_money=True)),
        ("Avg Loss ($)",  metrics.get("Avg Loss", 0),      dict(is_money=True)),
        ("T-Statistic",   metrics.get("T-Statistic", 0),   {}),
        ("P-Value",       metrics.get("P-Value", 1),        {}),
    ]
    for label, val, args in trade_rows:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")

    lines.append(sep)

    # 3. Hold Times
    lines.append(f"{'Max Hold Time':<{LABEL_W}}{_fmt_td(max_hold):>{COL_W}}")
    lines.append(f"{'Min Hold Time':<{LABEL_W}}{_fmt_td(min_hold):>{COL_W}}")
    lines.append(f"{'Avg Hold Time':<{LABEL_W}}{_fmt_td(avg_hold):>{COL_W}}")
    lines.append(sep + "\n")

    return "\n".join(lines)
