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
    decimals: Optional[int] = None,
) -> str:
    """
    Formats a scalar value as a right-aligned display string.

    Args:
        value: Numeric value to format.
        is_pct:   Format as percentage (e.g. 12.34%).
        is_money: Format as dollars (e.g. $1,234).
        is_int:   Format as integer with thousands separator.
        decimals: Number of decimal digits for floats.

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
        if value < 0:
            return f"-${abs(value):,.0f}"
        return f"${value:,.0f}"
    
    if decimals is not None:
        return f"{value:.{decimals}f}"
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
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    
    if len(parts) == 0 and minutes == 0:
        return "0h"
    if minutes > 0 or len(parts) == 0:
        parts.append(f"{minutes}m")
    
    return " ".join(parts)


def _fmt_p_value(value: Any) -> str:
    """Formats p-values with enough precision for significance interpretation."""
    if pd.isna(value) or value is None:
        return "NaN"
    p_value = float(value)
    if p_value < 0.0:
        return "NaN"
    if p_value < 0.0001:
        return "<0.0001"
    if p_value < 0.01:
        return f"{p_value:.4f}"
    return f"{p_value:.3f}"


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

    TOTAL_W: int = 48
    LABEL_W: int = 28
    COL_W: int = TOTAL_W - LABEL_W

    eq_sep:  str = "=" * TOTAL_W
    sep:     str = "-" * TOTAL_W

    lines.append("\n" + eq_sep)
    lines.append(f"{'BACKTEST RESULTS':^{TOTAL_W}}")
    lines.append(eq_sep)
    lines.append("")

    total_pnl: float = sum(extract_pnls(trades or []))

    # 1. PERFORMANCE
    lines.append("PERFORMANCE")
    lines.append(sep)
    perf_rows: List[Tuple] = [
        ("Total Return",  metrics.get("Total Return"),  dict(is_pct=True)),
        ("CAGR",          metrics.get("CAGR"),          dict(is_pct=True)),
        ("Total PnL ($)", total_pnl,                    dict(is_money=True)),
    ]
    for label, val, args in perf_rows:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")
    lines.append("")

    # 2. RISK
    lines.append("RISK")
    lines.append(sep)
    risk_rows: List[Tuple] = [
        ("Volatility",    metrics.get("Volatility"),    dict(is_pct=True)),
        ("Max Drawdown",  metrics.get("Max Drawdown"),  dict(is_pct=True)),
    ]
    for label, val, args in risk_rows:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")
    lines.append("")

    # 3. RISK-ADJUSTED METRICS
    lines.append("RISK-ADJUSTED METRICS")
    lines.append(sep)
    adj_rows: List[Tuple] = [
        ("Sharpe Ratio",         metrics.get("Sharpe Ratio"),          dict(decimals=2)),
        ("Deflated Sharpe Ratio", metrics.get("Deflated Sharpe Ratio"), dict(decimals=2)),
        ("Sortino Ratio",        metrics.get("Sortino Ratio"),         dict(decimals=2)),
        ("Calmar Ratio",         metrics.get("Calmar Ratio"),          dict(decimals=2)),
    ]
    for label, val, args in adj_rows:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")
    lines.append("")

    # 4. TRADE STATISTICS
    lines.append("TRADE STATISTICS")
    lines.append(sep)
    trade_rows_1: List[Tuple] = [
        ("Total Trades",  metrics.get("Total Trades", 0),  dict(is_int=True)),
        ("Win Rate",      metrics.get("Win Rate", 0),      dict(is_pct=True)),
        ("Profit Factor", metrics.get("Profit Factor", 0), dict(decimals=2)),
    ]
    for label, val, args in trade_rows_1:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")
    lines.append("")
    
    trade_rows_2: List[Tuple] = [
        ("Avg Trade ($)", metrics.get("Avg Trade", 0),     dict(is_money=True)),
        ("Avg Win ($)",   metrics.get("Avg Win", 0),       dict(is_money=True)),
        ("Avg Loss ($)",  metrics.get("Avg Loss", 0),      dict(is_money=True)),
    ]
    for label, val, args in trade_rows_2:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")
    lines.append("")

    # 5. STATISTICAL SIGNIFICANCE
    lines.append("STATISTICAL SIGNIFICANCE")
    lines.append(sep)
    stat_rows: List[Tuple[str, Any, Dict[str, Any]]] = [
        ("T-Statistic", metrics.get("T-Statistic", 0), dict(decimals=2)),
    ]
    for label, val, args in stat_rows:
        lines.append(f"{label:<{LABEL_W}}{_fmt(val, **args):>{COL_W}}")
    lines.append(
        f"{'P-Value':<{LABEL_W}}{_fmt_p_value(metrics.get('P-Value', 1)):>{COL_W}}"
    )
    lines.append("")

    # 6. EXECUTION STATS
    lines.append("EXECUTION STATS")
    lines.append(sep)
    lines.append(f"{'Max Hold Time':<{LABEL_W}}{_fmt_td(max_hold):>{COL_W}}")
    lines.append(f"{'Min Hold Time':<{LABEL_W}}{_fmt_td(min_hold):>{COL_W}}")
    lines.append(f"{'Avg Hold Time':<{LABEL_W}}{_fmt_td(avg_hold):>{COL_W}}")

    return "\n".join(lines)
