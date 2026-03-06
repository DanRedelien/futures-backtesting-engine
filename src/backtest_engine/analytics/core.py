"""
src/backtest_engine/analytics/core.py

Central PerformanceMetrics class.

Responsibility: Orchestrates metrics.py, trades.py, and report.py to deliver
the stable public API that engine.py and optimizers call.  No raw math lives
here — this class only coordinates and caches results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .metrics import (
    calc_annualised_vol,
    calc_bars_per_year,
    calc_cagr,
    calc_calmar,
    calc_max_drawdown,
    calc_sharpe,
    calc_sortino,
    calc_total_return,
    calc_years,
)
from .trades import calc_trade_stats
from .report import get_full_report_str as _build_report


class PerformanceMetrics:
    """
    High-level performance analytics for a completed backtest.

    Methodology:
        Acts as an orchestrator: delegates equity-curve math to metrics.py,
        trade-level math to trades.py, and formatting to report.py.
        Keeps the public interface stable so callers (engine, optimizers, WFO)
        are not affected by internal refactors.
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Args:
            risk_free_rate: Annualised risk-free rate for Sharpe/Sortino.
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        portfolio_history: pd.DataFrame,
        trades: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Computes all KPIs for a completed backtest.

        Methodology:
            1. Derives annualisation factor from actual bar density rather than a
               fixed constant, so the same class works for any bar resolution.
            2. Delegates equity maths to metrics.py (pure functions).
            3. Delegates trade statistics to trades.py.

        Args:
            portfolio_history: DataFrame with 'total_value' column indexed by timestamp.
            trades: List of Trade objects or dicts (optional).

        Returns:
            Dict of named KPIs (floats).
        """
        if portfolio_history.empty:
            return {}

        equity: pd.Series = portfolio_history["total_value"]
        returns: pd.Series = equity.pct_change().fillna(0)

        total_return: float = calc_total_return(equity)
        years:        float = calc_years(equity)
        cagr:         float = calc_cagr(total_return, years)
        bpy:          float = calc_bars_per_year(len(equity), years)
        vol:          float = calc_annualised_vol(returns, bpy)
        sharpe:       float = calc_sharpe(cagr, vol, self.risk_free_rate)
        sortino:      float = calc_sortino(cagr, returns, bpy, self.risk_free_rate)
        max_dd:       float = calc_max_drawdown(equity)
        calmar:       float = calc_calmar(cagr, max_dd)

        metrics: Dict[str, float] = {
            "Total Return":  total_return,
            "CAGR":          cagr,
            "Volatility":    vol,
            "Sharpe Ratio":  sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown":  max_dd,
            "Calmar Ratio":  calmar,
        }

        if trades:
            metrics.update(calc_trade_stats(trades))

        return metrics

    def get_full_report_str(
        self,
        metrics: Dict[str, float],
        trades: Optional[List[Any]],
    ) -> str:
        """
        Returns the verbatim terminal report as a string (delegates to report.py).

        Args:
            metrics: From calculate_metrics().
            trades: Raw trade list.

        Returns:
            Formatted multi-line report string.
        """
        return _build_report(metrics, trades)

    def print_full_report(
        self,
        metrics: Dict[str, float],
        trades: Optional[List[Any]],
    ) -> None:
        """
        Prints the full backtest report to stdout.

        Args:
            metrics: From calculate_metrics().
            trades: Raw trade list.
        """
        print(self.get_full_report_str(metrics, trades))
