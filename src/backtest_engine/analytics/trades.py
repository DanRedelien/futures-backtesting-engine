"""
src/backtest_engine/analytics/trades.py

Trade-level statistical analysis.

Responsibility: Accept a list of Trade objects or dicts and compute
closed-trade KPIs.  No dependencies on portfolio history or pandas DataFrames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats


def extract_pnls(trades: List[Any]) -> List[float]:
    """
    Extracts PnL values from a heterogeneous list of trade representations.

    Handles both dict trades (e.g. from WFO) and dataclass/object trades
    (from ExecutionHandler) so the rest of the analytics layer never has to
    care about the underlying trade format.

    Args:
        trades: List of Trade objects or dicts, each exposing a 'pnl' field.

    Returns:
        Flat list of float PnL values.
    """
    pnls: List[float] = []
    for t in trades:
        if isinstance(t, dict):
            pnls.append(t.get("pnl", 0.0))
        else:
            pnls.append(getattr(t, "pnl", 0.0))
    return pnls


def calc_trade_stats(trades: List[Any]) -> Dict[str, float]:
    """
    Calculates trade-level KPIs from a list of closed trades.

    Methodology / Financial Logic:
        - Win Rate = winners / total trades.
        - Profit Factor = Gross Profit / Gross Loss.  PF > 1 indicates
          net-positive edge.  Infinity if no losing trades; 0 if no winners.
        - T-test (H0: mean PnL == 0) checks whether the return distribution
          is statistically distinguishable from random noise.

    Args:
        trades: List of Trade objects or dicts.

    Returns:
        Dict with trade-level statistics.
    """
    if not trades:
        return {
            "Total Trades":  0,
            "Win Rate":      0.0,
            "Profit Factor": 0.0,
            "Avg Trade":     0.0,
            "Avg Win":       0.0,
            "Avg Loss":      0.0,
            "T-Statistic":   0.0,
            "P-Value":       1.0,
        }

    pnls: List[float] = extract_pnls(trades)
    winners: List[float] = [p for p in pnls if p > 0]
    losers:  List[float] = [p for p in pnls if p <= 0]

    total_trades: int = len(trades)
    win_rate: float = len(winners) / total_trades

    gross_profit: float = sum(winners)
    gross_loss:   float = abs(sum(losers))
    profit_factor: float = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0.0
    )

    # One-sample T-test: H0 = mean PnL == 0
    t_stat: float = 0.0
    p_val:  float = 1.0
    if total_trades > 1 and np.std(pnls) > 0:
        t_stat, p_val = stats.ttest_1samp(pnls, 0.0)

    return {
        "Total Trades":  total_trades,
        "Win Rate":      win_rate,
        "Profit Factor": profit_factor,
        "Avg Trade":     sum(pnls) / total_trades,
        "Avg Win":       sum(winners) / len(winners) if winners else 0.0,
        "Avg Loss":      sum(losers)  / len(losers)  if losers  else 0.0,
        "T-Statistic":   t_stat,
        "P-Value":       p_val,
    }
