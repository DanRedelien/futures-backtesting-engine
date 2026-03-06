"""
src/backtest_engine/analytics/metrics.py

Pure mathematical functions for equity-curve-level performance metrics.

Responsibility: Stateless computations that accept a price/returns series
and return a single scalar.  No I/O, no state, no side-effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_total_return(equity: pd.Series) -> float:
    """
    Calculates simple total return from first to last bar.

    Args:
        equity: Portfolio value series indexed by timestamp.

    Returns:
        Decimal total return (e.g. 0.15 = 15%).
    """
    return (equity.iloc[-1] / equity.iloc[0]) - 1


def calc_years(equity: pd.Series) -> float:
    """
    Calculates the elapsed calendar years of the equity curve.

    Args:
        equity: Portfolio value series indexed by timestamp.

    Returns:
        Float years (e.g. 1.5 = 18 months).
    """
    days: int = (equity.index[-1] - equity.index[0]).days
    return days / 365.25 if days > 0 else 1.0


def calc_cagr(total_return: float, years: float) -> float:
    """
    Calculates Compound Annual Growth Rate (CAGR).

    Methodology:
        CAGR = (1 + total_return)^(1/years) - 1
        Floored at 0 to avoid domain errors on negative equity.

    Args:
        total_return: Decimal total return.
        years: Elapsed years from calc_years().

    Returns:
        Annualised compound growth rate.
    """
    return max(0.0, 1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0


def calc_bars_per_year(n_bars: int, years: float) -> float:
    """
    Derives the annualisation factor from actual data density.

    Methodology:
        Using observed bars/year instead of a fixed constant (e.g. 252)
        correctly adapts to any bar resolution: 1m, 5m, 30m, daily, etc.

    Args:
        n_bars: Total number of bars in the equity series.
        years: Elapsed years.

    Returns:
        Annualised bars scalar (e.g. ~6500 for 30-minute ES data).
    """
    return n_bars / years if years > 0 else 252.0


def calc_annualised_vol(returns: pd.Series, bars_per_year: float) -> float:
    """
    Annualises the standard deviation of bar-level returns.

    Args:
        returns: Per-bar percentage return series.
        bars_per_year: From calc_bars_per_year().

    Returns:
        Annualised volatility.
    """
    return returns.std() * np.sqrt(bars_per_year)


def calc_sharpe(cagr: float, vol: float, risk_free_rate: float) -> float:
    """
    Calculates the Sharpe Ratio.

    Methodology:
        Sharpe = (CAGR - Rf) / σ_annualised
        Returns 0 when volatility is zero (no activity edge case).

    Args:
        cagr: From calc_cagr().
        vol: Annualised volatility from calc_annualised_vol().
        risk_free_rate: Annualised risk-free rate (e.g. 0.02 for 2%).

    Returns:
        Sharpe Ratio.
    """
    return (cagr - risk_free_rate) / vol if vol > 0 else 0.0


def calc_sortino(
    cagr: float,
    returns: pd.Series,
    bars_per_year: float,
    risk_free_rate: float,
) -> float:
    """
    Calculates the Sortino Ratio using downside deviation.

    Methodology:
        Sortino = (CAGR - Rf) / σ_downside
        Only negative returns enter the denominator; positive returns do not
        penalise the ratio.  This rewards strategies that have fat right tails.

    Args:
        cagr: From calc_cagr().
        returns: Per-bar return series.
        bars_per_year: From calc_bars_per_year().
        risk_free_rate: Annualised risk-free rate.

    Returns:
        Sortino Ratio.
    """
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(bars_per_year)
    return (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0.0


def calc_max_drawdown(equity: pd.Series) -> float:
    """
    Calculates the maximum peak-to-trough drawdown.

    Args:
        equity: Portfolio value series.

    Returns:
        Max drawdown as a negative decimal (e.g. -0.15 = 15% drawdown).
    """
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def calc_calmar(cagr: float, max_drawdown: float) -> float:
    """
    Calculates the Calmar Ratio.

    Methodology:
        Calmar = CAGR / |Max Drawdown|
        High Calmar indicates the strategy generates good returns relative to
        the worst loss episode experienced.

    Args:
        cagr: From calc_cagr().
        max_drawdown: From calc_max_drawdown() (negative value).

    Returns:
        Calmar Ratio.
    """
    return cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0
