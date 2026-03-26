"""
src/backtest_engine/analytics/metrics.py

Pure mathematical functions for equity-curve-level performance metrics.

Responsibility: Stateless computations that accept a price/returns series
and return a single scalar.  No I/O, no state, no side-effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats


def calc_sample_sharpe(returns: pd.Series, risk_free_rate_per_period: float = 0.0) -> float:
    """
    Calculates the sample Sharpe ratio directly from the observed return series.

    Methodology:
        Uses the same return sample that enters the DSR/PSR standard error
        formula, avoiding a mismatch with CAGR-based Sharpe definitions.

    Args:
        returns: Per-period return series.
        risk_free_rate_per_period: Risk-free rate per return observation.

    Returns:
        Non-annualised sample Sharpe ratio.
    """
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0

    excess = clean - risk_free_rate_per_period
    std = float(excess.std())
    if std <= 0.0 or pd.isna(std):
        return 0.0
    return float(excess.mean() / std)


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
        The geometric root is only valid when (1 + total_return) > 0, which is
        always true unless the portfolio was fully wiped out (total_return <= -1).
        No artificial floor: a losing strategy must produce a negative CAGR so
        that Sharpe / Sortino correctly reflect the loss direction.

    Args:
        total_return: Decimal total return.
        years: Elapsed years from calc_years().

    Returns:
        Annualised compound growth rate (negative for losing strategies).
    """
    base = 1.0 + total_return
    if years <= 0 or base <= 0.0:
        return 0.0
    return base ** (1.0 / years) - 1.0


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
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(clean.std() * np.sqrt(bars_per_year))


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
    downside = returns.replace([np.inf, -np.inf], np.nan).dropna()
    downside = downside[downside < 0]
    downside_std = float(downside.std() * np.sqrt(bars_per_year)) if not downside.empty else 0.0
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


def calc_dsr(
    returns: pd.Series,
    sharpe: float,
    trials: int = 1,
    trials_sharpe: pd.Series | list[float] | None = None
) -> float:
    """
    Calculates the Deflated Sharpe Ratio (DSR) to account for non-normal returns.

    Methodology:
        Corrects the Sharpe ratio for the effects of non-normal returns (skewness and kurtosis)
        and multiple testing (data mining/selection bias).
        Formula: DSR = Phi((SR - SR*) / sigma(SR))
        Where SR* is the expected maximum Sharpe ratio from M independent trials.

    Args:
        returns: Per-bar return series.
        sharpe: Legacy input kept for backwards compatibility. The function
            now derives the sample Sharpe from `returns` because the DSR
            standard-error formula requires the same return sample.
        trials: Number of trials or parameter combinations tested.
        trials_sharpe: List of Sharpe ratios from all trials.

    Returns:
        Deflated Sharpe Ratio (probability that the Sharpe ratio is not due to chance).
    """
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_returns) < 30:
        return 0.0

    sample_sharpe = calc_sample_sharpe(clean_returns)
    if sample_sharpe == 0.0:
        return 0.0

    n = len(clean_returns)
    skewness = float(stats.skew(clean_returns, nan_policy="omit"))
    kurt = float(stats.kurtosis(clean_returns, fisher=False, nan_policy="omit"))

    # Calculate standard error of Sharpe Ratio
    sigma_sr = np.sqrt(
        (1 - skewness * sample_sharpe + ((kurt - 1) / 4) * (sample_sharpe ** 2)) / (n - 1)
    )
    
    if sigma_sr == 0.0 or pd.isna(sigma_sr):
        return 0.0

    # Calculate expected max Sharpe (SR*)
    sr_star = 0.0
    if trials > 1 and trials_sharpe is not None and len(trials_sharpe) > 0:
        mu_sr = float(np.mean(trials_sharpe))
        sigma_trials = float(np.std(trials_sharpe))
        gamma = 0.5772156649  # Euler-Mascheroni constant
        
        term1 = (1 - gamma) * stats.norm.ppf(1 - 1 / trials)
        term2 = gamma * stats.norm.ppf(1 - 1 / (trials * np.e))
        sr_star = mu_sr + sigma_trials * (term1 + term2)

    dsr = stats.norm.cdf((sample_sharpe - sr_star) / sigma_sr)
    return float(dsr)


def calc_return_stats(returns: pd.Series) -> tuple[float, float]:
    """
    Calculates the T-Statistic and P-Value of the return series.

    Methodology:
        Uses a one-sample T-test (H0: mean return == 0) to determine
        if the strategy's mean per-period return is statistically
        distinguishable from zero.

    Args:
        returns: Per-bar return series.

    Returns:
        Tuple of (T-Statistic, P-Value).
    """
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_returns) < 2 or float(clean_returns.std()) == 0.0:
        return 0.0, 1.0

    t_stat, p_val = stats.ttest_1samp(clean_returns, 0.0)
    return float(t_stat), float(p_val)
