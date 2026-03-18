"""
src/backtest_engine/spread_model.py

Deterministic spread model for backtest execution cost simulation:

    static               — constant spread_ticks per fill, independent of market state.
    adaptive_volatility  — spread widens when realized volatility rises above a rolling
                           baseline and narrows back when it falls, using a linear step
                           function derived from recent close-to-close log returns.

Design contract:
    - No state is held between calls.  The function is pure given its inputs.
    - All parameters come from BacktestSettings to satisfy the no-magic-numbers rule.
    - Inputs are sliced to bar[t] or earlier so the model never uses future data.
    - When history is insufficient, both modes fall back to spread_ticks without raising.
"""

from __future__ import annotations

from typing import Optional

import math

import numpy as np
import pandas as pd


# ── Public API ────────────────────────────────────────────────────────────────


def compute_spread_ticks(
    mode: str,
    base_ticks: int,
    closes: Optional[pd.Series],
    vol_step_pct: float,
    step_multiplier: float,
    vol_lookback: int = 20,
    vol_baseline_lookback: int = 100,
) -> int:
    """
    Returns the deterministic number of spread ticks to apply at fill time.

    Methodology:
        static mode:
            Returns base_ticks unchanged.  Used when transaction costs should
            be fixed and independent of regime.

        adaptive_volatility mode:
            1. Estimates current realized volatility from the most recent
               vol_lookback close-to-close log returns.
            2. Estimates a rolling baseline volatility using the most recent
               vol_baseline_lookback returns from the same series.
            3. Computes the signed step count: how many vol_step_pct-wide bands
               the current vol lies above (positive) or below (negative) the
               baseline.
            4. Returns base_ticks * step_multiplier^steps, rounded to the
               nearest integer and floored at 0.

        Both modes fall back to base_ticks when the close series is too short
        to produce reliable vol estimates.

    Args:
        mode: Spread mode — 'static' or 'adaptive_volatility'.
        base_ticks: Fixed or base tick count from settings.spread_ticks.
        closes: Close price series up to and including the current bar.
                Required for adaptive_volatility mode; ignored for static.
        vol_step_pct: Fractional volatility band width per step, e.g. 0.10 = 10 %.
                      Each full band above baseline widens spread by one step;
                      each full band below narrows it by one step.
        step_multiplier: Multiplier applied per step, e.g. 1.5 means one step
                         above baseline triples the base spread (1.5^1 = 1.5x),
                         two steps gives 1.5^2 = 2.25x, etc.  Values below 1.0
                         narrow the spread on each step.
        vol_lookback: Bars used to estimate current realized volatility.
                      Corresponds to a short estimation window (e.g. 20 bars).
        vol_baseline_lookback: Bars used to estimate the rolling baseline vol.
                               Corresponds to a longer reference window (e.g. 100 bars).

    Returns:
        Adjusted integer tick count (>= 0).  A value of 0 means zero spread.
    """
    if mode == "static":
        return max(0, base_ticks)

    if mode == "adaptive_volatility":
        if closes is None or len(closes) < vol_lookback + 1:
            return max(0, base_ticks)

        current_vol = _realized_vol(closes, vol_lookback)
        baseline_vol = _realized_vol(closes, min(vol_baseline_lookback, len(closes) - 1))

        if baseline_vol <= 0.0 or current_vol <= 0.0 or vol_step_pct <= 0.0:
            return max(0, base_ticks)

        # Signed step count: positive = vol above baseline (widen), negative = below (narrow).
        # Truncate toward zero so only full bands count; no partial-band compounding.
        vol_gap_pct = (current_vol - baseline_vol) / baseline_vol
        steps = math.trunc(vol_gap_pct / vol_step_pct)

        if steps == 0:
            return max(0, base_ticks)

        adjusted = base_ticks * (step_multiplier ** steps)
        return max(0, round(adjusted))

    raise ValueError(
        f"Unknown spread_mode: {mode!r}. "
        "Valid values are 'static' and 'adaptive_volatility'."
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _realized_vol(closes: pd.Series, lookback: int) -> float:
    """
    Computes non-annualized realized volatility from close-to-close log returns.

    Methodology:
        Takes the ddof=1 standard deviation of log(close[t] / close[t-1]) over
        the last `lookback` bars.  Non-annualized because the spread model uses
        relative volatility ratios, not annualized dollar amounts.  Returns 0.0
        when the series is too short or variance is exactly zero.

    Args:
        closes: Close price series.  Must have at least lookback + 1 entries.
        lookback: Number of return observations to include.

    Returns:
        Bar-frequency realized volatility (e.g. 0.005 per 30-min bar).
    """
    window = closes.iloc[-(lookback + 1):]
    log_rets = np.log(window / window.shift(1)).dropna()

    if len(log_rets) < 2:
        return 0.0

    return float(np.std(log_rets, ddof=1))
