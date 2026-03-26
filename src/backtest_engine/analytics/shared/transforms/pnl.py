from __future__ import annotations
from typing import Dict, Optional
import scipy.stats as stats
import pandas as pd

_HORIZON_RULE: Dict[str, str] = {
    '1d': '1D',
    '1w': '1W',
    '1m': '1ME',
}

def build_bar_pnl_matrix(
    history: pd.DataFrame,
    slots: Dict[str, str],
) -> pd.DataFrame:
    """
    Extracts per-slot *incremental* bar PnL as a wide DataFrame.

    Methodology:
        Pulls each `slot_N_pnl` column from `history` and computes the
        first-difference so the result contains bar-level PnL, NOT cumulative
        equity.  Correlation must be computed on incremental returns — using
        cumulative equity would inflate correlations toward 1.0 regardless of
        true strategy independence.

    Args:
        history: Portfolio history DataFrame with `slot_N_pnl` columns.
        slots: Mapping {slot_id: strategy_name} from the ResultBundle.

    Returns:
        Wide DataFrame indexed by timestamp, one column per strategy (labelled
        by strategy name).  Empty DataFrame if no slot columns are found.
    """
    if not slots or history.empty:
        return pd.DataFrame()

    frames: Dict[str, pd.Series] = {}
    for slot_id, strat_name in slots.items():
        col = f"slot_{slot_id}_pnl"
        if col in history.columns:
            # diff() gives bar-level incremental PnL from cumulative column
            frames[strat_name] = history[col].diff().fillna(0.0)

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames, index=history.index)

def resample_pnl_to_horizon(
    bar_pnl_matrix: pd.DataFrame,
    horizon: str = "1d",
) -> pd.DataFrame:
    """
    Resamples an incremental bar PnL matrix to a coarser time horizon.

    Args:
        bar_pnl_matrix: Output of build_bar_pnl_matrix().
        horizon: One of '1d', '1w', '1m'.

    Returns:
        Resampled DataFrame with summed PnL per period.
    """
    rule = _HORIZON_RULE.get(horizon, "1D")
    if bar_pnl_matrix.empty:
        return bar_pnl_matrix
    return bar_pnl_matrix.resample(rule).sum()

def compute_pnl_dist_stats(
    daily_pnl: pd.Series,
    var_confidence: float = 0.95,
    tail_confidence: float = 0.99,
) -> Dict[str, float]:
    """
    Computes distribution statistics for the daily PnL series.

    Methodology:
        skew     — Fisher skewness (positive = right tail, preferred for long).
        kurtosis — Excess kurtosis (Fisher). > 0 = fat tails vs normal.
        VaR / CVaR are returned as positive loss magnitudes for dashboard
        display. Negative tail quantiles from raw PnL space are converted
        explicitly so cards and annotations use one sign convention.

    Args:
        daily_pnl: Daily net PnL series (not cumulative).
        var_confidence: Primary confidence level for VaR / CVaR (default 0.95).
        tail_confidence: Tail VaR confidence level (default 0.99).

    Returns:
        Dict with keys: skew, kurtosis, var_95, cvar_95, var_99, mean, std.
    """
    empty_stats: Dict[str, float] = {
        "skew": float("nan"), "kurtosis": float("nan"),
        "var_95": float("nan"), "cvar_95": float("nan"),
        "var_99": float("nan"), "mean": float("nan"), "std": float("nan"),
    }
    if daily_pnl is None:
        return empty_stats

    # Match dropna(): inf is not NaN, so replace inf before emptiness check
    clean: pd.Series = daily_pnl.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if clean.empty:
        return empty_stats

    # SciPy throws RuntimeWarning if length < 3 or variance is 0
    if len(clean) > 2 and float(clean.std()) > 0:
        skew_val: float  = float(stats.skew(clean))
        kurt_val: float  = float(stats.kurtosis(clean))  # excess (Fisher)
    else:
        skew_val: float  = float("nan")
        kurt_val: float  = float("nan")

    var_95_threshold = float(clean.quantile(1 - var_confidence))
    var_99_threshold = float(clean.quantile(1 - tail_confidence))
    cvar_tail = clean[clean <= var_95_threshold]

    var_95 = max(0.0, -var_95_threshold)
    var_99 = max(0.0, -var_99_threshold)
    cvar_95 = max(0.0, -float(cvar_tail.mean())) if not cvar_tail.empty else float("nan")

    return {
        "skew":     round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "var_95":   round(var_95, 2),
        "cvar_95":  round(cvar_95, 2),
        "var_99":   round(var_99, 2),
        "mean":     round(float(clean.mean()), 2),
        "std":      round(float(clean.std()), 2),
    }

def build_strategy_equity_curve(
    history: pd.DataFrame,
    slot_id: str,
    slot_weight: Optional[float] = None,
    slot_count: Optional[int] = None,
) -> pd.Series:
    """
    Builds a standalone equity curve for one portfolio strategy slot.

    Methodology:
        The portfolio history stores cumulative slot PnL, not a standalone
        strategy NAV. To analyse strategy risk without mixing in portfolio-only
        diversification effects, we reconstruct an isolated equity curve as:

            allocated_capital + cumulative_slot_pnl

        The allocated capital uses the manifest slot weight when available and
        falls back to an equal split across slots otherwise.

    Args:
        history: Portfolio history with `total_value` and `slot_{id}_pnl`.
        slot_id: Slot identifier from the portfolio manifest.
        slot_weight: Optional initial weight allocated to the slot.
        slot_count: Number of active slots, used for equal-weight fallback.

    Returns:
        Standalone strategy equity curve indexed by timestamp.
    """
    if history.empty or "total_value" not in history.columns:
        return pd.Series(dtype=float)

    pnl_col = f"slot_{slot_id}_pnl"
    if pnl_col not in history.columns:
        return pd.Series(dtype=float)

    initial_total_equity = float(history["total_value"].dropna().iloc[0])
    if slot_weight is not None:
        allocated_capital = initial_total_equity * float(slot_weight)
    elif slot_count and slot_count > 0:
        allocated_capital = initial_total_equity / float(slot_count)
    else:
        allocated_capital = initial_total_equity

    slot_pnl = history[pnl_col].ffill().fillna(0.0).astype(float)
    return (slot_pnl + allocated_capital).rename(None)

def derive_daily_pnl_from_equity(equity: pd.Series) -> pd.Series:
    """
    Resamples an equity curve to end-of-day snapshots and returns daily PnL.

    Methodology:
        Daily PnL is derived from end-of-day equity deltas rather than intrabar
        returns so that risk estimates remain interpretable in dollars.
    """
    if equity is None or equity.dropna().empty:
        return pd.Series(dtype=float)

    daily_equity = equity.dropna().astype(float).resample("1D").last().dropna()
    if daily_equity.empty:
        return pd.Series(dtype=float)
    return daily_equity.diff().fillna(0.0)

