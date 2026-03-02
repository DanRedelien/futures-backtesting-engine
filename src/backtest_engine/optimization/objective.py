"""
Objective function for Optuna optimization.

Composite scoring: blended Sharpe/Sortino with soft penalties
for activity and drawdown stability. Designed to produce a
smooth gradient surface for Bayesian optimization.
"""

import numpy as np
from typing import Dict, Any


# ═══════════════════════════════════════════════════════════════════
# SCORING WEIGHTS
# ═══════════════════════════════════════════════════════════════════
SHARPE_WEIGHT: float = 0.7
SORTINO_WEIGHT: float = 0.3


def objective_score(
    stats: Dict[str, Any],
    min_trades: int = 10,
    target_trades: int = 50,
    max_dd_limit: float = 0.40,
) -> float:
    """
    Calculate risk-adjusted composite score for Optuna maximization.

    Formula:
        BaseScore = SHARPE_WEIGHT * Sharpe + SORTINO_WEIGHT * Sortino
        Score     = BaseScore × ActivityPenalty × StabilityPenalty

    ActivityPenalty:
        Smooth ramp from 0→1 as trades approach ``target_trades``.
        If trades < min_trades, returns -1.0 (Hard Kill).

    StabilityPenalty:
        Soft quadratic decay as MaxDD approaches ``max_dd_limit``.
        Returns 0.0 if MaxDD exceeds the limit (hard floor).

    Args:
        stats: Dict with keys total_trades, sharpe_ratio,
               sortino_ratio, max_drawdown (as PERCENT, e.g. -25.0).
        min_trades: Hard floor for validation (e.g. 10).
        target_trades: Soft target for statistical significance (e.g. 50).
        max_dd_limit: MaxDD fraction ceiling (e.g. 0.25 = 25%).

    Returns:
        Non-negative composite score (higher is better).
    """
    trades = stats.get("total_trades", 0)
    sharpe = stats.get("sharpe_ratio", 0.0)
    sortino = stats.get("sortino_ratio", 0.0)
    max_dd = stats.get("max_drawdown", 0.0)

    # ── 1. Hard rejection (Floor) ──────────────────────────────────
    if trades < min_trades:
        return -1.0  # Immediate kill — Optuna learns to avoid

    # ── 2. Base score (blended risk-adjusted return) ───────────────
    base_score = SHARPE_WEIGHT * sharpe + SORTINO_WEIGHT * sortino

    # ── 3. Activity penalty (smooth ramp to target) ────────────────
    #   Returns 1.0 if trades >= target_trades.
    #   Linear ramp from 0 to 1? (trades / target)
    activity_penalty = min(1.0, trades / target_trades)

    # ── 4. Stability penalty (soft quadratic decay) ────────────────
    #   max_dd comes from PerformanceMetrics as PERCENT (e.g. -25.0).
    #   Convert to positive fraction [0.0, 1.0+].
    max_dd_frac = abs(max_dd) / 100.0

    if max_dd_frac > max_dd_limit:
        # Hard floor — beyond the limit the trial is worthless
        stability_penalty = 0.0
    else:
        # Quadratic decay: 1.0 at DD=0, drops to 0.0 at DD=limit
        ratio = max_dd_frac / max_dd_limit
        stability_penalty = 1.0 - ratio ** 2

    # ── 5. Composite ───────────────────────────────────────────────
    score = base_score * activity_penalty * stability_penalty

    return max(0.0, score)
