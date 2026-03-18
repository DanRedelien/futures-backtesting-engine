"""
src/backtest_engine/portfolio_layer/domain/policies.py

Execution and rebalancing policy descriptors.

Responsibility: Named enumerations and parameter bags for the two key
behavioural policies.  No computation here — the Scheduler and
ExecutionHandler contain the actual logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


def _default_commission_rate() -> float:
    from src.backtest_engine.settings import BacktestSettings
    return BacktestSettings().commission_rate

def _default_spread_ticks() -> int:
    from src.backtest_engine.settings import BacktestSettings
    return BacktestSettings().spread_ticks


class RebalancePolicy(str, Enum):
    """
    Controls when the Allocator recomputes target positions.

    Attributes:
        INTRABAR: Recompute on every bar (default).  The strategy's on_bar()
            return value directly drives allocation each step.
        DAILY: Recompute once per calendar day at the first available bar.
            Subsequent intraday bars hold the morning target unchanged.
    """
    INTRABAR = "intrabar"
    DAILY    = "daily"


@dataclass
class ExecutionPolicy:
    """
    Parameters governing order fill simulation.

    Attributes:
        commission_rate: Dollar commission charged per contract per fill.
        spread_ticks: Base tick count applied as a deterministic spread at fill.
            The active spread mode (static or adaptive_volatility) is configured
            in BacktestSettings.spread_mode.
    """
    commission_rate: float = field(default_factory=_default_commission_rate)
    spread_ticks: int = field(default_factory=_default_spread_ticks)
