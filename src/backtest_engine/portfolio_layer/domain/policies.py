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
    from src.backtest_engine.settings import get_settings
    return get_settings().commission_rate

def _default_max_slippage_ticks() -> int:
    from src.backtest_engine.settings import get_settings
    return get_settings().max_slippage_ticks


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
        max_slippage_ticks: Maximum random slippage ticks applied at fill.
            Actual slippage is drawn uniformly from [0, max_slippage_ticks].
    """
    commission_rate: float = field(default_factory=_default_commission_rate)
    max_slippage_ticks: int = field(default_factory=_default_max_slippage_ticks)
