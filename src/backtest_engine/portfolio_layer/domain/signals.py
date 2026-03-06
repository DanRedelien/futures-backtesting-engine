"""
src/backtest_engine/portfolio_layer/domain/signals.py

Directional signal and target-position contracts.

Responsibility: Pure data carriers between StrategyRunner → Allocator → Engine.
No computation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategySignal:
    """
    Directional intent emitted by a StrategyRunner for a specific (slot, symbol).

    The portfolio engine converts signals into target positions; sizing happens
    in the Allocator — not inside the strategy.

    Attributes:
        slot_id: Index of the originating StrategySlot in PortfolioConfig.slots.
        symbol: Ticker this signal targets.
        direction: +1 (long), -1 (short), 0 (flat / exit).
        reason: Human-readable tag (e.g. 'SIGNAL', 'SL', 'TP', 'EXIT').
        timestamp: Bar timestamp at which the signal was generated (close[t]).
    """
    slot_id: int
    symbol: str
    direction: int              # +1 / -1 / 0
    reason: str = "SIGNAL"
    timestamp: Optional[object] = None


@dataclass
class TargetPosition:
    """
    Desired signed contract quantity for a (slot_id, symbol) pair.

    Produced by the Allocator and consumed by the portfolio engine to
    compute order deltas.

    Attributes:
        slot_id: Originating strategy slot.
        symbol: Ticker.
        target_qty: Signed contracts (positive = long, negative = short).
    """
    slot_id: int
    symbol: str
    target_qty: float
