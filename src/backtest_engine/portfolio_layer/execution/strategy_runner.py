"""
src/backtest_engine/portfolio_layer/execution/strategy_runner.py

Per-slot strategy instance management and signal collection.

Responsibility: For each StrategySlot, maintains one strategy instance per
symbol (built via LegacyStrategyAdapter), calls on_bar(), and translates
returned Orders into StrategySignals.  No sizing logic here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtest_engine.execution import Order
from ..domain.contracts import PortfolioConfig
from ..domain.signals import StrategySignal
from ..adapters.legacy_strategy_adapter import LegacyStrategyAdapter


class StrategyRunner:
    """
    Manages one strategy instance per (slot_id, symbol) pair.

    Methodology:
        Strategies are constructed once before the bar loop via
        LegacyStrategyAdapter.build() — see that module for explicit
        limitations of the legacy BaseStrategy adapter contract.

        on_bar() returns List[Order].  We take the last order's side as
        the directional intent:
          BUY  → direction +1 (long)
          SELL → direction -1 (short)
          qty == 0 or empty list → no new directive (hold current target)

        Sizing is deferred entirely to the Allocator.
    """

    def __init__(
        self,
        config: PortfolioConfig,
        data_map: Dict[Tuple[int, str], pd.DataFrame],
        settings: Any,
    ) -> None:
        """
        Args:
            config: Validated PortfolioConfig.
            data_map: (slot_id, symbol) → full OHLCV DataFrame.
            settings: BacktestSettings instance.
        """
        self._config = config
        self._instances: Dict[Tuple[int, str], Any] = {}

        for slot_id, slot in enumerate(config.slots):
            for symbol in slot.symbols:
                df = data_map.get((slot_id, symbol), pd.DataFrame())
                instance = LegacyStrategyAdapter.build(
                    strategy_class=slot.strategy_class,
                    data=df,
                    symbol=symbol,
                    settings=settings,
                    params=slot.params,
                )
                self._instances[(slot_id, symbol)] = instance

    def collect_signals(
        self,
        bar_map: Dict[Tuple[int, str], Any],
        timestamp: Any,
        current_positions: Optional[Dict[Tuple[int, str], float]] = None,
    ) -> List[StrategySignal]:
        """
        Calls on_bar() for every (slot_id, symbol) and collects signals.

        Args:
            bar_map: (slot_id, symbol) → current OHLCV bar Series.
            timestamp: Current bar timestamp (close[t]).
            current_positions: Optional book positions (reserved for future use).

        Returns:
            List of StrategySignal objects.
        """
        signals: List[StrategySignal] = []

        for (slot_id, symbol), instance in self._instances.items():
            bar = bar_map.get((slot_id, symbol))
            if bar is None:
                continue

            try:
                orders: List[Order] = instance.on_bar(bar) or []
            except Exception as exc:
                print(f"[Runner] {instance.__class__.__name__}({symbol}) error: {exc}")
                orders = []

            if not orders:
                continue

            order = orders[-1]

            if getattr(instance, "_invested", False):
                pos_side = getattr(instance, "_position_side", None)
                if pos_side == "LONG":
                    direction = 1
                elif pos_side == "SHORT":
                    direction = -1
                else:
                    direction = 0
            else:
                direction = 0

            signals.append(StrategySignal(
                slot_id=slot_id,
                symbol=symbol,
                direction=direction,
                reason=order.reason,
                timestamp=timestamp,
            ))

        return signals
