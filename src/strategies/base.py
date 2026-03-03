"""
Base Strategy Interface.

All strategy implementations must inherit from this abstract class to be
compatible with the BacktestEngine. Provides a standardised hook contract
and a convenience reference to engine internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtest_engine.execution import Order

if TYPE_CHECKING:
    from src.backtest_engine.engine import BacktestEngine


class BaseStrategy(ABC):
    """
    Abstract base class for all single-asset backtesting strategies.

    Methodology:
        The engine calls on_bar() once per bar after filling any pending
        orders.  Strategies should pre-compute all indicators during
        __init__ via vectorised pandas/numpy operations on the full
        dataset. on_bar() simply performs an O(1)
        dictionary lookup against those pre-computed arrays.
        (Note: Do NOT shift indicators by 1. The engine naturally executes
        generated orders at the NEXT bar's Open, providing realistic 1-bar 
        execution lag without lookahead bias).

    Required:
        on_bar() — return a list of Orders (may be empty).

    Optional:
        on_start() — called once just before the event loop begins.
        get_search_space() — classmethod; returns WFO optimisation bounds.
    """

    def __init__(self, engine: "BacktestEngine") -> None:
        self.engine = engine
        self.settings = engine.settings

    # ── Abstract hook ──────────────────────────────────────────────────────────

    @abstractmethod
    def on_bar(self, bar: pd.Series) -> List[Order]:
        """
        Called once per bar by BacktestEngine.

        Args:
            bar: Current OHLCV bar (pd.Series with index open/high/low/close/volume).

        Returns:
            List of Order objects to queue for execution on the NEXT bar's open.
            Return an empty list if no new order is required.
        """
        ...

    # ── Optional hooks ─────────────────────────────────────────────────────────

    def on_start(self) -> None:
        """
        Called once by the engine immediately before the bar loop starts.

        Override to run any setup that requires the full dataset to be loaded
        but must happen just before live iteration (e.g. printing summaries).
        Default: no-op.
        """

    # ── WFO interface ──────────────────────────────────────────────────────────

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """
        Returns the Walk-Forward Optimisation (WFO) search space for Optuna.

        Each key maps to a strategy parameter name (must match __init__ args
        that the WFO engine will inject into settings).

        Supported value formats:
            - (start, stop, step)  → int or float range
            - [val1, val2, ...]    → categorical choice
            - (start, stop)        → continuous range without step

        Returns:
            Dictionary of {param_name: bounds}. Empty dict means no WFO.
        """
        return {}

    # ── Convenience helpers ────────────────────────────────────────────────────

    def get_position(self, symbol: Optional[str] = None) -> float:
        """
        Returns the current signed position quantity for a symbol.

        Args:
            symbol: Symbol to query. Defaults to settings.default_symbol.

        Returns:
            Signed quantity (positive = long, negative = short, 0 = flat).
        """
        sym = symbol or self.settings.default_symbol
        return self.engine.portfolio.positions.get(sym, 0.0)

    def is_flat(self, symbol: Optional[str] = None) -> bool:
        """Returns True when the strategy holds no open position."""
        return self.get_position(symbol) == 0.0

    def market_order(
        self,
        side: str,
        quantity: float,
        reason: str = "SIGNAL",
        symbol: Optional[str] = None,
        timestamp: Optional[Any] = None,
    ) -> Order:
        """
        Convenience factory for MARKET orders.

        Args:
            side: 'BUY' or 'SELL'.
            quantity: Absolute number of contracts/shares.
            reason: Exit/Entry reason tag (e.g. 'SL', 'TP', 'SIGNAL').
            symbol: Override asset symbol; defaults to settings.default_symbol.
            timestamp: Bar timestamp. Filled automatically when None.

        Returns:
            Order ready to be returned from on_bar().
        """
        return Order(
            symbol=symbol or self.settings.default_symbol,
            quantity=abs(quantity),
            side=side,
            order_type="MARKET",
            reason=reason,
            timestamp=timestamp,
        )
