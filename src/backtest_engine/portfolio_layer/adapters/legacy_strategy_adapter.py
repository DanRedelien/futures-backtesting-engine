"""
src/backtest_engine/portfolio_layer/adapters/legacy_strategy_adapter.py

Bridges the legacy single-asset BaseStrategy API to the portfolio engine.

Background
----------
All existing strategies inherit from BaseStrategy (src/strategies/base.py) and
expect a BacktestEngine-shaped object as their first constructor argument.  The
portfolio engine is structurally different (shared capital, multi-symbol), so
this module provides a thin shim that satisfies the strategy's __init__ contract
without introducing a circular dependency on BacktestEngine.

LIMITATIONS
-----------
1. self.engine.portfolio.positions[symbol] is a SINGLE-ASSET dict, not the
   shared PortfolioBook.  Strategies that call get_position() or is_flat()
   see the MockPortfolio — their own per-instance dict — not the real portfolio
   ledger.  This means internal SL/TP tracking inside the strategy is correct,
   but the strategy CANNOT see positions opened by other slots.

2. self.market_order(side, quantity) — the quantity field in the returned
   Order is ignored by the portfolio engine.  Only the DIRECTION (BUY/SELL)
   matters; actual sizing is deferred entirely to the Allocator.

3. Risk limits inside BaseStrategy (daily loss caps, drawdown halts) that
   depend on self.engine.portfolio are NOT connected to the real portfolio book.

These limitations are acceptable for the current single-lens strategy design.
If a strategy needs portfolio-aware sizing, it should be refactored to accept a
PortfolioContext injection directly.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class _MockPortfolio:
    """
    Minimal portfolio shim so BaseStrategy helpers don't crash.

    Instance-level dict prevents shared mutable state across strategy instances.
    """

    def __init__(self) -> None:
        self.positions: Dict[str, float] = {}  # instance attribute, never shared


class _MockEngine:
    """
    Minimal engine shim passed to strategy __init__.

    Satisfies the BaseStrategy(engine) constructor contract without requiring
    the full BacktestEngine class.
    """

    def __init__(self, settings: Any, data: pd.DataFrame, symbol: str) -> None:
        self.settings  = settings
        self.data      = data
        self.portfolio = _MockPortfolio()
        self._symbol   = symbol


class _PatchedSettings:
    """
    Wraps BacktestSettings and injects StrategySlot.params as direct attributes.

    This is the same mechanism WFO uses: parameters are injected into settings
    so strategies can read them via self.settings.param_name without any
    modification to the strategy itself.
    """

    def __init__(self, settings: Any, symbol: str, params: Dict[str, Any]) -> None:
        self._settings      = settings
        self.default_symbol = symbol
        for key, value in params.items():
            object.__setattr__(self, key, value)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._settings, item)


class LegacyStrategyAdapter:
    """
    Constructs a legacy single-asset strategy instance for use in the portfolio engine.

    Methodology:
        1. Creates a _MockEngine with the strategy's specific symbol data.
        2. Wraps BacktestSettings with _PatchedSettings to inject slot params.
        3. Instantiates the strategy class and calls on_start() if available.

    Args:
        strategy_class: The BaseStrategy subclass to instantiate.
        data: Full OHLCV DataFrame for this (slot, symbol) pair.
        symbol: Ticker symbol this instance will trade.
        settings: BacktestSettings instance (used as the base for patching).
        params: Slot-level param overrides (from StrategySlot.params).

    Returns:
        An instantiated, ready-to-call strategy instance.
    """

    @staticmethod
    def build(
        strategy_class: Any,
        data: pd.DataFrame,
        symbol: str,
        settings: Any,
        params: Dict[str, Any],
    ) -> Any:
        mock_engine = _MockEngine(settings, data, symbol)
        mock_engine.settings = _PatchedSettings(settings, symbol, params)
        instance = strategy_class(mock_engine)
        if hasattr(instance, "on_start"):
            instance.on_start()
        return instance
