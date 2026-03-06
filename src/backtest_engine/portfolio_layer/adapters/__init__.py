"""
src/backtest_engine/portfolio_layer/adapters/

Bridges from the legacy single-asset BaseStrategy API to the portfolio engine.
"""

from .legacy_strategy_adapter import LegacyStrategyAdapter

__all__ = ["LegacyStrategyAdapter"]
