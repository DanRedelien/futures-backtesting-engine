"""
Cost Model for Research-Grade Backtesting.

Handles transaction costs and slippage using per-symbol instrument specs
from BacktestSettings.
"""
from typing import Optional

from ..settings import BacktestSettings, get_settings


class CostModel:
    """
    Calculates transaction costs and slippage for a specific instrument.

    All instrument specs live in settings.py (Single Source of Truth).

    Usage:
        cost = CostModel("GC")
        total = cost.round_trip_cost(n_trades=100)
        specs = cost.specs
    """

    def __init__(
        self, symbol: str, settings: Optional[BacktestSettings] = None
    ) -> None:
        """
        Initialize cost model for a specific symbol.

        Args:
            symbol: Futures ticker (e.g. "GC", "ES").
            settings: Optional settings override.
        """
        self.symbol = symbol
        _settings = settings or get_settings()
        self.specs = _settings.get_instrument_spec(symbol)
        self.commission = _settings.commission_rate

    def round_trip_cost(self, n_trades: int = 1) -> float:
        """
        Calculate total cost for N round-trip trades (Entry + Exit).

        Args:
            n_trades: Number of round-trip trades.

        Returns:
            Total cost in USD (commission + slippage).
        """
        tick_value = self.specs.get("tick_size", 0.01) * self.specs.get(
            "multiplier", 1.0
        )
        comm = n_trades * 2 * self.commission
        slippage = n_trades * 2 * tick_value  # 1 tick slippage per side
        return comm + slippage

    def cost_per_trade(self) -> float:
        """Single round-trip cost (for per-trade deduction)."""
        return self.round_trip_cost(1)
