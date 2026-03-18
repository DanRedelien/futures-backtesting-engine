"""
src/backtest_engine/portfolio_layer/domain/contracts.py

Top-level portfolio configuration contracts.

Responsibility: PortfolioConfig and StrategySlot define the shape of user-
facing YAML configuration only.  No computation or I/O here.

Execution settings (commission_rate, spread_ticks, spread_mode) and kill-switch
thresholds (max_daily_loss, max_drawdown_pct, max_account_floor) live in
BacktestSettings (settings.py) to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from .policies import RebalancePolicy


@dataclass
class StrategySlot:
    """
    Wires a strategy class to a set of symbols with an allocation weight.

    Attributes:
        strategy_class: Any class that inherits from BaseStrategy.
        symbols: Tickers this strategy slot trades (e.g. ['ES', 'NQ']).
        weight: Capital fraction allocated to this slot (0-1).
                All slot weights must sum to 1.0.
        timeframe: Bar resolution to load for all symbols in this slot.
        params: Optional strategy-level kwargs injected as settings overrides.
                These are written through _PatchedSettings so strategies can
                read them via self.settings.param_name (same as WFO injection).
    """
    strategy_class: Type
    symbols: List[str]
    weight: float
    timeframe: str = "30m"
    params: Dict = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """
    Top-level portfolio backtest configuration.

    Only portfolio-specific settings live here.  Execution parameters
    (commission_rate, spread_ticks, spread_mode) and risk kill-switch thresholds
    are read from BacktestSettings to keep a single source of truth.

    Attributes:
        slots: List of StrategySlots defining the full multi-strat allocation.
        initial_capital: Total portfolio capital in dollars.
        rebalance_frequency: 'intrabar', 'daily', or 'weekly'.
        target_portfolio_vol: Annualised portfolio volatility target (e.g. 0.10 = 10 %).
        vol_lookback_bars: Rolling window (bars) used to estimate realised vol per symbol.
        max_contracts_per_slot: Hard cap on contracts per (slot, symbol) pair.
    """
    slots: List[StrategySlot]
    initial_capital: float
    rebalance_frequency: str
    target_portfolio_vol: float = 0.10
    vol_lookback_bars: int = 20
    max_contracts_per_slot: int = 3
    benchmark_symbol: Optional[str] = "ES"   # Buy-and-hold benchmark (None to disable)

    def validate(self) -> None:
        """
        Validates config invariants before the engine starts.

        Raises:
            ValueError: If weights do not sum to 1 or any slot has no symbols.
        """
        total_weight = sum(s.weight for s in self.slots)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"StrategySlot weights must sum to 1.0, got {total_weight:.4f}"
            )
        for slot in self.slots:
            if not slot.symbols:
                raise ValueError(f"StrategySlot {slot.strategy_class.__name__} has no symbols.")
            if slot.weight <= 0:
                raise ValueError(f"StrategySlot {slot.strategy_class.__name__} has weight <= 0.")
        if not (0.0 < self.target_portfolio_vol <= 1.0):
            raise ValueError(
                f"target_portfolio_vol must be in (0, 1], got {self.target_portfolio_vol}"
            )
        if self.vol_lookback_bars < 2:
            raise ValueError("vol_lookback_bars must be >= 2.")
        if self.max_contracts_per_slot < 1:
            raise ValueError("max_contracts_per_slot must be >= 1.")
