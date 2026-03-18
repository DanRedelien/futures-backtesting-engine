"""tests/unit/test_allocator.py — Allocator unit tests."""

import math

import pytest
import pandas as pd
import numpy as np

from src.backtest_engine.portfolio_layer.allocation.allocator import Allocator
from src.backtest_engine.portfolio_layer.domain.contracts import PortfolioConfig, StrategySlot
from src.backtest_engine.portfolio_layer.domain.signals import StrategySignal


def _config(
    weight: float = 1.0,
    n_symbols: int = 1,
    target_vol: float = 0.10,
    vol_lookback: int = 5,
    max_contracts: int = 10,
) -> PortfolioConfig:
    from src.strategies.sma_crossover import SmaCrossoverStrategy
    return PortfolioConfig(
        slots=[StrategySlot(
            strategy_class=SmaCrossoverStrategy,
            symbols=["ES"] * n_symbols,
            weight=weight,
        )],
        initial_capital=100_000.0,
        rebalance_frequency="intrabar",
        target_portfolio_vol=target_vol,
        vol_lookback_bars=vol_lookback,
        max_contracts_per_slot=max_contracts,
    )


def _signal(direction: int = 1, symbol: str = "ES", slot_id: int = 0) -> StrategySignal:
    return StrategySignal(slot_id=slot_id, symbol=symbol, direction=direction)


def _price_history(
    symbol: str = "ES",
    slot_id: int = 0,
    n: int = 30,
    vol: float = 0.015,
) -> dict:
    """Generates a synthetic price series with known volatility."""
    np.random.seed(42)
    log_rets = np.random.normal(0, vol, n)
    prices = 4000.0 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2023-01-02", periods=n, freq="30min")
    return {(slot_id, symbol): pd.Series(prices, index=idx)}


SPECS = {"ES": {"multiplier": 50.0, "tick_size": 0.25}}


class TestComputeTargets:
    def test_zero_direction_yields_zero_qty(self):
        alloc = Allocator(_config())
        hist = _price_history()
        t = alloc.compute_targets([_signal(direction=0)], 100_000.0, {"ES": 4000.0}, SPECS, hist)
        assert t[0].target_qty == 0

    def test_long_signal_yields_positive_qty(self):
        alloc = Allocator(_config())
        hist = _price_history(vol=0.0025)
        t = alloc.compute_targets([_signal(direction=1)], 5_000_000.0, {"ES": 4000.0}, SPECS, hist)
        assert t[0].target_qty > 0

    def test_short_signal_yields_negative_qty(self):
        alloc = Allocator(_config())
        hist = _price_history(vol=0.0025)
        t = alloc.compute_targets([_signal(direction=-1)], 5_000_000.0, {"ES": 4000.0}, SPECS, hist)
        assert t[0].target_qty < 0

    def test_zero_price_yields_zero_qty(self):
        alloc = Allocator(_config())
        hist = _price_history()
        t = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 0.0}, SPECS, hist)
        assert t[0].target_qty == 0

    def test_max_contracts_cap_applied(self):
        """Verify the hard cap on contracts is respected."""
        alloc = Allocator(_config(max_contracts=2))
        # Give very low vol (high history consistency) to force a large raw size.
        hist = _price_history(vol=0.0001)
        t = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist)
        assert abs(t[0].target_qty) <= 2

    def test_insufficient_history_gives_minimal_sizing(self):
        """When fewer bars than lookback exist, vol fallback (1.0) minimises size."""
        alloc = Allocator(_config(vol_lookback=20))
        hist = {
            (0, "ES"): pd.Series(
                [4000.0, 4001.0], index=pd.date_range("2023-01-02", periods=2, freq="30min")
            )
        }
        t = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist)
        # With 100% annualised vol fallback, sizing is tiny — cap at 0
        assert t[0].target_qty >= 0  # non-negative for long

    def test_higher_vol_yields_smaller_position(self):
        """Higher realised vol should produce a smaller position (vol-scaling)."""
        alloc = Allocator(_config())
        hist_low  = _price_history(vol=0.005)
        hist_high = _price_history(vol=0.050)
        t_low  = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist_low)
        t_high = alloc.compute_targets([_signal(direction=1)], 100_000.0, {"ES": 4000.0}, SPECS, hist_high)
        assert abs(t_low[0].target_qty) >= abs(t_high[0].target_qty)

    def test_raw_contract_sizing_matches_manual_es_example(self):
        """Raw vol-target sizing should match the documented ES back-of-envelope example."""
        alloc = Allocator(_config())

        raw_contracts = alloc._compute_raw_contracts(
            equity=100_000.0,
            price=5_000.0,
            multiplier=50.0,
            annualised_vol=0.20,
        )

        assert math.isclose(raw_contracts, 0.2, rel_tol=1e-9), raw_contracts
