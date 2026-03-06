"""tests/regression/test_exit_signals.py

Regression: SELL from a flat strategy should produce direction=-1 (short intent),
not be silently treated as an exit.  Previously SELL→-1 always, but the distinction
of intent vs. book-delta is the Allocator+Engine's job, NOT the runner's.
"""

import pytest
import pandas as pd

from src.backtest_engine.portfolio_layer.execution.strategy_runner import StrategyRunner
from src.backtest_engine.portfolio_layer.domain.contracts import PortfolioConfig, StrategySlot
from src.backtest_engine.settings import get_settings


class _AlwaysSell:
    """Minimal strategy that always emits a SELL order."""
    def __init__(self, engine):
        self._invested = True
        self._position_side = "SHORT"

    def on_bar(self, bar):
        from src.backtest_engine.execution import Order
        return [Order(symbol="ES", quantity=1, side="SELL",
                      order_type="MARKET", reason="TEST", timestamp=None)]


class _AlwaysBuy:
    """Minimal strategy that always emits a BUY order."""
    def __init__(self, engine):
        self._invested = True
        self._position_side = "LONG"

    def on_bar(self, bar):
        from src.backtest_engine.execution import Order
        return [Order(symbol="ES", quantity=1, side="BUY",
                      order_type="MARKET", reason="TEST", timestamp=None)]


def _make_config(strategy_class):
    return PortfolioConfig(
        slots=[StrategySlot(strategy_class=strategy_class, symbols=["ES"], weight=1.0)],
        initial_capital=100_000.0,
        rebalance_frequency="intrabar",
    )


def _make_data() -> pd.DataFrame:
    idx = pd.date_range("2023-01-02 09:30", periods=5, freq="30min")
    return pd.DataFrame(
        {"open": 4000.0, "high": 4010.0, "low": 3990.0, "close": 4005.0, "volume": 1000},
        index=idx,
    )


class TestExitSignalMapping:
    def test_sell_from_flat_maps_to_minus_one(self):
        """SELL → direction -1.  The engine decides whether it's a reversal or exit."""
        settings = get_settings()
        df = _make_data()
        config = _make_config(_AlwaysSell)
        data_map = {(0, "ES"): df}

        runner = StrategyRunner(config, data_map, settings)
        ts = df.index[0]
        bar_map = {(0, "ES"): df.loc[ts]}
        signals = runner.collect_signals(bar_map, ts)

        assert len(signals) == 1
        assert signals[0].direction == -1

    def test_buy_maps_to_plus_one(self):
        settings = get_settings()
        df = _make_data()
        config = _make_config(_AlwaysBuy)
        data_map = {(0, "ES"): df}

        runner = StrategyRunner(config, data_map, settings)
        ts = df.index[0]
        bar_map = {(0, "ES"): df.loc[ts]}
        signals = runner.collect_signals(bar_map, ts)

        assert len(signals) == 1
        assert signals[0].direction == 1

    def test_no_orders_yields_no_signal(self):
        class _Silent:
            def __init__(self, engine): pass
            def on_bar(self, bar): return []

        settings = get_settings()
        df = _make_data()
        config = _make_config(_Silent)
        data_map = {(0, "ES"): df}

        runner = StrategyRunner(config, data_map, settings)
        ts = df.index[0]
        bar_map = {(0, "ES"): df.loc[ts]}
        signals = runner.collect_signals(bar_map, ts)

        assert signals == []
