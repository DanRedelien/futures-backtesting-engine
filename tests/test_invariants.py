"""
Unit tests for engine invariants: Accounting, Risk Limits, and Execution Leakage.
"""

import math
from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.backtest_engine.engine import BacktestEngine
from src.backtest_engine.settings import BacktestSettings
from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy


def create_mock_data(n_bars: int = 10, start_price: float = 100.0) -> pd.DataFrame:
    """Helper to generate a simple sequential dataframe."""
    timestamps = [datetime(2025, 1, 1, 9, 30) + timedelta(minutes=30 * i) for i in range(n_bars)]
    data = []
    
    # We create a predictable price sequence where open=close for simplicity
    for i in range(n_bars):
        price = start_price + i
        data.append({
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000
        })
        
    df = pd.DataFrame(data, index=timestamps)
    return df


class MockSignalStrategy(BaseStrategy):
    """Fires a LONG order on the 2nd bar (index 1), exits on 5th bar (index 4)."""
    
    def on_bar(self, bar) -> list:
        # We find our numeric index by checking the close price vs start
        index = bar.close - 100.0 
        
        if index == 1.0: # 2nd bar
            return [Order(symbol=self.engine.settings.default_symbol, quantity=1, side="BUY")]
            
        if index == 4.0: # 5th bar
            return [Order(symbol=self.engine.settings.default_symbol, quantity=1, side="SELL", reason="EXIT")]
            
        return []


class MockDrawdownStrategy(BaseStrategy):
    """Buys immediately and holds while price plummets to test DD limit."""
    
    def on_bar(self, bar) -> list:
        if "BOUGHT" not in getattr(self, "state", []):
            self.state = ["BOUGHT"]
            return [Order(symbol=self.engine.settings.default_symbol, quantity=1, side="BUY")]
        return []


def test_no_leakage_execution_timing():
    """
    Ensures that a signal emitted at bar T executes at the Open of bar T+1.
    """
    data = create_mock_data()
    # At index 1 (T=1), close=101. Signal is fired.
    # It must execute at index 2 (T=2) open=102.
    
    settings = BacktestSettings(
        commission_rate=0.0,
        max_slippage_ticks=0,   # Set slippage to 0 for exact math
        initial_capital=10000.0
    )
    # Add dummy spec to ensure multiplier=1 for easy math
    settings.instrument_specs = {"NQ": {"tick_size": 1.0, "multiplier": 1.0}}
    
    engine = BacktestEngine(settings=settings, data=data)
    engine.run(MockSignalStrategy)
    
    fills = engine.execution.fills
    trades = engine.execution.trades
    
    assert len(fills) == 2, "Expected 1 entry and 1 exit fill"
    
    entry_fill = fills[0]
    expected_entry_time = data.index[2] # Signal bar 1, execute bar 2
    expected_entry_price = data.iloc[2]["open"] # 102.0
    
    assert entry_fill.timestamp == expected_entry_time, "Execution time leaked or was delayed"
    assert entry_fill.fill_price == expected_entry_price, "Execution price did not match next bar open"


def test_accounting_invariants():
    """
    Ensures total equity = cash + mark-to-market positions at every step.
    """
    data = create_mock_data()
    
    settings = BacktestSettings(commission_rate=0.0, max_slippage_ticks=0, initial_capital=10000.0)
    settings.instrument_specs = {"NQ": {"tick_size": 1.0, "multiplier": 1.0}}
    
    engine = BacktestEngine(settings=settings, data=data)
    engine.run(MockSignalStrategy)
    
    history = engine.portfolio.history
    
    for row in history:
        cash = row["cash"]
        holdings = row["holdings"]
        total = row["total_value"]
        
        # Invariant: cash + holdings must exactly equal total_value
        assert math.isclose(cash + holdings, total, rel_tol=1e-9), f"Accounting drift detected! {cash} + {holdings} != {total}"


def test_circuit_breaker_max_drawdown():
    """
    Forces a strategy to hold a losing position and verifies the engine aborts and liquidates.
    """
    # Create plummeting prices
    timestamps = [datetime(2025, 1, 1, 9, 30) + timedelta(minutes=30 * i) for i in range(10)]
    data = pd.DataFrame({
        "open":  [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        "high":  [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        "low":   [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        "close": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    }, index=timestamps)
    
    # Capital: $1000. Multiplier: 10. Start price: 100.
    # Buy 1 contract at 90 (bar 1). Notional value $900.
    # At price 60 (bar 4), loss is -$300. Max DD limit 20% of 1000 is 200.
    # Should halt when DD exceeds 20%.
    
    settings = BacktestSettings(
        commission_rate=0.0, 
        max_slippage_ticks=0, 
        initial_capital=1000.0,
        max_drawdown_pct=0.20 # 20% max DD
    )
    settings.instrument_specs = {"NQ": {"tick_size": 1.0, "multiplier": 10.0}}
    
    engine = BacktestEngine(settings=settings, data=data)
    engine.run(MockDrawdownStrategy)
    
    assert engine.trading_halted_permanently, "Engine failed to trigger permanent halt on max drawdown breach."
    
    # After a halt, the engine force-liquidates next bar.
    # Let's verify we closed out our position.
    final_positions = engine.portfolio.positions
    assert sum(abs(v) for v in final_positions.values()) == 0, "Positions were not liquidated after halt!"
