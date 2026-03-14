"""
cli/single.py

Single-asset backtest CLI handler.

Responsibility: Parse strategy name and run BacktestEngine.
Called by run.py --backtest.
"""

from __future__ import annotations

import sys
from typing import Any

from src.data.data_lake import DataLake
from src.strategies.registry import load_strategy_by_id


def _load_strategy(name: str) -> Any:
    """
    Returns the strategy class for the given short name.

    Args:
        name: Strategy identifier ('sma', 'mean_rev', 'ict_ob', etc.).

    Returns:
        Strategy class (subclass of BaseStrategy).
    """
    try:
        return load_strategy_by_id(name)
    except ValueError as e:
        print(f"[Error] {e}")
        sys.exit(1)

def run(strategy_name: str, settings: Any) -> None:
    """
    Runs a single-asset backtest.

    Args:
        strategy_name: Short strategy name (e.g. 'zscore').
        settings: BacktestSettings instance.
    """
    from src.backtest_engine.engine import BacktestEngine

    strategy_class = _load_strategy(strategy_name)

    print("=" * 60)
    print(f"  Backtest: {strategy_class.__name__}")
    print(f"  Symbol   : {settings.default_symbol}")
    print(f"  Timeframe: {settings.low_interval}")
    print(f"  Capital  : ${settings.initial_capital:,.0f}")
    print("=" * 60)

    data_lake = DataLake(settings)
    cache_errors = data_lake.validate_cache_requirements(
        requirements=[(settings.default_symbol, settings.low_interval)],
    )
    if cache_errors:
        print("[Data] Cache freshness check failed:")
        for err in cache_errors:
            print(f"  - {err}")
        print(
            f"[Data] Update cache first. "
            f"Max allowed age: {settings.max_cache_staleness_days} days."
        )
        print(f"[Data] Example: python run.py --download {settings.default_symbol}")
        sys.exit(1)

    engine = BacktestEngine(settings=settings)
    engine.run(strategy_class)
    engine.show_results()
