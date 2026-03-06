"""
cli/single.py

Single-asset backtest CLI handler.

Responsibility: Parse strategy name, run BacktestEngine, and optionally launch
the Streamlit dashboard.  Called by run.py --backtest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.strategies.registry import load_strategy_by_id, get_strategy_ids


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

def run(strategy_name: str, settings: Any, launch_dashboard: bool = False) -> None:
    """
    Runs a single-asset backtest and optionally launches the dashboard.

    Args:
        strategy_name: Short strategy name (e.g. 'zscore').
        settings: BacktestSettings instance.
        launch_dashboard: If True, launch Streamlit after the backtest.
    """
    from src.backtest_engine.engine import BacktestEngine

    strategy_class = _load_strategy(strategy_name)

    print("=" * 60)
    print(f"  Backtest: {strategy_class.__name__}")
    print(f"  Symbol   : {settings.default_symbol}")
    print(f"  Timeframe: {settings.low_interval}")
    print(f"  Capital  : ${settings.initial_capital:,.0f}")
    print("=" * 60)

    engine = BacktestEngine()
    engine.run(strategy_class)
    engine.show_results()

    if launch_dashboard:
        dashboard_path = (
            Path(__file__).parent.parent
            / "src" / "backtest_engine" / "analytics" / "dashboard" / "app.py"
        )
        print("\n[Dashboard] Launching Streamlit dashboard...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            cwd=str(Path(__file__).parent.parent),
            check=False,
        )
