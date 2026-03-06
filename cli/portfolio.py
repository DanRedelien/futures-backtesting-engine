"""
cli/portfolio.py

Portfolio backtest CLI handler.

Responsibility: Parse the YAML portfolio config and run PortfolioBacktestEngine.
This module is called exclusively by run.py --portfolio-backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path


def run(config_path: str, launch_dashboard: bool = False) -> None:
    """
    Loads a YAML portfolio config and runs PortfolioBacktestEngine.

    Methodology:
        Reads the YAML once with safe_load.  Portfolio-specific fields
        (target_portfolio_vol, vol_lookback_bars, max_contracts_per_slot,
        rebalance_frequency) come from the YAML.  Shared execution settings
        (commission_rate, max_slippage_ticks, initial_capital, kill-switch
        thresholds) are read from BacktestSettings (settings.py).

    Args:
        config_path: Path to the YAML config file (absolute or project-relative).
        launch_dashboard: If True, launch Streamlit after the backtest.
    """
    import yaml
    import subprocess

    from src.backtest_engine.portfolio_layer.engine import PortfolioBacktestEngine
    from src.backtest_engine.portfolio_layer.domain.contracts import (
        PortfolioConfig, StrategySlot,
    )
    from src.strategies.registry import get_strategy_class_by_name
    from src.backtest_engine.settings import get_settings

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[Portfolio] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    portfolio_cfg = raw.get("portfolio", {})
    settings = get_settings()

    slots = []
    for slot_cfg in raw["strategies"]:
        try:
            strategy_cls = get_strategy_class_by_name(slot_cfg["strategy"])
        except ValueError as e:
            print(f"[Portfolio] {e}")
            sys.exit(1)

        slots.append(StrategySlot(
            strategy_class=strategy_cls,
            symbols=slot_cfg["symbols"],
            weight=slot_cfg["weight"],
            timeframe=slot_cfg.get("timeframe", "30m"),
            params=slot_cfg.get("params", {}),
        ))

    config = PortfolioConfig(
        slots=slots,
        initial_capital=settings.initial_capital,
        rebalance_frequency=portfolio_cfg.get("rebalance_frequency", "intrabar"),
        target_portfolio_vol=portfolio_cfg.get("target_portfolio_vol", 0.10),
        vol_lookback_bars=int(portfolio_cfg.get("vol_lookback_bars", 20)),
        max_contracts_per_slot=int(portfolio_cfg.get("max_contracts_per_slot", 3)),
        benchmark_symbol=portfolio_cfg.get("benchmark_symbol", "ES") or None,
    )

    engine = PortfolioBacktestEngine(config)
    engine.run()

    # Load benchmark price series for reporting/dashboard
    benchmark_data = None
    if config.benchmark_symbol:
        from src.data.data_lake import DataLake
        try:
            dl = DataLake(settings)
            bdf = dl.load(config.benchmark_symbol, timeframe="30m")
            if not bdf.empty:
                benchmark_data = bdf[["close"]]
        except Exception as exc:
            print(f"[Portfolio] Benchmark load failed ({exc}), skipping.")

    engine.show_results(benchmark=benchmark_data)

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
