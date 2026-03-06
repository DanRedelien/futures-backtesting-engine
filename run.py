"""
Entry point for the Backtesting Engine.

This script supports three distinct execution modes:
1. Single-Asset Backtest    (--backtest)
2. Walk-Forward Optimization (--wfo)
3. Multi-Strategy Portfolio  (--portfolio-backtest)

── 1. Single-Asset Backtesting ──────────────────────────────────────────────
    python run.py --backtest --strategy zscore
    python run.py --backtest --strategy zscore --dashboard

Run `python run.py --help` to see all available strategies.

── 2. Walk-Forward Optimization (WFO) ───────────────────────────────────────
    python run.py --wfo --strategy zscore

── 3. Portfolio Backtesting ─────────────────────────────────────────────────
    python run.py --portfolio-backtest --dashboard
    python run.py --portfolio-backtest --portfolio-config path/to/config.yaml

── Data Management ──────────────────────────────────────────────────────────
    python run.py --download ES NQ CL GC
"""

import argparse
import sys

from src.backtest_engine.settings import get_settings
from src.data import IBFetcher


if __name__ == "__main__":
    from src.strategies.registry import STRATEGIES
    strategy_list = ", ".join(STRATEGIES.keys())
    
    parser = argparse.ArgumentParser(description="Backtesting Engine")
    parser.add_argument("--download", nargs="+", help="Download data for symbols via IB")
    parser.add_argument("--backtest", action="store_true", help="Run single-asset backtest")
    parser.add_argument("--wfo", action="store_true", help="Run Walk-Forward Optimization")
    parser.add_argument("--strategy", type=str, default="sma",
                        help=f"Strategy name ({strategy_list})")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Streamlit dashboard after backtest")
    parser.add_argument("--portfolio-backtest", action="store_true",
                        help="Run multi-strategy portfolio backtest")
    parser.add_argument("--portfolio-config", type=str,
                        default="src/backtest_engine/portfolio_layer/portfolio_config_example.yaml",
                        help="Path to YAML portfolio config (default: portfolio_layer/portfolio_config_example.yaml)")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()

    if args.download:
        print("=" * 60)
        print(f"  Downloading data: {args.download}")
        print("=" * 60)
        fetcher = IBFetcher(settings=settings)
        for sym in args.download:
            fetcher.fetch_all_timeframes(sym)
        print("Download complete.")

    if args.backtest:
        from cli.single import run as run_backtest
        run_backtest(args.strategy, settings, launch_dashboard=args.dashboard)

    if args.wfo:
        from cli.wfo import run as run_wfo
        run_wfo(args.strategy, settings)

    if getattr(args, "portfolio_backtest", False):
        from cli.portfolio import run as run_portfolio
        run_portfolio(args.portfolio_config, launch_dashboard=args.dashboard)
