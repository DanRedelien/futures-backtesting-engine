"""
Entry point for the Single-Asset Backtesting Engine.

Usage:
  python run.py --download GC ES NQ RTY YM CL SI
  python run.py --backtest --strategy sma
  python run.py --backtest --strategy mean_rev
  python run.py --backtest --strategy ict_ob
  python run.py --backtest --strategy zscore
  python run.py --wfo --strategy sma
  python run.py --wfo --strategy mean_rev
  python run.py --wfo --strategy ict_ob
  python run.py --wfo --strategy zscore
"""

import argparse
import sys

from src.backtest_engine.settings import get_settings
from src.data import IBFetcher


# ── Strategy registry ──────────────────────────────────────────────────────────
def _load_strategy(name: str):
    """
    Returns the strategy class for the given short name.

    Args:
        name: Strategy identifier ('sma', 'mean_rev', 'ict_ob', or 'zscore').

    Returns:
        Strategy class (subclass of BaseStrategy).
    """
    registry = {
        "sma":      "src.strategies.sma_crossover:SmaCrossoverStrategy",
        "mean_rev": "src.strategies.mean_reversion:MeanReversionStrategy",
        "ict_ob":   "src.strategies.ict_order_block:IctOrderBlockStrategy",
        "zscore":   "src.strategies.zscore_reversal:ZScoreReversalStrategy",
    }

    if name not in registry:
        print(f"[Error] Unknown strategy '{name}'. Available: {list(registry)}")
        sys.exit(1)

    module_path, class_name = registry[name].split(":")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Asset Backtest Engine")
    parser.add_argument(
        "--download", nargs="+",
        help="Download data for symbols via IB (e.g. --download ES NQ)",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run a single backtest for the selected strategy",
    )
    parser.add_argument(
        "--wfo", action="store_true",
        help="Run Walk-Forward Validation (WFV) with Skeptic analysis for the selected strategy",
    )
    parser.add_argument(
        "--strategy", type=str, default="sma",
        help="Strategy to use: 'sma', 'mean_rev', 'ict_ob', or 'zscore'",
    )
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()

    # ── Download ───────────────────────────────────────────────────────────────
    if args.download:
        print("=" * 60)
        print(f"  Downloading data: {args.download}")
        print("=" * 60)
        fetcher = IBFetcher(settings=settings)
        for sym in args.download:
            fetcher.fetch_all_timeframes(sym)
        print("Download complete.")

    # ── Backtest ───────────────────────────────────────────────────────────────
    if args.backtest:
        from src.backtest_engine.engine import BacktestEngine

        strategy_class = _load_strategy(args.strategy)

        print("=" * 60)
        print(f"  Backtest: {strategy_class.__name__}")
        print(f"  Symbol   : {settings.default_symbol}")
        print(f"  Timeframe: {settings.low_interval}")
        print(f"  Capital  : ${settings.initial_capital:,.0f}")
        print("=" * 60)

        engine = BacktestEngine()
        engine.run(strategy_class)
        engine.show_results()

    # ── Walk-Forward Validation ─────────────────────────────────────────────────
    if args.wfo:
        from src.backtest_engine.optimization.wfv_optimizer import WalkForwardOptimizer

        strategy_class = _load_strategy(args.strategy)

        print("=" * 60)
        print(f"  WFV: {strategy_class.__name__}")
        print(f"  Symbol   : {settings.default_symbol}")
        print(f"  Timeframe: {settings.low_interval}")
        print("=" * 60)

        wfv = WalkForwardOptimizer(settings=settings)
        wfv.run(strategy_class=strategy_class)

