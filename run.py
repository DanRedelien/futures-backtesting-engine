"""
Entry point for the Backtesting Engine.

TODO: create 4-step testing line, like:

1. python run.py --download ES NQ YM RTY CL GC SI
2. python run.py --backtest --strategy zscore (Create strategy and test in with Deep-Analysis)
3. python run.py --ultra fast single strategys (create ultra fast and very simple strategy tester, 
    he takes 10 selected strategys and runs them in single strategy mode, 
    then builds matplot lib with them all(only graph) without saving .png, just open in window.
    + 3 metrics, pnl%, Sharp ratio, MDD% per strategy, that all in one graph)
    Like protoflio mode, but without deep-analysis.
4. Optimizer (optimize best strategy)
5. python run.py --portfolio-backtest (take best and create uncorrletaed pnl strategys.)


Execution modes:
    1. Single-Asset Backtest      (--backtest)
    2. Walk-Forward Optimization  (--wfo)
    3. Multi-Strategy Portfolio   (--portfolio-backtest)
    4. Dashboard only             (--dashboard)

Strategies:
    To see the full list of available strategies, run:
        python run.py --help

── 1. Single-Asset Backtesting ──────────────────────────────────────────────
    python run.py --backtest --strategy zscore

── 2. Walk-Forward Optimization (WFO) ───────────────────────────────────────
    python run.py --wfo --strategy zscore

── 3. Portfolio Backtesting ─────────────────────────────────────────────────
    python run.py --portfolio-backtest
    python run.py --portfolio-backtest --portfolio-config my_config.yaml

── 4. Open Dashboard (standalone, no new backtest) ──────────────────────────
    python run.py --dashboard

To run a backtest AND open the dashboard immediately after:
    python run.py --backtest --strategy zscore --dashboard
    python run.py --portfolio-backtest --dashboard

── Data Management ──────────────────────────────────────────────────────────
    python run.py --download ES NQ YM RTY CL GC SI
"""

import argparse
import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).parent
_TERMINAL_DASHBOARD_APP = "src.backtest_engine.analytics.terminal_ui.app:app"
_TERMINAL_DASHBOARD_HOST = "127.0.0.1"
_TERMINAL_DASHBOARD_PORT = "8000"


def _launch_dashboard() -> None:
    """
    Launches the FastAPI terminal dashboard as a child process.

    Runs in the foreground so the terminal shows ASGI logs.
    The backtest process has already finished writing artifacts before
    this call — no race condition.
    """
    print("\n[Dashboard] Launching terminal dashboard...")
    print(f"[Dashboard] URL: http://{_TERMINAL_DASHBOARD_HOST}:{_TERMINAL_DASHBOARD_PORT}\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            _TERMINAL_DASHBOARD_APP,
            "--host",
            _TERMINAL_DASHBOARD_HOST,
            "--port",
            _TERMINAL_DASHBOARD_PORT,
        ],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )


if __name__ == "__main__":
    from src.strategies.registry import STRATEGIES
    strategy_list = ", ".join(STRATEGIES.keys())

    parser = argparse.ArgumentParser(
        description="Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download", nargs="+",
                        help="Download data for symbols via IB")
    parser.add_argument("--backtest", action="store_true",
                        help="Run single-asset backtest")
    parser.add_argument("--wfo", action="store_true",
                        help="Run Walk-Forward Optimization")
    parser.add_argument("--strategy", type=str, default="sma",
                        help=f"Strategy name ({strategy_list})")
    parser.add_argument("--portfolio-backtest", action="store_true",
                        help="Run multi-strategy portfolio backtest")
    parser.add_argument("--portfolio-config", type=str,
                        default="src/backtest_engine/portfolio_layer/portfolio_config_example.yaml",
                        help="Path to YAML portfolio config")
    parser.add_argument("--dashboard", action="store_true",
                        help=(
                            "Launch terminal dashboard. "
                            "When combined with --backtest/--portfolio-backtest, "
                            "opens AFTER the backtest completes. "
                            "Standalone: 'python run.py --dashboard'."
                        ))
    parser.add_argument("--results-subdir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-run-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-type", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-params-json", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # ── Mode: dashboard only (no backtest) ────────────────────────────────────
    if args.dashboard and not args.backtest and not args.portfolio_backtest and not args.wfo:
        _launch_dashboard()
        sys.exit(0)

    # ── Mode: data download ───────────────────────────────────────────────────
    from src.backtest_engine.settings import BacktestSettings
    settings = BacktestSettings()

    if args.download:
        from src.data import IBFetcher
        print("=" * 60)
        print(f"  Downloading data: {args.download}")
        print("=" * 60)
        fetcher = IBFetcher(settings=settings)
        for sym in args.download:
            fetcher.fetch_all_timeframes(sym)
        print("Download complete.")

    # ── Mode: single backtest ─────────────────────────────────────────────────
    if args.backtest:
        from cli.single import run as run_backtest
        run_backtest(args.strategy, settings)
        if args.dashboard:
            _launch_dashboard()

    # ── Mode: WFO ─────────────────────────────────────────────────────────────
    if args.wfo:
        from cli.wfo import run as run_wfo
        run_wfo(args.strategy, settings)

    # ── Mode: portfolio backtest ───────────────────────────────────────────────
    if getattr(args, "portfolio_backtest", False):
        from cli.portfolio import run as run_portfolio
        run_portfolio(
            args.portfolio_config,
            results_subdir=args.results_subdir,
            scenario_id=args.scenario_id,
            baseline_run_id=args.baseline_run_id,
            scenario_type=args.scenario_type,
            scenario_params_json=args.scenario_params_json,
        )
        if args.dashboard:
            _launch_dashboard()
