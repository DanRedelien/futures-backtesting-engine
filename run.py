"""
Backtesting Engine — entry point.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Install deps      pip install -r requirements.txt
  2. Download data     python run.py --download ES NQ YM RTY CL GC YM SI
  3. Run a backtest    python run.py --backtest --strategy sma --symbol ES --tf 1h
  4. Open dashboard    python run.py --dashboard
                       (or add --dashboard to any --backtest / --portfolio-backtest call)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE STRATEGIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ID                  Alias               Description
  ──────────────────  ──────────────────  ──────────────────────
  sma                 sma_crossover       Trend Following (SMA crossover)
  zscore              zscore_reversal     Mean Reversion  (Z-score)
  mean_rev            mean_reversion      Mean Reversion  (Bollinger)
  sma_pullback        —                   Trend Following (SMA pullback)
  intraday_momentum   —                   Momentum (opening range)
  stat_level          statistical_level   Statistical Support / Resistance
  ict_ob              ict_order_block     ICT Order Block

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN SYMBOLS  (pre-loaded in instrument_specs; others use generic fallback)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ES  NQ  YM  RTY   — US equity index futures
  CL  NG             — energy
  GC  SI  PL         — metals
  ZC  ZB             — grains / bonds
  6E                 — FX (EUR/USD CME)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ── Data download ─────────────────────────────────────────────────────────
  python run.py --download 6E

  ── Single backtest ───────────────────────────────────────────────────────
  python run.py --backtest --strategy stat_level --symbol ES --tf 1h --dashboard
  python run.py --backtest --strategy zscore --symbol NQ --tf 30m --dashboard

  ── Walk-Forward Optimization (single) ────────────────────────────────────
  python run.py --wfo --strategy stat_level --symbol ES --tf 1h

  ── Portfolio backtest ────────────────────────────────────────────────────
  python run.py --portfolio-backtest --dashboard
  python run.py --portfolio-backtest --portfolio-config path/to/config.yaml

  ── Batch: one strategy, many symbols / timeframes ────────────────────────
  python run.py batch --strategies sma mean_rev ict_ob zscore sma_pullback intraday_momentum stat_level --symbol CL NG ES GC NQ RTY SI YM --tf 1h
  python run.py batch --strategies sma zscore --symbol ES --tf 1h 30m

  ── WFO-Batch: full walk-forward sweep across scenarios ───────────────────
  python run.py wfo-batch --strategies sma zscore mean_rev sma_pullback intraday_momentum stat_level ict_ob --symbol ES --tf 1h
  python run.py wfo-batch --strategies sma --symbol ES NQ CL GC YM RTY --tf 1h

  ── Terminal dashboard (standalone) ───────────────────────────────────────
  python run.py --dashboard
  python run.py --dashboard --dashboard-port 8080

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Strategy IDs and aliases are interchangeable on the CLI.
  • --tf accepts the same timeframe labels used in data cache filenames (30m, 1h, 4h, 1D …).
  • batch / wfo-batch accept multiple --symbol values (space-separated) and multiple --tf values.
  • --workers N overrides the process-pool size for batch modes (default: settings.batch_max_workers).
  • Settings can be overridden via environment variables prefixed with QUANT_BACKTEST_
    or by editing a .env file in the project root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Sequence
import urllib.error
import urllib.request


_PROJECT_ROOT = Path(__file__).parent
_TERMINAL_DASHBOARD_APP = "src.backtest_engine.runtime.terminal_ui.app:app"
_TERMINAL_DASHBOARD_HOST = "127.0.0.1"
_TERMINAL_DASHBOARD_PORT = "8000"
_HEALTH_HEADER = "X-Quant-Terminal"
_HEALTH_HEADER_VALUE = "1"
_LIGHTWEIGHT_BATCH_COMMANDS = {"batch", "wfo-batch"}


def _resolve_preferred_dashboard_port(cli_port: int | None) -> int:
    """Resolves the preferred dashboard HTTP port."""
    if cli_port is not None:
        return cli_port
    raw = os.environ.get("TERMINAL_DASHBOARD_PORT", _TERMINAL_DASHBOARD_PORT)
    try:
        return int(raw)
    except ValueError:
        return int(_TERMINAL_DASHBOARD_PORT)


def _dashboard_already_running(host: str, port: int) -> bool:
    """True when this project's terminal dashboard is already listening."""
    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.headers.get(_HEALTH_HEADER) != _HEALTH_HEADER_VALUE:
                return False
            body = json.loads(resp.read().decode())
            return body.get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def _first_free_tcp_port(host: str, start: int, *, span: int = 32) -> int:
    """Returns the first free TCP port in a small Windows-safe range."""
    last_error: OSError | None = None
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError as exc:
                last_error = exc
                continue
            return port
    hint = f": {last_error}" if last_error else ""
    raise RuntimeError(
        f"No free TCP port found for terminal dashboard on {host} "
        f"(tried {start}..{start + span - 1}){hint}"
    )


def _launch_dashboard(*, dashboard_port: int | None = None) -> None:
    """
    Launches the FastAPI terminal dashboard as a child process.

    Methodology:
        The dashboard remains a separate uvicorn process.  The launcher avoids
        spawning duplicates and falls back to the next free port when needed.
    """
    host = _TERMINAL_DASHBOARD_HOST
    preferred = _resolve_preferred_dashboard_port(dashboard_port)

    if _dashboard_already_running(host, preferred):
        print(
            f"\n[Dashboard] Terminal UI already running - open "
            f"http://{host}:{preferred} (not starting a second server).\n"
        )
        return

    port = _first_free_tcp_port(host, preferred)
    if port != preferred:
        print(
            f"\n[Dashboard] Port {preferred} is in use; "
            f"binding terminal dashboard on {port} instead.\n"
        )

    print("\n[Dashboard] Launching terminal dashboard...")
    print(f"[Dashboard] URL: http://{host}:{port}\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            _TERMINAL_DASHBOARD_APP,
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )


def _build_lightweight_batch_parser(command_name: str) -> argparse.ArgumentParser:
    """
    Builds the dedicated parser for the new positional batch commands.

    Args:
        command_name: Either ``batch`` or ``wfo-batch``.

    Returns:
        Configured parser for that command.
    """
    parser = argparse.ArgumentParser(
        prog=f"python run.py {command_name}",
        description=(
            "Lightweight multi-scenario batch backtester."
            if command_name == "batch"
            else "Lightweight multi-scenario WFO batch runner."
        ),
    )
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument("--strategy", type=str, help="One strategy ID or alias")
    strategy_group.add_argument(
        "--strategies",
        nargs="+",
        help="One or many strategy IDs or aliases",
    )
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbol",
        dest="symbols",
        nargs="+",
        help="One or many futures symbols",
    )
    symbol_group.add_argument(
        "--symbols",
        dest="symbols",
        nargs="+",
        help="One or many futures symbols",
    )
    parser.add_argument(
        "--tf",
        nargs="+",
        required=True,
        metavar="TIMEFRAME",
        help="One or many timeframes (for example: 1h 30m 5m 1m)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker override for the batch process pool",
    )
    return parser


def _dispatch_lightweight_batch_command(argv: Sequence[str]) -> bool:
    """
    Dispatches the new lightweight batch commands before the legacy parser.

    Args:
        argv: Raw CLI arguments excluding the executable name.

    Returns:
        True when a lightweight command was handled.
    """
    if not argv:
        return False

    command_name = str(argv[0]).strip().lower()
    if command_name not in _LIGHTWEIGHT_BATCH_COMMANDS:
        return False

    parser = _build_lightweight_batch_parser(command_name)
    args = parser.parse_args(list(argv[1:]))

    from src.backtest_engine.settings import BacktestSettings

    settings = BacktestSettings()
    strategy_names = [args.strategy] if args.strategy else list(args.strategies or [])

    if command_name == "batch":
        from cli.batch import run as run_batch

        run_batch(
            strategy_names=strategy_names,
            symbols=args.symbols,
            timeframes=args.tf,
            settings=settings,
            max_workers=args.workers,
        )
    else:
        from cli.wfo_batch import run as run_wfo_batch

        run_wfo_batch(
            strategy_names=strategy_names,
            symbols=args.symbols,
            timeframes=args.tf,
            settings=settings,
            max_workers=args.workers,
        )

    return True


def _apply_single_mode_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    settings: "BacktestSettings",
) -> "BacktestSettings":
    """
    Applies legacy single-mode symbol/timeframe CLI overrides.

    Args:
        parser: Active argparse parser for validation failures.
        args: Parsed CLI namespace.
        settings: Base settings loaded from environment.

    Returns:
        Possibly updated settings instance.
    """
    updates: dict[str, object] = {}
    flag_symbol = str(args.symbol_override).strip() if args.symbol_override else ""
    positional_symbol = (
        str(args.symbol_positional).strip() if args.symbol_positional else ""
    )

    if flag_symbol and positional_symbol and flag_symbol.upper() != positional_symbol.upper():
        parser.error("Use either '--symbol ES' or positional 'ES', not both.")

    resolved_symbol = flag_symbol or positional_symbol
    if resolved_symbol:
        updates["default_symbol"] = resolved_symbol.upper()

    timeframes = list(args.timeframes or [])
    if timeframes:
        if len(timeframes) != 1:
            parser.error(
                "--tf accepts exactly one value for --backtest and --wfo. "
                "Use 'batch' or 'wfo-batch' for multi-timeframe sweeps."
            )
        updates["low_interval"] = str(timeframes[0]).strip()

    return settings.model_copy(update=updates) if updates else settings


if __name__ == "__main__":
    if _dispatch_lightweight_batch_command(sys.argv[1:]):
        sys.exit(0)

    from src.strategies.registry import get_strategy_ids

    strategy_list = ", ".join(get_strategy_ids(include_aliases=True))

    parser = argparse.ArgumentParser(
        description="Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--download", nargs="+", help="Download data for symbols via IB")
    parser.add_argument("--backtest", action="store_true", help="Run single-asset backtest")
    parser.add_argument("--wfo", action="store_true", help="Run Walk-Forward Optimization")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma",
        help=f"Strategy name ({strategy_list})",
    )
    parser.add_argument(
        "--symbol",
        dest="symbol_override",
        type=str,
        default=None,
        help="Override default symbol for --backtest or --wfo",
    )
    parser.add_argument(
        "--tf",
        dest="timeframes",
        nargs="+",
        default=None,
        metavar="TIMEFRAME",
        help="Override timeframe for --backtest or --wfo",
    )
    parser.add_argument(
        "--portfolio-backtest",
        action="store_true",
        help="Run multi-strategy portfolio backtest",
    )
    parser.add_argument(
        "--portfolio-config",
        type=str,
        default="src/backtest_engine/portfolio_layer/portfolio_config_example.yaml",
        help="Path to YAML portfolio config",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help=(
            "Launch terminal dashboard. "
            "When combined with --backtest/--portfolio-backtest, "
            "opens AFTER the backtest completes. "
            "Standalone: 'python run.py --dashboard'."
        ),
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        metavar="PORT",
        help=(
            "HTTP port for the terminal dashboard (default: 8000 or "
            "TERMINAL_DASHBOARD_PORT). If the port is busy, the next free port is used."
        ),
    )
    parser.add_argument("--results-subdir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-run-id", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-type", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-params-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("symbol_positional", nargs="?", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.dashboard and not args.backtest and not args.portfolio_backtest and not args.wfo:
        _launch_dashboard(dashboard_port=args.dashboard_port)
        sys.exit(0)

    from src.backtest_engine.settings import BacktestSettings

    settings = BacktestSettings()
    settings = _apply_single_mode_overrides(parser, args, settings)

    if args.download:
        from src.data import IBFetcher

        print("=" * 60)
        print(f"  Downloading data: {args.download}")
        print("=" * 60)
        fetcher = IBFetcher(settings=settings)
        for sym in args.download:
            fetcher.fetch_all_timeframes(sym)
        print("Download complete.")

    if args.backtest:
        from cli.single import run as run_backtest

        run_backtest(args.strategy, settings)
        if args.dashboard:
            _launch_dashboard(dashboard_port=args.dashboard_port)

    if args.wfo:
        from cli.wfo import run as run_wfo

        run_wfo(args.strategy, settings)

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
            _launch_dashboard(dashboard_port=args.dashboard_port)
