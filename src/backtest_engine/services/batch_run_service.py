"""
Lightweight multi-scenario single-strategy batch backtesting service.

Methodology:
    This workflow deliberately bypasses the heavy terminal dashboard and the
    artifact exporter.  Each worker runs one standard ``BacktestEngine``
    scenario, extracts only the requested KPIs plus the equity curve, and the
    parent process renders one Matplotlib popup when the batch completes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from src.backtest_engine.engine import BacktestEngine
from src.backtest_engine.services.batch_models import BatchScenario, SingleBatchResult
from src.backtest_engine.services.batch_plot_service import show_single_batch_plot
from src.backtest_engine.services.run_helpers import validate_cache_requirements_or_exit
from src.backtest_engine.settings import BacktestSettings
from src.strategies.registry import load_strategy_by_id, resolve_strategy_id


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    """Normalizes symbol inputs to uppercase, preserving order."""
    return [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]


def _normalize_timeframes(timeframes: Sequence[str]) -> List[str]:
    """Normalizes timeframe inputs, preserving order."""
    return [str(timeframe).strip() for timeframe in timeframes if str(timeframe).strip()]


def _build_scenarios(
    strategy_names: Sequence[str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> List[BatchScenario]:
    """
    Creates the Cartesian product of requested strategies, symbols, and timeframes.

    Args:
        strategy_names: Raw CLI strategy identifiers.
        symbols: Raw CLI symbols.
        timeframes: Raw CLI timeframe values.

    Returns:
        Ordered list of batch scenarios.
    """
    normalized_strategies = [resolve_strategy_id(name) for name in strategy_names]
    normalized_symbols = _normalize_symbols(symbols)
    normalized_timeframes = _normalize_timeframes(timeframes)
    return [
        BatchScenario(strategy_id=strategy_id, symbol=symbol, timeframe=timeframe)
        for strategy_id in normalized_strategies
        for symbol in normalized_symbols
        for timeframe in normalized_timeframes
    ]


def _validate_strategies_or_exit(strategy_names: Sequence[str]) -> None:
    """Fails fast when any requested strategy ID or alias is unknown."""
    try:
        for strategy_name in strategy_names:
            load_strategy_by_id(resolve_strategy_id(strategy_name))
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(1)


def _resolve_max_workers(
    scenario_count: int,
    settings: BacktestSettings,
    requested_workers: Optional[int],
) -> int:
    """Bounds worker count to a safe, useful range."""
    cpu_count = os.cpu_count() or int(settings.batch_max_workers)
    base_workers = int(requested_workers or settings.batch_max_workers)
    return max(1, min(base_workers, scenario_count, cpu_count))


def _build_capped_log_equity_curve(
    total_values: Sequence[float],
    initial_capital: float,
    floor_pct: float,
    ruin_equity_ratio: float,
) -> np.ndarray:
    """
    Converts portfolio values into a log-equity curve capped at the loss floor.

    Methodology:
        Batch plots should remain readable even when a strategy drives account
        equity below zero.  Once the account hits the configured floor, the
        displayed curve is pinned to a fixed positive surrogate so the log axis
        stays finite and other scenarios remain visible.

    Args:
        total_values: Portfolio total-value history.
        initial_capital: Starting capital used for normalization.
        floor_pct: Minimum allowed displayed total-return percentage.
        ruin_equity_ratio: Positive surrogate ratio used for log plotting at the floor.

    Returns:
        Numpy array with log-equity values ready for plotting.
    """
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive.")
    if ruin_equity_ratio <= 0.0:
        raise ValueError("ruin_equity_ratio must be positive for log plotting.")

    total_value_array = np.asarray(total_values, dtype=float)
    raw_return_ratio = (total_value_array / float(initial_capital)) - 1.0
    capped_return_ratio = np.maximum(raw_return_ratio, floor_pct / 100.0)
    display_equity_ratio = np.maximum(capped_return_ratio + 1.0, ruin_equity_ratio)
    return np.log(display_equity_ratio)


def _render_progress_bar(
    current: int,
    total: int,
    width: int,
    label: str,
    start_time: float,
) -> None:
    """
    Legacy shim kept so that the single-scenario fast path still compiles.
    The multi-scenario path now uses tqdm directly.
    """
    elapsed = time.monotonic() - start_time
    avg_sec = elapsed / current if current > 0 else 0.0
    safe_total = max(total, 1)
    remaining = avg_sec * (safe_total - current)

    def _fmt_t(secs: float) -> str:
        m, s = divmod(int(secs), 60)
        return f"{m:02d}:{s:02d}"

    timing = f"[{_fmt_t(elapsed)}<{_fmt_t(remaining)}, {avg_sec:.2f}s/it]"
    percent = f"{current / safe_total:3.0%}"
    filled = int(width * current / safe_total)
    bar = "█" * filled + "-" * (width - filled)
    message = f"\r[Batch] {percent}|{bar}| {current}/{total} {label} {timing}"
    sys.stdout.write(message)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _run_batch_worker(
    scenario: BatchScenario,
    settings_payload: Dict[str, Any],
) -> SingleBatchResult:
    """
    Executes one lightweight single-strategy batch scenario in a worker process.

    Args:
        scenario: Scenario definition.
        settings_payload: Serialized base settings from the parent process.

    Returns:
        Serializable batch result for chart rendering.
    """
    settings = BacktestSettings(**settings_payload).model_copy(
        update={
            "default_symbol": scenario.symbol,
            "low_interval": scenario.timeframe,
        }
    )

    try:
        strategy_class = load_strategy_by_id(scenario.strategy_id)
        engine = BacktestEngine(settings=settings)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            engine.run(strategy_class)

        history = engine.portfolio.get_history_df()
        if history.empty:
            return SingleBatchResult(
                scenario=scenario,
                status="failed",
                error="No portfolio history produced.",
            )

        metrics = engine.analytics.calculate_metrics(history, engine.execution.trades)
        total_values = history["total_value"].astype(float)
        floor_pct = float(settings.batch_equity_floor_pct)
        log_equity = _build_capped_log_equity_curve(
            total_values=total_values.to_numpy(dtype=float),
            initial_capital=float(settings.initial_capital),
            floor_pct=floor_pct,
            ruin_equity_ratio=float(settings.batch_plot_ruin_equity_ratio),
        )

        return SingleBatchResult(
            scenario=scenario,
            status="completed",
            timestamps=history.index.to_list(),
            log_equity=log_equity.tolist(),
            pnl_pct=max(float(metrics.get("Total Return", 0.0) * 100.0), floor_pct),
            max_drawdown_pct=max(float(metrics.get("Max Drawdown", 0.0) * 100.0), floor_pct),
            sharpe_ratio=float(metrics.get("Sharpe Ratio", 0.0)),
        )
    except Exception as exc:
        return SingleBatchResult(
            scenario=scenario,
            status="failed",
            error=str(exc),
        )


def run_batch_backtests(
    strategy_names: Sequence[str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
    settings: BacktestSettings,
    max_workers: Optional[int] = None,
) -> None:
    """
    Runs many independent single-strategy backtests and renders one popup chart.

    Methodology:
        The service validates all required caches upfront, then distributes
        independent scenarios across worker processes.  Only compact metrics and
        normalized equity paths are returned to the parent process.

    Args:
        strategy_names: Strategy identifiers or accepted aliases.
        symbols: One or many futures symbols.
        timeframes: One or many timeframe strings.
        settings: Base backtest settings.
        max_workers: Optional worker override.
    """
    _validate_strategies_or_exit(strategy_names)
    scenarios = _build_scenarios(strategy_names, symbols, timeframes)
    if not scenarios:
        print("[Batch] No scenarios requested.")
        return

    validate_cache_requirements_or_exit(
        requirements=[(scenario.symbol, scenario.timeframe) for scenario in scenarios],
        settings=settings,
    )

    resolved_workers = _resolve_max_workers(
        scenario_count=len(scenarios),
        settings=settings,
        requested_workers=max_workers,
    )
    settings_payload = settings.model_dump(mode="python")

    print("=" * 72)
    print("  Lightweight Batch Backtest")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Workers   : {resolved_workers}")
    print("=" * 72)

    results: List[SingleBatchResult] = []

    bar = tqdm(
        total=len(scenarios),
        desc="[Batch]",
        unit="scenario",
        ncols=80,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n}/{total} {postfix} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    if len(scenarios) == 1:
        result = _run_batch_worker(scenarios[0], settings_payload)
        results.append(result)
        bar.set_postfix_str(scenarios[0].legend_label)
        bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = {
                executor.submit(_run_batch_worker, scenario, settings_payload): scenario
                for scenario in scenarios
            }
            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = SingleBatchResult(
                        scenario=scenario,
                        status="failed",
                        error=str(exc),
                    )
                results.append(result)
                bar.set_postfix_str(scenario.legend_label)
                bar.update(1)

    bar.close()

    successful_results = [
        result for result in results if result.status == "completed" and result.log_equity
    ]
    failed_results = [result for result in results if result.status != "completed"]

    if successful_results:
        show_single_batch_plot(
            results=successful_results,
            figure_width=float(settings.batch_plot_figure_width),
            figure_height=float(settings.batch_plot_figure_height),
        )
    else:
        print("[Batch] No successful scenarios to plot.")

    if failed_results:
        print("[Batch] Failed scenarios:")
        for result in failed_results:
            print(f"  - {result.scenario.legend_label}: {result.error}")
