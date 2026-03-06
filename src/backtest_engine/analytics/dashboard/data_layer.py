"""
src/backtest_engine/analytics/dashboard/data_layer.py

Unified Data Transfer Object (DTO) for backtest results.

Responsibility: Loads, parses, and standardizes artifacts from the `results/`
directory, automatically handling both Single-Asset and Portfolio runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest_engine.analytics.dashboard.components import get_results_dir


@dataclass
class ResultBundle:
    """Unified DTO containing all loaded artifacts for the dashboard."""
    run_type: str                   # 'single' or 'portfolio'
    
    # Core time-series
    history: pd.DataFrame           # Equity curve + slot_N_pnl columns
    trades: pd.DataFrame            # All completed trades
    
    # Optional time-series
    benchmark: Optional[pd.DataFrame] = None
    exposure: Optional[pd.DataFrame] = None
    strategy_pnl_daily: Optional[pd.DataFrame] = None
    
    # Metadata and Reporting
    metrics: Dict[str, Any] = None
    report: str = ""
    slots: Dict[str, str] = None    # {slot_id: strategy_name}


def load_result_bundle(results_dir: Optional[Path] = None) -> Optional[ResultBundle]:
    """
    Auto-detects run type and loads all artifacts into a unified ResultBundle.

    Methodology:
        Checks modification times of `history.parquet` in both `results/` 
        and `results/portfolio/`. Whichever is newer dictates whether we 
        are visualizing a Single-Asset or Portfolio backtest.

    Args:
        results_dir: Optional override for testing. Defaults to project `results/`.

    Returns:
        ResultBundle if artifacts are found, None otherwise.
    """
    base_dir = results_dir or get_results_dir()
    
    marker_path = base_dir / ".run_type"
    portfolio_dir = base_dir / "portfolio"
    
    if marker_path.exists():
        mode = marker_path.read_text(encoding="utf-8").strip()
    else:
        # Fallback to single if marker is missing
        mode = "single"

    if mode == "portfolio" and (portfolio_dir / "manifest.json").exists():
        run_type = "portfolio"
        target_dir = portfolio_dir
        manifest = json.loads((portfolio_dir / "manifest.json").read_text(encoding="utf-8"))
        slots = manifest.get("slots", {})
    else:
        run_type = "single"
        target_dir = base_dir
        slots = {}

    # Required artifacts
    history_path = target_dir / "history.parquet"
    trades_path  = target_dir / "trades.parquet"
    
    if not history_path.exists() or not trades_path.exists():
        return None

    history = pd.read_parquet(history_path)
    trades  = pd.read_parquet(trades_path)

    # Standardize index types for timeline joins
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
        
    if "exit_time" in trades.columns:
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])

    # Optional artifacts
    bundle = ResultBundle(run_type=run_type, history=history, trades=trades, slots=slots)

    # Load report
    report_path = target_dir / "report.txt"
    if report_path.exists():
        bundle.report = report_path.read_text(encoding="utf-8")

    # Load metrics
    metrics_path = target_dir / "metrics.json"
    if metrics_path.exists():
        bundle.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Load exposure
    exposure_path = target_dir / "exposure.parquet"
    if exposure_path.exists():
        bundle.exposure = pd.read_parquet(exposure_path)
        if not isinstance(bundle.exposure.index, pd.DatetimeIndex):
            bundle.exposure.index = pd.to_datetime(bundle.exposure.index)

    # Load strategy daily PnL (mostly specific to portfolio)
    spd_path = target_dir / "strategy_pnl_daily.parquet"
    if spd_path.exists():
        bundle.strategy_pnl_daily = pd.read_parquet(spd_path)
        if not isinstance(bundle.strategy_pnl_daily.index, pd.DatetimeIndex):
            bundle.strategy_pnl_daily.index = pd.to_datetime(bundle.strategy_pnl_daily.index)

    # Load benchmark (usually specific to single-asset right now)
    bench_path = target_dir / "benchmark.parquet"
    if bench_path.exists():
        bundle.benchmark = pd.read_parquet(bench_path)
        if not isinstance(bundle.benchmark.index, pd.DatetimeIndex):
            bundle.benchmark.index = pd.to_datetime(bundle.benchmark.index)

    return bundle
