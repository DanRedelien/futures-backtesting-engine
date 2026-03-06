"""
src/backtest_engine/analytics/exporter.py

Backtest Results Exporter.

Responsibility:
    Persists backtest artifacts to results/ so the Streamlit dashboard
    can load them independently of the engine run.

Workflow (quant standard):
    1. engine.run()         — event loop
    2. exporter.save()      — write Parquet + JSON
    3. streamlit dashboard  — reads files, renders charts

Files written per run:
    results/history.parquet   — portfolio equity curve (indexed by timestamp)
    results/trades.parquet    — closed trade log
    results/benchmark.parquet — buy-and-hold price series (optional)
    results/report.txt        — verbatim terminal report string
    results/metrics.json      — KPIs as a JSON object
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest_engine.settings import get_settings


def save_backtest_results(
    history: pd.DataFrame,
    trades: Optional[List[Any]],
    report_str: str,
    metrics: Dict[str, float],
    benchmark: Optional[pd.Series] = None,
) -> Path:
    """
    Saves all backtest artifacts to the configured results directory.

    Methodology:
        Uses Parquet for columnar time-series storage (fast load, typed).
        Metrics are serialised as JSON so the dashboard can rebuild KPI cards
        without re-running analytics.  The verbatim report string is stored as
        plain text to guarantee the console log and the Streamlit panel are
        byte-for-byte identical.

    Args:
        history: Portfolio equity curve DataFrame (index = timestamp).
        trades: List of Trade objects from ExecutionHandler.
        report_str: Pre-formatted terminal report from get_full_report_str().
        metrics: KPI dict from calculate_metrics().
        benchmark: Optional buy-and-hold close price Series.

    Returns:
        Path to the results directory where files were written.
    """
    results_dir: Path = get_settings().get_results_path()

    # Equity curve
    if not history.empty:
        history.to_parquet(results_dir / "history.parquet")

    # Trades
    if trades:
        rows = []
        for t in trades:
            if isinstance(t, dict):
                rows.append(t)
            else:
                rows.append({
                    "entry_time":  getattr(t, "entry_time",  None),
                    "exit_time":   getattr(t, "exit_time",   None),
                    "direction":   getattr(t, "direction",   ""),
                    "pnl":         getattr(t, "pnl",         0.0),
                    "exit_reason": getattr(t, "exit_reason", ""),
                })
        pd.DataFrame(rows).to_parquet(results_dir / "trades.parquet", index=False)

    # Benchmark
    if benchmark is not None:
        benchmark.to_frame(name="close").to_parquet(results_dir / "benchmark.parquet")

    # Text report
    (results_dir / "report.txt").write_text(report_str, encoding="utf-8")

    # Metrics JSON (numpy scalars must be cast for JSON serialisation)
    json_metrics: Dict[str, Any] = {
        k: float(v) if isinstance(v, (float, int)) else str(v)
        for k, v in metrics.items()
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(json_metrics, indent=2), encoding="utf-8"
    )

    # Run type marker for the dashboard
    base_results_dir = get_settings().base_dir / get_settings().results_dir
    (base_results_dir / ".run_type").write_text("single", encoding="utf-8")

    print(f"[Exporter] Results saved → {results_dir}")
    return results_dir
