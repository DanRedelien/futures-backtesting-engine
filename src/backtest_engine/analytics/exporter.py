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

from src.backtest_engine.settings import BacktestSettings


def save_backtest_results(
    history: pd.DataFrame,
    trades: Optional[List[Any]],
    report_str: str,
    metrics: Dict[str, float],
    benchmark: Optional[pd.Series] = None,
    data_map: Optional[Dict[str, pd.DataFrame]] = None,
    settings: Optional[BacktestSettings] = None,
    strategy: Optional[Any] = None,
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
    from src.backtest_engine.analytics.exit_analysis import enrich_trades_with_exit_analytics
    
    _settings = settings or BacktestSettings()
    results_dir: Path = _settings.get_results_path()

    # Extract dynamic vol regimes if strategy is available
    vol_params = {}
    if strategy and hasattr(strategy, "config"):
        cfg = strategy.config
        vol_params = {
            "regime_window": getattr(cfg, "vol_regime_window", _settings.vol_regime_window_default),
            "history_window": getattr(cfg, "vol_history_window", _settings.vol_history_window_default),
            "vol_min_pct": getattr(cfg, "vol_min_pct", _settings.vol_min_pct_default),
            "vol_max_pct": getattr(cfg, "vol_max_pct", _settings.vol_max_pct_default),
        }

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
                    "symbol":      getattr(t, "symbol",      ""),
                    "entry_price": getattr(t, "entry_price", 0.0),
                    "exit_price":  getattr(t, "exit_price",  0.0),
                    "quantity":    getattr(t, "quantity",    0.0),
                    "commission":  getattr(t, "commission",  0.0),
                    "slippage":    getattr(t, "slippage",    0.0),
                    "entry_time":  getattr(t, "entry_time",  None),
                    "exit_time":   getattr(t, "exit_time",   None),
                    "direction":   getattr(t, "direction",   ""),
                    "pnl":         getattr(t, "pnl",         0.0),
                    "exit_reason": getattr(t, "exit_reason", ""),
                })
        
        trades_df = pd.DataFrame(rows)
        if not trades_df.empty and data_map:
            trades_df = enrich_trades_with_exit_analytics(
                trades_df, 
                data_map, 
                **vol_params
            )
            
        trades_df.to_parquet(results_dir / "trades.parquet", index=False)

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

    # Save vol_params to manifest for dashboard context
    manifest_extras = {
        "vol_regime_config": vol_params
    }

    # Run type marker for the dashboard
    base_results_dir = _settings.base_dir / _settings.results_dir
    (base_results_dir / ".run_type").write_text("single", encoding="utf-8")

    print(f"[Exporter] Results saved → {results_dir}")
    return results_dir
