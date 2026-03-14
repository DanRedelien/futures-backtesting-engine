"""
src/backtest_engine/portfolio_layer/reporting/results.py

Portfolio result artifact serialisation.

Responsibility: Saves all artifacts to results/portfolio/:
  - history.parquet               Bar-by-bar equity curve + slot_N_pnl columns.
  - exposure.parquet              Per-bar qty + notional per (slot, symbol).
  - strategy_pnl_daily.parquet   Per-slot incremental daily PnL.
  - trades.parquet                All completed round-trip trades.
  - metrics.json                  Scalar performance metrics.
  - report.txt                    Human-readable terminal report.
  - manifest.json                 Run metadata (run_type, schema_version, artifacts).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtest_engine.analytics.artifact_contract import (
    ARTIFACT_SCHEMA_VERSION,
    build_artifact_identity,
)
from src.backtest_engine.serialization import dumps_json


# ── Versioning ─────────────────────────────────────────────────────────────────
SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = [
    "history.parquet",
    "exposure.parquet",
    "strategy_pnl_daily.parquet",
    "trades.parquet",
    "metrics.json",
    "report.txt",
    "manifest.json",
]


def _portfolio_results_dir(output_dir: Optional[Path] = None) -> Path:
    """Creates and returns the target portfolio results directory."""
    path = output_dir if output_dir is not None else Path("results") / "portfolio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_portfolio_results(
    history: pd.DataFrame,
    exposure_df: pd.DataFrame,
    slot_trades: Dict[int, List[Any]],
    report_str: str,
    metrics: Dict[str, Any],
    slot_names: Optional[Dict[int, str]] = None,
    benchmark: Optional[pd.DataFrame] = None,
    data_map: Optional[Dict[Any, pd.DataFrame]] = None,
    slot_weights: Optional[Dict[int, float]] = None,
    instrument_specs: Optional[Dict[str, Dict]] = None,
    output_dir: Optional[Path] = None,
    manifest_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Serialises all portfolio artifacts to results/portfolio/.

    Artifacts:
        history.parquet:             Bar-by-bar equity curve + per-slot PnL columns.
        exposure.parquet:            Per-bar qty and notional per (slot, symbol).
        strategy_pnl_daily.parquet:  Per-slot incremental daily PnL.
        trades.parquet:              All completed round-trip trades with slot metadata.
        metrics.json:                Scalar performance metrics dict.
        report.txt:                  Human-readable terminal report.
        manifest.json:               Run metadata for dashboard auto-detection.

    Args:
        history: DataFrame from PortfolioBook.get_history_df().
        exposure_df: DataFrame from PortfolioBook.get_exposure_df().
        slot_trades: {slot_id -> List[Trade]} from ExecutionHandlers.
        report_str: The formatted text report.
        metrics: Scalar metrics dict from PerformanceMetrics.
        slot_names: Optional {slot_id -> strategy class name} for manifest.
        instrument_specs: Optional {symbol -> {multiplier, tick_size}} override.
                          Falls back to BacktestSettings.instrument_specs.
    """
    # Load instrument specs so gross_pnl is correctly scaled by the contract
    # multiplier (e.g. 50 for ES, 100 for GC).  Without the multiplier, futures
    # P&L is systematically understated by the multiplier factor.
    _specs: Dict[str, Dict] = instrument_specs or {}
    if not _specs:
        try:
            from src.backtest_engine.settings import BacktestSettings
            _specs = BacktestSettings().instrument_specs
        except Exception:
            pass

    out = _portfolio_results_dir(output_dir=output_dir)
    saved: List[str] = []

    # 1. Equity curve (includes slot_N_pnl columns from PortfolioBook)
    history.to_parquet(out / "history.parquet")
    saved.append("history.parquet")

    # 2. Exposure: qty + notional per (slot, symbol) per bar
    if not exposure_df.empty:
        exposure_df.to_parquet(out / "exposure.parquet")
        saved.append("exposure.parquet")

    # 3. Benchmark buy-and-hold close prices
    if benchmark is not None and not benchmark.empty:
        benchmark.to_parquet(out / "benchmark.parquet")
        saved.append("benchmark.parquet")

    # 3.5. Instrument Close Prices (for Alpha/Beta calculation)
    if data_map:
        closes_dict = {}
        for key, df in data_map.items():
            # key can be symbol or (slot_id, symbol)
            # We want to extract just the symbol string to serve as the column name
            symbol = key[1] if isinstance(key, tuple) else key
            if "close" in df.columns:
                closes_dict[symbol] = df["close"]
                
        if closes_dict:
            # Drop timezone when concatenating if there's any discrepancy, or just concat
            # pandas handles Index alignment
            inst_closes_df = pd.DataFrame(closes_dict)
            if not isinstance(inst_closes_df.index, pd.DatetimeIndex):
                inst_closes_df.index = pd.to_datetime(inst_closes_df.index)
                
            # Since Alpha/Beta correlations only need daily data, we must resample the 
            # intraday close prices (potentially 100k+ bars) down to daily bars to save memory.
            inst_closes_df = inst_closes_df.resample('1D').last().dropna(how='all')
                
            inst_closes_df.to_parquet(out / "instrument_closes.parquet")
            saved.append("instrument_closes.parquet")

    # 3. Per-slot incremental daily PnL derived from end-of-day slot snapshots.
    pnl_cols = [c for c in history.columns if c.startswith("slot_") and c.endswith("_pnl")]
    if pnl_cols:
        pnl_df = history[pnl_cols].copy()
        pnl_df.index = pd.DatetimeIndex(pnl_df.index)
        pnl_snapshots = pnl_df.resample("D").last().fillna(0.0)
        pnl_daily = pnl_snapshots.diff()
        if not pnl_daily.empty:
            pnl_daily.iloc[0] = pnl_snapshots.iloc[0]
        pnl_daily = pnl_daily.fillna(0.0)
        pnl_daily.to_parquet(out / "strategy_pnl_daily.parquet")
        saved.append("strategy_pnl_daily.parquet")

    # 4. All trades (flatten across slots, enrich with slot metadata)
    all_trade_rows = []
    for slot_id, trades in slot_trades.items():
        strategy_name = (slot_names or {}).get(slot_id, f"slot_{slot_id}")
        for t in trades:
            qty       = getattr(t, "quantity", 0)
            ep        = getattr(t, "entry_price", 0.0)
            xp        = getattr(t, "exit_price", 0.0)
            comm      = getattr(t, "commission", 0.0)
            slip      = getattr(t, "slippage", 0.0)
            direction = getattr(t, "direction", "LONG")
            sign      = 1.0 if direction == "LONG" else -1.0
            symbol    = getattr(t, "symbol", "")
            # Look up the contract multiplier so gross_pnl is in real dollars.
            # Without the multiplier, one ES point shows as $1 instead of $50.
            spec       = _specs.get(symbol, {"multiplier": 1.0, "tick_size": 0.01})
            multiplier = float(spec.get("multiplier", 1.0))
            gross_pnl  = sign * (xp - ep) * abs(qty) * multiplier
            net_pnl    = getattr(t, "pnl", 0.0)
            all_trade_rows.append({
                "slot_id":       slot_id,
                "strategy":      strategy_name,
                "symbol":        symbol,
                "direction":     direction,
                "entry_time":    getattr(t, "entry_time", None),
                "exit_time":     getattr(t, "exit_time", None),
                "entry_price":   ep,
                "exit_price":    xp,
                "quantity":      qty,
                "gross_pnl":     round(gross_pnl, 2),
                "commission":    comm,
                "slippage":      slip,
                "pnl":           net_pnl,
                "exit_reason":   getattr(t, "exit_reason", ""),
            })
    if all_trade_rows:
        trades_df = pd.DataFrame(all_trade_rows)
        
        if not trades_df.empty and data_map:
            from src.backtest_engine.analytics.exit_analysis import enrich_trades_with_exit_analytics
            trades_df = enrich_trades_with_exit_analytics(trades_df, data_map)
            
        trades_df.to_parquet(out / "trades.parquet")
        saved.append("trades.parquet")

    # 5. Scalar metrics JSON
    (out / "metrics.json").write_text(
        dumps_json(metrics), encoding="utf-8"
    )
    saved.append("metrics.json")

    # 6. Human report
    (out / "report.txt").write_text(report_str, encoding="utf-8")
    saved.append("report.txt")

    # 7. Manifest — schema contract for the dashboard auto-detection
    saved.append("manifest.json")
    manifest = {
        "run_type": "portfolio",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc),
        "artifacts": saved,
        "slots": slot_names or {},
        "slot_weights": slot_weights or {},
    }
    if manifest_metadata:
        manifest.update(manifest_metadata)
    manifest.update(
        build_artifact_identity(
            run_type="portfolio",
            artifact_path=out,
            project_root=_PROJECT_ROOT,
        )
    )
    (out / "manifest.json").write_text(dumps_json(manifest), encoding="utf-8")

    # Run type marker for the dashboard
    (out.parent / ".run_type").write_text("portfolio", encoding="utf-8")

    print(f"[Portfolio Exporter] Results saved -> {out.resolve()}")
    return out
