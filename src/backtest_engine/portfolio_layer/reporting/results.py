"""
src/backtest_engine/portfolio_layer/reporting/results.py

Portfolio result artifact serialisation.

Responsibility: Saves all artifacts to results/portfolio/:
  - history.parquet               Bar-by-bar equity curve + slot_N_pnl columns.
  - exposure.parquet              Per-bar qty + notional per (slot, symbol).
  - strategy_pnl_daily.parquet   Per-slot daily PnL (slot_N_pnl columns).
  - trades.parquet                All completed round-trip trades.
  - metrics.json                  Scalar performance metrics.
  - report.txt                    Human-readable terminal report.
  - manifest.json                 Run metadata (run_type, schema_version, artifacts).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Versioning ─────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
ARTIFACTS = [
    "history.parquet",
    "exposure.parquet",
    "strategy_pnl_daily.parquet",
    "trades.parquet",
    "metrics.json",
    "report.txt",
    "manifest.json",
]


def _portfolio_results_dir() -> Path:
    """Creates and returns the results/portfolio directory."""
    path = Path("results") / "portfolio"
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
) -> None:
    """
    Serialises all portfolio artifacts to results/portfolio/.

    Artifacts:
        history.parquet:             Bar-by-bar equity curve + per-slot PnL columns.
        exposure.parquet:            Per-bar qty and notional per (slot, symbol).
        strategy_pnl_daily.parquet:  Per-slot daily PnL resampled to calendar days.
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
    """
    out = _portfolio_results_dir()
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

    # 3. Per-slot daily PnL (resample slot_N_pnl columns to calendar day)
    pnl_cols = [c for c in history.columns if c.startswith("slot_") and c.endswith("_pnl")]
    if pnl_cols:
        pnl_df = history[pnl_cols].copy()
        pnl_df.index = pd.DatetimeIndex(pnl_df.index)
        pnl_daily = pnl_df.resample("D").last()   # total PnL snapshot at end of day
        pnl_daily.to_parquet(out / "strategy_pnl_daily.parquet")
        saved.append("strategy_pnl_daily.parquet")

    # 4. All trades (flatten across slots, enrich with slot metadata)
    all_trade_rows = []
    for slot_id, trades in slot_trades.items():
        strategy_name = (slot_names or {}).get(slot_id, f"slot_{slot_id}")
        for t in trades:
            all_trade_rows.append({
                "slot_id":       slot_id,
                "strategy":      strategy_name,
                "symbol":        t.symbol,
                "direction":     t.direction,
                "entry_time":    t.entry_time,
                "exit_time":     t.exit_time,
                "entry_price":   t.entry_price,
                "exit_price":    t.exit_price,
                "quantity":      t.quantity,
                "pnl":           t.pnl,
                "commission":    t.commission,
                "exit_reason":   t.exit_reason,
            })
    if all_trade_rows:
        trades_df = pd.DataFrame(all_trade_rows)
        trades_df.to_parquet(out / "trades.parquet")
        saved.append("trades.parquet")

    # 5. Scalar metrics JSON
    serialisable = {
        k: (float(v) if hasattr(v, "__float__") else str(v))
        for k, v in metrics.items()
    }
    (out / "metrics.json").write_text(
        json.dumps(serialisable, indent=2), encoding="utf-8"
    )
    saved.append("metrics.json")

    # 6. Human report
    (out / "report.txt").write_text(report_str, encoding="utf-8")
    saved.append("report.txt")

    # 7. Manifest — schema contract for the dashboard auto-detection
    saved.append("manifest.json")
    manifest = {
        "run_type":       "portfolio",
        "schema_version": SCHEMA_VERSION,
        "generated_at":   datetime.now(tz=timezone.utc).isoformat(),
        "artifacts":      saved,
        "slots":          slot_names or {},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # Run type marker for the dashboard
    (out.parent / ".run_type").write_text("portfolio", encoding="utf-8")

    print(f"[Portfolio Exporter] Results saved -> {out.resolve()}")
