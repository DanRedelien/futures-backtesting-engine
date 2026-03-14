"""
cli/portfolio.py

Portfolio backtest CLI handler.

Responsibility: Parse the YAML portfolio config and run PortfolioBacktestEngine.
This module is called exclusively by run.py --portfolio-backtest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DATA_VERSION_DIGEST_LENGTH = 16


def _compute_data_version(
    data_lake: "DataLake",
    requirements: List[Tuple[str, str]],
) -> str:
    """
    Builds a lightweight cache fingerprint for rerun compatibility metadata.

    Methodology:
        This uses a pragmatic provenance marker that changes when any
        required cached input file changes. Using ordered cache file mtimes is
        sufficient for the rerun guard without broadening the current scope.

    Args:
        data_lake: Cache resolver used for required market data inputs.
        requirements: Required `(symbol, timeframe)` pairs for the run.

    Returns:
        Short SHA-256 digest representing the current cache state.
    """
    digest = hashlib.sha256()
    for symbol, timeframe in sorted(requirements):
        cache_file = data_lake._get_cache_file(symbol, timeframe)
        if cache_file.exists():
            digest.update(f"{symbol}:{timeframe}:{cache_file.stat().st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:_DATA_VERSION_DIGEST_LENGTH]


def run(
    config_path: str,
    results_subdir: Optional[str] = None,
    scenario_id: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    scenario_type: Optional[str] = None,
    scenario_params_json: Optional[str] = None,
) -> None:
    """
    Loads a YAML portfolio config and runs PortfolioBacktestEngine.

    Methodology:
        Reads the YAML once with safe_load.  Portfolio-specific fields
        (target_portfolio_vol, vol_lookback_bars, max_contracts_per_slot,
        rebalance_frequency) come from the YAML.  Shared execution settings
        (commission_rate, max_slippage_ticks, initial_capital, kill-switch
        thresholds) are read from BacktestSettings (settings.py).

    Args:
        config_path: Path to the YAML config file (absolute or project-relative).
        results_subdir: Optional project-relative or absolute artifact directory.
        scenario_id: Optional scenario identifier for manifest metadata.
        baseline_run_id: Optional baseline reference for scenario manifests.
        scenario_type: Optional scenario classification stored in manifest metadata.
        scenario_params_json: Optional JSON payload describing rerun parameters.
    """
    import yaml
    from src.backtest_engine.portfolio_layer.engine import PortfolioBacktestEngine
    from src.backtest_engine.portfolio_layer.domain.contracts import (
        PortfolioConfig, StrategySlot,
    )
    from src.strategies.registry import get_strategy_class_by_name
    from src.backtest_engine.settings import BacktestSettings
    from src.data.data_lake import DataLake

    project_root = Path(__file__).parent.parent
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[Portfolio] Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    portfolio_cfg = raw.get("portfolio", {})
    settings = BacktestSettings()

    slots = []
    for slot_cfg in raw["strategies"]:
        try:
            strategy_cls = get_strategy_class_by_name(slot_cfg["strategy"])
        except ValueError as e:
            print(f"[Portfolio] {e}")
            sys.exit(1)

        slots.append(StrategySlot(
            strategy_class=strategy_cls,
            symbols=slot_cfg["symbols"],
            weight=slot_cfg["weight"],
            timeframe=slot_cfg.get("timeframe", "30m"),
            params=slot_cfg.get("params", {}),
        ))

    config = PortfolioConfig(
        slots=slots,
        initial_capital=settings.initial_capital,
        rebalance_frequency=portfolio_cfg.get("rebalance_frequency", "intrabar"),
        target_portfolio_vol=portfolio_cfg.get("target_portfolio_vol", 0.10),
        vol_lookback_bars=int(portfolio_cfg.get("vol_lookback_bars", 20)),
        max_contracts_per_slot=int(portfolio_cfg.get("max_contracts_per_slot", 3)),
        benchmark_symbol=portfolio_cfg.get("benchmark_symbol", "ES") or None,
    )

    requirements: List[Tuple[str, str]] = []
    seen = set()
    for slot in slots:
        for symbol in slot.symbols:
            key = (symbol, slot.timeframe)
            if key not in seen:
                seen.add(key)
                requirements.append(key)

    data_lake = DataLake(settings)
    cache_errors = data_lake.validate_cache_requirements(requirements=requirements)
    if cache_errors:
        print("[Data] Cache freshness check failed:")
        for err in cache_errors:
            print(f"  - {err}")
        print(
            f"[Data] Update cache first. "
            f"Max allowed age: {settings.max_cache_staleness_days} days."
        )
        symbols_str = " ".join(sorted({symbol for symbol, _ in requirements}))
        print(f"[Data] Example: python run.py --download {symbols_str}")
        sys.exit(1)

    data_version = _compute_data_version(data_lake=data_lake, requirements=requirements)
    engine = PortfolioBacktestEngine(config, settings=settings)
    engine.run()

    scenario_params: Optional[Dict[str, Any]] = None
    if scenario_params_json:
        try:
            parsed = json.loads(scenario_params_json)
            scenario_params = parsed if isinstance(parsed, dict) else {"payload": parsed}
        except json.JSONDecodeError as exc:
            print(f"[Portfolio] Invalid scenario params JSON: {exc}")
            sys.exit(1)

    output_dir: Optional[Path] = None
    if results_subdir:
        output_dir = Path(results_subdir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    manifest_metadata: Dict[str, Any] = {
        "run_kind": "scenario" if scenario_id else "baseline",
        "source_config_path": str(cfg_path.resolve()),
        "config_hash": config_hash,
        "run_seed": settings.random_seed,
        "data_version": data_version,
    }
    if scenario_id:
        manifest_metadata["scenario_id"] = scenario_id
    if baseline_run_id:
        manifest_metadata["baseline_run_id"] = baseline_run_id
    if scenario_type:
        manifest_metadata["scenario_type"] = scenario_type
    if scenario_params is not None:
        manifest_metadata["scenario_params"] = scenario_params

    # Load benchmark price series for reporting and analytics views.
    benchmark_data = None
    if config.benchmark_symbol:
        from src.data.data_lake import DataLake
        try:
            dl = DataLake(settings)
            bdf = dl.load(config.benchmark_symbol, timeframe="30m")
            if not bdf.empty:
                benchmark_data = bdf[["close"]]
        except Exception as exc:
            print(f"[Portfolio] Benchmark load failed ({exc}), skipping.")

    engine.show_results(
        benchmark=benchmark_data,
        output_dir=output_dir,
        manifest_metadata=manifest_metadata,
    )
