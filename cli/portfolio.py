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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.data.data_lake import DataLake


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


def _parse_scenario_params_payload(
    scenario_params_json: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Parses the optional scenario payload passed through the CLI boundary.

    Methodology:
        The CLI accepts a JSON string because the child backtest process is
        launched through subprocess. The payload is normalized here once so the
        engine and manifest logic can consume a consistent dictionary shape.
    """

    if not scenario_params_json:
        return None
    try:
        parsed = json.loads(scenario_params_json)
    except json.JSONDecodeError as exc:
        print(f"[Portfolio] Invalid scenario params JSON: {exc}")
        sys.exit(1)
    return parsed if isinstance(parsed, dict) else {"payload": parsed}


def _parse_optional_datetime(raw_value: Any) -> Optional[datetime]:
    """Parses one optional ISO timestamp used by replay-window filters."""

    if raw_value in (None, ""):
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _resolve_replay_window_filters(
    scenario_params: Optional[Dict[str, Any]],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Resolves engine date filters from the typed scenario payload.

    Methodology:
        Replay windows are expressed in the scenario contract rather than hidden
        inside YAML so scenario preparation can restrict the engine input span
        without changing the public CLI flags in Plan A.
    """

    if not scenario_params:
        return None, None
    replay_window = None
    artifact_manifest = scenario_params.get("artifact_manifest")
    if isinstance(artifact_manifest, dict):
        selection_metadata = artifact_manifest.get("selection_metadata")
        if isinstance(selection_metadata, dict):
            replay_window = selection_metadata.get("replay_window")
    if replay_window is None:
        scenario_spec = scenario_params.get("scenario_spec")
        if isinstance(scenario_spec, dict):
            replay_window = scenario_spec.get("replay_window")
    if not isinstance(replay_window, dict):
        return None, None
    date_range = replay_window.get("date_range")
    if not isinstance(date_range, dict):
        return None, None
    return (
        _parse_optional_datetime(date_range.get("start")),
        _parse_optional_datetime(date_range.get("end")),
    )


def _merge_scenario_manifest_metadata(
    manifest_metadata: Dict[str, Any],
    scenario_params: Optional[Dict[str, Any]],
) -> None:
    """Promotes normalized scenario manifest fields into the saved portfolio manifest."""

    if not scenario_params:
        return
    artifact_manifest = scenario_params.get("artifact_manifest")
    if not isinstance(artifact_manifest, dict):
        return
    for field_name in (
        "artifact_family",
        "artifact_version",
        "job_type",
        "scenario_family",
        "simulation_family",
        "baseline_reference",
        "input_contract",
        "execution_contract",
        "reproducibility",
        "selection_metadata",
    ):
        value = artifact_manifest.get(field_name)
        if value is not None:
            manifest_metadata[field_name] = value


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
        (commission_rate, spread_ticks, spread_mode, initial_capital,
        kill-switch thresholds) are read from BacktestSettings (settings.py).

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
    scenario_params = _parse_scenario_params_payload(scenario_params_json)
    start_date, end_date = _resolve_replay_window_filters(scenario_params)

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
    engine = PortfolioBacktestEngine(
        config,
        settings=settings,
        start_date=start_date,
        end_date=end_date,
    )
    engine.run()

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
        "spread_mode": settings.spread_mode,
        "spread_ticks": settings.spread_ticks,
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
    _merge_scenario_manifest_metadata(manifest_metadata=manifest_metadata, scenario_params=scenario_params)

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
