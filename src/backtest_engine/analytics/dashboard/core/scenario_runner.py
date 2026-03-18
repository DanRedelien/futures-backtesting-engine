from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.core.paths import (
    get_project_root as resolve_project_root,
    get_results_dir as resolve_results_dir,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.settings import BacktestSettings


def get_project_root() -> Path:
    """Returns the shared project root from the neutral path helper."""
    return resolve_project_root()


def get_results_dir() -> Path:
    """Returns the shared results root from the neutral path helper."""
    return resolve_results_dir()


def get_scenarios_root() -> Path:
    """Returns the shared scenario-results root under `results/scenarios/`."""
    root = get_results_dir() / "scenarios"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_baseline_run_id(bundle: ResultBundle) -> str:
    """
    Returns the baseline identifier used to link scenario artifacts back to source.

    Methodology:
        Existing baselines may not yet carry an explicit `run_id`, so the loader
        falls back to the manifest generation timestamp. Scenario manifests always
        persist the resolved identifier they were derived from.
    """
    manifest = bundle.manifest or {}
    return str(manifest.get("run_id") or manifest.get("generated_at") or "baseline")


def resolve_portfolio_config_path(bundle: ResultBundle) -> Path:
    """
    Resolves the source portfolio config path for scenario reruns.

    Methodology:
        The loader does not fall back to an example portfolio config.
        Older artifacts without reproducibility metadata stay view-only until
        their rerun contract is explicit and safe.
    """
    compatibility = bundle.compatibility
    if compatibility is not None and not compatibility.is_rerunnable:
        raise ValueError(compatibility.reason)

    manifest = bundle.manifest or {}
    source_path = manifest.get("source_config_path")
    if source_path:
        path = Path(str(source_path))
        if path.exists():
            return path

    raise ValueError(
        "Baseline artifact is view-only because `source_config_path` is missing "
        "or no longer exists."
    )


def _build_scenario_id() -> str:
    """
    Builds a collision-resistant scenario artifact identifier.

    Methodology:
        Scenario naming stays collision-resistant so concurrent or
        repeated reruns cannot collide on second-level timestamps alone.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"scenario-{timestamp}-{uuid4().hex[:8]}"


def _build_scenario_payload(
    bundle: ResultBundle,
    stress: StressMultipliers,
    settings: BacktestSettings,
    base_target_vol: float,
) -> Dict[str, Any]:
    """Builds the reproducible scenario parameter payload stored in metadata."""
    return {
        "preview_control_values": {
            "volatility_multiplier": float(stress.volatility),
            "slippage_multiplier": float(stress.slippage),
            "commission_multiplier": float(stress.commission),
        },
        "rerun_interpretation": {
            "target_portfolio_vol": float(base_target_vol) * float(stress.volatility),
            "commission_rate": float(settings.commission_rate) * float(stress.commission),
            "spread_ticks": max(0, int(round(float(settings.spread_ticks) * float(stress.slippage)))),
        },
        "baseline_reference": {
            "run_id": get_baseline_run_id(bundle),
            "source_config_path": str(resolve_portfolio_config_path(bundle)),
        },
    }


def _write_scenario_config(
    source_config_path: Path,
    target_path: Path,
    volatility_multiplier: float,
) -> float:
    """
    Writes a derived portfolio config for a real scenario rerun.

    Methodology:
        The current scenario rerun architecture interprets the portfolio
        volatility control as a multiplier on the YAML portfolio
        `target_portfolio_vol` setting. This differs intentionally from the
        fast preview transform, which only rescales realised daily PnL.
    """
    with source_config_path.open(encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}

    scenario_config = copy.deepcopy(raw_config)
    portfolio_cfg = scenario_config.setdefault("portfolio", {})
    base_target_vol = float(portfolio_cfg.get("target_portfolio_vol", 0.10))
    portfolio_cfg["target_portfolio_vol"] = base_target_vol * float(volatility_multiplier)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8")
    return base_target_vol


def run_portfolio_scenario(
    bundle: ResultBundle,
    stress: StressMultipliers,
    timeout_seconds: Optional[int] = None,
) -> Path:
    """
    Launches a real portfolio rerun into a separate scenario artifact namespace.

    Args:
        bundle: Loaded portfolio artifact bundle used as the rerun baseline.
        stress: User-selected stress multipliers for the rerun.
        timeout_seconds: Optional subprocess timeout for async worker control.

    Returns:
        Path to the scenario results root that contains `.run_type` and the
        `portfolio/` artifact folder.
    """
    if bundle.run_type != "portfolio":
        raise ValueError("Scenario reruns are only supported for portfolio bundles.")

    compatibility = bundle.compatibility
    if compatibility is not None and not compatibility.is_rerunnable:
        raise ValueError(compatibility.reason)

    settings = BacktestSettings()
    source_config_path = resolve_portfolio_config_path(bundle)
    scenario_id = _build_scenario_id()
    scenario_root = get_scenarios_root() / scenario_id
    scenario_artifacts_dir = scenario_root / "portfolio"
    scenario_config_path = scenario_root / "scenario_portfolio_config.yaml"
    base_target_vol = _write_scenario_config(
        source_config_path=source_config_path,
        target_path=scenario_config_path,
        volatility_multiplier=stress.volatility,
    )

    scenario_payload = _build_scenario_payload(
        bundle=bundle,
        stress=stress,
        settings=settings,
        base_target_vol=base_target_vol,
    )
    env = os.environ.copy()
    env["QUANT_BACKTEST_COMMISSION_RATE"] = str(
        float(settings.commission_rate) * float(stress.commission)
    )
    env["QUANT_BACKTEST_SPREAD_TICKS"] = str(
        max(0, int(round(float(settings.spread_ticks) * float(stress.slippage))))
    )

    command = [
        sys.executable,
        "run.py",
        "--portfolio-backtest",
        "--portfolio-config",
        str(scenario_config_path),
        "--results-subdir",
        str(scenario_artifacts_dir),
        "--scenario-id",
        scenario_id,
        "--baseline-run-id",
        get_baseline_run_id(bundle),
        "--scenario-type",
        "stress_rerun",
        "--scenario-params-json",
        json.dumps(scenario_payload),
    ]
    result = subprocess.run(
        command,
        cwd=str(get_project_root()),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Scenario rerun failed.\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    return scenario_root


def list_portfolio_scenarios() -> List[Dict[str, Any]]:
    """Lists available portfolio scenario artifact roots sorted newest-first."""
    scenarios: List[Dict[str, Any]] = []
    root = get_scenarios_root()
    for scenario_root in root.iterdir():
        if not scenario_root.is_dir():
            continue
        manifest_path = scenario_root / "portfolio" / "manifest.json"
        legacy_manifest_path = scenario_root / "manifest.json"
        if manifest_path.exists():
            resolved_manifest_path = manifest_path
        elif legacy_manifest_path.exists():
            resolved_manifest_path = legacy_manifest_path
        else:
            continue
        try:
            manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        scenarios.append(
            {
                "root": scenario_root,
                "manifest": manifest,
                "label": (
                    f"{manifest.get('scenario_id', scenario_root.name)}"
                    f" | {manifest.get('generated_at', 'unknown')}"
                ),
            }
        )

    scenarios.sort(key=lambda item: str(item["manifest"].get("generated_at", "")), reverse=True)
    return scenarios


def scenario_matches_baseline(baseline_bundle: ResultBundle, scenario_bundle: ResultBundle) -> bool:
    """Checks whether a scenario bundle explicitly references the active baseline."""
    scenario_manifest = scenario_bundle.manifest or {}
    return str(scenario_manifest.get("baseline_run_id", "")) == get_baseline_run_id(baseline_bundle)
