from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import yaml

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.core.paths import (
    get_project_root as resolve_project_root,
    get_results_dir as resolve_results_dir,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.scenario_engine import (
    ArtifactFamily,
    BaselineReference,
    ExecutionMutation,
    JobType,
    MarketDataMutation,
    OutputSummary,
    ReproducibilityMetadata,
    ScenarioFamily,
    ScenarioSpec,
    build_artifact_manifest,
    get_artifact_run_root,
)
from src.backtest_engine.settings import BacktestSettings


@dataclass(frozen=True)
class PreparedScenarioExecution:
    """
    Captures the normalized scenario execution inputs before the child run starts.

    Methodology:
        Preparation stays explicit so job metadata, artifact manifests, and the
        child CLI invocation all consume the same normalized bundle.
    """

    run_identifier: str
    scenario_root: Path
    scenario_artifacts_dir: Path
    scenario_config_path: Path
    scenario_spec: ScenarioSpec
    artifact_manifest: Dict[str, Any]
    child_payload: Dict[str, Any]
    command: List[str]
    env: Dict[str, str]


def get_project_root() -> Path:
    """Returns the shared project root from the neutral path helper."""
    return resolve_project_root()


def get_results_dir() -> Path:
    """Returns the shared results root from the neutral path helper."""
    return resolve_results_dir()


def get_scenarios_root() -> Path:
    """Returns the shared scenario-results root under `results/scenarios/`."""
    root = get_results_dir() / ArtifactFamily.SCENARIOS.value
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_simulation_analysis_root() -> Path:
    """Returns the reserved root for future simulation-analysis artifacts."""
    root = get_results_dir() / ArtifactFamily.SIMULATION_ANALYSIS.value
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


def _build_run_identifier(artifact_family: ArtifactFamily) -> str:
    """
    Builds a collision-resistant artifact identifier for one scenario-family run.

    Methodology:
        Run identifiers stay family-aware so future simulation-analysis jobs can
        share the same helper without colliding in the artifact namespace.
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    prefix = "simulation" if artifact_family == ArtifactFamily.SIMULATION_ANALYSIS else "scenario"
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


def _compute_config_hash(config_path: Path) -> str:
    """Returns the SHA-256 digest for one source portfolio config file."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Writes one JSON payload with stable indentation for debugging and diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_stress_scenario_spec(
    bundle: ResultBundle,
    stress: StressMultipliers,
    settings: Optional[BacktestSettings] = None,
) -> ScenarioSpec:
    """
    Adapts the existing stress-slider inputs into the neutral scenario contract.

    Methodology:
        Plan A keeps the current UI payload working by translating it once into
        a typed scenario specification before queueing or execution begins.
    """

    resolved_settings = settings or BacktestSettings()
    source_config_path = resolve_portfolio_config_path(bundle)
    manifest = bundle.manifest or {}
    return ScenarioSpec(
        name=f"stress-rerun-{get_baseline_run_id(bundle)}",
        job_type=JobType.STRESS_RERUN,
        scenario_family=ScenarioFamily.EXECUTION_SHOCK,
        artifact_family=ArtifactFamily.SCENARIOS,
        market_data_mutation=MarketDataMutation(
            regime_label="volatility_shift",
            volatility_multiplier=float(stress.volatility),
        ),
        execution_mutation=ExecutionMutation(
            commission_rate=float(resolved_settings.commission_rate) * float(stress.commission),
            spread_mode=str(resolved_settings.spread_mode),
            spread_base_ticks=max(
                0,
                int(round(float(resolved_settings.spread_ticks) * float(stress.slippage))),
            ),
            vol_step_pct=float(resolved_settings.spread_volatility_step_pct),
            step_multiplier=float(resolved_settings.spread_step_multiplier),
            latency_ms=int(resolved_settings.scenario_engine.default_latency_ms),
        ),
        reproducibility=ReproducibilityMetadata(
            input_contract_version=resolved_settings.scenario_engine.scenario_contract_version,
            baseline_run_id=get_baseline_run_id(bundle),
            source_config_path=str(source_config_path),
            config_hash=str(manifest.get("config_hash") or _compute_config_hash(source_config_path)),
            data_version=str(manifest.get("data_version") or ""),
        ),
    )


def _coerce_scenario_spec(
    bundle: ResultBundle,
    scenario_input: Union[ScenarioSpec, StressMultipliers],
) -> ScenarioSpec:
    """
    Normalizes legacy stress-multiplier inputs into the typed scenario contract.

    Methodology:
        The Streamlit dashboard still exists during the migration window, so the
        runner preserves backward compatibility by adapting the older slider
        payload at the execution boundary instead of forcing every caller to
        know about the new typed contract immediately.
    """

    if isinstance(scenario_input, ScenarioSpec):
        return scenario_input
    return build_stress_scenario_spec(bundle=bundle, stress=scenario_input)


def _write_scenario_config(
    source_config_path: Path,
    target_path: Path,
    market_data_mutation: MarketDataMutation,
) -> float:
    """
    Writes a derived portfolio config for a prepared scenario rerun.

    Methodology:
        Plan A keeps the engine hot loop intact by materializing config-level
        overrides before the child backtest starts. Replay windows are passed
        separately through the scenario payload because the engine already
        supports direct date filters.
    """

    with source_config_path.open(encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh) or {}

    scenario_config = copy.deepcopy(raw_config)
    portfolio_cfg = scenario_config.setdefault("portfolio", {})
    base_target_vol = float(portfolio_cfg.get("target_portfolio_vol", 0.10))
    portfolio_cfg["target_portfolio_vol"] = (
        base_target_vol * float(market_data_mutation.volatility_multiplier)
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8")
    return base_target_vol


def _build_child_scenario_payload(
    scenario_spec: ScenarioSpec,
    artifact_manifest: Dict[str, Any],
    settings: BacktestSettings,
    base_target_vol: float,
) -> Dict[str, Any]:
    """Builds the normalized child payload passed through `--scenario-params-json`."""

    spread_ticks = scenario_spec.execution_mutation.spread_base_ticks or 0
    commission_rate = scenario_spec.execution_mutation.commission_rate or 0.0
    base_spread_ticks = max(1, int(settings.spread_ticks))
    base_commission_rate = float(settings.commission_rate) or 1.0
    return {
        "scenario_spec": scenario_spec.model_dump(mode="json", exclude_none=True),
        "artifact_manifest": artifact_manifest,
        "preview_control_values": {
            "volatility_multiplier": float(scenario_spec.market_data_mutation.volatility_multiplier),
            "slippage_multiplier": float(spread_ticks) / float(base_spread_ticks),
            "commission_multiplier": float(commission_rate) / float(base_commission_rate),
        },
        "rerun_interpretation": {
            "target_portfolio_vol": float(base_target_vol)
            * float(scenario_spec.market_data_mutation.volatility_multiplier),
            "commission_rate": commission_rate,
            "spread_ticks": spread_ticks,
            "spread_mode": scenario_spec.execution_mutation.spread_mode,
        },
    }


def _build_execution_env(settings: BacktestSettings, scenario_spec: ScenarioSpec) -> Dict[str, str]:
    """Builds child-process environment overrides from the execution contract."""

    env = os.environ.copy()
    execution = scenario_spec.execution_mutation
    if execution.commission_rate is not None:
        env["QUANT_BACKTEST_COMMISSION_RATE"] = str(execution.commission_rate)
    if execution.spread_base_ticks is not None:
        env["QUANT_BACKTEST_SPREAD_TICKS"] = str(execution.spread_base_ticks)
    if execution.spread_mode:
        env["QUANT_BACKTEST_SPREAD_MODE"] = execution.spread_mode
    if execution.vol_step_pct is not None:
        env["QUANT_BACKTEST_SPREAD_VOLATILITY_STEP_PCT"] = str(execution.vol_step_pct)
    if execution.step_multiplier is not None:
        env["QUANT_BACKTEST_SPREAD_STEP_MULTIPLIER"] = str(execution.step_multiplier)
    env.setdefault(
        "QUANT_BACKTEST_SCENARIO_CONTRACT_VERSION",
        settings.scenario_engine.scenario_contract_version,
    )
    return env


def _prepare_portfolio_scenario(
    bundle: ResultBundle,
    scenario_spec: ScenarioSpec,
) -> PreparedScenarioExecution:
    """
    Prepares the normalized scenario bundle before the child backtest starts.

    Methodology:
        Preparation validates the contract, materializes config overrides, and
        writes the root manifest before execution so failures remain inspectable.
    """

    if bundle.run_type != "portfolio":
        raise ValueError("Scenario reruns are only supported for portfolio bundles.")
    compatibility = bundle.compatibility
    if compatibility is not None and not compatibility.is_rerunnable:
        raise ValueError(compatibility.reason)
    if scenario_spec.job_type == JobType.SIMULATION:
        raise NotImplementedError("Simulation execution is reserved for a later plan.")

    settings = BacktestSettings()
    source_config_path = Path(scenario_spec.reproducibility.source_config_path)
    run_identifier = _build_run_identifier(scenario_spec.artifact_family)
    scenario_root = get_artifact_run_root(
        results_dir=get_results_dir(),
        artifact_family=scenario_spec.artifact_family,
        run_identifier=run_identifier,
    )
    scenario_artifacts_dir = scenario_root / "portfolio"
    scenario_config_path = scenario_root / "scenario_portfolio_config.yaml"
    base_target_vol = _write_scenario_config(
        source_config_path=source_config_path,
        target_path=scenario_config_path,
        market_data_mutation=scenario_spec.market_data_mutation,
    )
    artifact_manifest_model = build_artifact_manifest(
        spec=scenario_spec,
        run_identifier=run_identifier,
        baseline_reference=BaselineReference(
            run_id=scenario_spec.baseline_run_id,
            source_config_path=str(source_config_path),
        ),
    )
    artifact_manifest = artifact_manifest_model.model_dump(mode="json", exclude_none=True)
    child_payload = _build_child_scenario_payload(
        scenario_spec=scenario_spec,
        artifact_manifest=artifact_manifest,
        settings=settings,
        base_target_vol=base_target_vol,
    )
    _write_json(scenario_root / "scenario_manifest.json", artifact_manifest)
    command = [
        sys.executable,
        "run.py",
        "--portfolio-backtest",
        "--portfolio-config",
        str(scenario_config_path),
        "--results-subdir",
        str(scenario_artifacts_dir),
        "--scenario-id",
        run_identifier,
        "--baseline-run-id",
        scenario_spec.baseline_run_id,
        "--scenario-type",
        scenario_spec.job_type.value,
        "--scenario-params-json",
        json.dumps(child_payload),
    ]
    return PreparedScenarioExecution(
        run_identifier=run_identifier,
        scenario_root=scenario_root,
        scenario_artifacts_dir=scenario_artifacts_dir,
        scenario_config_path=scenario_config_path,
        scenario_spec=scenario_spec,
        artifact_manifest=artifact_manifest,
        child_payload=child_payload,
        command=command,
        env=_build_execution_env(settings=settings, scenario_spec=scenario_spec),
    )


def _finalize_prepared_scenario(prepared: PreparedScenarioExecution) -> None:
    """Merges the normalized scenario manifest into the final artifact manifest."""

    final_manifest_path = prepared.scenario_artifacts_dir / "manifest.json"
    if not final_manifest_path.exists():
        raise RuntimeError("Scenario rerun completed without a portfolio manifest.")

    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    artifact_names = final_manifest.get("artifacts", [])
    artifact_paths = [
        str((prepared.scenario_artifacts_dir / artifact_name).resolve())
        for artifact_name in artifact_names
        if isinstance(artifact_name, str)
    ]
    finalized_root_manifest = build_artifact_manifest(
        spec=prepared.scenario_spec,
        run_identifier=prepared.run_identifier,
        baseline_reference=BaselineReference(
            run_id=prepared.scenario_spec.baseline_run_id,
            source_config_path=prepared.scenario_spec.reproducibility.source_config_path,
        ),
        output_summary=OutputSummary(
            output_artifact_path=str(prepared.scenario_artifacts_dir.resolve()),
            artifact_paths=artifact_paths,
        ),
    ).model_dump(mode="json", exclude_none=True)
    finalized_root_manifest["generated_at"] = final_manifest.get(
        "generated_at",
        finalized_root_manifest["generated_at"],
    )
    final_manifest.update(finalized_root_manifest)
    final_manifest["scenario_type"] = prepared.scenario_spec.job_type.value
    final_manifest_path.write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    _write_json(prepared.scenario_root / "scenario_manifest.json", finalized_root_manifest)


def run_portfolio_scenario(
    bundle: ResultBundle,
    scenario_spec: Union[ScenarioSpec, StressMultipliers],
    timeout_seconds: Optional[int] = None,
) -> Path:
    """
    Launches one prepared portfolio rerun into the scenario artifact namespace.

    Args:
        bundle: Loaded portfolio artifact bundle used as the rerun baseline.
        scenario_spec: Typed scenario contract to execute, or legacy stress
            multipliers adapted for compatibility during the UI migration.
        timeout_seconds: Optional subprocess timeout for async worker control.

    Returns:
        Path to the scenario results root that contains `.run_type` and the
        `portfolio/` artifact folder.
    """

    resolved_spec = _coerce_scenario_spec(bundle=bundle, scenario_input=scenario_spec)
    prepared = _prepare_portfolio_scenario(bundle=bundle, scenario_spec=resolved_spec)
    result = subprocess.run(
        prepared.command,
        cwd=str(get_project_root()),
        env=prepared.env,
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
    _finalize_prepared_scenario(prepared)
    return prepared.scenario_root


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
    return str(scenario_manifest.get("baseline_run_id", "")) == get_baseline_run_id(
        baseline_bundle
    )
