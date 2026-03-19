from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cli.portfolio import _resolve_replay_window_filters
from src.backtest_engine.analytics.dashboard.core import scenario_runner
from src.backtest_engine.settings import BacktestSettings
from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.core.scenario_runner import build_stress_scenario_spec
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.scenario_engine import (
    ArtifactFamily,
    BaselineReference,
    DateRange,
    ExecutionMutation,
    JobType,
    MarketDataMutation,
    ReplaySelectionMethod,
    ReplayWindowSelection,
    ReproducibilityMetadata,
    ScenarioFamily,
    ScenarioSpec,
    build_artifact_manifest,
    get_progress_stages,
)


def test_build_stress_scenario_spec_adapts_slider_payload(tmp_path: Path) -> None:
    """Stress-slider inputs should be normalized into the typed scenario contract."""

    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("portfolio:\n  target_portfolio_vol: 0.10\n", encoding="utf-8")
    bundle = ResultBundle(
        run_type="portfolio",
        history=pd.DataFrame({"total_value": []}),
        trades=pd.DataFrame(),
        manifest={
            "run_id": "baseline-001",
            "source_config_path": str(config_path),
            "config_hash": "abc123",
            "data_version": "deadbeef12345678",
            "run_seed": 42,
        },
    )

    spec = build_stress_scenario_spec(
        bundle=bundle,
        stress=StressMultipliers(volatility=2.0, slippage=3.0, commission=2.0),
    )

    assert spec.job_type == JobType.STRESS_RERUN
    assert spec.scenario_family == ScenarioFamily.EXECUTION_SHOCK
    assert spec.artifact_family == ArtifactFamily.SCENARIOS
    assert spec.market_data_mutation.volatility_multiplier == 2.0
    assert spec.execution_mutation.commission_rate == 5.0
    assert spec.execution_mutation.spread_base_ticks == 3
    assert spec.reproducibility.input_contract_version == "scenario-spec.v1"
    assert spec.reproducibility.baseline_run_id == "baseline-001"


def test_run_portfolio_scenario_accepts_legacy_stress_multipliers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Legacy Streamlit callers should still reach the runner via StressMultipliers."""

    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("portfolio:\n  target_portfolio_vol: 0.10\n", encoding="utf-8")
    bundle = ResultBundle(
        run_type="portfolio",
        history=pd.DataFrame({"total_value": []}),
        trades=pd.DataFrame(),
        manifest={
            "run_id": "baseline-001",
            "source_config_path": str(config_path),
            "config_hash": "abc123",
            "data_version": "deadbeef12345678",
            "run_seed": 42,
        },
    )
    prepared = scenario_runner.PreparedScenarioExecution(
        run_identifier="scenario-001",
        scenario_root=tmp_path / "results" / "scenarios" / "scenario-001",
        scenario_artifacts_dir=tmp_path / "results" / "scenarios" / "scenario-001" / "portfolio",
        scenario_config_path=tmp_path / "results" / "scenarios" / "scenario-001" / "scenario_portfolio_config.yaml",
        scenario_spec=build_stress_scenario_spec(
            bundle=bundle,
            stress=StressMultipliers(volatility=2.0, slippage=3.0, commission=2.0),
        ),
        artifact_manifest={},
        child_payload={},
        command=["python", "run.py"],
        env={},
    )
    captured: dict[str, object] = {}

    def _fake_prepare_portfolio_scenario(*, bundle: ResultBundle, scenario_spec: ScenarioSpec):
        captured["bundle"] = bundle
        captured["scenario_spec"] = scenario_spec
        return prepared

    monkeypatch.setattr(scenario_runner, "_prepare_portfolio_scenario", _fake_prepare_portfolio_scenario)
    monkeypatch.setattr(
        scenario_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(scenario_runner, "_finalize_prepared_scenario", lambda prepared: None)
    monkeypatch.setattr(scenario_runner, "get_project_root", lambda: tmp_path)

    scenario_root = scenario_runner.run_portfolio_scenario(
        bundle=bundle,
        scenario_spec=StressMultipliers(volatility=2.0, slippage=3.0, commission=2.0),
    )

    assert scenario_root == prepared.scenario_root
    resolved_spec = captured["scenario_spec"]
    assert isinstance(resolved_spec, ScenarioSpec)
    assert resolved_spec.job_type == JobType.STRESS_RERUN
    assert resolved_spec.market_data_mutation.volatility_multiplier == 2.0


def test_build_artifact_manifest_preserves_replay_window_metadata() -> None:
    """Artifact manifests should persist replay-window selection metadata explicitly."""

    spec = ScenarioSpec(
        name="manual-replay",
        job_type=JobType.MARKET_REPLAY,
        scenario_family=ScenarioFamily.MARKET_REPLAY,
        artifact_family=ArtifactFamily.SCENARIOS,
        market_data_mutation=MarketDataMutation(regime_label="replay", volatility_multiplier=1.0),
        execution_mutation=ExecutionMutation(spread_mode="adaptive_volatility", spread_base_ticks=2),
        replay_window=ReplayWindowSelection(
            date_range=DateRange(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
            selection_method=ReplaySelectionMethod.MANUAL,
            selection_reason="Operator selected a harsh local window.",
        ),
        reproducibility=ReproducibilityMetadata(
            input_contract_version="scenario-spec.v1",
            baseline_run_id="baseline-001",
            source_config_path="C:/tmp/portfolio.yaml",
        ),
    )

    manifest = build_artifact_manifest(
        spec=spec,
        run_identifier="scenario-001",
        baseline_reference=BaselineReference(
            run_id="baseline-001",
            source_config_path="C:/tmp/portfolio.yaml",
        ),
    )

    replay_window = manifest.selection_metadata["replay_window"]
    assert manifest.job_type == "market_replay"
    assert manifest.scenario_id == "scenario-001"
    assert replay_window["selection_method"] == "manual"
    assert replay_window["date_range"]["start"].startswith("2024-01-01")


def test_progress_stage_contract_is_normalized_for_stress_jobs() -> None:
    """Scenario jobs should expose the canonical seven-stage lifecycle."""

    stages = get_progress_stages(JobType.STRESS_RERUN)

    assert len(stages) == 7
    assert stages[0].stage_id.value == "load_baseline"
    assert stages[-1].stage_id.value == "finalize_metadata"
    assert stages[-1].stage_count == 7


def test_build_artifact_manifest_honors_configured_version(monkeypatch) -> None:
    """Artifact manifests should use the configured scenario artifact version."""

    monkeypatch.setenv(
        "QUANT_BACKTEST_SCENARIO_ENGINE",
        '{"scenario_artifact_version":"9.9"}',
    )
    spec = ScenarioSpec(
        name="manual-replay",
        job_type=JobType.MARKET_REPLAY,
        scenario_family=ScenarioFamily.MARKET_REPLAY,
        artifact_family=ArtifactFamily.SCENARIOS,
        market_data_mutation=MarketDataMutation(regime_label="replay", volatility_multiplier=1.0),
        execution_mutation=ExecutionMutation(spread_mode="adaptive_volatility", spread_base_ticks=2),
        reproducibility=ReproducibilityMetadata(
            input_contract_version="scenario-spec.v1",
            baseline_run_id="baseline-001",
            source_config_path="C:/tmp/portfolio.yaml",
        ),
    )

    assert BacktestSettings().scenario_engine.scenario_artifact_version == "9.9"
    manifest = build_artifact_manifest(
        spec=spec,
        run_identifier="scenario-001",
        baseline_reference=BaselineReference(
            run_id="baseline-001",
            source_config_path="C:/tmp/portfolio.yaml",
        ),
    )

    assert manifest.artifact_version == "9.9"


def test_simulation_family_stays_freeform_metadata() -> None:
    """Simulation subtype metadata should remain a string, not a fixed enum."""

    spec = ScenarioSpec(
        name="future-simulation",
        job_type=JobType.SIMULATION,
        scenario_family=ScenarioFamily.SIMULATION,
        artifact_family=ArtifactFamily.SIMULATION_ANALYSIS,
        reproducibility=ReproducibilityMetadata(
            input_contract_version="scenario-spec.v1",
            baseline_run_id="baseline-001",
            source_config_path="C:/tmp/portfolio.yaml",
        ),
        simulation_family="custom-sampler-family",
    )

    assert spec.simulation_family == "custom-sampler-family"


def test_resolve_replay_window_filters_reads_typed_payload() -> None:
    """The portfolio CLI should translate replay metadata into engine date filters."""

    start_date, end_date = _resolve_replay_window_filters(
        {
            "artifact_manifest": {
                "selection_metadata": {
                    "replay_window": {
                        "date_range": {
                            "start": "2024-01-01T00:00:00+00:00",
                            "end": "2024-02-01T00:00:00+00:00",
                        }
                    }
                }
            }
        }
    )

    assert start_date == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert end_date == datetime(2024, 2, 1, tzinfo=timezone.utc)
