from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .contracts import ArtifactFamily, ScenarioSpec


ARTIFACT_MANIFEST_VERSION = "1.0"


def resolve_artifact_manifest_version() -> str:
    """
    Resolves the active scenario artifact manifest version from settings.

    Methodology:
        Plan A added a dedicated scenario settings namespace, so manifest
        builders should honor that setting instead of silently hardcoding the
        version at emit time.
    """

    try:
        from src.backtest_engine.settings import BacktestSettings

        return str(BacktestSettings().scenario_engine.scenario_artifact_version)
    except Exception:
        return ARTIFACT_MANIFEST_VERSION


class BaselineReference(BaseModel):
    """
    Identifies the baseline artifacts used to prepare one scenario run.

    Methodology:
        Baseline references stay separate from output artifacts so later replay
        and comparison tooling can reconstruct provenance without guessing from
        directory names.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    source_config_path: str


class OutputSummary(BaseModel):
    """
    Summarizes the scenario outputs written by one completed worker run.

    Methodology:
        Output summaries stay compact and JSON-safe so job metadata and manifest
        readers do not need to scan the artifact directory on every request.
    """

    model_config = ConfigDict(frozen=True)

    output_artifact_path: str = ""
    artifact_paths: list[str] = Field(default_factory=list)


class ScenarioArtifactManifest(BaseModel):
    """
    Defines the persisted manifest contract for scenario or simulation outputs.

    Methodology:
        The manifest is intentionally richer than the current CLI scenario flags
        so future workers, loaders, and retention policies can share one durable
        contract instead of branching on ad-hoc metadata.
    """

    model_config = ConfigDict(frozen=True)

    artifact_family: ArtifactFamily
    artifact_version: str = Field(default_factory=resolve_artifact_manifest_version)
    scenario_id: Optional[str] = None
    simulation_id: Optional[str] = None
    job_type: str
    scenario_family: str
    simulation_family: Optional[str] = None
    baseline_run_id: str
    baseline_reference: BaselineReference
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_contract: dict[str, Any]
    execution_contract: dict[str, Any]
    reproducibility: dict[str, Any]
    selection_metadata: dict[str, Any] = Field(default_factory=dict)
    output_summary: OutputSummary = Field(default_factory=OutputSummary)


def get_artifact_family_root(results_dir: Path, artifact_family: ArtifactFamily) -> Path:
    """Returns the root directory for one artifact family."""

    root = results_dir / artifact_family.value
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_artifact_run_root(
    results_dir: Path,
    artifact_family: ArtifactFamily,
    run_identifier: str,
) -> Path:
    """Returns the run-specific directory for one artifact family."""

    root = get_artifact_family_root(results_dir=results_dir, artifact_family=artifact_family) / run_identifier
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_artifact_manifest(
    spec: ScenarioSpec,
    *,
    run_identifier: str,
    baseline_reference: BaselineReference,
    output_summary: Optional[OutputSummary] = None,
) -> ScenarioArtifactManifest:
    """
    Builds the persisted artifact manifest for one prepared scenario run.

    Methodology:
        The manifest stores the typed contracts as plain JSON-safe payloads so
        worker code can evolve without losing the original scenario definition.
    """

    selection_metadata: dict[str, Any] = {}
    if spec.replay_window is not None:
        selection_metadata["replay_window"] = spec.replay_window.model_dump(mode="json")

    manifest_kwargs = {
        "artifact_family": spec.artifact_family,
        "artifact_version": resolve_artifact_manifest_version(),
        "job_type": spec.job_type.value,
        "scenario_family": spec.scenario_family.value,
        "simulation_family": spec.simulation_family,
        "baseline_run_id": spec.baseline_run_id,
        "baseline_reference": baseline_reference,
        "input_contract": {
            "name": spec.name,
            "job_type": spec.job_type.value,
            "scenario_family": spec.scenario_family.value,
            "artifact_family": spec.artifact_family.value,
            "market_data_mutation": spec.market_data_mutation.model_dump(mode="json", exclude_none=True),
            "replay_window": (
                spec.replay_window.model_dump(mode="json", exclude_none=True)
                if spec.replay_window is not None
                else None
            ),
        },
        "execution_contract": spec.execution_mutation.model_dump(mode="json", exclude_none=True),
        "reproducibility": spec.reproducibility.model_dump(mode="json", exclude_none=True),
        "selection_metadata": selection_metadata,
        "output_summary": output_summary or OutputSummary(),
    }
    if spec.artifact_family == ArtifactFamily.SIMULATION_ANALYSIS:
        manifest_kwargs["simulation_id"] = run_identifier
    else:
        manifest_kwargs["scenario_id"] = run_identifier
    return ScenarioArtifactManifest(**manifest_kwargs)
