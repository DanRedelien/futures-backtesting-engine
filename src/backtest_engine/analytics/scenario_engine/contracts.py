from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobType(str, Enum):
    """Enumerates the supported async scenario job families."""

    STRESS_RERUN = "stress_rerun"
    MARKET_REPLAY = "market_replay"
    TAIL_EVENT_RERUN = "tail_event_rerun"
    SIMULATION = "simulation"


class ScenarioFamily(str, Enum):
    """Categorizes the scenario intent independent of one worker job type."""

    EXECUTION_SHOCK = "execution_shock"
    MARKET_REPLAY = "market_replay"
    TAIL_EVENT = "tail_event"
    SIMULATION = "simulation"


class ArtifactFamily(str, Enum):
    """Separates deterministic scenario artifacts from future simulation outputs."""

    SCENARIOS = "scenarios"
    SIMULATION_ANALYSIS = "simulation_analysis"


class ReplaySelectionMethod(str, Enum):
    """Describes how a replay window was chosen."""

    MANUAL = "manual"
    DATASET_BOUNDED = "dataset_bounded"


class DateRange(BaseModel):
    """
    Represents one inclusive replay window for scenario execution.

    Methodology:
        Replay windows must stay explicit so future replay and ranking logic can
        be added without falling back to ad-hoc string payloads.
    """

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> "DateRange":
        """Rejects inverted date windows before execution starts."""
        if self.end < self.start:
            raise ValueError("Replay window end must be greater than or equal to start.")
        return self


class ReplayWindowSelection(BaseModel):
    """
    Captures the selected replay window and how it was chosen.

    Methodology:
        The selection method is tracked alongside the date range so future
        ranking heuristics can coexist with honest manual selection metadata.
    """

    model_config = ConfigDict(frozen=True)

    date_range: DateRange
    selection_method: ReplaySelectionMethod = ReplaySelectionMethod.MANUAL
    selection_reason: str = ""


class MarketDataMutation(BaseModel):
    """
    Describes how scenario execution mutates or constrains market inputs.

    Methodology:
        Plan A keeps this contract intentionally small: one regime-oriented
        volatility hook, one optional price-shift placeholder, and replay-window
        selection tracked separately in the top-level scenario spec.
    """

    model_config = ConfigDict(frozen=True)

    regime_label: str = ""
    volatility_multiplier: float = Field(default=1.0, ge=0.0)
    price_shift_pct: Optional[float] = None
    tail_event_label: str = ""


class ExecutionMutation(BaseModel):
    """
    Describes execution-model overrides applied before the rerun.

    Methodology:
        These fields align with the current deterministic execution and spread
        settings so Plan A can widen the contract before deeper market-model
        work exists.
    """

    model_config = ConfigDict(frozen=True)

    commission_rate: Optional[float] = Field(default=None, ge=0.0)
    spread_mode: str = ""
    spread_base_ticks: Optional[int] = Field(default=None, ge=0)
    vol_step_pct: Optional[float] = Field(default=None, ge=0.0)
    step_multiplier: Optional[float] = Field(default=None, ge=0.0)
    latency_ms: Optional[int] = Field(default=None, ge=0)


class ReproducibilityMetadata(BaseModel):
    """
    Stores the minimum metadata needed to reconstruct one scenario contract.

    Methodology:
        Scenario runs should stay tied to a baseline artifact and source config
        path even before full simulation reproducibility work lands.
    """

    model_config = ConfigDict(frozen=True)

    input_contract_version: str
    baseline_run_id: str
    source_config_path: str
    config_hash: Optional[str] = None
    data_version: Optional[str] = None
    seed: Optional[int] = None


class ScenarioSpec(BaseModel):
    """
    Defines one typed scenario contract for deterministic reruns or simulations.

    Methodology:
        A single composable contract keeps UI, workers, and manifests aligned
        while avoiding a proliferation of thin scenario-specific payload types.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    job_type: JobType
    scenario_family: ScenarioFamily
    artifact_family: ArtifactFamily = ArtifactFamily.SCENARIOS
    market_data_mutation: MarketDataMutation = Field(default_factory=MarketDataMutation)
    execution_mutation: ExecutionMutation = Field(default_factory=ExecutionMutation)
    replay_window: Optional[ReplayWindowSelection] = None
    reproducibility: ReproducibilityMetadata
    simulation_family: Optional[str] = None

    @property
    def input_contract_version(self) -> str:
        """Returns the declared input contract version for job metadata."""
        return self.reproducibility.input_contract_version

    @property
    def baseline_run_id(self) -> str:
        """Returns the baseline reference reused by queue metadata and manifests."""
        return self.reproducibility.baseline_run_id

    @property
    def seed(self) -> Optional[int]:
        """Returns the optional random seed reserved for future simulation paths."""
        return self.reproducibility.seed

    @model_validator(mode="after")
    def validate_family_consistency(self) -> "ScenarioSpec":
        """Keeps job, artifact, and scenario-family combinations internally consistent."""
        if self.job_type == JobType.SIMULATION:
            if self.artifact_family != ArtifactFamily.SIMULATION_ANALYSIS:
                raise ValueError("Simulation jobs must write to simulation_analysis artifacts.")
            if self.scenario_family != ScenarioFamily.SIMULATION:
                raise ValueError("Simulation jobs must use the simulation scenario family.")
            if not str(self.simulation_family or "").strip():
                raise ValueError("Simulation jobs must declare simulation_family metadata.")
        else:
            if self.artifact_family != ArtifactFamily.SCENARIOS:
                raise ValueError("Non-simulation jobs must write to the scenarios artifact family.")
            if self.simulation_family is not None:
                raise ValueError("simulation_family is only valid for simulation jobs.")
        return self
