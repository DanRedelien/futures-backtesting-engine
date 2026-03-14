from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

try:
    _redis_module = importlib.import_module("redis")
    _redis_exceptions_module = importlib.import_module("redis.exceptions")
    Redis = _redis_module.Redis
    RedisError = _redis_exceptions_module.RedisError
except Exception:  # pragma: no cover - optional dependency import safety
    Redis = None  # type: ignore[assignment]

    class RedisError(Exception):
        """Fallback Redis error when the redis package is unavailable."""

try:
    _rq_module = importlib.import_module("rq")
    Queue = _rq_module.Queue
    Retry = _rq_module.Retry
except Exception:  # pragma: no cover - optional dependency import safety
    Queue = None  # type: ignore[assignment]
    Retry = None  # type: ignore[assignment]

from src.backtest_engine.analytics.dashboard.core.data_layer import (
    ResultBundle,
    load_result_bundle_uncached,
)
from src.backtest_engine.analytics.dashboard.core.paths import get_results_dir
from src.backtest_engine.analytics.dashboard.core.scenario_runner import (
    get_baseline_run_id,
    run_portfolio_scenario,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers


ScenarioJobStatus = Literal["queued", "running", "completed", "failed", "timeout"]
FINAL_SCENARIO_JOB_STATES = {"completed", "failed", "timeout"}


@dataclass(frozen=True)
class TerminalQueueConfig:
    """Execution policy for terminal-driven async scenario jobs."""

    redis_url: Optional[str]
    queue_name: str
    timeout_seconds: int
    max_retries: int
    sse_max_updates_per_second: float


@dataclass
class ScenarioJobMetadata:
    """Persistent metadata for one queued or completed scenario rerun."""

    job_id: str
    status: ScenarioJobStatus
    created_at: str
    baseline_results_dir: str
    baseline_run_id: str
    scenario_type: str
    scenario_params: Dict[str, Any]
    timeout_seconds: int
    max_retries: int
    failure_state: str
    queue_name: str
    progress_current: int = 0
    progress_total: int = 0
    progress_message: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: Optional[float] = None
    output_artifact_path: str = ""
    artifact_paths: List[str] = field(default_factory=list)
    rq_job_id: str = ""
    last_error: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        """Returns JSON-safe metadata for UI responses and SSE events."""
        data = asdict(self)
        total = max(0, int(self.progress_total))
        current = max(0, int(self.progress_current))
        data["progress_pct"] = (
            round(current / total * 100.0, 1)
            if total > 0
            else 0.0
        )
        return data


class ScenarioJobStore:
    """File-backed metadata store for queued and completed scenario jobs."""

    def __init__(self, results_dir: Optional[str] = None) -> None:
        self.results_root = Path(results_dir) if results_dir is not None else get_results_dir()
        self.jobs_dir = self.results_root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        """Returns the metadata file path for one job identifier."""
        return self.jobs_dir / f"{job_id}.json"

    def save(self, metadata: ScenarioJobMetadata) -> ScenarioJobMetadata:
        """Persists one job metadata record."""
        path = self._job_path(metadata.job_id)
        path.write_text(json.dumps(metadata.to_public_dict(), indent=2), encoding="utf-8")
        return metadata

    def get(self, job_id: str) -> Optional[ScenarioJobMetadata]:
        """Loads one job metadata record by identifier."""
        path = self._job_path(job_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        raw.pop("progress_pct", None)
        return ScenarioJobMetadata(**raw)

    def list(self, limit: int = 20) -> List[ScenarioJobMetadata]:
        """Lists recent jobs newest-first."""
        records: List[ScenarioJobMetadata] = []
        for path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            job = self.get(path.stem)
            if job is not None:
                records.append(job)
            if len(records) >= limit:
                break
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records


def _utc_now_iso() -> str:
    """Returns the current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _scenario_job_id() -> str:
    """Builds a unique identifier for one async scenario job."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"scenario-job-{timestamp}-{uuid4().hex[:8]}"


def _update_job_metadata(
    store: ScenarioJobStore,
    job_id: str,
    **updates: Any,
) -> Optional[ScenarioJobMetadata]:
    """Loads, mutates, and persists one job record."""
    metadata = store.get(job_id)
    if metadata is None:
        return None
    for key, value in updates.items():
        setattr(metadata, key, value)
    return store.save(metadata)


def run_portfolio_scenario_job(
    job_id: str,
    baseline_results_dir: str,
    stress_payload: Dict[str, float],
    timeout_seconds: int,
) -> Dict[str, Any]:
    """
    Executes one queued scenario rerun inside an RQ worker process.

    Methodology:
        Workers always load the baseline bundle afresh from persisted artifacts,
        update file-backed metadata before and after the expensive subprocess
        step, and write final scenario outputs back to Parquet artifacts. Redis
        holds queue state, while durable metadata remains in `results/jobs/`.
    """
    store = ScenarioJobStore(results_dir=baseline_results_dir)
    started_at = _utc_now_iso()
    _update_job_metadata(
        store,
        job_id,
        status="running",
        started_at=started_at,
        progress_total=3,
        progress_current=1,
        progress_message="Loading baseline artifacts.",
    )

    try:
        bundle = load_result_bundle_uncached(results_dir=baseline_results_dir)
        if bundle is None or bundle.run_type != "portfolio":
            raise ValueError("Baseline portfolio artifacts are unavailable for scenario rerun.")

        stress = StressMultipliers(
            volatility=float(stress_payload["volatility"]),
            slippage=float(stress_payload["slippage"]),
            commission=float(stress_payload["commission"]),
        )

        _update_job_metadata(
            store,
            job_id,
            progress_current=2,
            progress_message="Running child portfolio backtest.",
        )
        scenario_root = run_portfolio_scenario(
            bundle=bundle,
            stress=stress,
            timeout_seconds=timeout_seconds,
        )

        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        duration_seconds = (completed_dt - started_dt).total_seconds()
        artifact_path = str((scenario_root / "portfolio").resolve())

        _update_job_metadata(
            store,
            job_id,
            status="completed",
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            progress_current=3,
            progress_message="Scenario artifacts completed.",
            output_artifact_path=artifact_path,
            artifact_paths=[artifact_path],
        )
        return {"job_id": job_id, "output_artifact_path": artifact_path}
    except subprocess.TimeoutExpired as exc:
        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        _update_job_metadata(
            store,
            job_id,
            status="timeout",
            completed_at=completed_at,
            duration_seconds=(completed_dt - started_dt).total_seconds(),
            progress_message="Scenario rerun timed out.",
            last_error=str(exc),
        )
        raise
    except Exception as exc:
        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        _update_job_metadata(
            store,
            job_id,
            status="failed",
            completed_at=completed_at,
            duration_seconds=(completed_dt - started_dt).total_seconds(),
            progress_message="Scenario rerun failed.",
            last_error=str(exc),
        )
        raise


class ScenarioJobService:
    """
    Queues and tracks scenario reruns through RQ plus Redis.

    Methodology:
        Redis and RQ hold queue semantics, while file-backed metadata remains
        the durable source for UI monitoring, SSE progress, and completed job
        inspection even after Redis TTL or worker restarts.
    """

    def __init__(
        self,
        *,
        results_dir: Optional[str],
        config: TerminalQueueConfig,
    ) -> None:
        self.results_dir = results_dir
        self.config = config
        self.store = ScenarioJobStore(results_dir=results_dir)
        self._redis_client: Optional[Redis] = None

    def list_jobs(self, limit: int = 20) -> List[ScenarioJobMetadata]:
        """Lists recent scenario jobs newest-first."""
        return self.store.list(limit=limit)

    def get_job(self, job_id: str) -> Optional[ScenarioJobMetadata]:
        """Returns one scenario job record by identifier."""
        return self.store.get(job_id)

    def queue_status(self) -> Dict[str, Any]:
        """Returns queue availability and execution policy details."""
        available = self._get_queue() is not None
        return {
            "available": available,
            "queue_name": self.config.queue_name,
            "timeout_seconds": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
            "failure_state": "failed",
            "limitations": "RQ queueing only; no DAG orchestration or distributed compute.",
        }

    def enqueue_portfolio_scenario(
        self,
        *,
        bundle: ResultBundle,
        stress: StressMultipliers,
        baseline_results_dir: Optional[str],
    ) -> ScenarioJobMetadata:
        """
        Queues one portfolio scenario rerun and persists initial job metadata.

        Args:
            bundle: Active baseline artifact bundle.
            stress: Scenario rerun multipliers from the terminal UI.
            baseline_results_dir: Results root used by the worker to reload artifacts.

        Returns:
            Newly created job metadata in `queued` or immediate `failed` state.
        """
        resolved_results_dir = str(
            Path(baseline_results_dir).resolve()
            if baseline_results_dir is not None
            else get_results_dir().resolve()
        )
        job_id = _scenario_job_id()
        metadata = ScenarioJobMetadata(
            job_id=job_id,
            status="queued",
            created_at=_utc_now_iso(),
            baseline_results_dir=resolved_results_dir,
            baseline_run_id=get_baseline_run_id(bundle),
            scenario_type="stress_rerun",
            scenario_params={
                "volatility": float(stress.volatility),
                "slippage": float(stress.slippage),
                "commission": float(stress.commission),
            },
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            failure_state="failed",
            queue_name=self.config.queue_name,
            progress_total=3,
            progress_current=0,
            progress_message="Queued for execution.",
        )
        self.store.save(metadata)

        queue = self._get_queue()
        if queue is None:
            metadata.status = "failed"
            metadata.completed_at = _utc_now_iso()
            metadata.progress_message = "Redis or RQ is unavailable for async scenario execution."
            metadata.last_error = "Queue backend unavailable."
            self.store.save(metadata)
            return metadata

        retry_policy = Retry(max=self.config.max_retries) if Retry is not None else None
        rq_job = queue.enqueue(
            run_portfolio_scenario_job,
            kwargs={
                "job_id": job_id,
                "baseline_results_dir": resolved_results_dir,
                "stress_payload": metadata.scenario_params,
                "timeout_seconds": self.config.timeout_seconds,
            },
            job_timeout=self.config.timeout_seconds,
            retry=retry_policy,
        )
        metadata.rq_job_id = str(rq_job.id)
        self.store.save(metadata)
        return metadata

    def _get_queue(self) -> Optional[Queue]:
        """Returns the configured RQ queue when Redis is reachable."""
        client = self._get_redis_client()
        if client is None or Queue is None:
            return None
        return Queue(name=self.config.queue_name, connection=client)

    def _get_redis_client(self) -> Optional[Redis]:
        """Returns a connected Redis client when configured and reachable."""
        if not self.config.redis_url or Redis is None:
            return None
        if self._redis_client is not None:
            return self._redis_client

        try:
            client = Redis.from_url(self.config.redis_url, decode_responses=True)
            client.ping()
        except (RedisError, ValueError):
            return None

        self._redis_client = client
        return self._redis_client
