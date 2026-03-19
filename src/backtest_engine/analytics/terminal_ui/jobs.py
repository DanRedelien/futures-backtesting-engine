from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional
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
    build_stress_scenario_spec,
    get_baseline_run_id,
    run_portfolio_scenario,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.scenario_engine import (
    ArtifactFamily,
    JobType,
    ProgressStageId,
    ScenarioSpec,
    build_progress_metadata,
    get_progress_stages,
)

if TYPE_CHECKING:
    from src.backtest_engine.analytics.terminal_ui.worker_manager import LocalRedisManager, LocalWorkerManager


ScenarioJobStatus = Literal["queued", "running", "completed", "failed", "timeout"]
FINAL_SCENARIO_JOB_STATES = {"completed", "failed", "timeout"}
SUPPORTED_QUEUE_JOB_TYPES: tuple[JobType, ...] = (JobType.STRESS_RERUN,)


def _resolve_redis_bindings() -> tuple[Optional[type[Any]], type[Exception]]:
    """Resolves Redis bindings dynamically so newly installed packages are picked up."""
    try:
        redis_module = importlib.import_module("redis")
        redis_exceptions_module = importlib.import_module("redis.exceptions")
        return redis_module.Redis, redis_exceptions_module.RedisError
    except Exception:
        return None, RedisError


def _resolve_rq_bindings() -> tuple[Optional[type[Any]], Optional[type[Any]]]:
    """
    Resolves RQ bindings dynamically so readiness checks are not process-stale.

    Methodology:
        rq 2.x uses multiprocessing fork at import time, which is unavailable on
        Windows. Pin rq<2.0.0 in requirements.txt to avoid this failure.
        Retry is optional — jobs queue correctly without it (no retry policy applied).
    """
    try:
        rq_module = importlib.import_module("rq")
        retry_class = getattr(rq_module, "Retry", None)
        return rq_module.Queue, retry_class
    except Exception:
        return None, None


@dataclass(frozen=True)
class TerminalQueueConfig:
    """Execution policy for terminal-driven async scenario jobs."""

    redis_url: Optional[str]
    queue_name: str
    timeout_seconds: int
    max_retries: int
    sse_max_updates_per_second: float
    worker_start_grace_seconds: float = 2.0
    worker_stop_timeout_seconds: float = 2.0


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
    job_type: str = JobType.STRESS_RERUN.value
    scenario_family: str = ""
    simulation_family: str = ""
    artifact_family: str = ArtifactFamily.SCENARIOS.value
    progress_stage_id: str = ""
    progress_stage_label: str = ""
    progress_stage_order: int = 0
    progress_stage_count: int = 0
    input_contract_version: str = ""
    seed: Optional[int] = None
    scenario_spec: Dict[str, Any] = field(default_factory=dict)
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

    def __post_init__(self) -> None:
        """Backfills compatibility fields when loading older job metadata."""
        if not self.job_type:
            self.job_type = self.scenario_type or JobType.STRESS_RERUN.value
        if not self.scenario_type:
            self.scenario_type = self.job_type
        if not self.artifact_family:
            self.artifact_family = ArtifactFamily.SCENARIOS.value
        if not self.progress_stage_count and self.progress_total:
            self.progress_stage_count = int(self.progress_total)
        if not self.progress_stage_order and self.progress_current:
            self.progress_stage_order = int(self.progress_current)

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


def _update_job_stage(
    store: ScenarioJobStore,
    job_id: str,
    *,
    job_type: str,
    stage_id: ProgressStageId,
    progress_message: str,
    **updates: Any,
) -> Optional[ScenarioJobMetadata]:
    """Applies one normalized stage transition to the persisted job record."""

    stage_updates = build_progress_metadata(job_type=job_type, stage_id=stage_id)
    stage_updates["progress_message"] = progress_message
    stage_updates.update(updates)
    return _update_job_metadata(store, job_id, **stage_updates)


def run_portfolio_scenario_job(
    job_id: str,
    baseline_results_dir: str,
    scenario_spec_payload: Dict[str, Any],
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
    scenario_spec = ScenarioSpec.model_validate(scenario_spec_payload)
    started_at = _utc_now_iso()
    _update_job_stage(
        store,
        job_id,
        job_type=scenario_spec.job_type.value,
        stage_id=ProgressStageId.LOAD_BASELINE,
        progress_message="Loading baseline artifacts.",
        status="running",
        started_at=started_at,
    )

    try:
        bundle = load_result_bundle_uncached(results_dir=baseline_results_dir)
        if bundle is None or bundle.run_type != "portfolio":
            raise ValueError("Baseline portfolio artifacts are unavailable for scenario rerun.")
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.BUILD_SCENARIO_INPUTS,
            progress_message="Validating scenario contract.",
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.PREPARE_EXECUTION_MODEL,
            progress_message="Preparing execution overrides.",
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.RUN_BACKTEST_OR_SIMULATION,
            progress_message="Running child portfolio backtest.",
        )
        scenario_root = run_portfolio_scenario(
            bundle=bundle,
            scenario_spec=scenario_spec,
            timeout_seconds=timeout_seconds,
        )

        completed_at = _utc_now_iso()
        started_dt = datetime.fromisoformat(started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        duration_seconds = (completed_dt - started_dt).total_seconds()
        artifact_path = str((scenario_root / "portfolio").resolve())
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.COMPUTE_POST_METRICS,
            progress_message="Collecting scenario output metadata.",
            output_artifact_path=artifact_path,
            artifact_paths=[artifact_path],
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.WRITE_ARTIFACTS,
            progress_message="Writing final scenario manifests.",
            output_artifact_path=artifact_path,
            artifact_paths=[artifact_path],
        )
        _update_job_stage(
            store,
            job_id,
            job_type=scenario_spec.job_type.value,
            stage_id=ProgressStageId.FINALIZE_METADATA,
            progress_message="Scenario artifacts completed.",
            status="completed",
            completed_at=completed_at,
            duration_seconds=duration_seconds,
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
        worker_manager: Optional["LocalWorkerManager"] = None,
        redis_manager: Optional["LocalRedisManager"] = None,
    ) -> None:
        self.results_dir = results_dir
        self.config = config
        self.store = ScenarioJobStore(results_dir=results_dir)
        self._redis_client: Optional[Redis] = None
        self.worker_manager = worker_manager
        self.redis_manager = redis_manager

    def list_jobs(self, limit: int = 20) -> List[ScenarioJobMetadata]:
        """Lists recent scenario jobs newest-first."""
        jobs = self.store.list(limit=limit)
        return [self._sync_job_status(job) for job in jobs]

    def get_job(self, job_id: str) -> Optional[ScenarioJobMetadata]:
        """Returns one scenario job record by identifier."""
        metadata = self.store.get(job_id)
        if metadata is None:
            return None
        return self._sync_job_status(metadata)

    def cancel_job(self, job_id: str) -> Optional["ScenarioJobMetadata"]:
        """
        Cancels a queued or running job.

        Attempts to remove the job from the Redis queue first, then marks
        the local metadata record as cancelled regardless of whether the
        Redis-side cancellation succeeded (handles stale/orphaned jobs).

        Returns the updated metadata, or None if the job does not exist.
        """
        metadata = self.store.get(job_id)
        if metadata is None:
            return None

        if metadata.rq_job_id:
            try:
                redis_client = self._get_redis_client()
                if redis_client is not None:
                    rq_module = importlib.import_module("rq")
                    rq_job = rq_module.job.Job.fetch(metadata.rq_job_id, connection=redis_client)
                    rq_job.cancel()
            except Exception:
                pass

        metadata.status = "cancelled"
        metadata.last_error = "Cancelled by user."
        self.store.save(metadata)
        return metadata

    def _worker_start_command(self) -> str:
        """Returns the expected RQ worker command for the configured terminal queue."""
        if self.worker_manager is not None:
            return self.worker_manager.snapshot().command
        base_command = f'"{sys.executable}" -m rq worker'
        if self.config.redis_url:
            return f'{base_command} --url "{self.config.redis_url}" {self.config.queue_name}'
        return f"{base_command} {self.config.queue_name}"

    def _module_readiness(self) -> Dict[str, Any]:
        """Returns Python dependency availability for the current queue backend."""
        queue_class, retry_class = _resolve_rq_bindings()
        redis_class, _redis_error_class = _resolve_redis_bindings()
        rq_installed = queue_class is not None and retry_class is not None
        redis_installed = redis_class is not None
        missing_packages: List[str] = []
        if not rq_installed:
            missing_packages.append("rq")
        if not redis_installed:
            missing_packages.append("redis")
        return {
            "rq_installed": rq_installed,
            "redis_installed": redis_installed,
            "missing_packages": missing_packages,
        }

    def _backend_readiness(self, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Returns Redis configuration and reachability state for queue execution."""
        redis_url_configured = bool(self.config.redis_url)
        redis_reachable = False
        backend_state = "not_configured"
        backend_message = (
            "Redis backend is not configured for this dashboard session."
            if not redis_url_configured
            else ""
        )
        if redis_url_configured and bool(dependencies.get("redis_installed")):
            redis_reachable = self._get_redis_client() is not None
            backend_state = "ready" if redis_reachable else "unreachable"
            backend_message = (
                "Redis backend is reachable."
                if redis_reachable
                else f"Redis is configured but unreachable at {self.config.redis_url}."
            )
        return {
            "redis_url_configured": redis_url_configured,
            "redis_reachable": redis_reachable,
            "backend_state": backend_state,
            "backend_message": backend_message,
            "redis_url": self.config.redis_url or "",
        }

    def _worker_snapshot(self) -> Dict[str, Any]:
        """Returns the managed-worker snapshot as a JSON-safe dictionary."""
        if self.worker_manager is None:
            return {
                "state": "stopped",
                "is_running": False,
                "started_by_app": False,
                "pid": None,
                "started_at": "",
                "exit_code": None,
                "last_error": "",
                "log_path": "",
                "command": self._worker_start_command(),
            }
        return self.worker_manager.snapshot().to_public_dict()

    def _redis_manager_snapshot(self) -> Dict[str, Any]:
        """Returns the managed-redis snapshot as a JSON-safe dictionary."""
        if self.redis_manager is None:
            return {
                "state": "stopped",
                "is_live": False,
                "started_by_app": False,
                "pid": None,
                "started_at": "",
                "exit_code": None,
                "last_error": "",
                "log_path": "",
                "host": "",
                "port": 0,
            }
        return self.redis_manager.snapshot().to_public_dict()

    def _missing_dependency_message(self, missing_packages: List[str]) -> str:
        """Builds a human-readable message for missing Python dependencies."""
        if not missing_packages:
            return ""
        if len(missing_packages) == 1:
            return (
                f"Background worker is unavailable because Python package "
                f"{missing_packages[0]} is not installed in this environment."
            )
        packages = ", ".join(missing_packages)
        return (
            "Background worker is unavailable because these Python packages are "
            f"missing from this environment: {packages}."
        )

    def _readiness_summary(
        self,
        dependencies: Dict[str, Any],
        backend: Dict[str, Any],
        worker: Dict[str, Any],
        redis_mgr: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Builds the user-facing readiness summary for Stress Testing."""
        missing_packages = list(dependencies.get("missing_packages", []))
        worker_state = str(worker.get("state", "stopped"))
        redis_state = str(redis_mgr.get("state", "stopped"))
        has_redis_manager = self.redis_manager is not None
        queueing_available = (
            bool(dependencies.get("rq_installed"))
            and bool(dependencies.get("redis_installed"))
            and bool(backend.get("redis_url_configured"))
            and bool(backend.get("redis_reachable"))
        )
        can_stop_redis = has_redis_manager and redis_state == "live"

        if missing_packages:
            return {
                "readiness_state": "missing_dependencies",
                "readiness_message": self._missing_dependency_message(missing_packages),
                "worker_status_label": "Install requirements first.",
                "can_start_worker": False,
                "ready_to_queue": False,
                "can_start_redis": False,
                "can_stop_redis": False,
            }
        if not bool(backend.get("redis_url_configured")):
            return {
                "readiness_state": "backend_not_configured",
                "readiness_message": "Redis URL is not configured. Set REDIS_URL in your environment or .env file.",
                "worker_status_label": "Redis not configured.",
                "can_start_worker": False,
                "ready_to_queue": False,
                "can_start_redis": False,
                "can_stop_redis": False,
            }
        if not bool(backend.get("redis_reachable")):
            if redis_state == "starting":
                return {
                    "readiness_state": "redis_starting",
                    "readiness_message": "Redis is starting. The panel will refresh automatically.",
                    "worker_status_label": "Redis is starting.",
                    "can_start_worker": False,
                    "ready_to_queue": False,
                    "can_start_redis": False,
                    "can_stop_redis": False,
                }
            can_start_redis = has_redis_manager and redis_state not in {"starting", "live"}
            return {
                "readiness_state": "backend_unreachable",
                "readiness_message": (
                    f"Redis is not running at {self.config.redis_url}."
                ),
                "worker_status_label": "Redis offline.",
                "can_start_worker": False,
                "ready_to_queue": False,
                "can_start_redis": can_start_redis,
                "can_stop_redis": False,
            }
        if worker_state == "crashed":
            return {
                "readiness_state": "worker_crashed",
                "readiness_message": (
                    str(worker.get("last_error", "")).strip()
                    or "Worker started, but exited immediately."
                ),
                "worker_status_label": "Managed worker crashed.",
                "can_start_worker": True,
                "ready_to_queue": False,
                "can_start_redis": False,
                "can_stop_redis": can_stop_redis,
            }
        if worker_state == "starting":
            return {
                "readiness_state": "worker_starting",
                "readiness_message": "Local worker is starting. The panel will refresh automatically.",
                "worker_status_label": "Managed worker is starting.",
                "can_start_worker": False,
                "ready_to_queue": False,
                "can_start_redis": False,
                "can_stop_redis": can_stop_redis,
            }
        if worker_state == "running":
            return {
                "readiness_state": "ready",
                "readiness_message": "Ready. Queue a stress test now.",
                "worker_status_label": "Managed worker is running.",
                "can_start_worker": False,
                "ready_to_queue": queueing_available,
                "can_start_redis": False,
                "can_stop_redis": can_stop_redis,
            }
        return {
            "readiness_state": "worker_stopped",
            "readiness_message": "Redis is live. Start the local worker to begin stress testing.",
            "worker_status_label": "Managed worker is stopped.",
            "can_start_worker": queueing_available,
            "ready_to_queue": False,
            "can_start_redis": False,
            "can_stop_redis": can_stop_redis,
        }

    def _sync_job_status(self, metadata: ScenarioJobMetadata) -> ScenarioJobMetadata:
        """
        Reconciles file-backed metadata with live RQ state when possible.

        Methodology:
            Job execution can fail before worker-side stage metadata is written
            (for example on platform-specific RQ runtime errors). In that case,
            the UI would otherwise keep showing "queued". This reconciliation
            keeps metadata aligned with Redis/RQ state for active jobs.
        """
        if metadata.status in FINAL_SCENARIO_JOB_STATES:
            return metadata
        if not metadata.rq_job_id:
            return metadata
        redis_client = self._get_redis_client()
        if redis_client is None:
            return metadata
        try:
            rq_module = importlib.import_module("rq")
            rq_job = rq_module.job.Job.fetch(metadata.rq_job_id, connection=redis_client)
            rq_status = str(rq_job.get_status(refresh=True) or "").strip().lower()
        except Exception:
            return metadata

        if rq_status in {"started", "busy"} and metadata.status != "running":
            metadata.status = "running"
            if not metadata.started_at:
                metadata.started_at = _utc_now_iso()
            if not metadata.progress_message:
                metadata.progress_message = "Worker picked up the job."
            self.store.save(metadata)
            return metadata

        if rq_status in {"queued", "deferred", "scheduled"}:
            return metadata

        if rq_status in {"failed", "stopped", "canceled", "cancelled"}:
            metadata.status = "failed"
            if not metadata.completed_at:
                metadata.completed_at = _utc_now_iso()
            if metadata.started_at and metadata.duration_seconds is None:
                started_dt = datetime.fromisoformat(metadata.started_at)
                completed_dt = datetime.fromisoformat(metadata.completed_at)
                metadata.duration_seconds = (completed_dt - started_dt).total_seconds()
            exc_info = str(getattr(rq_job, "exc_info", "") or "").strip()
            if exc_info:
                metadata.last_error = exc_info.splitlines()[-1][:500]
            if not metadata.last_error:
                metadata.last_error = (
                    "Worker failed before scenario stage updates were persisted. "
                    "Check results/jobs/managed-worker.log."
                )
            if metadata.progress_message in {"", "Queued for execution.", "Waiting for worker."}:
                metadata.progress_message = "Scenario rerun failed in the worker process."
            self.store.save(metadata)
            return metadata

        return metadata

    def queue_status(self) -> Dict[str, Any]:
        """Returns queue availability and execution policy details."""
        dependencies = self._module_readiness()
        backend = self._backend_readiness(dependencies)
        worker = self._worker_snapshot()
        redis_mgr = self._redis_manager_snapshot()
        readiness = self._readiness_summary(dependencies, backend, worker, redis_mgr)
        queueing_available = (
            bool(dependencies.get("rq_installed"))
            and bool(dependencies.get("redis_installed"))
            and bool(backend.get("redis_url_configured"))
            and bool(backend.get("redis_reachable"))
        )
        return {
            "available": queueing_available,
            "backend": "Redis/RQ",
            **dependencies,
            **backend,
            "queueing_available": queueing_available,
            "queue_name": self.config.queue_name,
            "timeout_seconds": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
            "failure_state": "failed",
            "supported_job_types": [job_type.value for job_type in SUPPORTED_QUEUE_JOB_TYPES],
            "worker": worker,
            "redis_manager": redis_mgr,
            "worker_start_command": self._worker_start_command(),
            "worker_refresh_interval_ms": int(max(1.0, float(self.config.worker_start_grace_seconds)) * 1000.0),
            **readiness,
        }

    def start_managed_worker(self) -> Dict[str, Any]:
        """Starts the app-owned local worker when the environment is ready."""
        status = self.queue_status()
        if not bool(status.get("can_start_worker")):
            raise RuntimeError(str(status.get("readiness_message", "Worker cannot be started right now.")))
        if self.worker_manager is None:
            raise RuntimeError("Managed worker support is unavailable in this app session.")
        return self.worker_manager.start_worker().to_public_dict()

    def start_managed_redis(self) -> Dict[str, Any]:
        """Starts the app-owned local redis-server."""
        if self.redis_manager is None:
            raise RuntimeError("Managed Redis support is unavailable in this session.")
        self._redis_client = None
        return self.redis_manager.start_redis().to_public_dict()

    def stop_managed_redis(self) -> Dict[str, Any]:
        """Stops the app-owned local redis-server."""
        if self.redis_manager is None:
            raise RuntimeError("Managed Redis support is unavailable in this session.")
        self._redis_client = None
        return self.redis_manager.stop_redis().to_public_dict()

    def _assert_publicly_queueable(self, scenario_spec: ScenarioSpec) -> None:
        """
        Rejects scenario job types that are not yet supported by the public queue surface.

        Methodology:
            Only the currently executable job types are exposed so future consumers
            cannot enqueue reserved families by accident.
        """

        if scenario_spec.job_type not in SUPPORTED_QUEUE_JOB_TYPES:
            raise NotImplementedError(
                f"Public queueing for `{scenario_spec.job_type.value}` is reserved for a later plan."
            )

    def enqueue_scenario_spec(
        self,
        *,
        bundle: ResultBundle,
        scenario_spec: ScenarioSpec,
        baseline_results_dir: Optional[str],
    ) -> ScenarioJobMetadata:
        """
        Queues one typed scenario specification through the public job service.

        Methodology:
            This is the public queue boundary for normalized scenario contracts.
            It rejects reserved job families up front so unsupported payloads do
            not persist misleading job records or reach workers accidentally.
        """

        self._assert_publicly_queueable(scenario_spec)
        resolved_results_dir = str(
            Path(baseline_results_dir).resolve()
            if baseline_results_dir is not None
            else get_results_dir().resolve()
        )
        stage_count = len(get_progress_stages(scenario_spec.job_type))
        job_id = _scenario_job_id()
        metadata = ScenarioJobMetadata(
            job_id=job_id,
            status="queued",
            created_at=_utc_now_iso(),
            baseline_results_dir=resolved_results_dir,
            baseline_run_id=get_baseline_run_id(bundle),
            scenario_type=scenario_spec.job_type.value,
            scenario_params=scenario_spec.model_dump(mode="json", exclude_none=True),
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            failure_state="failed",
            queue_name=self.config.queue_name,
            job_type=scenario_spec.job_type.value,
            scenario_family=scenario_spec.scenario_family.value,
            simulation_family=scenario_spec.simulation_family or "",
            artifact_family=scenario_spec.artifact_family.value,
            progress_stage_count=stage_count,
            input_contract_version=scenario_spec.input_contract_version,
            seed=scenario_spec.seed,
            scenario_spec=scenario_spec.model_dump(mode="json", exclude_none=True),
            progress_total=stage_count,
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

        queue_class, retry_class = _resolve_rq_bindings()
        retry_policy = retry_class(max=self.config.max_retries) if retry_class is not None else None
        rq_job = queue.enqueue(
            run_portfolio_scenario_job,
            kwargs={
                "job_id": job_id,
                "baseline_results_dir": resolved_results_dir,
                "scenario_spec_payload": metadata.scenario_spec,
                "timeout_seconds": self.config.timeout_seconds,
            },
            # rq uses signal.SIGALRM for job timeout enforcement, which does not
            # exist on Windows. job_timeout=-1 disables rq's death-penalty
            # entirely; the job function's own timeout_seconds handles subprocess limits.
            job_timeout=-1,
            retry=retry_policy,
        )
        metadata.rq_job_id = str(rq_job.id)
        self.store.save(metadata)
        return metadata

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
        scenario_spec = build_stress_scenario_spec(bundle=bundle, stress=stress)
        return self.enqueue_scenario_spec(
            bundle=bundle,
            scenario_spec=scenario_spec,
            baseline_results_dir=baseline_results_dir,
        )

    def _get_queue(self) -> Optional[Queue]:
        """Returns the configured RQ queue when Redis is reachable."""
        queue_class, _retry_class = _resolve_rq_bindings()
        client = self._get_redis_client()
        if client is None or queue_class is None:
            return None
        return queue_class(name=self.config.queue_name, connection=client)

    def _get_redis_client(self) -> Optional[Redis]:
        """Returns a connected Redis client when configured and reachable."""
        redis_class, redis_error_class = _resolve_redis_bindings()
        if not self.config.redis_url or redis_class is None:
            return None
        if self._redis_client is not None:
            return self._redis_client

        try:
            # RQ stores job data as pickled bytes. decode_responses=True causes
            # 'utf-8' codec errors when Job.fetch reads binary payloads.
            client = redis_class.from_url(self.config.redis_url, decode_responses=False)
            client.ping()
        except (redis_error_class, ValueError):
            return None

        self._redis_client = client
        return self._redis_client
