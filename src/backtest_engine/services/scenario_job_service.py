"""
Framework-neutral scenario job queue service.

Methodology:
    Scenario job metadata, queue configuration, and the RQ-facing service
    boundary live here so the terminal UI stays a thin HTTP shell. The module
    remains the public import surface, while storage and worker concerns are
    split into adjacent helpers to keep responsibilities local and testable.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

from src.backtest_engine.analytics.shared.risk_models import StressMultipliers
from src.backtest_engine.analytics.scenario_engine import (
    ScenarioSpec,
    get_progress_stages,
)
from src.backtest_engine.services.artifact_service import ResultBundle
from src.backtest_engine.services.paths import get_results_dir
from src.backtest_engine.services.scenario_runner_service import (
    build_stress_scenario_spec,
    get_baseline_run_id,
)

from .scenario_job_models import (
    FINAL_SCENARIO_JOB_STATES,
    SUPPORTED_QUEUE_JOB_TYPES,
    ScenarioJobMetadata,
    ScenarioJobStatus,
    TerminalQueueConfig,
)
from .scenario_job_readiness import (
    build_readiness_summary,
    build_redis_manager_snapshot,
    build_worker_snapshot,
    build_worker_start_command,
)
from .scenario_job_store import ScenarioJobStore
from .scenario_job_worker import (
    _scenario_job_id,
    _update_job_metadata,
    _update_job_stage,
    _utc_now_iso,
    run_portfolio_scenario_job,
)

if TYPE_CHECKING:
    from src.backtest_engine.services.worker_manager import LocalRedisManager, LocalWorkerManager


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
        Retry is optional - jobs queue correctly without it (no retry policy applied).
    """
    try:
        rq_module = importlib.import_module("rq")
        retry_class = getattr(rq_module, "Retry", None)
        return rq_module.Queue, retry_class
    except Exception:
        return None, None


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

    def cancel_job(self, job_id: str) -> Optional[ScenarioJobMetadata]:
        """
        Cancels a queued or running job.

        Attempts to remove the job from the Redis queue first, then marks
        the local metadata record as cancelled regardless of whether the
        Redis-side cancellation succeeded.
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

        metadata.status = "cancelled"  # type: ignore[assignment]
        metadata.last_error = "Cancelled by user."
        self.store.save(metadata)
        return metadata

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

    def _sync_job_status(self, metadata: ScenarioJobMetadata) -> ScenarioJobMetadata:
        """
        Reconciles file-backed metadata with live RQ state when possible.

        Methodology:
            Job execution can fail before worker-side stage metadata is written.
            This reconciliation keeps metadata aligned with Redis/RQ state for
            active jobs without making the UI dependent on Redis durability.
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
        worker_start_command = build_worker_start_command(self.config, self.worker_manager)
        worker = build_worker_snapshot(
            worker_manager=self.worker_manager,
            worker_start_command=worker_start_command,
        )
        redis_mgr = build_redis_manager_snapshot(self.redis_manager)
        readiness = build_readiness_summary(
            config=self.config,
            redis_manager=self.redis_manager,
            dependencies=dependencies,
            backend=backend,
            worker=worker,
            redis_mgr=redis_mgr,
        )
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
            "worker_start_command": worker_start_command,
            "worker_refresh_interval_ms": int(
                max(1.0, float(self.config.worker_start_grace_seconds)) * 1000.0
            ),
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
        """Queues one typed scenario specification through the public job service."""
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

        _queue_class, retry_class = _resolve_rq_bindings()
        retry_policy = retry_class(max=self.config.max_retries) if retry_class is not None else None
        rq_job = queue.enqueue(
            run_portfolio_scenario_job,
            kwargs={
                "job_id": job_id,
                "baseline_results_dir": resolved_results_dir,
                "scenario_spec_payload": metadata.scenario_spec,
                "timeout_seconds": self.config.timeout_seconds,
            },
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
        """Queues one portfolio scenario rerun and persists initial job metadata."""
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
            client = redis_class.from_url(self.config.redis_url, decode_responses=False)
            client.ping()
        except (redis_error_class, ValueError):
            return None

        self._redis_client = client
        return self._redis_client


__all__ = [
    "FINAL_SCENARIO_JOB_STATES",
    "SUPPORTED_QUEUE_JOB_TYPES",
    "Redis",
    "RedisError",
    "Queue",
    "Retry",
    "ScenarioJobMetadata",
    "ScenarioJobService",
    "ScenarioJobStatus",
    "ScenarioJobStore",
    "TerminalQueueConfig",
    "run_portfolio_scenario_job",
    "_resolve_redis_bindings",
    "_resolve_rq_bindings",
    "_scenario_job_id",
    "_update_job_metadata",
    "_update_job_stage",
    "_utc_now_iso",
]
