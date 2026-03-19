from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

import pandas as pd
from fastapi.testclient import TestClient

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.scenario_engine import (
    ArtifactFamily,
    JobType,
    ReproducibilityMetadata,
    ScenarioFamily,
    ScenarioSpec,
)
from src.backtest_engine.analytics.terminal_ui.app import create_terminal_dashboard_app
from src.backtest_engine.analytics.terminal_ui.cache import (
    TerminalCachePolicy,
    TerminalCacheService,
)
from src.backtest_engine.analytics.terminal_ui.jobs import (
    ScenarioJobService,
    ScenarioJobStore,
    TerminalQueueConfig,
)
from src.backtest_engine.analytics.terminal_ui.worker_manager import LocalWorkerManager


def _queue_status_payload(
    *,
    readiness_state: str = "ready",
    readiness_message: str = "Ready. Queue a stress test now.",
    can_start_worker: bool = False,
    ready_to_queue: bool = True,
    worker_state: str = "running",
    worker_last_error: str = "",
    missing_packages: list[str] | None = None,
    redis_url_configured: bool = True,
    redis_reachable: bool = True,
    can_start_redis: bool = False,
    can_stop_redis: bool = False,
) -> dict[str, object]:
    """Builds a deterministic queue-status payload for terminal UI route tests."""

    resolved_missing_packages = list(missing_packages or [])
    return {
        "available": ready_to_queue,
        "backend": "Redis/RQ",
        "rq_installed": "rq" not in resolved_missing_packages,
        "redis_installed": "redis" not in resolved_missing_packages,
        "missing_packages": resolved_missing_packages,
        "redis_url_configured": redis_url_configured,
        "redis_reachable": redis_reachable,
        "backend_state": "ready" if redis_reachable else "unreachable",
        "backend_message": "Redis backend is reachable." if redis_reachable else "Redis backend unavailable.",
        "redis_url": "redis://127.0.0.1:6379/0" if redis_url_configured else "",
        "queueing_available": redis_reachable and not resolved_missing_packages,
        "queue_name": "terminal-scenarios",
        "timeout_seconds": 1800,
        "max_retries": 2,
        "failure_state": "failed",
        "supported_job_types": ["stress_rerun"],
        "worker": {
            "state": worker_state,
            "is_running": worker_state in {"starting", "running"},
            "started_by_app": worker_state in {"starting", "running", "crashed"},
            "pid": 12345 if worker_state in {"starting", "running"} else None,
            "started_at": "2026-03-14T00:00:00+00:00",
            "exit_code": 1 if worker_state == "crashed" else None,
            "last_error": worker_last_error,
            "log_path": "C:/tmp/managed-worker.log",
            "command": f'"{sys.executable}" -m rq worker terminal-scenarios',
        },
        "redis_manager": {
            "state": "stopped",
            "is_live": False,
            "started_by_app": False,
            "pid": None,
            "started_at": "",
            "exit_code": None,
            "last_error": "",
            "log_path": "",
            "host": "127.0.0.1",
            "port": 6379,
        },
        "worker_start_command": f'"{sys.executable}" -m rq worker terminal-scenarios',
        "worker_refresh_interval_ms": 2000,
        "readiness_state": readiness_state,
        "readiness_message": readiness_message,
        "worker_status_label": readiness_message,
        "can_start_worker": can_start_worker,
        "ready_to_queue": ready_to_queue,
        "can_start_redis": can_start_redis,
        "can_stop_redis": can_stop_redis,
    }


def test_cache_key_format() -> None:
    """Cache keys must expose metric, artifact, parameter hash, and schema."""
    cache = TerminalCacheService(
        redis_url=None,
        policy=TerminalCachePolicy(correlation_ttl_seconds=600, risk_ttl_seconds=300),
    )

    key = cache.build_cache_key(
        metric_name="corr_matrix",
        artifact_id="artifact-001",
        schema_version="1.1",
        parameters={"window": "1d", "scope": "portfolio"},
    )

    parts = key.split(":")
    assert parts[0] == "terminal"
    assert parts[1] == "corr_matrix"
    assert parts[2] == "artifact-001"
    assert len(parts[3]) == 16
    assert parts[4] == "1.1"


def test_scenario_job_store_persists_metadata(
    tmp_path: Path,
    seed_scenario_job: Callable[..., object],
) -> None:
    """Scenario job metadata should persist outside Redis for UI monitoring."""
    results_root = tmp_path / "results"
    results_root.mkdir()

    saved = seed_scenario_job(results_root)
    store = ScenarioJobStore(results_dir=str(results_root))
    loaded = store.get(saved.job_id)
    listed = store.list(limit=5)

    assert loaded is not None
    assert loaded.job_id == saved.job_id
    assert loaded.status == "completed"
    assert loaded.job_type == "stress_rerun"
    assert loaded.progress_stage_id == "finalize_metadata"
    assert len(listed) == 1
    assert listed[0].job_id == saved.job_id


def test_operations_panel_renders_monitor_and_backlog(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    seed_scenario_job: Callable[..., object],
) -> None:
    """Operations should focus on diagnostics while Stress Testing owns the launcher."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    seed_scenario_job(results_root, status="running")

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=operations")

    assert response.status_code == 200
    assert "Operations" in response.text
    assert "Launch new reruns from Stress Testing." in response.text
    assert "Expected worker command" in response.text
    assert "Active Jobs" in response.text
    assert "Recent Jobs" in response.text
    assert "scenario-job-seeded" in response.text
    assert "Queue Stress Test" not in response.text


def test_stress_testing_panel_renders_launcher_and_multiple_active_jobs(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    seed_scenario_job: Callable[..., object],
    monkeypatch,
) -> None:
    """Stress Testing should expose the launcher plus multiple concurrent job cards."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    seed_scenario_job(
        results_root,
        job_id="scenario-job-running",
        status="running",
        created_at="2026-03-14T00:00:00+00:00",
    )
    seed_scenario_job(
        results_root,
        job_id="scenario-job-queued",
        status="queued",
        created_at="2026-03-14T00:01:00+00:00",
    )
    seed_scenario_job(
        results_root,
        job_id="scenario-job-completed",
        status="completed",
        created_at="2026-03-14T00:02:00+00:00",
    )
    monkeypatch.setattr(
        ScenarioJobService,
        "queue_status",
        lambda self: _queue_status_payload(),
    )

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=stress-testing")

    assert response.status_code == 200
    assert "Stress Testing" in response.text
    assert "Queue Stress Test" in response.text
    assert "Execution Shock" in response.text
    assert "scenario-job-running" in response.text
    assert "scenario-job-queued" in response.text
    assert "scenario-job-completed" in response.text
    assert "Simulation Backlog" in response.text


def test_queue_status_only_advertises_publicly_queueable_job_types() -> None:
    """Queue capabilities should not advertise reserved job families as executable."""

    job_service = ScenarioJobService(
        results_dir=None,
        config=TerminalQueueConfig(
            redis_url=None,
            queue_name="terminal-scenarios",
            timeout_seconds=1800,
            max_retries=2,
            sse_max_updates_per_second=2.0,
        ),
    )

    queue_status = job_service.queue_status()

    assert queue_status["supported_job_types"] == ["stress_rerun"]
    assert "simulation" not in queue_status["supported_job_types"]
    assert "worker_start_command" in queue_status


def test_queue_status_reports_missing_dependencies(monkeypatch) -> None:
    """Readiness should distinguish missing Python modules from Redis reachability issues."""

    import src.backtest_engine.analytics.terminal_ui.jobs as jobs_module

    def _missing_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(jobs_module.importlib, "import_module", _missing_import)
    job_service = ScenarioJobService(
        results_dir=None,
        config=TerminalQueueConfig(
            redis_url="redis://127.0.0.1:6379/0",
            queue_name="terminal-scenarios",
            timeout_seconds=1800,
            max_retries=2,
            sse_max_updates_per_second=2.0,
        ),
    )

    queue_status = job_service.queue_status()

    assert queue_status["readiness_state"] == "missing_dependencies"
    assert queue_status["can_start_worker"] is False
    assert queue_status["ready_to_queue"] is False
    assert "rq" in queue_status["missing_packages"]
    assert "redis" in queue_status["missing_packages"]


def test_queue_status_rechecks_optional_modules_dynamically(monkeypatch) -> None:
    """Dependency readiness should re-import optional backends instead of using process-stale globals."""

    import src.backtest_engine.analytics.terminal_ui.jobs as jobs_module

    class FakeRedisClient:
        def ping(self) -> None:
            return None

    class FakeRedis:
        @classmethod
        def from_url(cls, url: str, decode_responses: bool = True) -> FakeRedisClient:
            return FakeRedisClient()

    class FakeQueue:
        def __init__(self, name: str, connection: object) -> None:
            self.name = name
            self.connection = connection

    class FakeRetry:
        def __init__(self, max: int) -> None:
            self.max = max

    def _fake_import_module(name: str) -> object:
        if name == "rq":
            return SimpleNamespace(Queue=FakeQueue, Retry=FakeRetry)
        if name == "redis":
            return SimpleNamespace(Redis=FakeRedis)
        if name == "redis.exceptions":
            return SimpleNamespace(RedisError=RuntimeError)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(jobs_module.importlib, "import_module", _fake_import_module)
    monkeypatch.setattr(jobs_module, "Queue", None)
    monkeypatch.setattr(jobs_module, "Retry", None)
    monkeypatch.setattr(jobs_module, "Redis", None)
    job_service = ScenarioJobService(
        results_dir=None,
        config=TerminalQueueConfig(
            redis_url="redis://127.0.0.1:6379/0",
            queue_name="terminal-scenarios",
            timeout_seconds=1800,
            max_retries=2,
            sse_max_updates_per_second=2.0,
        ),
    )

    queue_status = job_service.queue_status()

    assert queue_status["rq_installed"] is True
    assert queue_status["redis_installed"] is True
    assert queue_status["redis_reachable"] is True
    assert queue_status["readiness_state"] == "worker_stopped"


def test_queue_status_worker_command_includes_custom_redis_url() -> None:
    """Worker guidance should include the configured Redis URL when one is required."""

    job_service = ScenarioJobService(
        results_dir=None,
        config=TerminalQueueConfig(
            redis_url="redis://127.0.0.1:6380/5",
            queue_name="terminal-scenarios",
            timeout_seconds=1800,
            max_retries=2,
            sse_max_updates_per_second=2.0,
        ),
    )

    queue_status = job_service.queue_status()

    assert '--url "redis://127.0.0.1:6380/5"' in queue_status["worker_start_command"]
    assert queue_status["worker_start_command"].endswith(" terminal-scenarios")
    assert f'"{sys.executable}" -m rq worker' in queue_status["worker_start_command"]


def test_public_enqueue_surface_rejects_simulation_specs_before_persisting_jobs(
    tmp_path: Path,
) -> None:
    """Unsupported simulation specs should be rejected before any job record is persisted."""

    results_root = tmp_path / "results"
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
    simulation_spec = ScenarioSpec(
        name="future-simulation",
        job_type=JobType.SIMULATION,
        scenario_family=ScenarioFamily.SIMULATION,
        artifact_family=ArtifactFamily.SIMULATION_ANALYSIS,
        reproducibility=ReproducibilityMetadata(
            input_contract_version="scenario-spec.v1",
            baseline_run_id="baseline-001",
            source_config_path=str(config_path),
        ),
        simulation_family="custom-sampler-family",
    )
    job_service = ScenarioJobService(
        results_dir=str(results_root),
        config=TerminalQueueConfig(
            redis_url=None,
            queue_name="terminal-scenarios",
            timeout_seconds=1800,
            max_retries=2,
            sse_max_updates_per_second=2.0,
        ),
    )

    try:
        job_service.enqueue_scenario_spec(
            bundle=bundle,
            scenario_spec=simulation_spec,
            baseline_results_dir=str(results_root),
        )
        assert False, "Simulation specs should not be queueable through the public service."
    except NotImplementedError as exc:
        assert "reserved for a later plan" in str(exc)

    assert job_service.list_jobs(limit=5) == []


def test_stress_testing_panel_shows_install_requirements_message_when_dependencies_missing(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """The panel should tell non-technical users to install requirements before queueing."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    monkeypatch.setattr(
        ScenarioJobService,
        "queue_status",
        lambda self: _queue_status_payload(
            readiness_state="missing_dependencies",
            readiness_message="Background worker is unavailable because Python package rq is not installed in this environment.",
            can_start_worker=False,
            ready_to_queue=False,
            worker_state="stopped",
            missing_packages=["rq"],
            redis_reachable=False,
        ),
    )

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=stress-testing")

    assert response.status_code == 200
    assert "Install Packages" in response.text
    assert "Queue Stress Test" not in response.text
    assert "Start Local Worker" not in response.text


def test_stress_testing_panel_shows_start_worker_button_when_worker_stopped(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """The panel should guide users to start the local worker before queueing."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    monkeypatch.setattr(
        ScenarioJobService,
        "queue_status",
        lambda self: _queue_status_payload(
            readiness_state="worker_stopped",
            readiness_message="Queue backend is ready. Start the local worker to enable one-click stress tests.",
            can_start_worker=True,
            ready_to_queue=False,
            worker_state="stopped",
        ),
    )

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=stress-testing")

    assert response.status_code == 200
    assert "Start Local Worker" in response.text
    assert "Queue Stress Test" not in response.text


def test_stress_testing_panel_marks_starting_worker_for_auto_refresh(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """The panel should mark worker-starting state for automatic refresh instead of manual reload."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    monkeypatch.setattr(
        ScenarioJobService,
        "queue_status",
        lambda self: _queue_status_payload(
            readiness_state="worker_starting",
            readiness_message="Local worker is starting. Wait a moment before queueing a rerun.",
            can_start_worker=False,
            ready_to_queue=False,
            worker_state="starting",
        ),
    )

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=stress-testing")

    assert response.status_code == 200
    assert 'data-worker-readiness-state="worker_starting"' in response.text
    assert 'data-worker-refresh-ms="' in response.text
    assert "The panel will refresh automatically" in response.text


def test_stress_testing_panel_hides_stop_redis_button(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """Stress Testing should not expose a Redis stop button in the user-facing setup flow."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    monkeypatch.setattr(
        ScenarioJobService,
        "queue_status",
        lambda self: _queue_status_payload(
            readiness_state="worker_stopped",
            readiness_message="Redis is live. Start the local worker to begin stress testing.",
            can_start_worker=True,
            ready_to_queue=False,
            worker_state="stopped",
            can_stop_redis=True,
            redis_reachable=True,
        ),
    )

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=stress-testing")

    assert response.status_code == 200
    assert "Stop Redis" not in response.text


def test_jobs_api_and_sse_expose_persisted_job_metadata(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    seed_scenario_job: Callable[..., object],
) -> None:
    """The jobs API should expose list and SSE views over job metadata."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    seeded = seed_scenario_job(results_root, status="completed")

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    jobs_response = client.get("/api/jobs")
    stream_response = client.get(f"/api/jobs/{seeded.job_id}/events")

    assert jobs_response.status_code == 200
    payload = jobs_response.json()
    assert "queue" in payload
    assert payload["jobs"][0]["job_id"] == seeded.job_id
    assert payload["jobs"][0]["job_type"] == "stress_rerun"
    assert payload["jobs"][0]["progress_stage_id"] == "finalize_metadata"
    assert stream_response.status_code == 200
    assert "event: status" in stream_response.text
    assert seeded.job_id in stream_response.text
    assert '"status": "completed"' in stream_response.text


def test_queue_scenario_post_re_renders_stress_testing_panel(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """Queue POST should return the Stress Testing panel rather than the old Operations launcher view."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    monkeypatch.setattr(
        ScenarioJobService,
        "queue_status",
        lambda self: _queue_status_payload(),
    )
    monkeypatch.setattr(
        ScenarioJobService,
        "enqueue_portfolio_scenario",
        lambda self, **kwargs: type(
            "QueuedJob",
            (),
            {"status": "queued", "job_id": "scenario-job-queued"},
        )(),
    )
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.post(
        "/partials/queue-scenario",
        data={
            "tab": "stress-testing",
            "launch_family": "execution_shock",
            "stress_volatility": "2.0",
            "stress_slippage": "3.0",
            "stress_commission": "2.0",
        },
    )

    assert response.status_code == 200
    assert "Stress Testing" in response.text
    assert "Queue Stress Test" in response.text


def test_start_worker_post_re_renders_stress_testing_panel_with_queue_button(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """Starting the managed worker should re-render the panel into a queue-ready state."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    state = {"started": False}

    def _fake_queue_status(self) -> dict[str, object]:
        if state["started"]:
            return _queue_status_payload(
                readiness_state="ready",
                readiness_message="System is ready. You can queue a real scenario rerun now.",
                can_start_worker=False,
                ready_to_queue=True,
                worker_state="running",
            )
        return _queue_status_payload(
            readiness_state="worker_stopped",
            readiness_message="Queue backend is ready. Start the local worker to enable one-click stress tests.",
            can_start_worker=True,
            ready_to_queue=False,
            worker_state="stopped",
        )

    def _fake_start_managed_worker(self) -> dict[str, object]:
        state["started"] = True
        return _queue_status_payload(worker_state="running")["worker"]  # type: ignore[index]

    monkeypatch.setattr(ScenarioJobService, "queue_status", _fake_queue_status)
    monkeypatch.setattr(ScenarioJobService, "start_managed_worker", _fake_start_managed_worker)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.post(
        "/partials/worker/start",
        data={
            "tab": "stress-testing",
            "stress_volatility": "2.0",
            "stress_slippage": "3.0",
            "stress_commission": "2.0",
        },
    )

    assert response.status_code == 200
    assert "Local worker is running." in response.text
    assert "Queue Stress Test" in response.text


def test_dashboard_shutdown_stops_managed_worker(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    monkeypatch,
) -> None:
    """Closing the dashboard app should stop the app-owned managed worker."""

    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    stopped = {"called": False}

    def _fake_stop_worker(self) -> dict[str, object]:
        stopped["called"] = True
        return {
            "state": "stopped",
            "is_running": False,
            "started_by_app": True,
            "pid": None,
            "started_at": "",
            "exit_code": 0,
            "last_error": "",
            "log_path": "",
            "command": "",
        }

    monkeypatch.setattr(LocalWorkerManager, "stop_worker", _fake_stop_worker)

    with TestClient(create_terminal_dashboard_app(results_dir=str(results_root))) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200

    assert stopped["called"] is True
