from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from src.backtest_engine.analytics.terminal_ui.app import create_terminal_dashboard_app
from src.backtest_engine.analytics.terminal_ui.cache import (
    TerminalCachePolicy,
    TerminalCacheService,
)
from src.backtest_engine.analytics.terminal_ui.jobs import ScenarioJobStore


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
    assert len(listed) == 1
    assert listed[0].job_id == saved.job_id


def test_operations_panel_renders_monitor_and_backlog(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
    seed_scenario_job: Callable[..., object],
) -> None:
    """The operations tab should expose job monitoring and simulation backlog scope."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    seed_scenario_job(results_root, status="running")

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/partials/bottom-panel?tab=operations")

    assert response.status_code == 200
    assert "Scenario Operations" in response.text
    assert "Active Job" in response.text
    assert "Recent Jobs" in response.text
    assert "Simulation Backlog" in response.text
    assert "scenario-job-seeded" in response.text


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
    assert stream_response.status_code == 200
    assert "event: status" in stream_response.text
    assert seeded.job_id in stream_response.text
    assert '"status": "completed"' in stream_response.text
