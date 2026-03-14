from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from src.backtest_engine.analytics.terminal_ui.jobs import (
    ScenarioJobMetadata,
    ScenarioJobStore,
)


@pytest.fixture
def make_single_bundle() -> Callable[..., None]:
    """Returns a helper that writes a minimal single-run artifact bundle."""

    def _make_single_bundle(
        root: Path,
        *,
        artifact_id: str = "single-001",
        include_marker: bool = True,
    ) -> None:
        history = pd.DataFrame(
            {"total_value": [100_000.0, 100_250.0, 100_180.0]},
            index=pd.to_datetime(
                ["2024-01-01 09:30:00", "2024-01-01 10:00:00", "2024-01-01 10:30:00"]
            ),
        )
        trades = pd.DataFrame(
            {
                "strategy": ["SmaStrategy", "SmaStrategy"],
                "symbol": ["ES", "ES"],
                "direction": ["LONG", "SHORT"],
                "entry_time": pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 10:00:00"]),
                "exit_time": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:30:00"]),
                "pnl": [250.0, -70.0],
                "mfe": [300.0, 20.0],
                "mae": [-40.0, -90.0],
                "pnl_decay_60m": [200.0, -50.0],
                "exit_reason": ["target", "stop"],
            }
        )

        root.mkdir(parents=True, exist_ok=True)
        history.to_parquet(root / "history.parquet")
        trades.to_parquet(root / "trades.parquet", index=False)
        (root / "report.txt").write_text("SINGLE REPORT", encoding="utf-8")
        (root / "metrics.json").write_text(
            json.dumps(
                {
                    "Total Return": 0.12,
                    "CAGR": 0.08,
                    "Win Rate": 0.5,
                    "Total Trades": 2,
                }
            ),
            encoding="utf-8",
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "run_type": "single",
                    "artifact_id": artifact_id,
                    "run_id": artifact_id,
                    "schema_version": "1.1",
                    "engine_version": "test-version",
                    "artifact_created_at": "2026-03-14T00:00:00+00:00",
                    "artifact_path": str(root.resolve()),
                }
            ),
            encoding="utf-8",
        )
        if include_marker:
            (root / ".run_type").write_text("single", encoding="utf-8")

    return _make_single_bundle


@pytest.fixture
def make_portfolio_bundle() -> Callable[..., None]:
    """Returns a helper that writes a minimal portfolio artifact bundle."""

    def _make_portfolio_bundle(
        root: Path,
        *,
        artifact_id: str = "portfolio-001",
        run_seed: int = 42,
    ) -> None:
        portfolio_root = root / "portfolio"
        history = pd.DataFrame(
            {
                "total_value": [1_000_000.0, 1_000_800.0, 1_001_100.0, 1_000_950.0],
                "slot_0_pnl": [0.0, 450.0, 700.0, 620.0],
                "slot_1_pnl": [0.0, 350.0, 400.0, 330.0],
            },
            index=pd.to_datetime(
                [
                    "2024-01-01 09:30:00",
                    "2024-01-01 10:00:00",
                    "2024-01-02 09:30:00",
                    "2024-01-02 10:00:00",
                ]
            ),
        )
        trades = pd.DataFrame(
            {
                "strategy": ["StrategyA", "StrategyA", "StrategyB", "StrategyB"],
                "symbol": ["ES", "ES", "NQ", "NQ"],
                "direction": ["LONG", "SHORT", "LONG", "SHORT"],
                "entry_time": pd.to_datetime(
                    [
                        "2024-01-01 09:30:00",
                        "2024-01-02 09:30:00",
                        "2024-01-01 09:30:00",
                        "2024-01-02 09:30:00",
                    ]
                ),
                "exit_time": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00",
                        "2024-01-02 10:00:00",
                        "2024-01-01 10:00:00",
                        "2024-01-02 10:00:00",
                    ]
                ),
                "pnl": [450.0, 170.0, 350.0, -70.0],
                "mfe": [500.0, 220.0, 420.0, 20.0],
                "mae": [-60.0, -30.0, -50.0, -90.0],
                "pnl_decay_60m": [410.0, 150.0, 320.0, -40.0],
                "exit_reason": ["target", "target", "target", "stop"],
            }
        )
        exposure = pd.DataFrame(
            {
                "slot_0_ES_notional": [100_000.0, 110_000.0, 105_000.0, 108_000.0],
                "slot_1_NQ_notional": [80_000.0, 82_000.0, 79_000.0, 81_000.0],
            },
            index=history.index,
        )
        benchmark = pd.DataFrame(
            {"close": [5000.0, 5010.0, 5025.0, 5020.0]},
            index=history.index,
        )

        portfolio_root.mkdir(parents=True, exist_ok=True)
        history.to_parquet(portfolio_root / "history.parquet")
        trades.to_parquet(portfolio_root / "trades.parquet", index=False)
        exposure.to_parquet(portfolio_root / "exposure.parquet")
        benchmark.to_parquet(portfolio_root / "benchmark.parquet")
        (portfolio_root / "report.txt").write_text("PORTFOLIO REPORT", encoding="utf-8")
        (portfolio_root / "metrics.json").write_text(
            json.dumps(
                {
                    "Total Return": 0.21,
                    "CAGR": 0.14,
                    "Win Rate": 0.75,
                    "Total Trades": 4,
                }
            ),
            encoding="utf-8",
        )
        config_path = root / "portfolio.yaml"
        config_path.write_text("portfolio: {}\n", encoding="utf-8")
        (portfolio_root / "manifest.json").write_text(
            json.dumps(
                {
                    "run_type": "portfolio",
                    "artifact_id": artifact_id,
                    "run_id": artifact_id,
                    "schema_version": "1.1",
                    "engine_version": "test-version",
                    "artifact_created_at": "2026-03-14T00:00:00+00:00",
                    "artifact_path": str(portfolio_root.resolve()),
                    "source_config_path": str(config_path.resolve()),
                    "run_seed": run_seed,
                    "config_hash": "abc123",
                    "data_version": "deadbeef12345678",
                    "slots": {"0": "StrategyA", "1": "StrategyB"},
                    "slot_weights": {"0": 0.5, "1": 0.5},
                }
            ),
            encoding="utf-8",
        )
        (root / ".run_type").write_text("portfolio", encoding="utf-8")

    return _make_portfolio_bundle


@pytest.fixture
def seed_scenario_job() -> Callable[..., ScenarioJobMetadata]:
    """Returns a helper that persists one synthetic scenario job."""

    def _seed_scenario_job(
        results_root: Path,
        *,
        status: str = "completed",
        baseline_run_id: str = "portfolio-001",
    ) -> ScenarioJobMetadata:
        store = ScenarioJobStore(results_dir=str(results_root))
        metadata = ScenarioJobMetadata(
            job_id="scenario-job-seeded",
            status=status,
            created_at="2026-03-14T00:00:00+00:00",
            baseline_results_dir=str(results_root.resolve()),
            baseline_run_id=baseline_run_id,
            scenario_type="stress_rerun",
            scenario_params={"volatility": 2.0, "slippage": 3.0, "commission": 2.0},
            timeout_seconds=1800,
            max_retries=2,
            failure_state="failed",
            queue_name="terminal-scenarios",
            progress_current=3 if status == "completed" else 1,
            progress_total=3,
            progress_message=(
                "Scenario artifacts completed."
                if status == "completed"
                else "Queued for execution."
            ),
            started_at="2026-03-14T00:00:05+00:00",
            completed_at="2026-03-14T00:00:12+00:00" if status == "completed" else "",
            duration_seconds=7.0 if status == "completed" else None,
            output_artifact_path=str(
                (results_root / "scenarios" / "scenario-job-seeded" / "portfolio").resolve()
            ),
            artifact_paths=[
                str((results_root / "scenarios" / "scenario-job-seeded" / "portfolio").resolve())
            ],
            rq_job_id="rq-job-seeded",
            last_error="",
        )
        return store.save(metadata)

    return _seed_scenario_job
