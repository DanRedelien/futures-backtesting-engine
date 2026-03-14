from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from src.backtest_engine.analytics.artifact_contract import resolve_engine_version
from src.backtest_engine.analytics.dashboard.core.data_layer import (
    ResultBundle,
    inspect_result_bundle,
    load_result_bundle_uncached,
    result_bundle_service,
)
from src.backtest_engine.analytics.dashboard.core.scenario_runner import (
    resolve_portfolio_config_path,
)
from src.backtest_engine.portfolio_layer.reporting.results import (
    _PROJECT_ROOT,
    save_portfolio_results,
)


def test_inspect_result_bundle_distinguishes_missing_incomplete_and_valid(
    tmp_path: Path,
    make_single_bundle: Callable[..., None],
) -> None:
    """The loader should distinguish missing, incomplete, and valid single bundles."""
    results_root = tmp_path / "results"

    missing = inspect_result_bundle(results_dir=str(results_root))
    assert missing.state == "missing"

    incomplete_root = results_root / "incomplete"
    incomplete_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"total_value": [1000.0]}).to_parquet(incomplete_root / "history.parquet")
    incomplete = inspect_result_bundle(results_dir=str(incomplete_root))
    assert incomplete.state == "incomplete"
    assert "manifest.json" in incomplete.missing_files

    valid_root = results_root / "valid"
    make_single_bundle(valid_root, include_marker=False)
    valid = inspect_result_bundle(results_dir=str(valid_root))
    assert valid.state == "valid"
    assert valid.run_type == "single"


def test_load_result_bundle_rejects_portfolio_namespace_without_marker(
    tmp_path: Path,
    make_single_bundle: Callable[..., None],
) -> None:
    """Portfolio artifacts without `.run_type` should stay unreadable until the bundle is explicit."""
    scenario_root = tmp_path / "results" / "scenario-a"
    portfolio_root = scenario_root / "portfolio"
    make_single_bundle(portfolio_root, include_marker=False)

    status = inspect_result_bundle(results_dir=str(scenario_root))
    bundle = load_result_bundle_uncached(results_dir=str(scenario_root))

    assert status.state == "incomplete"
    assert bundle is None


def test_result_bundle_service_loads_same_contract_as_function(
    tmp_path: Path,
    make_single_bundle: Callable[..., None],
) -> None:
    """The framework-neutral service wrapper should expose the same bundle contract."""
    valid_root = tmp_path / "results" / "valid"
    make_single_bundle(valid_root, include_marker=False)

    bundle = result_bundle_service.load_bundle(results_dir=str(valid_root), use_cache=False)
    status = result_bundle_service.inspect_bundle(results_dir=str(valid_root))

    assert bundle is not None
    assert bundle.run_type == "single"
    assert status.state == "valid"


def test_result_bundle_marks_incomplete_rerun_metadata_as_view_only() -> None:
    """Old baselines without reproducibility metadata must not rerun by default."""
    bundle = ResultBundle(
        run_type="portfolio",
        history=pd.DataFrame({"total_value": []}),
        trades=pd.DataFrame(),
        manifest={
            "source_config_path": "portfolio.yaml",
            "run_seed": 42,
            "config_hash": "abc123",
        },
    )

    assert bundle.compatibility is not None
    assert not bundle.compatibility.is_rerunnable
    assert "data_version" in bundle.compatibility.missing_fields


def test_result_bundle_with_complete_metadata_is_rerunnable(tmp_path: Path) -> None:
    """Complete reproducibility metadata should keep new baselines rerunnable."""
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("portfolio: {}\n", encoding="utf-8")

    bundle = ResultBundle(
        run_type="portfolio",
        history=pd.DataFrame({"total_value": []}),
        trades=pd.DataFrame(),
        manifest={
            "source_config_path": str(config_path),
            "run_seed": 42,
            "config_hash": "abc123",
            "data_version": "deadbeef12345678",
        },
    )

    assert bundle.compatibility is not None
    assert bundle.compatibility.is_rerunnable


def test_resolve_portfolio_config_path_rejects_view_only_baseline() -> None:
    """Scenario reruns must not fall back to example configs for legacy artifacts."""
    bundle = ResultBundle(
        run_type="portfolio",
        history=pd.DataFrame({"total_value": []}),
        trades=pd.DataFrame(),
        manifest={"generated_at": "legacy-baseline"},
    )

    try:
        resolve_portfolio_config_path(bundle)
    except ValueError as exc:
        assert "view-only" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a non-rerunnable baseline.")


def test_save_portfolio_results_writes_artifact_identity_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Portfolio manifests should carry identity metadata for future caches and jobs."""
    monkeypatch.chdir(tmp_path)

    history = pd.DataFrame(
        {"total_value": [1000.0, 1010.0]},
        index=pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 10:00:00"]),
    )
    output_dir = tmp_path / "results" / "portfolio"

    save_portfolio_results(
        history=history,
        exposure_df=pd.DataFrame(),
        slot_trades={},
        report_str="report",
        metrics={"finite": 1.0},
        slot_names={0: "StrategyA"},
        slot_weights={0: 1.0},
        output_dir=output_dir,
        manifest_metadata={"source_config_path": str(tmp_path / "portfolio.yaml")},
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["artifact_id"]
    assert manifest["run_id"] == manifest["artifact_id"]
    assert manifest["schema_version"]
    assert manifest["engine_version"] == resolve_engine_version(_PROJECT_ROOT)
    assert manifest["artifact_created_at"]
    assert manifest["artifact_path"] == str(output_dir.resolve())
