from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import src.backtest_engine.analytics.dashboard.core.scenario_runner as scenario_runner
from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.core.transforms.stress import _build_trade_cost_series
from src.backtest_engine.analytics.exit_analysis import enrich_trades_with_exit_analytics
from src.backtest_engine.analytics.exporter import save_backtest_results
from src.backtest_engine.execution import ExecutionHandler, Order
from src.backtest_engine.portfolio_layer.reporting.results import save_portfolio_results
from src.backtest_engine.settings import BacktestSettings


@dataclass
class StubSettings:
    commission_rate: float = 2.5
    max_slippage_ticks: int = 0
    random_seed: int = 42

    def get_instrument_spec(self, symbol: str) -> dict:
        return {"tick_size": 0.25, "multiplier": 50.0}


class FixedRandom:
    def __init__(self, value: int) -> None:
        self.value = value

    def randint(self, _low: int, _high: int) -> int:
        return self.value


def _bar(timestamp: str, open_price: float) -> pd.Series:
    return pd.Series({"open": open_price, "close": open_price}, name=pd.Timestamp(timestamp))


def test_partial_fill_commission_residue_does_not_inflate() -> None:
    """FIFO residue trackers must preserve proportional commission after partial closes."""
    handler = ExecutionHandler(StubSettings(max_slippage_ticks=0))

    handler.execute_order(Order(symbol="ES", quantity=4, side="BUY"), _bar("2024-01-01 09:30:00", 100.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:00:00", 101.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:30:00", 102.0))

    commissions = [trade.commission for trade in handler.trades]
    assert len(commissions) == 2
    assert commissions == [10.0, 10.0]
    assert sum(commissions) == 20.0
    assert handler.fills[0].order.quantity == 4


def test_partial_fill_residue_keeps_per_contract_slippage_convention() -> None:
    """Residue tracking must preserve per-contract slippage for later matched fragments."""
    handler = ExecutionHandler(StubSettings(max_slippage_ticks=1))
    handler._random = FixedRandom(1)

    handler.execute_order(Order(symbol="ES", quantity=4, side="BUY"), _bar("2024-01-01 09:30:00", 100.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:00:00", 101.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:30:00", 102.0))

    slippages = [trade.slippage for trade in handler.trades]
    assert slippages == [50.0, 50.0]
    assert sum(slippages) == 100.0


def test_slippage_propagates_to_exported_trades_and_daily_cost_series(tmp_path: Path) -> None:
    """Non-zero fill slippage must survive into trade artifacts and daily stress inputs."""
    handler = ExecutionHandler(StubSettings(max_slippage_ticks=1))
    handler._random = FixedRandom(1)

    handler.execute_order(Order(symbol="ES", quantity=1, side="BUY"), _bar("2024-01-01 09:30:00", 100.0))
    handler.execute_order(Order(symbol="ES", quantity=1, side="SELL"), _bar("2024-01-01 10:00:00", 101.0))

    settings = BacktestSettings(base_dir=tmp_path, results_dir=Path("results"))
    history = pd.DataFrame(
        {"total_value": [100_000.0, 100_025.0]},
        index=pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 10:00:00"]),
    )
    save_backtest_results(
        history=history,
        trades=handler.trades,
        report_str="report",
        metrics={"finite": np.float64(1.25), "nan": np.nan, "inf": np.inf},
        settings=settings,
    )

    trades_df = pd.read_parquet(tmp_path / "results" / "trades.parquet")
    metrics = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))

    assert float(trades_df.loc[0, "slippage"]) == 25.0
    assert metrics["finite"] == 1.25
    assert metrics["nan"] is None
    assert metrics["inf"] is None

    commission_daily, slippage_daily = _build_trade_cost_series(trades_df, {})
    assert float(commission_daily.iloc[0]) == 5.0
    assert float(slippage_daily.iloc[0]) == 25.0


def test_portfolio_results_metrics_are_strict_json_and_daily_pnl_is_incremental(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Portfolio artifacts must keep strict JSON and truthful incremental daily PnL semantics."""
    monkeypatch.chdir(tmp_path)

    history = pd.DataFrame(
        {
            "total_value": [1000.0, 1010.0, 1012.0, 1007.0],
            "slot_0_pnl": [0.0, 10.0, 12.0, 7.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-01 10:00:00",
                "2024-01-01 15:00:00",
                "2024-01-02 10:00:00",
                "2024-01-02 15:00:00",
            ]
        ),
    )

    save_portfolio_results(
        history=history,
        exposure_df=pd.DataFrame(),
        slot_trades={},
        report_str="report",
        metrics={"finite": np.float64(2.5), "nan": np.nan, "inf": np.inf},
        slot_names={0: "StrategyA"},
        slot_weights={0: 1.0},
    )

    metrics = json.loads((tmp_path / "results" / "portfolio" / "metrics.json").read_text(encoding="utf-8"))
    strategy_pnl_daily = pd.read_parquet(tmp_path / "results" / "portfolio" / "strategy_pnl_daily.parquet")

    assert metrics["finite"] == 2.5
    assert metrics["nan"] is None
    assert metrics["inf"] is None
    assert strategy_pnl_daily["slot_0_pnl"].tolist() == [10.0, -3.0]


def test_portfolio_results_can_write_to_namespaced_scenario_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Scenario reruns must write to a separate namespace with reconstructible metadata."""
    monkeypatch.chdir(tmp_path)

    history = pd.DataFrame(
        {
            "total_value": [1000.0, 1010.0],
            "slot_0_pnl": [0.0, 10.0],
        },
        index=pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 15:00:00"]),
    )
    output_dir = tmp_path / "results" / "scenarios" / "scenario-001" / "portfolio"

    saved_dir = save_portfolio_results(
        history=history,
        exposure_df=pd.DataFrame(),
        slot_trades={},
        report_str="report",
        metrics={"finite": 1.0},
        slot_names={0: "StrategyA"},
        slot_weights={0: 1.0},
        output_dir=output_dir,
        manifest_metadata={
            "run_kind": "scenario",
            "scenario_id": "scenario-001",
            "baseline_run_id": "baseline-abc",
            "scenario_type": "costs",
            "scenario_params": {"commission_multiplier": 2.0},
        },
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    marker = (output_dir.parent / ".run_type").read_text(encoding="utf-8").strip()

    assert saved_dir == output_dir
    assert manifest["run_kind"] == "scenario"
    assert manifest["scenario_id"] == "scenario-001"
    assert manifest["baseline_run_id"] == "baseline-abc"
    assert manifest["scenario_type"] == "costs"
    assert manifest["scenario_params"] == {"commission_multiplier": 2.0}
    assert marker == "portfolio"


def test_exit_analytics_excludes_entry_bar_path() -> None:
    """MFE/MAE should start after entry so the pre-position entry bar is excluded."""
    idx = pd.to_datetime(
        [
            "2024-01-01 09:30:00",
            "2024-01-01 10:00:00",
            "2024-01-01 10:30:00",
        ]
    )
    market = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.5],
            "high": [150.0, 101.0, 103.0],
            "low": [99.0, 99.5, 98.0],
            "close": [100.0, 100.5, 102.0],
        },
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "slot_id": [0],
            "symbol": ["TEST"],
            "direction": ["LONG"],
            "entry_time": [idx[0]],
            "exit_time": [idx[2]],
            "entry_price": [100.0],
            "quantity": [1.0],
            "commission": [0.0],
            "slippage": [0.0],
        }
    )

    enriched = enrich_trades_with_exit_analytics(trades, {(0, "TEST"): market})

    assert float(enriched.loc[0, "mfe"]) == 3.0
    assert float(enriched.loc[0, "mae"]) == -2.0


def test_exit_analytics_populates_pnl_decay_columns() -> None:
    """PnL decay should be populated by forward close lookup at T+N (not left as NaN)."""
    idx = pd.to_datetime(
        [
            "2024-01-01 09:30:00",
            "2024-01-01 09:35:00",
            "2024-01-01 09:45:00",
        ]
    )
    market = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
        },
        index=idx,
    )
    trades = pd.DataFrame(
        {
            "slot_id": [0],
            "symbol": ["TEST"],
            "direction": ["LONG"],
            "entry_time": [idx[0]],
            "exit_time": [idx[2]],
            "entry_price": [100.0],
            "quantity": [1.0],
            "commission": [0.0],
            "slippage": [0.0],
        }
    )

    enriched = enrich_trades_with_exit_analytics(trades, {(0, "TEST"): market})

    assert float(enriched.loc[0, "pnl_decay_5m"]) == 1.0
    assert float(enriched.loc[0, "pnl_decay_15m"]) == 2.0


def test_list_portfolio_scenarios_reads_namespaced_manifests(tmp_path: Path, monkeypatch) -> None:
    """Scenario discovery should find namespaced portfolio manifests and sort newest first."""
    results_root = tmp_path / "results"
    scenarios_root = results_root / "scenarios"
    first_manifest = scenarios_root / "scenario-a" / "portfolio" / "manifest.json"
    second_manifest = scenarios_root / "scenario-b" / "portfolio" / "manifest.json"
    first_manifest.parent.mkdir(parents=True, exist_ok=True)
    second_manifest.parent.mkdir(parents=True, exist_ok=True)
    first_manifest.write_text(
        json.dumps({"scenario_id": "scenario-a", "generated_at": "2026-03-14T01:00:00+00:00"}),
        encoding="utf-8",
    )
    second_manifest.write_text(
        json.dumps({"scenario_id": "scenario-b", "generated_at": "2026-03-14T02:00:00+00:00"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(scenario_runner, "get_results_dir", lambda: results_root)

    scenarios = scenario_runner.list_portfolio_scenarios()

    assert [item["manifest"]["scenario_id"] for item in scenarios] == ["scenario-b", "scenario-a"]


def test_list_portfolio_scenarios_supports_legacy_root_manifest_layout(tmp_path: Path, monkeypatch) -> None:
    """Scenario discovery should remain compatible with the earlier root-manifest layout."""
    results_root = tmp_path / "results"
    scenarios_root = results_root / "scenarios"
    legacy_manifest = scenarios_root / "scenario-legacy" / "manifest.json"
    legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
    legacy_manifest.write_text(
        json.dumps(
            {
                "scenario_id": "scenario-legacy",
                "generated_at": "2026-03-14T03:00:00+00:00",
                "baseline_run_id": "baseline-123",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(scenario_runner, "get_results_dir", lambda: results_root)

    scenarios = scenario_runner.list_portfolio_scenarios()

    assert scenarios[0]["manifest"]["scenario_id"] == "scenario-legacy"


def test_scenario_matches_baseline_requires_explicit_reference() -> None:
    """Scenario comparison must reject bundles that do not reference the active baseline."""
    empty_history = pd.DataFrame({"total_value": []})
    empty_trades = pd.DataFrame()
    baseline_bundle = ResultBundle(
        run_type="portfolio",
        history=empty_history,
        trades=empty_trades,
        manifest={"generated_at": "baseline-123"},
    )
    matching_scenario = ResultBundle(
        run_type="portfolio",
        history=empty_history,
        trades=empty_trades,
        manifest={"baseline_run_id": "baseline-123"},
    )
    mismatched_scenario = ResultBundle(
        run_type="portfolio",
        history=empty_history,
        trades=empty_trades,
        manifest={"baseline_run_id": "other-baseline"},
    )

    assert scenario_runner.scenario_matches_baseline(baseline_bundle, matching_scenario)
    assert not scenario_runner.scenario_matches_baseline(baseline_bundle, mismatched_scenario)
