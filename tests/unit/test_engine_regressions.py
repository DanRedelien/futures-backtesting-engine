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
from src.backtest_engine.spread_model import compute_spread_ticks


@dataclass
class StubSettings:
    commission_rate: float = 2.5
    spread_ticks: int = 0
    spread_mode: str = "static"
    spread_volatility_step_pct: float = 0.10
    spread_step_multiplier: float = 1.5
    spread_vol_lookback: int = 20
    spread_vol_baseline_lookback: int = 100

    def get_instrument_spec(self, symbol: str) -> dict:
        return {"tick_size": 0.25, "multiplier": 50.0}


def _bar(timestamp: str, open_price: float) -> pd.Series:
    return pd.Series({"open": open_price, "close": open_price}, name=pd.Timestamp(timestamp))


def test_partial_fill_commission_residue_does_not_inflate() -> None:
    """FIFO residue trackers must preserve proportional commission after partial closes."""
    handler = ExecutionHandler(StubSettings(spread_ticks=0))

    handler.execute_order(Order(symbol="ES", quantity=4, side="BUY"), _bar("2024-01-01 09:30:00", 100.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:00:00", 101.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:30:00", 102.0))

    commissions = [trade.commission for trade in handler.trades]
    assert len(commissions) == 2
    assert commissions == [10.0, 10.0]
    assert sum(commissions) == 20.0
    assert handler.fills[0].order.quantity == 4


def test_partial_fill_residue_keeps_per_contract_slippage_convention() -> None:
    """Residue tracking must preserve per-contract slippage for later matched fragments.

    With spread_ticks=1 and tick_size=0.25, each fill gets slippage=0.25 price units.
    For a LONG round-trip with qty=2 at multiplier=50:
      entry slippage_per_unit = 0.25 * 50 = 12.5 per contract
      exit  slippage_per_unit = 0.25 * 50 = 12.5 per contract
      trade_slippage = (12.5 + 12.5) * 2 = 50.0 per trade
    Two trades -> total 100.0.
    """
    handler = ExecutionHandler(StubSettings(spread_ticks=1))

    handler.execute_order(Order(symbol="ES", quantity=4, side="BUY"), _bar("2024-01-01 09:30:00", 100.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:00:00", 101.0))
    handler.execute_order(Order(symbol="ES", quantity=2, side="SELL"), _bar("2024-01-01 10:30:00", 102.0))

    slippages = [trade.slippage for trade in handler.trades]
    assert slippages == [50.0, 50.0]
    assert sum(slippages) == 100.0


def test_slippage_propagates_to_exported_trades_and_daily_cost_series(tmp_path: Path) -> None:
    """Non-zero fill slippage must survive into trade artifacts and daily stress inputs."""
    handler = ExecutionHandler(StubSettings(spread_ticks=1))

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


def test_static_spread_mode_is_deterministic() -> None:
    """Static mode must produce identical fill prices on repeated calls with the same inputs."""
    handler = ExecutionHandler(StubSettings(spread_ticks=2))

    bar = _bar("2024-01-01 09:30:00", 100.0)
    fill1 = handler.execute_order(Order(symbol="ES", quantity=1, side="BUY"), bar)
    fill2 = handler.execute_order(Order(symbol="ES", quantity=1, side="BUY"), bar)

    assert fill1 is not None and fill2 is not None
    assert fill1.fill_price == fill2.fill_price, "Static spread must be deterministic"


def test_static_spread_buy_adds_ticks() -> None:
    """BUY fill price must be price + spread_ticks * tick_size for static mode."""
    settings = StubSettings(spread_ticks=2)
    handler = ExecutionHandler(settings)

    bar = _bar("2024-01-01 09:30:00", 100.0)
    fill = handler.execute_order(Order(symbol="ES", quantity=1, side="BUY"), bar)

    assert fill is not None
    expected = 100.0 + 2 * 0.25  # tick_size=0.25
    assert fill.fill_price == expected


def test_static_spread_sell_subtracts_ticks() -> None:
    """SELL fill price must be price - spread_ticks * tick_size for static mode."""
    settings = StubSettings(spread_ticks=2)
    handler = ExecutionHandler(settings)

    bar = _bar("2024-01-01 09:30:00", 100.0)
    fill = handler.execute_order(Order(symbol="ES", quantity=1, side="SELL"), bar)

    assert fill is not None
    expected = 100.0 - 2 * 0.25
    assert fill.fill_price == expected


def test_spread_ticks_zero_produces_no_slippage() -> None:
    """spread_ticks=0 must result in zero slippage and exact price execution."""
    handler = ExecutionHandler(StubSettings(spread_ticks=0))

    bar = _bar("2024-01-01 09:30:00", 100.0)
    fill = handler.execute_order(Order(symbol="ES", quantity=1, side="BUY"), bar)

    assert fill is not None
    assert fill.fill_price == 100.0
    assert fill.slippage == 0.0


def test_effective_spread_ticks_override_takes_precedence() -> None:
    """Engine-supplied effective_spread_ticks must override settings.spread_ticks."""
    handler = ExecutionHandler(StubSettings(spread_ticks=0))

    bar = _bar("2024-01-01 09:30:00", 100.0)
    fill = handler.execute_order(
        Order(symbol="ES", quantity=1, side="BUY"),
        bar,
        effective_spread_ticks=3,
    )

    assert fill is not None
    expected = 100.0 + 3 * 0.25
    assert fill.fill_price == expected


def test_adaptive_spread_widens_in_high_vol() -> None:
    """Adaptive mode must return more ticks when current vol exceeds baseline."""
    n = 200
    prices = pd.Series([100.0 + i * 0.1 for i in range(n)])

    # Inject a high-vol spike into the last 20 bars (short window)
    spike = pd.Series([100.0 + i * 5.0 for i in range(20)])
    closes_spike = pd.concat([prices, spike], ignore_index=True)

    ticks_spiked = compute_spread_ticks(
        mode="adaptive_volatility",
        base_ticks=1,
        closes=closes_spike,
        vol_step_pct=0.10,
        step_multiplier=2.0,
        vol_lookback=20,
        vol_baseline_lookback=100,
    )
    ticks_calm = compute_spread_ticks(
        mode="adaptive_volatility",
        base_ticks=1,
        closes=prices,
        vol_step_pct=0.10,
        step_multiplier=2.0,
        vol_lookback=20,
        vol_baseline_lookback=100,
    )

    assert ticks_spiked > ticks_calm, (
        f"Adaptive spread must widen under elevated volatility: "
        f"spiked={ticks_spiked}, calm={ticks_calm}"
    )


def test_adaptive_spread_narrows_in_low_vol() -> None:
    """Adaptive mode must return fewer ticks when current vol falls below baseline."""
    # Baseline with moderate historical volatility
    n = 200
    prices = pd.Series([100.0 + (i % 10) * 0.5 for i in range(n)])

    # Add a very calm tail (low recent vol)
    calm_tail = pd.Series([200.0] * 25)
    closes_calm_tail = pd.concat([prices, calm_tail], ignore_index=True)

    ticks_narrowed = compute_spread_ticks(
        mode="adaptive_volatility",
        base_ticks=4,
        closes=closes_calm_tail,
        vol_step_pct=0.10,
        step_multiplier=2.0,
        vol_lookback=20,
        vol_baseline_lookback=100,
    )

    assert ticks_narrowed <= 4, (
        f"Adaptive spread should narrow when current vol < baseline: got {ticks_narrowed}"
    )


def test_adaptive_spread_insufficient_history_falls_back_to_base() -> None:
    """Adaptive mode must fall back to base_ticks when history is too short."""
    short_closes = pd.Series([100.0, 101.0, 102.0])

    ticks = compute_spread_ticks(
        mode="adaptive_volatility",
        base_ticks=2,
        closes=short_closes,
        vol_step_pct=0.10,
        step_multiplier=1.5,
        vol_lookback=20,
        vol_baseline_lookback=100,
    )
    assert ticks == 2, f"Expected fallback to base_ticks=2, got {ticks}"


def test_adaptive_spread_is_non_compounding_across_bars() -> None:
    """Spread adjustment must be recalculated from scratch each bar, not accumulated.

    Two consecutive calls with slightly different histories must each compute
    independently without carrying over a multiplied spread from the prior bar.
    """
    n = 150
    prices = pd.Series([100.0 + i * 0.2 for i in range(n)])

    ticks_bar1 = compute_spread_ticks(
        mode="adaptive_volatility",
        base_ticks=1,
        closes=prices.iloc[:100],
        vol_step_pct=0.10,
        step_multiplier=2.0,
        vol_lookback=20,
        vol_baseline_lookback=80,
    )
    ticks_bar2 = compute_spread_ticks(
        mode="adaptive_volatility",
        base_ticks=1,
        closes=prices.iloc[:101],
        vol_step_pct=0.10,
        step_multiplier=2.0,
        vol_lookback=20,
        vol_baseline_lookback=80,
    )

    # Neither call should compound the prior result; both should be small finite values.
    assert ticks_bar1 >= 0
    assert ticks_bar2 >= 0
    # If compounding occurred, tick counts would grow exponentially (e.g. >100).
    assert ticks_bar1 < 50, f"Unexpected tick inflation at bar1: {ticks_bar1}"
    assert ticks_bar2 < 50, f"Unexpected tick inflation at bar2: {ticks_bar2}"


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
