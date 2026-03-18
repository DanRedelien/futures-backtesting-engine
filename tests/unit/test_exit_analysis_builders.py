"""
test_exit_analysis_builders.py

Unit tests for the exit-analysis payload builders in exit_chart_builders.py.

Covers:
  - MFE vs MAE scatter payload structure and break-even diagonal
  - PnL decay category-line payload + legacy max-hold cutoff
  - Holding time bar payload bucket count
  - Volatility regime bar payload graceful empty state
  - Exit reason bar payload and breakdown stats table
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.terminal_ui.exit_chart_builders import (
    build_exit_holding_time_payload,
    build_exit_mfe_mae_payload,
    build_exit_pnl_decay_payload,
    build_exit_reason_breakdown_stats,
    build_exit_reason_payload,
    build_exit_vol_regime_payload,
)
from src.backtest_engine.analytics.terminal_ui.service import load_terminal_bundle
from src.backtest_engine.analytics.terminal_ui.table_builders import build_exit_detail_table


# ---------------------------------------------------------------------------
# MFE vs MAE scatter
# ---------------------------------------------------------------------------


def test_exit_mfe_mae_payload_returns_winner_loser_series(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    MFE/MAE payload must produce two series (Winners, Losers) when trades with
    both positive and negative PnL are present.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    payload = build_exit_mfe_mae_payload(bundle, strategy_name="__all__")

    assert payload["title"] == "MFE vs MAE"
    assert len(payload["series"]) == 2
    series_names = {s["name"] for s in payload["series"]}
    assert series_names == {"Winners", "Losers"}
    assert payload["xAxisReversed"] is True
    winners = next(s for s in payload["series"] if s["name"] == "Winners")
    assert len(winners["points"]) > 0
    assert all("x" in p and "y" in p for p in winners["points"])


def test_exit_mfe_mae_payload_includes_break_even_diagonal(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    MFE/MAE payload must include a 'diagonal' key with the break-even line
    endpoints so the renderer can draw the y = -x reference boundary.

    Methodology:
    The break-even diagonal (MFE = |MAE|) is a visual parity requirement from
    the legacy Streamlit dashboard. It helps analysts distinguish trades that
    recovered from adverse excursion (above the line) from those that did not.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    payload = build_exit_mfe_mae_payload(bundle, strategy_name="__all__")

    assert "diagonal" in payload, "Payload must expose the break-even diagonal endpoints"
    diagonal = payload["diagonal"]
    assert diagonal["x1"] == 0.0
    assert diagonal["y1"] == 0.0
    assert diagonal["x2"] < 0.0
    assert abs(diagonal["x2"]) == pytest.approx(diagonal["y2"])


def test_exit_mfe_mae_payload_filtered_by_strategy(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Filtering by strategy must reduce the point count compared to __all__."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    all_payload = build_exit_mfe_mae_payload(bundle, strategy_name="__all__")
    strategy_payload = build_exit_mfe_mae_payload(bundle, strategy_name="StrategyA")

    all_point_count = sum(len(s["points"]) for s in all_payload["series"])
    strategy_point_count = sum(len(s["points"]) for s in strategy_payload["series"])
    assert strategy_point_count < all_point_count


# ---------------------------------------------------------------------------
# PnL decay category-line
# ---------------------------------------------------------------------------


def test_exit_pnl_decay_payload_returns_single_series(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """PnL decay payload must include at least the 60m horizon (guaranteed column)."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    payload = build_exit_pnl_decay_payload(bundle, strategy_name="__all__")

    assert payload["title"] == "PnL Decay (Forward Horizon)"
    assert len(payload["series"]) == 1
    series = payload["series"][0]
    assert "60m" in payload["categories"] or "1h" in payload["categories"]
    assert len(series["values"]) == len(payload["categories"])
    assert len(payload["thresholds"]) >= 1
    assert any("Actual Avg" in str(t.get("label", "")) for t in payload["thresholds"])
    assert "verticalMarkers" in payload


def test_exit_pnl_decay_payload_honours_max_hold_cutoff() -> None:
    """
    PnL decay must stop at the first horizon >= max observed hold time.

    Methodology:
    Legacy exit_decomposition.py breaks out of the horizon loop once h >= max_hold,
    so strategies with 60m max hold do not show 120m, 240m, ... 1440m horizons.
    This test constructs trades with a 30m hold and confirms the returned
    categories do not extend beyond the 60m bucket (first >= 30).
    """
    index = pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 10:00:00"])
    trades = pd.DataFrame(
        {
            "strategy": ["S", "S"],
            "pnl": [100.0, -50.0],
            "entry_time": pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 09:30:00"]),
            "exit_time": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:00:00"]),
            "pnl_decay_5m": [80.0, -30.0],
            "pnl_decay_15m": [90.0, -40.0],
            "pnl_decay_60m": [95.0, -45.0],
            "pnl_decay_120m": [85.0, -55.0],
            "pnl_decay_1440m": [70.0, -60.0],
        }
    )
    history = pd.DataFrame({"total_value": [1_000_000.0, 1_000_100.0]}, index=index)
    bundle = ResultBundle(run_type="single", history=history, trades=trades)

    payload = build_exit_pnl_decay_payload(bundle, strategy_name="__all__")

    assert "120m" not in payload["categories"] and "2h" not in payload["categories"], (
        "PnL decay must not include horizons beyond the max hold cutoff"
    )
    assert "1440m" not in payload["categories"] and "24h" not in payload["categories"]
    assert len(payload["categories"]) <= 4


# ---------------------------------------------------------------------------
# Holding time
# ---------------------------------------------------------------------------


def test_exit_holding_time_payload_returns_five_buckets(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Holding time chart must always produce exactly 5 buckets."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    payload = build_exit_holding_time_payload(bundle, strategy_name="__all__")

    assert payload["title"] == "Avg PnL by Holding Time"
    assert len(payload["categories"]) == 5
    assert len(payload["series"]) == 1
    values = payload["series"][0]["values"]
    item_colors = payload["series"][0]["itemColors"]
    assert len(values) == 5
    assert len(item_colors) == 5
    assert all(c in ("#22C55E", "#EF4444") for c in item_colors)


# ---------------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------------


def test_exit_vol_regime_payload_returns_empty_state_without_column(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    Vol regime payload must return a graceful empty state when the
    entry_volatility enrichment column is absent from the bundle.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    payload = build_exit_vol_regime_payload(bundle, strategy_name="__all__")

    assert payload["title"] == "Avg PnL by Entry Volatility"
    assert payload["categories"] == []
    assert payload["series"] == []
    assert "emptyReason" in payload
    assert len(payload["emptyReason"]) > 0


# ---------------------------------------------------------------------------
# Exit reason
# ---------------------------------------------------------------------------


def test_exit_reason_payload_returns_bar_payload(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Exit reason chart must return categories matching exit_reason values in the bundle."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    payload = build_exit_reason_payload(bundle, strategy_name="__all__")

    assert payload["title"] == "Total PnL by Exit Reason"
    assert len(payload["categories"]) > 0
    assert "target" in payload["categories"] or "stop" in payload["categories"]
    assert len(payload["series"]) == 1
    assert len(payload["series"][0]["itemColors"]) == len(payload["categories"])


def test_exit_reason_breakdown_stats_returns_rows(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Breakdown stats must return one row per distinct exit reason."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    rows = build_exit_reason_breakdown_stats(bundle, strategy_name="__all__")

    assert len(rows) > 0
    for row in rows:
        assert "Exit Reason" in row
        assert "Count" in row
        assert "Win Rate" in row
        assert "Avg PnL" in row
        assert "Total PnL" in row
        assert row["Win Rate"].endswith("%")
        assert row["Avg PnL"].startswith("$")


def test_exit_detail_trade_log_formats_numeric_columns_to_two_decimals() -> None:
    """Trade-log view must format selected numeric columns with two decimals."""
    index = pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 10:00:00"])
    trades = pd.DataFrame(
        {
            "strategy": ["S", "S"],
            "symbol": ["GC", "GC"],
            "direction": ["LONG", "SHORT"],
            "entry_time": pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 09:40:00"]),
            "exit_time": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:10:00"]),
            "pnl": [155.123456, -12.1],
            "mfe": [191.999999, 0.0],
            "mae": [-270.0000000136, -5.5678],
            "pnl_decay_60m": [-225.000000003727, 9.2],
            "exit_reason": ["STOP_LOSS", "TAKE_PROFIT"],
        }
    )
    history = pd.DataFrame({"total_value": [1_000_000.0, 1_000_100.0]}, index=index)
    bundle = ResultBundle(run_type="single", history=history, trades=trades)

    frame, total_rows = build_exit_detail_table(
        bundle,
        strategy_name="__all__",
        page=1,
        page_size=50,
    )

    assert total_rows == 2
    assert frame.loc[0, "pnl"] == "155.12"
    assert frame.loc[1, "pnl"] == "-12.10"
    assert frame.loc[0, "mfe"] == "192.00"
    assert frame.loc[1, "mfe"] == "0.00"
    assert frame.loc[0, "mae"] == "-270.00"
    assert frame.loc[1, "mae"] == "-5.57"
    assert frame.loc[0, "pnl_decay_60m"] == "-225.00"
    assert frame.loc[1, "pnl_decay_60m"] == "9.20"


def test_exit_pnl_decay_payload_adds_time_stop_vertical_marker_when_available() -> None:
    """PnL decay payload must expose a vertical marker for TIME_STOP exits."""
    index = pd.to_datetime(["2024-01-01 09:30:00", "2024-01-01 10:00:00", "2024-01-01 10:30:00"])
    trades = pd.DataFrame(
        {
            "strategy": ["S", "S", "S"],
            "pnl": [100.0, -50.0, -25.0],
            "entry_time": pd.to_datetime(
                ["2024-01-01 09:30:00", "2024-01-01 09:30:00", "2024-01-01 10:00:00"]
            ),
            "exit_time": pd.to_datetime(
                ["2024-01-01 10:00:00", "2024-01-01 10:00:00", "2024-01-01 10:30:00"]
            ),
            "pnl_decay_60m": [95.0, -45.0, -20.0],
            "exit_reason": ["TAKE_PROFIT", "TIME_STOP", "TIME_STOP_1BAR"],
        }
    )
    history = pd.DataFrame({"total_value": [1_000_000.0, 1_000_100.0, 1_000_050.0]}, index=index)
    bundle = ResultBundle(run_type="single", history=history, trades=trades)

    payload = build_exit_pnl_decay_payload(bundle, strategy_name="__all__")

    labels = [str(t.get("label", "")) for t in payload["thresholds"]]
    assert any("Actual Avg" in label for label in labels)
    assert payload["verticalMarkers"], "Expected one time-stop marker"
    marker = payload["verticalMarkers"][0]
    assert marker["legend"] == "Time Stop Hold"
    assert marker["category"] in payload["categories"]
    assert "Time Stop" in marker["label"]
