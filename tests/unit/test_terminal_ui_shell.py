"""
test_terminal_ui_shell.py

Tests for the terminal-UI application shell:
  - Root page rendering (portfolio & single-asset modes)
  - Core chart / partial endpoints
  - PnL-distribution scope resolution
  - Fragment-level error handling for HTMX partials
  - Equity chart drawdown overlay contract
  - Dashboard resize handles
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi.testclient import TestClient

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.app import create_terminal_dashboard_app
from src.backtest_engine.analytics.terminal_ui.chart_builders import (
    build_equity_chart_payload,
    build_pnl_distribution_payload,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    _build_risk_profile_for_scope,
    load_terminal_bundle,
    load_terminal_runtime_context,
)


# ---------------------------------------------------------------------------
# Shell rendering
# ---------------------------------------------------------------------------


def test_terminal_ui_root_renders_portfolio_shell(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """The terminal UI root should render the fixed shell for a valid portfolio bundle."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/")

    assert response.status_code == 200
    assert "Quant Terminal" in response.text
    assert "PnL Distribution" in response.text
    assert "Stress Testing" in response.text
    assert "Correlations" in response.text
    assert "Open Stress Testing" in response.text


def test_terminal_ui_chart_endpoints_return_json_payloads(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Chart endpoints should return JSON payloads backed by canonical transforms."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    equity_response = client.get("/api/charts/equity")
    correlation_response = client.get("/api/charts/strategy-correlation?horizon=1d")
    risk_response = client.get("/api/charts/risk-var?risk_scope=portfolio")

    assert equity_response.status_code == 200
    assert len(equity_response.json()["series"]) >= 1
    assert correlation_response.status_code == 200
    assert "values" in correlation_response.json()
    assert risk_response.status_code == 200
    assert "series" in risk_response.json()


def test_pnl_distribution_payload_changes_with_strategy_risk_scope(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """PnL distribution must use selected strategy scope, not only portfolio equity."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    portfolio_payload = build_pnl_distribution_payload(bundle, risk_scope="portfolio")
    strategy_payload = build_pnl_distribution_payload(bundle, risk_scope="StrategyA")

    assert portfolio_payload["summary"]["mean"] == 75.0
    assert strategy_payload["summary"]["mean"] == 85.0
    assert strategy_payload["summary"]["mean"] != portfolio_payload["summary"]["mean"]


def test_pnl_distribution_chart_endpoint_respects_risk_scope(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Chart endpoint should accept risk_scope and return scope-specific stats."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    portfolio_response = client.get("/api/charts/pnl-distribution?risk_scope=portfolio")
    strategy_response = client.get("/api/charts/pnl-distribution?risk_scope=StrategyA")

    assert portfolio_response.status_code == 200
    assert strategy_response.status_code == 200
    assert portfolio_response.json()["summary"]["mean"] == 75.0
    assert strategy_response.json()["summary"]["mean"] == 85.0


def test_pnl_distribution_scope_matching_tolerates_plus_vs_space(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Strategy scope resolution should treat plus-sign and space labels equally."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None

    bundle.slots["0"] = "Strategy+Alpha"
    plus_payload = build_pnl_distribution_payload(bundle, risk_scope="Strategy+Alpha")
    space_payload = build_pnl_distribution_payload(bundle, risk_scope="Strategy Alpha")

    assert plus_payload["summary"]["mean"] == 85.0
    assert space_payload["summary"]["mean"] == plus_payload["summary"]["mean"]


def test_risk_profile_scope_matching_tolerates_plus_vs_space(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Risk scope fallback to portfolio should not happen on plus/space label drift."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    bundle = load_terminal_bundle(results_dir=str(results_root))
    assert bundle is not None
    bundle.slots["0"] = "Strategy+Alpha"

    runtime = load_terminal_runtime_context()
    stress = StressMultipliers(volatility=1.0, slippage=1.0, commission=1.0)
    profile_plus = _build_risk_profile_for_scope(bundle, runtime, "Strategy+Alpha", stress)
    profile_space = _build_risk_profile_for_scope(bundle, runtime, "Strategy Alpha", stress)
    profile_portfolio = _build_risk_profile_for_scope(bundle, runtime, "portfolio", stress)

    assert profile_space.summary["total_pnl"] == profile_plus.summary["total_pnl"]
    assert profile_space.summary["total_pnl"] != profile_portfolio.summary["total_pnl"]


def test_terminal_ui_single_mode_hides_portfolio_only_tabs(
    tmp_path: Path,
    make_single_bundle: Callable[..., None],
) -> None:
    """Single mode should reuse the same shell while hiding unavailable portfolio panels."""
    results_root = tmp_path / "results"
    make_single_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    response = client.get("/")

    assert response.status_code == 200
    assert "Strategy Stats" in response.text
    assert "Stress Testing" in response.text
    assert "Decomposition" not in response.text
    assert "Correlations" not in response.text


# ---------------------------------------------------------------------------
# Fragment-level error HTML for partial routes
# ---------------------------------------------------------------------------


def _partial_routes() -> list[str]:
    return [
        "/partials/top-ribbon",
        "/partials/main-stage",
        "/partials/bottom-panel?tab=pnl-distribution",
    ]


def test_partial_routes_return_fragment_error_when_no_bundle(tmp_path: Path) -> None:
    """
    When no bundle is present, partial endpoints must return a compact HTML
    fragment, not the full dashboard.html shell.

    Methodology:
        HTMX swaps the response into a named target element. Returning the full
        shell here would replace e.g. #top-ribbon with an entire page, breaking
        the layout. The fragment must be swappable inline without a full reload.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    client = TestClient(create_terminal_dashboard_app(results_dir=str(empty_dir)))

    for route in _partial_routes():
        response = client.get(route)

        assert response.status_code == 200, f"{route} should return 200 for HTMX swap"
        assert "terminal-fragment-error" in response.text, (
            f"{route} must return a fragment-level error, not a full page"
        )
        assert "<!DOCTYPE" not in response.text, (
            f"{route} must not return the full dashboard shell on error"
        )
        assert "terminal-shell" not in response.text, (
            f"{route} must not embed the shell layout in the error fragment"
        )


def test_partial_routes_return_fragment_not_full_page_on_incomplete_bundle(tmp_path: Path) -> None:
    """
    Incomplete artifact roots (marker present but files missing) should also
    produce a fragment error, not a full-page error response.
    """
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    (incomplete_dir / ".run_type").write_text("single", encoding="utf-8")
    client = TestClient(create_terminal_dashboard_app(results_dir=str(incomplete_dir)))

    for route in _partial_routes():
        response = client.get(route)

        assert response.status_code == 200, f"{route} should be 200 even on incomplete bundle"
        assert "terminal-fragment-error" in response.text, (
            f"{route} must return an inline error for incomplete bundles"
        )
        assert "<!DOCTYPE" not in response.text


def test_root_returns_full_shell_error_when_no_bundle(tmp_path: Path) -> None:
    """
    GET / is the only route allowed to return the full dashboard.html error shell,
    confirming the split between full-page and fragment error renderers.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    client = TestClient(create_terminal_dashboard_app(results_dir=str(empty_dir)))
    response = client.get("/")

    assert response.status_code == 200
    assert "<!DOCTYPE" in response.text
    assert "terminal-error-card" in response.text
    assert "terminal-fragment-error" not in response.text


# ---------------------------------------------------------------------------
# Equity chart drawdown overlay
# ---------------------------------------------------------------------------


def test_equity_chart_includes_drawdown_overlay_series_for_portfolio(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    The equity chart JSON must include a drawdown series with priceScaleId
    'drawdown' so the TradingView renderer can place it on a secondary axis.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    payload = client.get("/api/charts/equity").json()

    drawdown_series = [s for s in payload["series"] if s.get("priceScaleId") == "drawdown"]
    assert len(drawdown_series) == 1, "Equity payload must contain exactly one drawdown overlay series"

    series = drawdown_series[0]
    assert series["name"] == "Drawdown %"
    assert series["color"] == "#EF4444"
    assert len(series["points"]) > 0

    values = [p["value"] for p in series["points"]]
    assert all(v <= 0.001 for v in values), (
        "Drawdown values must be ≤ 0 (drawdown is always non-positive)"
    )


def test_equity_chart_includes_drawdown_overlay_series_for_single(
    tmp_path: Path,
    make_single_bundle: Callable[..., None],
) -> None:
    """
    Drawdown overlay must be present in single-asset mode too, since the
    contract applies to all equity chart renders regardless of run type.
    """
    results_root = tmp_path / "results"
    make_single_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    payload = client.get("/api/charts/equity").json()

    drawdown_series = [s for s in payload["series"] if s.get("priceScaleId") == "drawdown"]
    assert len(drawdown_series) == 1, "Single-mode equity payload must also have a drawdown overlay"

    values = [p["value"] for p in drawdown_series[0]["points"]]
    assert all(v <= 0.001 for v in values)


def test_equity_chart_drawdown_series_is_separate_from_equity_series(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    Non-drawdown series must not carry priceScaleId='drawdown', ensuring the
    overlay does not corrupt the primary equity scale.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    payload = client.get("/api/charts/equity").json()

    equity_series = [s for s in payload["series"] if s.get("priceScaleId") != "drawdown"]
    assert len(equity_series) >= 1, "At least one equity series must exist alongside the overlay"
    for s in equity_series:
        assert s.get("priceScaleId", "right") != "drawdown"


def test_equity_chart_payload_keeps_full_history_for_long_portfolio_runs() -> None:
    """Main Equity should bypass the generic chart point cap and keep full history."""
    point_count = 2505
    index = pd.date_range("2024-01-01 09:30:00", periods=point_count, freq="min")
    history = pd.DataFrame(
        {
            "total_value": 1_000_000.0 + pd.Series(range(point_count), index=index).astype(float).values,
            "slot_0_pnl": pd.Series(range(point_count), index=index).astype(float).values,
            "slot_1_pnl": (pd.Series(range(point_count), index=index) * 0.5).astype(float).values,
        },
        index=index,
    )
    benchmark = pd.DataFrame(
        {"close": 5000.0 + pd.Series(range(point_count), index=index).astype(float).values},
        index=index,
    )
    bundle = ResultBundle(
        run_type="portfolio",
        history=history,
        trades=pd.DataFrame(),
        benchmark=benchmark,
        manifest={"slots": {"0": "StrategyA", "1": "StrategyB"}},
        slots={"0": "StrategyA", "1": "StrategyB"},
    )
    runtime = replace(load_terminal_runtime_context(), max_chart_points=2000)

    payload = build_equity_chart_payload(bundle, runtime)

    assert payload["series"], "Equity payload should contain benchmark, strategy, total, and drawdown series"
    assert all(len(series["points"]) == point_count for series in payload["series"]), (
        "Main Equity must include the complete history for every rendered series"
    )


# ---------------------------------------------------------------------------
# Resize handles
# ---------------------------------------------------------------------------


def test_dashboard_shell_contains_sidebar_resize_handle(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    The rendered shell must include the column resize handle between the sidebar
    and the main area so users can drag to adjust sidebar width.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    html = client.get("/").text

    assert 'id="resize-sidebar"' in html
    assert "terminal-resize-handle--col" in html


def test_dashboard_shell_contains_bottom_panel_resize_handle(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    The rendered shell must include the row resize handle between the main stage
    and the bottom panel so users can drag to adjust bottom panel height.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)

    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))
    html = client.get("/").text

    assert 'id="resize-bottom"' in html
    assert "terminal-resize-handle--row" in html


def test_dashboard_shell_resize_handles_absent_on_error_page(tmp_path: Path) -> None:
    """
    The error shell (no artifacts) must not contain resize handles, since there
    is no functional layout to resize.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    client = TestClient(create_terminal_dashboard_app(results_dir=str(empty_dir)))
    html = client.get("/").text

    assert "terminal-error-card" in html
    assert "resize-sidebar" not in html
    assert "resize-bottom" not in html
