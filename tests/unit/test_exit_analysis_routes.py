"""
test_exit_analysis_routes.py

Integration tests for the exit-analysis HTTP chart endpoints and partial routes.

Covers:
  - /api/charts/exit-* JSON responses
  - strategy query-parameter filtering
  - /partials/bottom-panel?tab=exit-analysis master-detail scaffold
  - /partials/exit-analysis/detail sub-view rendering
  - Empty-state guard for portfolio __all__ selection
  - Single-asset mode auto-load bypass
  - Out-of-range pagination clamping
  - Exit-breakdown stats table presence
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from src.backtest_engine.analytics.terminal_ui.app import create_terminal_dashboard_app


# ---------------------------------------------------------------------------
# /api/charts/exit-* endpoints
# ---------------------------------------------------------------------------


def test_exit_chart_endpoints_return_json_payloads(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """All five exit-analysis chart endpoints must return valid JSON with expected keys."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    endpoints = [
        "/api/charts/exit-mfe-mae",
        "/api/charts/exit-pnl-decay",
        "/api/charts/exit-holding-time",
        "/api/charts/exit-vol-regime",
        "/api/charts/exit-reason",
    ]
    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200, f"{endpoint} returned {response.status_code}"
        body = response.json()
        assert "title" in body, f"{endpoint} missing 'title' key"


def test_exit_chart_endpoints_accept_strategy_filter(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Exit chart endpoints must accept a strategy query param without error."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get("/api/charts/exit-mfe-mae?strategy=StrategyA")
    assert response.status_code == 200
    payload = response.json()
    assert "series" in payload

    response = client.get("/api/charts/exit-reason?strategy=StrategyB")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /partials/bottom-panel?tab=exit-analysis
# ---------------------------------------------------------------------------


def test_exit_analysis_bottom_panel_renders_master_detail_scaffold(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    Exit-analysis tab must render the summary table and the detail workspace
    container with HTMX attributes for sub-view loading.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get("/partials/bottom-panel?tab=exit-analysis")

    assert response.status_code == 200
    html = response.text
    assert "Exit Analysis" in html
    assert "exit-detail-workspace" in html
    assert "/partials/exit-analysis/detail" in html


# ---------------------------------------------------------------------------
# /partials/exit-analysis/detail sub-views
# ---------------------------------------------------------------------------


def test_exit_analysis_detail_trade_log_returns_paginated_table(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Trade-log detail view must render trade rows and pagination controls for a specific strategy."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get(
        "/partials/exit-analysis/detail?exit_detail_view=trade-log&exit_strategy=StrategyA"
    )

    assert response.status_code == 200
    assert "Page" in response.text


def test_exit_analysis_detail_shows_empty_state_for_portfolio_all_strategies(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    Portfolio mode with exit_strategy=__all__ must return the placeholder
    empty-state partial rather than a real sub-view.

    Methodology:
    The detail workspace is intentionally blank until the user picks a specific
    strategy from the summary table.  Showing aggregate data for __all__ would
    contradict the master-detail UX contract described in the implementation plan.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get(
        "/partials/exit-analysis/detail?exit_detail_view=trade-log&exit_strategy=__all__"
    )

    assert response.status_code == 200
    html = response.text
    assert "Select a strategy" in html
    assert "Page" not in html


def test_exit_analysis_detail_auto_loads_for_single_asset_mode(
    tmp_path: Path,
    make_single_bundle: Callable[..., None],
) -> None:
    """
    Single-asset mode must bypass the empty-state guard and render the trade-log
    even when exit_strategy=__all__, because __all__ is the only valid selection.
    """
    results_root = tmp_path / "results"
    make_single_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get(
        "/partials/exit-analysis/detail?exit_detail_view=trade-log&exit_strategy=__all__"
    )

    assert response.status_code == 200
    html = response.text
    assert "Page" in html
    assert "Select a strategy" not in html


def test_exit_analysis_detail_sub_views_render_without_error(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """All four sub-view types must render without server errors for a specific strategy."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    for view in ("trade-log", "execution-quality", "time-context", "exit-breakdown"):
        response = client.get(
            f"/partials/exit-analysis/detail?exit_detail_view={view}&exit_strategy=StrategyA"
        )
        assert response.status_code == 200, f"Sub-view '{view}' returned {response.status_code}"


def test_exit_analysis_detail_preserves_strategy_filter_across_sub_views(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    Strategy filter must be embedded in chart endpoint URLs of the rendered
    detail partial so each chart loads scoped data.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get(
        "/partials/exit-analysis/detail"
        "?exit_detail_view=execution-quality&exit_strategy=StrategyA"
    )

    assert response.status_code == 200
    assert "StrategyA" in response.text


def test_exit_analysis_detail_trade_log_out_of_range_page_is_clamped(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """
    An out-of-range page must be clamped to the last valid page.

    Methodology:
    If page=99 is requested but only 1 page of trades exists, the template
    must show 'Page 1 / 1' (or 'Page N / N') rather than 'Page 99 / 1'.
    Showing an impossible page number alongside an empty table would give
    users no actionable navigation path.
    """
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get(
        "/partials/exit-analysis/detail"
        "?exit_detail_view=trade-log&exit_strategy=StrategyA&page=99"
    )

    assert response.status_code == 200
    html = response.text
    assert "Page 99" not in html
    assert "Page" in html


def test_exit_analysis_detail_exit_breakdown_contains_stats_table(
    tmp_path: Path,
    make_portfolio_bundle: Callable[..., None],
) -> None:
    """Exit-breakdown sub-view must include the breakdown stats table alongside the chart."""
    results_root = tmp_path / "results"
    make_portfolio_bundle(results_root)
    client = TestClient(create_terminal_dashboard_app(results_dir=str(results_root)))

    response = client.get(
        "/partials/exit-analysis/detail"
        "?exit_detail_view=exit-breakdown&exit_strategy=StrategyA"
    )

    assert response.status_code == 200
    html = response.text
    assert "Exit Reason" in html
    assert "Win Rate" in html
