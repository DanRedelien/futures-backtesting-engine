from __future__ import annotations

import math

import matplotlib
import numpy as np
import pytest

from src.backtest_engine.services.batch_models import BatchScenario, SingleBatchResult
from src.backtest_engine.services.batch_plot_service import (
    _format_summary_scenario_label,
    show_single_batch_plot,
)
from src.backtest_engine.services.batch_run_service import _build_capped_log_equity_curve


def test_build_capped_log_equity_curve_clamps_losses_at_configured_floor() -> None:
    """
    Batch log-equity curves should stop at the configured ruin floor even when
    raw account equity turns negative.
    """
    log_equity = _build_capped_log_equity_curve(
        total_values=[100_000.0, 50_000.0, 0.0, -25_000.0],
        initial_capital=100_000.0,
        floor_pct=-100.0,
        ruin_equity_ratio=0.01,
    )

    assert np.isclose(log_equity[0], 0.0)
    assert np.isclose(log_equity[1], math.log(0.5))
    assert np.isclose(log_equity[2], math.log(0.01))
    assert np.isclose(log_equity[3], math.log(0.01))


def test_build_capped_log_equity_curve_rejects_non_positive_ruin_ratio() -> None:
    """
    Log plotting requires a strictly positive surrogate ratio at the loss floor.
    """
    with pytest.raises(ValueError, match="ruin_equity_ratio must be positive"):
        _build_capped_log_equity_curve(
            total_values=[100_000.0],
            initial_capital=100_000.0,
            floor_pct=-100.0,
            ruin_equity_ratio=0.0,
        )


def test_show_single_batch_plot_makes_legend_entries_pickable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Legend entries should be clickable so users can hide individual curves.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    result = SingleBatchResult(
        scenario=BatchScenario(strategy_id="sma", symbol="ES", timeframe="30m"),
        status="completed",
        timestamps=[1, 2, 3],
        log_equity=[0.0, -0.1, -0.2],
        pnl_pct=-18.0,
        max_drawdown_pct=-22.0,
        sharpe_ratio=-0.3,
    )

    show_single_batch_plot(results=[result], figure_width=8.0, figure_height=4.0)

    figure = plt.gcf()
    chart_ax = figure.axes[0]
    legend = chart_ax.get_legend()

    assert legend is not None
    assert legend.get_lines()[0].get_picker() is True
    assert legend.get_texts()[0].get_picker() is True

    plt.close(figure)


def test_format_summary_scenario_label_stacks_strategy_and_meta() -> None:
    """
    Summary labels should keep strategy and market metadata readable in narrow tables.
    """
    result = SingleBatchResult(
        scenario=BatchScenario(
            strategy_id="intraday_momentum",
            symbol="NQ",
            timeframe="30m",
        ),
        status="completed",
    )

    label = _format_summary_scenario_label(result)

    assert "intraday_" in label
    assert "momentum" in label
    assert label.endswith("NQ | 30m")
