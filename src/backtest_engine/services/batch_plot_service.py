"""
Matplotlib renderers for lightweight batch workflows.

Methodology:
    The batch commands intentionally avoid the heavy terminal dashboard and
    instead open one standard Matplotlib window in the parent process after all
    worker scenarios finish.  Plotting stays isolated here so orchestration
    services only deal with typed results.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import numpy as np

from src.backtest_engine.services.batch_models import SingleBatchResult, WfoBatchResult


def _format_summary_scenario_label(result: SingleBatchResult) -> str:
    """
    Formats batch scenario labels for narrow summary tables.

    Methodology:
        The chart legend can keep the full one-line label, but the summary table
        benefits from a stacked layout where the strategy ID is prominent and
        symbol/timeframe metadata sit on a second line.
    """
    strategy_line = "\n".join(
        textwrap.wrap(result.scenario.strategy_id, width=14, break_long_words=False)
    )
    meta_line = f"{result.scenario.symbol} | {result.scenario.timeframe}"
    return f"{strategy_line}\n{meta_line}" if strategy_line else meta_line


def show_single_batch_plot(
    results: Sequence[SingleBatchResult],
    figure_width: float,
    figure_height: float,
) -> None:
    """
    Renders one popup with log-PnL curves and a compact KPI table.

    Args:
        results: Successful batch backtest results.
        figure_width: Matplotlib figure width in inches.
        figure_height: Matplotlib figure height in inches.
    """
    if not results:
        return

    import matplotlib.pyplot as plt

    ordered_results = sorted(results, key=lambda item: item.pnl_pct, reverse=True)
    fig, (chart_ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [3.7, 1.8]},
    )

    plotted_lines = []
    for result in ordered_results:
        (line,) = chart_ax.plot(
            result.timestamps,
            result.log_equity,
            linewidth=1.4,
            label=result.scenario.legend_label,
        )
        plotted_lines.append(line)

    chart_ax.axhline(0.0, color="#7F7F7F", linewidth=0.9, linestyle="--")
    chart_ax.set_title("Batch Single-Strategy Log PnL")
    chart_ax.set_xlabel("Date")
    chart_ax.set_ylabel("log(total_value / initial_capital)")
    chart_ax.grid(alpha=0.2)
    legend = chart_ax.legend(loc="upper left", fontsize=8)

    legend_toggle_registry = {}
    for plotted_line, legend_line, legend_text in zip(
        plotted_lines,
        legend.get_lines(),
        legend.get_texts(),
    ):
        legend_line.set_picker(True)
        legend_line.set_pickradius(8)
        legend_text.set_picker(True)
        legend_toggle_registry[legend_line] = (plotted_line, legend_line, legend_text)
        legend_toggle_registry[legend_text] = (plotted_line, legend_line, legend_text)

    def _handle_legend_pick(event: object) -> None:
        artist = getattr(event, "artist", None)
        target = legend_toggle_registry.get(artist)
        if target is None:
            return

        plotted_line, legend_line, legend_text = target
        is_visible = not plotted_line.get_visible()
        plotted_line.set_visible(is_visible)
        item_alpha = 1.0 if is_visible else 0.25
        legend_line.set_alpha(item_alpha)
        legend_text.set_alpha(item_alpha)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _handle_legend_pick)

    table_ax.axis("off")
    table_rows = [
        [
            _format_summary_scenario_label(result),
            f"{result.pnl_pct:+.2f}%",
            f"{result.max_drawdown_pct:.2f}%",
            f"{result.sharpe_ratio:.2f}",
        ]
        for result in ordered_results
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Scenario", "PnL%", "MDD%", "Sharpe"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.50, 0.17, 0.17, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.08, 1.75)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
        if col_idx == 0:
            cell.get_text().set_wrap(True)
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("right")
    table_ax.set_title("Summary", pad=12)

    fig.tight_layout()
    plt.show()


def show_wfo_batch_heatmap(
    results: Sequence[WfoBatchResult],
    figure_width: float,
    figure_height: float,
) -> None:
    """
    Renders one verdict heatmap with rows=strategy and cols=symbol|timeframe.

    Methodology:
        Multi-timeframe batch WFO is naturally three-dimensional
        (strategy x symbol x timeframe).  To keep one readable heatmap, the
        column axis is flattened into a combined ``symbol timeframe`` label.

    Args:
        results: Successful WFO batch results.
        figure_width: Matplotlib figure width in inches.
        figure_height: Matplotlib figure height in inches.
    """
    if not results:
        return

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    strategies = sorted({result.scenario.strategy_id for result in results})
    targets = sorted(
        {
            f"{result.scenario.symbol} {result.scenario.timeframe}"
            for result in results
        }
    )
    matrix = np.zeros((len(strategies), len(targets)), dtype=float)
    text_matrix = [["FAIL" for _ in targets] for _ in strategies]

    verdict_map = {"FAIL": 0.0, "WARNING": 1.0, "PASS": 2.0}
    for result in results:
        row_idx = strategies.index(result.scenario.strategy_id)
        col_label = f"{result.scenario.symbol} {result.scenario.timeframe}"
        col_idx = targets.index(col_label)
        matrix[row_idx, col_idx] = verdict_map.get(result.verdict, 0.0)
        text_matrix[row_idx][col_idx] = result.verdict

    cmap = ListedColormap(["#FFFFFF", "#FACC15", "#22C55E"])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    image.set_clim(-0.5, 2.5)

    ax.set_title("WFO Batch Verdict Heatmap")
    ax.set_xlabel("Symbol | Timeframe")
    ax.set_ylabel("Strategy")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strategies)

    for row_idx in range(len(strategies)):
        for col_idx in range(len(targets)):
            ax.text(
                col_idx,
                row_idx,
                text_matrix[row_idx][col_idx],
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )

    legend_handles = [
        Patch(facecolor="#22C55E", edgecolor="#444444", label="PASS"),
        Patch(facecolor="#FACC15", edgecolor="#444444", label="WARNING"),
        Patch(facecolor="#FFFFFF", edgecolor="#444444", label="FAIL"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    plt.show()
