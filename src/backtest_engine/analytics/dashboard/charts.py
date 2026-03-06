"""
src/backtest_engine/analytics/dashboard/charts.py

Plotly figure builders for the backtest dashboard.

Responsibility: Accept pre-loaded DataFrames and return Plotly Figure objects.
No Streamlit calls, no file I/O — pure chart construction.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import pandas as pd


# ── Color palette (mirrors visualizer.py constants) ───────────────────────────
_C: dict = {
    "combined":  "#2980B9",
    "long":      "#27AE60",
    "short":     "#E74C3C",
    "bench":     "#BDC3C7",
    "dd_fill":   "#FADBD8",
    "dd_line":   "#E74C3C",
    "winner":    "#27AE60",
    "loser":     "#E74C3C",
    "text":      "#2C3E50",
}


def build_equity_figure(
    history: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    benchmark: Optional[pd.DataFrame],
    run_type: str = "single",
    slots: Optional[dict] = None,
) -> go.Figure:
    """
    Builds an interactive Plotly equity curve figure.

    Methodology:
        All series are expressed as cumulative dollar PnL from $0 so that
        portfolio equity, per-slot PnL, and the buy-and-hold benchmark are
        directly comparable on a single Y-axis without scale mismatch.

        portfolio_pnl  = total_value - initial_capital
        benchmark_pnl  = (price / price_0 - 1) * initial_capital  (same capital base)
        slot_pnl       = slot_N_pnl column (already starts at $0)

        Single mode additionally draws long/short cumulative PnL decomposition.

    Args:
        history: Portfolio history with 'total_value' indexed by timestamp.
        trades: Trades DataFrame with columns: direction, exit_time, pnl.
        benchmark: Optional single-column 'close' price DataFrame.
        run_type: "single" or "portfolio"
        slots: Optional dict mapping slot_id string to strategy name.
    """
    fig = go.Figure()
    initial_cap: float = history["total_value"].iloc[0]
    idx = history.index

    # Portfolio equity as cumulative dollar PnL (starts at $0)
    portfolio_pnl = history["total_value"] - initial_cap

    # Buy-and-hold benchmark: dollar return on same starting capital
    if benchmark is not None and not benchmark.empty:
        b = benchmark["close"]
        common = idx.intersection(b.index)
        if len(common) > 1:
            b_pnl = (b.loc[common] / b.loc[common].iloc[0] - 1.0) * initial_cap
            fig.add_trace(go.Scatter(
                x=common, y=b_pnl,
                mode="lines",
                name="B&H (Buy-and-Hold)",
                line=dict(color=_C["bench"], width=1.5, dash="dash"),
                opacity=1.0,
            ))

    # Long / short PnL decomposition — single-asset mode only
    if run_type == "single" and trades is not None and not trades.empty and "exit_time" in trades.columns:
        for direction, color, label in [
            ("LONG",  _C["long"],  "Long"),
            ("SHORT", _C["short"], "Short"),
        ]:
            sub = trades[trades["direction"] == direction].copy()
            if sub.empty:
                continue
            pnl_s = sub.set_index("exit_time")["pnl"].sort_index()
            pnl_s = pnl_s.groupby(level=0).sum()
            full_idx = idx.union(pnl_s.index)
            cum = (
                pnl_s.reindex(full_idx, fill_value=0)
                .reindex(idx, fill_value=0)
                .cumsum()
            )
            fig.add_trace(go.Scatter(
                x=idx, y=cum,
                mode="lines",
                name=label,
                line=dict(color=color, width=1.2),
                opacity=0.85,
            ))

    # Per-slot PnL (already cumulative from $0, no transformation needed)
    if run_type == "portfolio" and slots:
        colors = ["#9B59B6", "#F1C40F", "#E67E22", "#1ABC9C", "#D35400", "#8E44AD"]
        for i, (str_slot_id, strat_name) in enumerate(slots.items()):
            col_name = f"slot_{str_slot_id}_pnl"
            if col_name in history.columns:
                fig.add_trace(go.Scatter(
                    x=idx, y=history[col_name],
                    mode="lines",
                    name=f"{strat_name} PnL (S{str_slot_id})",
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.75,
                ))

    # Portfolio total PnL (bold, on top)
    fig.add_trace(go.Scatter(
        x=idx, y=portfolio_pnl,
        mode="lines",
        name="Strategy" if run_type == "single" else "Portfolio Total PnL",
        line=dict(color=_C["combined"], width=2),
    ))

    fig.add_hline(
        y=0,
        line_dash="dash", line_color=_C["bench"],
        line_width=0.8, opacity=0.6,
    )

    fig.update_layout(
        title=dict(text="Equity Curve — Cumulative PnL ($)", font_size=13, x=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Cumulative PnL ($)"),
        legend=dict(
            orientation="h", y=-0.10, x=0,
            font_size=10, bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=34, b=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=_C["text"],
    )
    return fig




def build_drawdown_figure(history: pd.DataFrame) -> go.Figure:
    """
    Builds a filled area drawdown chart.

    Methodology:
        Drawdown at each bar = (equity - running_peak) / running_peak * 100
        Displayed as a percentage to make severity immediately legible.

    Args:
        history: Portfolio history with 'total_value' column.

    Returns:
        Plotly Figure.
    """
    running_max = history["total_value"].cummax()
    dd = (history["total_value"] - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.25)",
        line=dict(color=_C["dd_line"], width=1),
        name="Drawdown",
    ))

    fig.update_layout(
        title=dict(text="Drawdown %", font_size=12, x=0),
        yaxis=dict(ticksuffix="%"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=_C["text"],
    )
    return fig


def build_pnl_hist_figure(trades: Optional[pd.DataFrame]) -> go.Figure:
    """
    Builds a semi-transparent overlay P&L distribution histogram.

    Methodology:
        Losers and Winners are drawn as separate histogram traces with
        barmode='overlay' and opacity=0.7.  This produces the classic
        overlapping distribution view where the overlap region clearly shows
        the proximity of winning and losing trades.

    Args:
        trades: Trades DataFrame with 'pnl' column.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    if trades is None or trades.empty or "pnl" not in trades.columns:
        fig.add_annotation(
            text="No Trades", x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False, font_size=14,
        )
        fig.update_layout(height=240, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        return fig

    pnls    = trades["pnl"].dropna()
    winners = pnls[pnls > 0]
    losers  = pnls[pnls <= 0]

    bin_size = (pnls.max() - pnls.min()) / 50
    bins = dict(start=pnls.min(), end=pnls.max(), size=bin_size)

    # Losers first so Winners overlay on top
    fig.add_trace(go.Histogram(
        x=losers, xbins=bins,
        name="Losers", marker_color=_C["loser"], opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=winners, xbins=bins,
        name="Winners", marker_color=_C["winner"], opacity=0.7,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=_C["text"], line_width=0.8)

    fig.update_layout(
        title=dict(text="P&L Distribution", font_size=12, x=0),
        barmode="overlay",
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
        margin=dict(l=0, r=0, t=30, b=0),
        height=240,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color=_C["text"],
    )
    return fig
