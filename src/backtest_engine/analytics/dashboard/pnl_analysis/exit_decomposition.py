"""
src/backtest_engine/analytics/dashboard/exit_charts.py

Figure builders for Phase 3 Exit Analysis.
Visualises data enriched by exit_analysis.py inside trades.parquet.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from src.backtest_engine.settings import BacktestSettings
from typing import Optional

from src.backtest_engine.analytics.dashboard.core.styles import PALETTE

def build_holding_time_chart(trades: pd.DataFrame) -> go.Figure:
    """
    Builds a histogram / bucketed bar chart of PnL by holding time.
    """
    fig = go.Figure()
    if trades.empty or "holding_time" not in trades.columns:
        return _empty_fig("Holding Time Analysis Not Available")

    # Convert Timedelta to total minutes
    if pd.api.types.is_timedelta64_dtype(trades["holding_time"]):
        holding_mins = trades["holding_time"].dt.total_seconds() / 60.0
    else:
        holding_mins = trades["holding_time"]

    max_hold = holding_mins.max()
    if pd.isna(max_hold) or max_hold <= 0:
        max_hold = 60.0

    if max_hold <= 60:
        step = 15; unit = "m"
    elif max_hold <= 240:
        step = 60; unit = "m"
    elif max_hold <= 1440:
        step = 360; unit = "h"
    else:
        step = 1440; unit = "d"

    b1, b2, b3, b4 = step, step*2, step*3, step*4
    bins = [0, b1, b2, b3, b4, float('inf')]
    
    def fmt(m):
        if unit == "m": return f"{int(m)}m"
        elif unit == "h": return f"{int(m/60)}h"
        return f"{int(m/1440)}d"

    labels = [
        f"<{fmt(b1)}", f"{fmt(b1)}-{fmt(b2)}",
        f"{fmt(b2)}-{fmt(b3)}", f"{fmt(b3)}-{fmt(b4)}", f">{fmt(b4)}"
    ]
    
    trades_copy = trades.copy()
    trades_copy['hold_bucket'] = pd.cut(holding_mins, bins=bins, labels=labels, right=False)
    
    grouped = trades_copy.groupby('hold_bucket', observed=False).agg(
        avg_pnl=('pnl', 'mean'),
        count=('pnl', 'count')
    ).fillna(0)
    
    colors = [PALETTE["winner"] if val >= 0 else PALETTE["loser"] for val in grouped['avg_pnl']]

    fig.add_trace(go.Bar(
        x=grouped.index,
        y=grouped['avg_pnl'],
        text=[f"${v:,.0f}<br>(n={c})" for v, c in zip(grouped['avg_pnl'], grouped['count'])],
        textposition='auto',
        marker_color=colors,
        name="Avg PnL"
    ))

    fig.update_layout(
        title=dict(text="Avg PnL by Holding Time", font_size=13, x=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        xaxis=dict(title="Holding Time Bucket"),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        showlegend=False
    )
    return fig


def build_pnl_decay_chart(trades: pd.DataFrame) -> go.Figure:
    """
    Builds a line chart showing the trajectory of hypothetical PnL 
    if the position was held exactly 5m, 15m, 30m, 60m.
    """
    fig = go.Figure()
    if trades.empty:
        return _empty_fig("PnL Decay Not Available")

    all_horizons = [5, 15, 30, 60, 120, 240, 480, 720, 1440]
    
    if "holding_time" in trades.columns:
        holding_s = trades["holding_time"]
        if pd.api.types.is_timedelta64_dtype(holding_s):
            max_hold = holding_s.dt.total_seconds().max() / 60.0
        else:
            max_hold = holding_s.max()
    else:
        max_hold = 60.0
        
    if pd.isna(max_hold) or max_hold <= 0:
        max_hold = 60.0

    horizons = []
    for h in all_horizons:
        horizons.append(h)
        if h >= max_hold:
            break

    # Base entry is 0m, 0 PnL (ignoring costs at exact t=0 for trajectory shape)
    x_vals = [0] + horizons
    y_vals = [0.0]
    
    for h in horizons:
        col = f"pnl_decay_{h}m"
        if col in trades.columns:
            avg_pnl = trades[col].mean()
            y_vals.append(avg_pnl if pd.notna(avg_pnl) else 0.0)
        else:
            y_vals.append(0.0)

    def fmt_m(m):
        if m == 0: return "Entry"
        if m < 60: return f"{m}m"
        elif m < 1440: return f"{int(m/60)}h"
        return f"{int(m/1440)}d"

    x_labels = [fmt_m(x) for x in x_vals]

    # Actual avg PnL at actual exit time
    actual_avg = trades["pnl"].mean()

    fig.add_trace(go.Scatter(
        x=x_labels,
        y=y_vals,
        mode="lines+markers",
        line=dict(color=PALETTE["combined"], width=2),
        marker=dict(size=8),
        name="Hypothetical Decay"
    ))
    
    # Add a horizontal line for actual achieved PnL
    fig.add_hline(
        y=actual_avg, 
        line_dash="dash", 
        line_color=PALETTE["winner"] if actual_avg >= 0 else PALETTE["loser"],
        annotation_text=f"Actual Avg: ${actual_avg:.0f}",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title=dict(text="PnL Decay (Forward Horizon)", font_size=13, x=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f", title="Avg Net PnL"),
        xaxis=dict(title="Time since Entry", type="category", categoryorder="array", categoryarray=x_labels),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        showlegend=False
    )
    return fig


def build_mfe_mae_scatter(trades: pd.DataFrame) -> go.Figure:
    """
    Builds a scatter plot of MAE (x-axis) vs MFE (y-axis), colored by PnL.
    """
    fig = go.Figure()
    if trades.empty or "mfe" not in trades.columns or "mae" not in trades.columns:
        return _empty_fig("MFE/MAE Not Available")

    # Filter out NaNs
    df = trades.dropna(subset=["mfe", "mae", "pnl"])
    if df.empty:
        return _empty_fig("MFE/MAE Not Available (No data)")

    wins = df[df["pnl"] >= 0]
    losses = df[df["pnl"] < 0]

    # Winners
    fig.add_trace(go.Scatter(
        x=wins["mae"], y=wins["mfe"],
        mode="markers",
        marker=dict(color=PALETTE["winner"], size=6, opacity=0.7),
        name="Winning Trades",
        hovertemplate="MAE: $%{x:,.0f}<br>MFE: $%{y:,.0f}<br>Net PnL: $%{customdata:,.0f}<extra></extra>",
        customdata=wins["pnl"]
    ))

    # Losers
    fig.add_trace(go.Scatter(
        x=losses["mae"], y=losses["mfe"],
        mode="markers",
        marker=dict(color=PALETTE["loser"], size=6, opacity=0.7),
        name="Losing Trades",
        hovertemplate="MAE: $%{x:,.0f}<br>MFE: $%{y:,.0f}<br>Net PnL: $%{customdata:,.0f}<extra></extra>",
        customdata=losses["pnl"]
    ))
    
    # 45-degree line y = -x
    # Set the limit to the maximum actual MAE to avoid stretching the X-axis uncontrollably
    min_mae = df["mae"].min() if not df.empty else -100
    fig.add_shape(
        type="line", line=dict(dash="dash", color="gray", width=1),
        x0=0, y0=0, x1=min_mae, y1=abs(min_mae)
    )

    fig.update_layout(
        title=dict(text="MFE vs MAE Scatter", font_size=13, x=0),
        xaxis=dict(title="MAE ($) [Adverse]", autorange="reversed"),  # negative values
        yaxis=dict(title="MFE ($) [Favorable]"),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        legend=dict(orientation="h", y=-0.2, x=0, font_size=10)
    )
    return fig


def build_exit_reason_chart(trades: pd.DataFrame) -> go.Figure:
    """
    Builds a grouped bar chart of total PnL and trade count by exit_reason.
    """
    fig = go.Figure()
    if trades.empty or "exit_reason" not in trades.columns:
        return _empty_fig("Exit Reason Not Available")

    grouped = trades.groupby("exit_reason").agg(
        total_pnl=("pnl", "sum"),
        count=("pnl", "count")
    ).sort_values("total_pnl", ascending=False)
    
    if grouped.empty:
        return _empty_fig("No Exit Reason Data")

    colors = [PALETTE["winner"] if val >= 0 else PALETTE["loser"] for val in grouped["total_pnl"]]

    fig.add_trace(go.Bar(
        x=grouped.index.astype(str),
        y=grouped["total_pnl"],
        text=[f"${v:,.0f}<br>(n={c})" for v, c in zip(grouped["total_pnl"], grouped["count"])],
        textposition="auto",
        marker_color=colors,
    ))

    fig.update_layout(
        title=dict(text="Total PnL by Exit Reason", font_size=13, x=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        showlegend=False
    )
    return fig


def build_vol_regime_chart(trades: pd.DataFrame) -> go.Figure:
    """
    Builds a bar chart showing PnL in different entry volatility regimes.
    """
    fig = go.Figure()
    if trades.empty or "entry_volatility" not in trades.columns:
        return _empty_fig("Vol Regime Not Available")

    df = trades.dropna(subset=["entry_volatility", "pnl"]).copy()
    if df.empty:
        return _empty_fig("No Volatility Data")

    # Use fixed percentile bins matching the strategy's filtering logic
    # Try to read dynamic thresholds from data, fallback to global settings
    _settings = BacktestSettings()
    v_min = trades["vol_min_pct"].iloc[0] if "vol_min_pct" in trades.columns and not pd.isna(trades["vol_min_pct"].iloc[0]) else _settings.vol_min_pct_default
    v_max = trades["vol_max_pct"].iloc[0] if "vol_max_pct" in trades.columns and not pd.isna(trades["vol_max_pct"].iloc[0]) else _settings.vol_max_pct_default

    try:
        df['vol_bucket'] = pd.cut(
            df['entry_volatility'], 
            bins=[0.0, v_min, v_max, 1.0], 
            labels=['Compression', 'Normal', 'Panic'],
            include_lowest=True
        )
    except Exception:
        # Fallback if bins are problematic
        return _empty_fig("Invalid Volatility Distribution")

    grouped = df.groupby("vol_bucket", observed=False).agg(
        avg_pnl=("pnl", "mean"),
        count=("pnl", "count")
    )
    
    colors = [PALETTE["winner"] if val >= 0 else PALETTE["loser"] for val in grouped["avg_pnl"]]

    fig.add_trace(go.Bar(
        x=grouped.index.astype(str),
        y=grouped["avg_pnl"],
        text=[f"${v:,.0f}<br>(n={c})" for v, c in zip(grouped["avg_pnl"], grouped["count"])],
        textposition="auto",
        marker_color=colors,
    ))

    fig.update_layout(
        title=dict(text="Avg PnL by Entry Volatility", font_size=13, x=0),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color=PALETTE["text"],
        showlegend=False
    )
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="gray"))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig
