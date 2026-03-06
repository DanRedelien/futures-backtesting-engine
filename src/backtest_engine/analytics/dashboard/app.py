"""
src/backtest_engine/analytics/dashboard/app.py

Streamlit entry point for the backtest research dashboard.

Responsibility: Only the page layout and orchestration of chart/component
calls.  No chart building (charts.py), no file I/O (components.py).

Layout (per approved wireframe):
    Row 1 : Equity Curve [left 70%]  | Terminal Report Log [right 30%]
    Row 2 : Drawdown % [full width]
    Row 3 : P&L Distribution [left 50%] | Exit Breakdown table [right 50%]

Usage:
    python run.py --backtest --strategy zscore --dashboard
    # or manually:
    streamlit run src/backtest_engine/analytics/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest_engine.analytics.dashboard.data_layer import load_result_bundle
from src.backtest_engine.analytics.dashboard.charts import (
    build_equity_figure,
    build_drawdown_figure,
    build_pnl_hist_figure,
)
from src.backtest_engine.analytics.dashboard.components import render_exit_table


def main() -> None:
    """
    Streamlit page entry point.

    Methodology / Workflow:
        Pure read-only viewer — the engine does NOT run inside Streamlit.
        Follows the quant standard:
            1. run backtest → 2. save Parquet → 3. open dashboard.
    """
    st.set_page_config(
        page_title="Backtest Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 0rem; }
            pre { font-size: 0.72rem; line-height: 1.35; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 1. Load Data Layer
    bundle = load_result_bundle()

    if bundle is None:
        st.title("Backtest Dashboard")
        st.error(
            "No backtest results found in `results/`. "
            "Run a backtest first (`python run.py --backtest --strategy <name>`) "
            "then refresh this page."
        )
        return

    # Dynamic Title
    mode_label = "Portfolio" if bundle.run_type == "portfolio" else "Single-Asset"
    st.title(f"Backtest Dashboard — {mode_label} Mode")

    # 2. Main Tab Structure (Phase 1)
    tab_pnl, tab_risk = st.tabs(["PnL Analysis", "Temporary / Risk"])

    with tab_pnl:
        # ── Row 1: Equity Curve | Terminal Log ────────────────────────────────────
        col_eq, col_log = st.columns([7, 3])

        with col_eq:
            fig_eq = build_equity_figure(
                history=bundle.history, 
                trades=bundle.trades, 
                benchmark=bundle.benchmark,
                run_type=bundle.run_type,
                slots=bundle.slots,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_log:
            st.markdown("**Terminal Report**")
            st.code(bundle.report or "No report available.", language="")

        # ── Row 2: P&L Distribution | Exit Breakdown ──────────────────────────────
        col_pnl_dist, col_exit = st.columns(2)

        with col_pnl_dist:
            fig_pnl = build_pnl_hist_figure(bundle.trades)
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_exit:
            st.markdown("**Exit Breakdown**")
            render_exit_table(bundle.trades)


    with tab_risk:
        st.markdown("*(Temporary placement for Phase 2 Risk Module)*")
        # ── Row 1: Drawdown (full width) ───────────────────────────────────────────
        fig_dd = build_drawdown_figure(bundle.history)
        st.plotly_chart(fig_dd, use_container_width=True)


if __name__ == "__main__":
    main()

