"""
src/backtest_engine/analytics/dashboard/app.py

Streamlit entry point for the backtest research dashboard.

Responsibility: Only page layout and orchestration.
No chart building (charts.py / *_charts.py), no file I/O (components.py),
no math (transforms.py).

Layout — PnL Analysis tab:
    Row 1 : Equity Curve [left 70%]  | Terminal Report Log [right 30%]
    Row 2 : Drawdown % [full width]
    Row 3 : PnL Distribution [left 50%] | Exit Breakdown table [right 50%]
    -- Portfolio mode only --
    Row 4 : Strategy PnL Decomposition table (full width)
    Row 5 : Decomposition bar chart (full width)
    Row 6 : Strategy Correlation [left 50%] | Exposure Correlation [right 50%]

Usage:
    streamlit run src/backtest_engine/analytics/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest_engine.analytics.dashboard.core.data_layer import (
    ResultBundle,
    inspect_result_bundle,
    load_result_bundle,
)
from src.backtest_engine.analytics.dashboard.pnl_analysis.drawdown_chart import build_drawdown_figure
from src.backtest_engine.analytics.dashboard.pnl_analysis.distribution_chart import build_pnl_distribution_figure
from src.backtest_engine.analytics.dashboard.pnl_analysis.correlation_heatmap import build_correlation_heatmap
from src.backtest_engine.analytics.dashboard.pnl_analysis.equity_chart import (
    build_portfolio_equity_figure,
    build_decomp_chart,
    build_single_equity_figure,
)
from src.backtest_engine.analytics.dashboard.pnl_analysis.exit_decomposition import (
    build_holding_time_chart,
    build_pnl_decay_chart,
    build_mfe_mae_scatter,
    build_exit_reason_chart,
    build_vol_regime_chart,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.risk_tab import render_risk_tab
from src.backtest_engine.analytics.dashboard.risk_analysis.models import (
    RiskDashboardConfig,
    StressMultipliers,
)
from src.backtest_engine.analytics.dashboard.simulation_analysis.sim_placeholder import render_simulation_tab
from src.backtest_engine.analytics.dashboard.core.components import (
    render_decomp_table,
    render_correlation_horizon_selector,
    render_dataframe,
)
from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_bar_pnl_matrix,
    compute_strategy_decomp,
    compute_strategy_correlation,
    compute_exposure_correlation,
    compute_rolling_sharpe,
    compute_pnl_dist_stats,
    compute_per_strategy_summary,
    compute_exit_summary,
)


# ── Cached computation wrappers ────────────────────────────────────────────────
# @st.cache_data lives here (not in transforms package) to keep transforms
# importable without a live Streamlit session — critical for unit tests.

@st.cache_data(show_spinner=False)
def _cached_bar_pnl_matrix(history: pd.DataFrame, slots: dict) -> pd.DataFrame:
    """Cache the bar PnL matrix by DataFrame hash."""
    return build_bar_pnl_matrix(history, slots)

@st.cache_data(show_spinner=False)
def _cached_rolling_sharpe(history: pd.DataFrame, window_days: int) -> pd.Series:
    """Cache rolling Sharpe by DataFrame hash + window config."""
    return compute_rolling_sharpe(history, window_days=window_days)

@st.cache_data(show_spinner=False)
def _cached_derive_daily_pnl(history: pd.DataFrame) -> pd.Series:
    """Cache the daily PnL derivation."""
    bar_pnl = history["total_value"].diff().fillna(0.0)
    return bar_pnl.resample("1D").sum()

@st.cache_data(show_spinner=False)
def _cached_compute_pnl_dist_stats(daily_pnl: pd.Series) -> dict:
    """Cache distribution stats computation."""
    return compute_pnl_dist_stats(daily_pnl)

@st.cache_data(show_spinner=False)
def _cached_compute_per_strategy_summary(trades: pd.DataFrame, slots: dict, history: pd.DataFrame, instrument_closes: pd.DataFrame, slot_weights: dict) -> dict:
    """Cache per-strategy stats computation."""
    return compute_per_strategy_summary(trades, slots, history, instrument_closes, slot_weights)

@st.cache_data(show_spinner=False)
def _cached_compute_exit_summary(trades: pd.DataFrame, slots: dict) -> pd.DataFrame:
    """Cache exit summary computation."""
    return compute_exit_summary(trades, slots)

@st.cache_data(show_spinner=False)
def _cached_compute_strategy_decomp(
    trades_df: pd.DataFrame,
    history: pd.DataFrame,
    slots: dict,
    tail_confidence: float,
) -> pd.DataFrame:
    """Cache strategy decomposition computation."""
    return compute_strategy_decomp(
        trades_df=trades_df,
        history=history,
        slots=slots,
        tail_confidence=tail_confidence,
    )

@st.cache_data(show_spinner=False)
def _cached_compute_strategy_correlation(bar_pnl_matrix: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Cache strategy correlation matrix computation."""
    return compute_strategy_correlation(bar_pnl_matrix, horizon=horizon)

@st.cache_data(show_spinner=False)
def _cached_compute_exposure_correlation(exposure: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, list[str]]:
    """Cache exposure correlation matrix computation."""
    return compute_exposure_correlation(exposure, horizon=horizon)


def _render_pnl_tab(bundle: ResultBundle, window_days: int, tail_confidence: float) -> None:
    """
    Renders the complete PnL Analysis tab for both modes.

    Args:
        bundle: Fully loaded ResultBundle from data_layer.
        window_days: Rolling Sharpe window from settings.
        tail_confidence: Confidence used for tail-day decomposition metrics.
    """
    is_portfolio: bool = bundle.run_type == "portfolio"

    # ── Pre-compute ────────────────────────────────────────────────────────────
    daily_pnl: pd.Series = _cached_derive_daily_pnl(bundle.history)
    dist_stats: dict     = _cached_compute_pnl_dist_stats(daily_pnl)

    rolling_sharpe    = None
    strategy_summaries = None
    bar_pnl_matrix    = None

    if is_portfolio and bundle.slots:
        bar_pnl_matrix   = _cached_bar_pnl_matrix(bundle.history, bundle.slots)
        strategy_summaries = _cached_compute_per_strategy_summary(
            bundle.trades, bundle.slots, bundle.history, bundle.instrument_closes, getattr(bundle, "slot_weights", None)
        )
        rolling_sharpe   = _cached_rolling_sharpe(
            bundle.history, window_days=window_days
        )

    # ── Row 1: Equity Curve & Drawdown | Terminal Report ───────────────────────
    col_eq, col_log = st.columns([7, 3])

    with col_eq:
        if is_portfolio:
            fig_eq = build_portfolio_equity_figure(
                history=bundle.history,
                benchmark=bundle.benchmark,
                slots=bundle.slots,
                rolling_sharpe=rolling_sharpe,
                strategy_summaries=strategy_summaries,
            )
        else:
            fig_eq = build_single_equity_figure(
                history=bundle.history,
                trades=bundle.trades,
                benchmark=bundle.benchmark,
            )
        st.plotly_chart(fig_eq, width="stretch")

        fig_dd = build_drawdown_figure(bundle.history)
        st.plotly_chart(fig_dd, width="stretch")

    with col_log:
        st.code(bundle.report or "No report available.", language="")

    st.divider()

    # ── Row 3: Exit Analysis Summary (Screener Table) ──────────────────────────
    st.markdown("#### Exit Analysis Summary")
    st.caption("Select a strategy row to open detailed exit charts.")

    st_summ = pd.DataFrame()
    if is_portfolio and bundle.slots:
        st_summ = _cached_compute_exit_summary(bundle.trades, bundle.slots)
    else:
        st_summ = _cached_compute_exit_summary(bundle.trades, {"single": "Single Asset"})

    if not st_summ.empty:
        event = render_dataframe(
            st_summ,
            selection_mode="single-row",
            on_select="rerun"
        )
        if event and event.selection and event.selection.rows:
            selected_idx = event.selection.rows[0]
            strat_name = st_summ.iloc[selected_idx]["Strategy"]
            if is_portfolio:
                strat_trades = bundle.trades[bundle.trades["strategy"] == strat_name] if not bundle.trades.empty else pd.DataFrame()
            else:
                strat_trades = bundle.trades

            _show_exit_analysis_dialog(strat_name, strat_trades)
    else:
        st.info("No detailed exit data available.")

    st.divider()

    # ── Row 4: PnL Distribution (full width since legacy table is removed) ─────
    st.markdown("#### Daily PnL Distribution")
    fig_dist = build_pnl_distribution_figure(daily_pnl, dist_stats)
    
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.plotly_chart(fig_dist, width="stretch")

    if not is_portfolio:
        return

    st.divider()

    # ── Row 5: Strategy PnL Decomposition table (full width) ──────────────────
    st.markdown("#### Strategy PnL Decomposition")
    decomp_df = _cached_compute_strategy_decomp(
        trades_df=bundle.trades,
        history=bundle.history,
        slots=bundle.slots or {},
        tail_confidence=tail_confidence,
    )
    render_decomp_table(decomp_df)

    # ── Row 5: Decomposition bar chart (full width, below table) ──────────────
    fig_decomp = build_decomp_chart(decomp_df)
    st.plotly_chart(fig_decomp, width="stretch")

    st.divider()

    # ── Row 6: Correlations ────────────────────────────────────────────────────
    st.markdown("#### Correlations")
    st.caption(
        "Strategy: incremental bar PnL — not cumulative equity.  "
        "Exposure: avg instrument exposure ratio per period."
    )
    # Shared horizon selector — same window applies to BOTH heatmaps so they
    # are directly comparable (no confusion from different time scales).
    horizon = render_correlation_horizon_selector(key="corr_horizon")

    col_strat_corr, col_exp_corr = st.columns(2)

    with col_strat_corr:
        st.markdown("**Strategy PnL Correlation**")
        strat_corr_matrix = _cached_compute_strategy_correlation(
            bar_pnl_matrix if bar_pnl_matrix is not None else pd.DataFrame(),
            horizon=horizon,
        )
        fig_strat_corr = build_correlation_heatmap(
            strat_corr_matrix,
            title=f"Strategy PnL Correlation ({horizon})",
        )
        st.plotly_chart(fig_strat_corr, width="stretch")

    with col_exp_corr:
        st.markdown("**Exposure Correlation (by Instrument)**")
        if bundle.exposure is not None and not bundle.exposure.empty:
            exp_corr_matrix, dropped_inst = _cached_compute_exposure_correlation(
                bundle.exposure, horizon=horizon
            )
            if exp_corr_matrix.empty:
                if dropped_inst:
                    st.info(
                        f"Insufficient data for instrument(s): {', '.join(dropped_inst)}. "
                        "Need at least 2 valid instruments to compute correlation.",
                        icon="ℹ️"
                    )
                else:
                    st.info(
                        "Exposure Correlation requires at least 2 distinct instruments.",
                        icon="ℹ️"
                    )
            else:
                fig_exp_corr = build_correlation_heatmap(
                    exp_corr_matrix,
                    title=f"Exposure Correlation ({horizon})",
                )
                st.plotly_chart(fig_exp_corr, width="stretch")
                
                if dropped_inst:
                    st.info(f"Insufficient data for instrument(s): {', '.join(dropped_inst)}", icon="ℹ️")
        else:
            st.info("No exposure data (`exposure.parquet` missing from results/portfolio/).")



def main() -> None:
    """
    Streamlit page entry point.

    Primarily a baseline artifact viewer.

    Methodology:
        The dashboard reads existing artifacts for baseline analysis. The Risk
        Analysis tab may also launch explicit child portfolio scenario reruns on
        button press, but passive widget changes never trigger engine work.
    """
    st.set_page_config(
        page_title="Backtest Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 0rem; }
            pre { font-size: 0.55rem; line-height: 1.08; }
            
            /* Expand dialog width to fit screen content better */
            div[data-testid="stDialog"] > div[role="dialog"],
            div[data-testid="stModal"] > div[role="dialog"],
            div[role="dialog"] {
                width: 90vw !important;
                max-width: 1200px !important;
            }
            
            div[role="dialog"] .block-container {
                max-width: 100% !important;
                width: 100% !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Settings (read-only, no engine import paths needed in Streamlit) ───────
    settings_warning: str | None = None
    try:
        from src.backtest_engine.settings import BacktestSettings
        _settings = BacktestSettings()
        terminal_ui_settings = _settings.terminal_ui
        window_days: int  = int(terminal_ui_settings.rolling_sharpe_window_days)
        risk_free_rate: float = float(_settings.risk_free_rate)
        instrument_specs: dict = _settings.instrument_specs
        risk_config = RiskDashboardConfig(
            var_confidence_primary=float(terminal_ui_settings.dashboard_risk_var_primary_confidence),
            var_confidence_tail=float(terminal_ui_settings.dashboard_risk_var_tail_confidence),
            rolling_var_window_days=int(terminal_ui_settings.dashboard_risk_rolling_var_window_days),
            rolling_vol_windows=(
                int(terminal_ui_settings.dashboard_risk_rolling_vol_window_short_days),
                int(terminal_ui_settings.dashboard_risk_rolling_vol_window_medium_days),
                int(terminal_ui_settings.dashboard_risk_rolling_vol_window_long_days),
            ),
            stress_slider_min=float(terminal_ui_settings.dashboard_stress_slider_min_multiplier),
            stress_slider_max=float(terminal_ui_settings.dashboard_stress_slider_max_multiplier),
            stress_slider_step=float(terminal_ui_settings.dashboard_stress_slider_step),
            stress_defaults=StressMultipliers(
                volatility=float(terminal_ui_settings.dashboard_stress_volatility_default_multiplier),
                slippage=float(terminal_ui_settings.dashboard_stress_slippage_default_multiplier),
                commission=float(terminal_ui_settings.dashboard_stress_commission_default_multiplier),
            ),
        )
    except Exception as exc:
        settings_warning = (
            "Dashboard settings could not be loaded from `BacktestSettings.terminal_ui`; "
            f"falling back to safe defaults. Details: {exc}"
        )
        window_days = 90   # safe default
        risk_free_rate = 0.0
        instrument_specs = {}
        risk_config = RiskDashboardConfig(
            var_confidence_primary=0.95,
            var_confidence_tail=0.99,
            rolling_var_window_days=60,
            rolling_vol_windows=(20, 50, 100),
            stress_slider_min=1.0,
            stress_slider_max=5.0,
            stress_slider_step=0.5,
            stress_defaults=StressMultipliers(
                volatility=2.0,
                slippage=3.0,
                commission=2.0,
            ),
        )

    if settings_warning:
        st.warning(settings_warning)

    with st.sidebar:
        if st.button("Refresh Artifacts", key="refresh_artifacts_button"):
            load_result_bundle.clear()

    # ── Load results ───────────────────────────────────────────────────────────
    bundle = load_result_bundle()

    if bundle is None:
        load_status = inspect_result_bundle()
        st.title("Backtest Dashboard")
        if load_status.state == "incomplete":
            missing_suffix = ""
            if load_status.missing_files:
                missing_suffix = f" Missing: {', '.join(load_status.missing_files)}."
            st.error(
                "Backtest artifacts are incomplete and were not loaded. "
                f"{load_status.reason}{missing_suffix}"
            )
        else:
            st.error(
                "No backtest results found in `results/`. "
                "Run a backtest first:\n\n"
                "```\npython run.py --backtest --strategy <name>\n"
                "python run.py --portfolio-backtest\n```\n\n"
                "Then open the dashboard separately:\n\n"
                "```\nstreamlit run src/backtest_engine/analytics/dashboard/app.py\n```"
            )
        return

    mode_label = "Portfolio" if bundle.run_type == "portfolio" else "Single-Asset"
    st.title(f"Backtest Dashboard — {mode_label} Mode")

    tab_pnl, tab_risk, tab_sim = st.tabs(["PnL Analysis", "Risk Analysis", "Simulation Analysis"])

    with tab_pnl:
        _render_pnl_tab(
            bundle,
            window_days=window_days,
            tail_confidence=risk_config.var_confidence_primary,
        )

    with tab_risk:
        render_risk_tab(
            bundle,
            config=risk_config,
            instrument_specs=instrument_specs,
            risk_free_rate=risk_free_rate,
        )
        
    with tab_sim:
        render_simulation_tab()

@st.dialog("Detailed Exit Analysis", width="large")
def _show_exit_analysis_dialog(title: str, trades: pd.DataFrame):
    st.markdown(f"**{title} Exits**")
    if trades.empty:
        st.write("No trades available.")
        return
        
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_mfe_mae_scatter(trades), width="stretch")
    with c2:
        st.plotly_chart(build_pnl_decay_chart(trades), width="stretch")
        
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(build_holding_time_chart(trades), width="stretch")
    with c4:
        st.plotly_chart(build_vol_regime_chart(trades), width="stretch")
        
    st.plotly_chart(build_exit_reason_chart(trades), width="stretch")



if __name__ == "__main__":
    main()
