"""
Risk Analysis tab renderer.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from src.backtest_engine.analytics.dashboard.core.data_layer import (
    ResultBundle,
    inspect_result_bundle,
    load_result_bundle,
)
from src.backtest_engine.analytics.dashboard.core.scenario_runner import (
    list_portfolio_scenarios,
    run_portfolio_scenario,
    scenario_matches_baseline,
)
from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_risk_profile,
    build_strategy_equity_curve,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.charts import (
    build_scenario_comparison_figure,
    build_drawdown_curve_figure,
    build_drawdown_distribution_figure,
    build_equity_curve_figure,
    build_risk_distribution_figure,
    build_rolling_volatility_figure,
    build_stress_test_figure,
    build_var_es_figure,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import (
    RiskDashboardConfig,
    RiskProfile,
    StressMultipliers,
)


def _fmt_currency(value: float) -> str:
    """Formats a dollar value for dashboard metrics."""
    if pd.isna(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _fmt_pct(value: float) -> str:
    """Formats a percentage value for dashboard metrics."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def _fmt_days(value: float) -> str:
    """Formats a duration in days for dashboard metrics."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}d"


def _render_formatted_table(
    frame: pd.DataFrame,
    *,
    currency_cols: tuple[str, ...] = (),
    pct_cols: tuple[str, ...] = (),
    float_cols: tuple[str, ...] = (),
) -> None:
    """Renders a dataframe while keeping numeric columns sortable underneath."""
    format_map: Dict[str, object] = {}
    for col in currency_cols:
        if col in frame.columns:
            format_map[col] = _fmt_currency
    for col in pct_cols:
        if col in frame.columns:
            format_map[col] = _fmt_pct
    for col in float_cols:
        if col in frame.columns:
            format_map[col] = "{:.2f}"

    styled = frame.style.format(format_map, na_rep="N/A") if format_map else frame
    st.dataframe(styled, hide_index=True, width="stretch")


def _render_metric_comparison_table(rows: list[dict]) -> None:
    """Renders baseline/scenario comparison rows while preserving numeric sorting."""
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No scenario comparison metrics available.")
        return

    value_columns = ["Baseline", "Scenario", "Delta"]
    metric_type = table.pop("_metric_type")
    styled = table.style.format(na_rep="N/A")

    currency_rows = table.index[metric_type == "currency"]
    pct_rows = table.index[metric_type == "pct"]
    float_rows = table.index[metric_type == "float"]

    if len(currency_rows) > 0:
        styled = styled.format(_fmt_currency, subset=pd.IndexSlice[currency_rows, value_columns])
    if len(pct_rows) > 0:
        styled = styled.format(_fmt_pct, subset=pd.IndexSlice[pct_rows, value_columns])
    if len(float_rows) > 0:
        styled = styled.format("{:.2f}", subset=pd.IndexSlice[float_rows, value_columns])

    st.dataframe(styled, hide_index=True, width="stretch")


def _render_stress_controls(key_prefix: str, config: RiskDashboardConfig) -> StressMultipliers:
    """Renders stress-test sliders and returns the selected multipliers."""
    st.markdown("##### Stress Preview")
    st.caption(
        "This is a fast post-processing preview, not a backtest rerun. "
        "It transforms the realized daily PnL path and derived equity curve only. "
        "It does not change signals, trades, fills, stops, sizing, or allocation decisions. "
        "Volatility preview rescales realized daily PnL dispersion, while slippage and commission preview "
        "adjust total realized trading costs relative to baseline."
    )
    col_container, _ = st.columns([2, 1])
    with col_container:
        col_vol, col_slip, col_comm = st.columns(3)
    with col_vol:
        volatility = st.slider(
            "Volatility x",
            min_value=float(config.stress_slider_min),
            max_value=float(config.stress_slider_max),
            value=float(config.stress_defaults.volatility),
            step=float(config.stress_slider_step),
            key=f"{key_prefix}_stress_volatility",
        )
    with col_slip:
        slippage = st.slider(
            "Slippage x",
            min_value=float(config.stress_slider_min),
            max_value=float(config.stress_slider_max),
            value=float(config.stress_defaults.slippage),
            step=float(config.stress_slider_step),
            key=f"{key_prefix}_stress_slippage",
        )
    with col_comm:
        commission = st.slider(
            "Commission x",
            min_value=float(config.stress_slider_min),
            max_value=float(config.stress_slider_max),
            value=float(config.stress_defaults.commission),
            step=float(config.stress_slider_step),
            key=f"{key_prefix}_stress_commission",
        )

    return StressMultipliers(
        volatility=float(volatility),
        slippage=float(slippage),
        commission=float(commission),
    )


def _get_stress_state(key_prefix: str, config: RiskDashboardConfig) -> StressMultipliers:
    """
    Returns the current stress state from Streamlit session state.

    Methodology:
        Risk tables and drilldowns must agree on the same live preview inputs
        during a rerun. Reading the widget-backed session state before the
        controls are rendered keeps upstream profile construction aligned with
        the currently selected slider values instead of silently falling back to
        config defaults.
    """
    return StressMultipliers(
        volatility=float(st.session_state.get(f"{key_prefix}_stress_volatility", config.stress_defaults.volatility)),
        slippage=float(st.session_state.get(f"{key_prefix}_stress_slippage", config.stress_defaults.slippage)),
        commission=float(st.session_state.get(f"{key_prefix}_stress_commission", config.stress_defaults.commission)),
    )


def _render_summary_cards(profile: RiskProfile, config: RiskDashboardConfig) -> None:
    """Renders top-level scalar risk metrics."""
    summary = profile.summary
    conf_primary = int(config.var_confidence_primary * 100)
    conf_tail = int(config.var_confidence_tail * 100)

    row_1 = st.columns(4)
    row_1[0].metric(f"VaR {conf_primary} Loss", _fmt_currency(summary.get("var_primary", float("nan"))))
    row_1[1].metric(f"VaR {conf_tail} Loss", _fmt_currency(summary.get("var_tail", float("nan"))))
    row_1[2].metric(f"ES {conf_primary} Loss", _fmt_currency(summary.get("es_primary", float("nan"))))
    row_1[3].metric(f"ES {conf_tail} Loss", _fmt_currency(summary.get("es_tail", float("nan"))))

    row_2 = st.columns(4)
    row_2[0].metric("Max DD", _fmt_pct(summary.get("max_drawdown_pct", float("nan"))))
    row_2[1].metric("DD 95", _fmt_pct(summary.get("drawdown_95_pct", float("nan"))))
    row_2[2].metric("Max DD Duration", _fmt_days(summary.get("max_drawdown_duration_days", float("nan"))))
    row_2[3].metric("Latest Vol", _fmt_pct(summary.get("latest_vol_pct", float("nan"))))


def _render_stress_table(profile: RiskProfile, config: RiskDashboardConfig) -> None:
    """Renders the stress-test summary table."""
    if not profile.stress_results:
        st.info("No stress preview scenarios available.")
        return

    primary_label = int(config.var_confidence_primary * 100)
    rows = []
    for scenario in profile.stress_results:
        rows.append(
            {
                "Scenario": scenario.label,
                "Final PnL": float(scenario.metrics.get("final_pnl", float("nan"))),
                "Delta vs Baseline": float(scenario.pnl_delta),
                f"VaR {primary_label} Loss": float(scenario.metrics.get("var_primary", float("nan"))),
                f"ES {primary_label} Loss": float(scenario.metrics.get("es_primary", float("nan"))),
                "Max DD": float(scenario.metrics.get("max_drawdown_pct", float("nan"))),
                "Sharpe": float(scenario.metrics.get("sharpe", float("nan"))),
            }
        )

    table = pd.DataFrame(rows)
    _render_formatted_table(
        table,
        currency_cols=("Final PnL", "Delta vs Baseline", f"VaR {primary_label} Loss", f"ES {primary_label} Loss"),
        pct_cols=("Max DD",),
        float_cols=("Sharpe",),
    )


def _render_risk_profile_core(profile: RiskProfile, config: RiskDashboardConfig) -> None:
    """Renders charts and summary blocks except stress-preview outputs."""
    _render_summary_cards(profile, config)

    st.markdown("##### Tail Risk")
    col_var, col_dist = st.columns(2)
    with col_var:
        st.plotly_chart(
            build_var_es_figure(
                profile.rolling_var,
                primary_confidence=config.var_confidence_primary,
                tail_confidence=config.var_confidence_tail,
                title="Rolling Historical VaR / ES",
            ),
            width="stretch",
        )
    with col_dist:
        st.plotly_chart(
            build_risk_distribution_figure(
                profile.daily_pnl,
                profile.summary,
                primary_confidence=config.var_confidence_primary,
                tail_confidence=config.var_confidence_tail,
                title="Daily PnL Tail Distribution",
            ),
            width="stretch",
        )

    st.markdown("##### Drawdown Analysis")
    col_eq, col_dd = st.columns(2)
    with col_eq:
        st.plotly_chart(
            build_equity_curve_figure(profile.equity, title="Equity Curve"),
            width="stretch",
        )
    with col_dd:
        st.plotly_chart(
            build_drawdown_curve_figure(profile.drawdown, title="Drawdown Curve"),
            width="stretch",
        )

    col_dd_dist, col_vol = st.columns(2)
    with col_dd_dist:
        st.plotly_chart(
            build_drawdown_distribution_figure(
                profile.drawdown_episodes,
                title="Drawdown Distribution",
            ),
            width="stretch",
        )
    with col_vol:
        st.plotly_chart(
            build_rolling_volatility_figure(
                profile.rolling_vol,
                title="Rolling Volatility (Annualized Returns)",
            ),
            width="stretch",
        )


def _render_stress_analysis(profile: RiskProfile, config: RiskDashboardConfig) -> None:
    """Renders the stress-preview charts and table."""
    st.plotly_chart(
        build_stress_test_figure(profile.stress_results, title="Stress Preview Equity Paths"),
        width="stretch",
    )
    _render_stress_table(profile, config)


def _build_portfolio_profile_from_bundle(
    bundle: ResultBundle,
    config: RiskDashboardConfig,
    instrument_specs: Dict[str, Dict[str, float]],
    risk_free_rate: float,
    label: str,
) -> RiskProfile:
    """Builds a realized portfolio risk profile from an artifact bundle."""
    return build_risk_profile(
        label=label,
        equity=bundle.history["total_value"],
        trades_df=bundle.trades,
        instrument_specs=instrument_specs,
        primary_confidence=config.var_confidence_primary,
        tail_confidence=config.var_confidence_tail,
        rolling_var_window_days=config.rolling_var_window_days,
        rolling_vol_windows=config.rolling_vol_windows,
        stress_multipliers=StressMultipliers(volatility=1.0, slippage=1.0, commission=1.0),
        risk_free_rate=risk_free_rate,
    )


def _render_scenario_comparison_table(
    baseline_profile: RiskProfile,
    scenario_profile: RiskProfile,
    config: RiskDashboardConfig,
) -> None:
    """Renders a baseline-vs-scenario metric comparison table."""
    primary_label = int(config.var_confidence_primary * 100)
    rows = [
        {
            "Metric": "Final PnL",
            "Baseline": float(baseline_profile.summary.get("total_pnl", float("nan"))),
            "Scenario": float(scenario_profile.summary.get("total_pnl", float("nan"))),
            "Delta": float(scenario_profile.summary.get("total_pnl", float("nan")))
            - float(baseline_profile.summary.get("total_pnl", float("nan"))),
            "_metric_type": "currency",
        },
        {
            "Metric": "End Equity",
            "Baseline": float(baseline_profile.equity.iloc[-1]) if not baseline_profile.equity.empty else float("nan"),
            "Scenario": float(scenario_profile.equity.iloc[-1]) if not scenario_profile.equity.empty else float("nan"),
            "Delta": (float(scenario_profile.equity.iloc[-1]) if not scenario_profile.equity.empty else float("nan"))
            - (float(baseline_profile.equity.iloc[-1]) if not baseline_profile.equity.empty else float("nan")),
            "_metric_type": "currency",
        },
        {
            "Metric": f"VaR {primary_label} Loss",
            "Baseline": float(baseline_profile.summary.get("var_primary", float("nan"))),
            "Scenario": float(scenario_profile.summary.get("var_primary", float("nan"))),
            "Delta": float(scenario_profile.summary.get("var_primary", float("nan")))
            - float(baseline_profile.summary.get("var_primary", float("nan"))),
            "_metric_type": "currency",
        },
        {
            "Metric": f"ES {primary_label} Loss",
            "Baseline": float(baseline_profile.summary.get("es_primary", float("nan"))),
            "Scenario": float(scenario_profile.summary.get("es_primary", float("nan"))),
            "Delta": float(scenario_profile.summary.get("es_primary", float("nan")))
            - float(baseline_profile.summary.get("es_primary", float("nan"))),
            "_metric_type": "currency",
        },
        {
            "Metric": "Max DD",
            "Baseline": float(baseline_profile.summary.get("max_drawdown_pct", float("nan"))),
            "Scenario": float(scenario_profile.summary.get("max_drawdown_pct", float("nan"))),
            "Delta": float(scenario_profile.summary.get("max_drawdown_pct", float("nan")))
            - float(baseline_profile.summary.get("max_drawdown_pct", float("nan"))),
            "_metric_type": "pct",
        },
        {
            "Metric": "Sharpe",
            "Baseline": float(baseline_profile.summary.get("sharpe", float("nan"))),
            "Scenario": float(scenario_profile.summary.get("sharpe", float("nan"))),
            "Delta": float(scenario_profile.summary.get("sharpe", float("nan")))
            - float(baseline_profile.summary.get("sharpe", float("nan"))),
            "_metric_type": "float",
        },
    ]
    _render_metric_comparison_table(rows)


def _render_scenario_backtest_section(
    bundle: ResultBundle,
    config: RiskDashboardConfig,
    instrument_specs: Dict[str, Dict[str, float]],
    risk_free_rate: float,
    stress_multipliers: StressMultipliers,
) -> None:
    """Renders explicit engine-rerun workflow and comparison for portfolio mode."""
    st.divider()
    st.markdown("#### Scenario Backtest")
    st.caption(
        "Runs a real child portfolio backtest only on button press. "
        "Scenario artifacts are written under `results/scenarios/<scenario_id>/` and do not overwrite the baseline. "
        "Current rerun interpretation: volatility scales YAML `target_portfolio_vol`, "
        "commission scales `commission_rate`, and slippage scales `spread_ticks`."
    )

    rerun_block_reason = ""
    if bundle.compatibility is not None and not bundle.compatibility.is_rerunnable:
        rerun_block_reason = bundle.compatibility.reason
        st.info(rerun_block_reason)

    if st.button(
        "Run Scenario Backtest",
        key="run_portfolio_scenario_button",
        disabled=bool(rerun_block_reason),
        help=rerun_block_reason or None,
    ):
        with st.spinner("Running scenario backtest..."):
            try:
                scenario_root = run_portfolio_scenario(bundle, stress_multipliers)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["active_scenario_root"] = str(scenario_root)
                load_result_bundle.clear()
                st.success(f"Scenario run completed: `{scenario_root.name}`")

    scenarios = list_portfolio_scenarios()
    if not scenarios:
        st.info("No scenario backtests available yet. Run one with the button above.")
        return

    active_root = st.session_state.get("active_scenario_root")
    labels = [item["label"] for item in scenarios]
    default_index = 0
    if active_root is not None:
        for idx, item in enumerate(scenarios):
            if str(item["root"]) == str(active_root):
                default_index = idx
                break

    selected_label = st.selectbox(
        "Scenario Artifacts",
        labels,
        index=default_index,
        key="scenario_artifact_selector",
    )
    selected_scenario = next(item for item in scenarios if item["label"] == selected_label)
    scenario_bundle = load_result_bundle(results_dir=str(selected_scenario["root"]))
    if scenario_bundle is None:
        load_status = inspect_result_bundle(results_dir=str(selected_scenario["root"]))
        if load_status.state == "incomplete":
            st.error(f"Selected scenario artifacts are incomplete: {load_status.reason}")
        else:
            st.error("Selected scenario artifacts could not be loaded.")
        return
    if not scenario_matches_baseline(bundle, scenario_bundle):
        st.warning(
            "This scenario bundle does not declare compatibility with the active baseline, "
            "so comparison is blocked."
        )
        return

    scenario_manifest = scenario_bundle.manifest or {}
    st.caption(
        f"Comparing baseline `{(bundle.manifest or {}).get('generated_at', 'unknown')}` "
        f"vs scenario `{scenario_manifest.get('scenario_id', 'unknown')}` "
        f"generated at `{scenario_manifest.get('generated_at', 'unknown')}`."
    )

    baseline_profile = _build_portfolio_profile_from_bundle(
        bundle=bundle,
        config=config,
        instrument_specs=instrument_specs,
        risk_free_rate=risk_free_rate,
        label="Baseline",
    )
    scenario_profile = _build_portfolio_profile_from_bundle(
        bundle=scenario_bundle,
        config=config,
        instrument_specs=instrument_specs,
        risk_free_rate=risk_free_rate,
        label=str(scenario_manifest.get("scenario_id", "Scenario")),
    )

    st.plotly_chart(
        build_scenario_comparison_figure(
            baseline_equity=baseline_profile.equity,
            scenario_equity=scenario_profile.equity,
            title="Baseline vs Scenario Backtest",
        ),
        width="stretch",
    )
    _render_scenario_comparison_table(baseline_profile, scenario_profile, config)


def _build_strategy_profiles(
    bundle: ResultBundle,
    config: RiskDashboardConfig,
    instrument_specs: Dict[str, Dict[str, float]],
    risk_free_rate: float,
    stress_multipliers: StressMultipliers,
) -> Dict[str, RiskProfile]:
    """Builds baseline strategy risk profiles for the portfolio strategy table."""
    profiles: Dict[str, RiskProfile] = {}
    strategy_count = len(bundle.slots or {})
    for slot_id, strategy_name in (bundle.slots or {}).items():
        strategy_equity = build_strategy_equity_curve(
            bundle.history,
            slot_id=str(slot_id),
            slot_weight=float(bundle.slot_weights.get(slot_id)) if bundle.slot_weights and slot_id in bundle.slot_weights else None,
            slot_count=strategy_count,
        )
        strategy_trades = (
            bundle.trades[bundle.trades["strategy"] == strategy_name]
            if bundle.trades is not None and not bundle.trades.empty and "strategy" in bundle.trades.columns
            else pd.DataFrame()
        )
        profiles[strategy_name] = build_risk_profile(
            label=strategy_name,
            equity=strategy_equity,
            trades_df=strategy_trades,
            instrument_specs=instrument_specs,
            primary_confidence=config.var_confidence_primary,
            tail_confidence=config.var_confidence_tail,
            rolling_var_window_days=config.rolling_var_window_days,
            rolling_vol_windows=config.rolling_vol_windows,
            stress_multipliers=stress_multipliers,
            risk_free_rate=risk_free_rate,
        )
    return profiles


def _render_strategy_snapshot_table(
    strategy_profiles: Dict[str, RiskProfile],
    config: RiskDashboardConfig,
) -> None:
    """Renders a compact cross-strategy risk summary for portfolio mode."""
    if not strategy_profiles:
        st.info("No strategy-level risk profiles available.")
        return

    primary_label = int(config.var_confidence_primary * 100)
    rows = []
    for strategy_name, profile in strategy_profiles.items():
        rows.append(
            {
                "Strategy": strategy_name,
                f"VaR {primary_label} Loss": float(profile.summary.get("var_primary", float("nan"))),
                f"ES {primary_label} Loss": float(profile.summary.get("es_primary", float("nan"))),
                "Max DD": float(profile.summary.get("max_drawdown_pct", float("nan"))),
                "DD 95": float(profile.summary.get("drawdown_95_pct", float("nan"))),
                "Latest Vol": float(profile.summary.get("latest_vol_pct", float("nan"))),
                "Sharpe": float(profile.summary.get("sharpe", float("nan"))),
            }
        )

    table = pd.DataFrame(rows).sort_values(by="Strategy")
    _render_formatted_table(
        table,
        currency_cols=(f"VaR {primary_label} Loss", f"ES {primary_label} Loss"),
        pct_cols=("Max DD", "DD 95", "Latest Vol"),
        float_cols=("Sharpe",),
    )


def render_risk_tab(
    bundle: ResultBundle,
    config: RiskDashboardConfig,
    instrument_specs: Dict[str, Dict[str, float]],
    risk_free_rate: float,
) -> None:
    """
    Renders the Risk Analysis tab.

    Methodology:
        Portfolio aggregate and per-strategy drilldown are rendered as separate
        sections because strategy views must not inherit portfolio-only effects
        such as diversification and cross-strategy netting.
    """
    if bundle is None or bundle.history is None or bundle.history.empty:
        st.info("No backtest history available for risk analysis.")
        return

    if bundle.run_type == "portfolio":
        st.markdown("#### Portfolio Risk")
        st.caption(
            "This section uses aggregate portfolio equity. Risk metrics here include diversification, "
            "netting and cross-strategy path interactions."
        )
        portfolio_top = st.container()
        portfolio_stress = _render_stress_controls("portfolio", config)
        
        portfolio_profile = build_risk_profile(
            label="Portfolio",
            equity=bundle.history["total_value"],
            trades_df=bundle.trades,
            instrument_specs=instrument_specs,
            primary_confidence=config.var_confidence_primary,
            tail_confidence=config.var_confidence_tail,
            rolling_var_window_days=config.rolling_var_window_days,
            rolling_vol_windows=config.rolling_vol_windows,
            stress_multipliers=portfolio_stress,
            risk_free_rate=risk_free_rate,
        )
        
        with portfolio_top:
            _render_risk_profile_core(portfolio_profile, config)
        _render_stress_analysis(portfolio_profile, config)
        _render_scenario_backtest_section(
            bundle=bundle,
            config=config,
            instrument_specs=instrument_specs,
            risk_free_rate=risk_free_rate,
            stress_multipliers=portfolio_stress,
        )
        st.divider()
        st.markdown("#### Strategy Risk Drilldown")
        st.caption(
            "The table and charts below isolate one strategy slot at a time. "
            "Portfolio-only diversification effects are intentionally excluded."
        )

        strategy_stress_state = _get_stress_state("strategy", config)
        strategy_profiles = _build_strategy_profiles(
            bundle,
            config,
            instrument_specs,
            risk_free_rate,
            strategy_stress_state,
        )
        _render_strategy_snapshot_table(strategy_profiles, config)

        strategy_names = sorted(strategy_profiles.keys())
        if not strategy_names:
            st.info("No strategy-level data available.")
            return

        selected_strategy = st.selectbox(
            "Strategy",
            strategy_names,
            key="risk_strategy_selector",
        )
        strategy_top = st.container()
        strategy_stress = _render_stress_controls("strategy", config)

        slot_lookup = {strategy_name: slot_id for slot_id, strategy_name in (bundle.slots or {}).items()}
        selected_slot_id = slot_lookup[selected_strategy]
        strategy_equity = build_strategy_equity_curve(
            bundle.history,
            slot_id=str(selected_slot_id),
            slot_weight=float(bundle.slot_weights.get(selected_slot_id)) if bundle.slot_weights and selected_slot_id in bundle.slot_weights else None,
            slot_count=len(bundle.slots or {}),
        )
        strategy_trades = (
            bundle.trades[bundle.trades["strategy"] == selected_strategy]
            if bundle.trades is not None and not bundle.trades.empty and "strategy" in bundle.trades.columns
            else pd.DataFrame()
        )
        strategy_profile = build_risk_profile(
            label=selected_strategy,
            equity=strategy_equity,
            trades_df=strategy_trades,
            instrument_specs=instrument_specs,
            primary_confidence=config.var_confidence_primary,
            tail_confidence=config.var_confidence_tail,
            rolling_var_window_days=config.rolling_var_window_days,
            rolling_vol_windows=config.rolling_vol_windows,
            stress_multipliers=strategy_stress,
            risk_free_rate=risk_free_rate,
        )
        with strategy_top:
            _render_risk_profile_core(strategy_profile, config)
        _render_stress_analysis(strategy_profile, config)
        return

    st.markdown("#### Strategy Risk")
    st.caption(
        "Single-asset mode analyses only the standalone strategy equity and daily PnL. "
        "No portfolio-only methods are applied here."
    )
    single_top = st.container()
    single_stress = _render_stress_controls("single", config)
    
    single_profile = build_risk_profile(
        label="Single Asset Strategy",
        equity=bundle.history["total_value"],
        trades_df=bundle.trades,
        instrument_specs=instrument_specs,
        primary_confidence=config.var_confidence_primary,
        tail_confidence=config.var_confidence_tail,
        rolling_var_window_days=config.rolling_var_window_days,
        rolling_vol_windows=config.rolling_vol_windows,
        stress_multipliers=single_stress,
        risk_free_rate=risk_free_rate,
    )
    with single_top:
        _render_risk_profile_core(single_profile, config)
    _render_stress_analysis(single_profile, config)
