from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.constants import (
    DEFAULT_RISK_VOL_WINDOW_DAYS,
    DEFAULT_RISK_SHARPE_HORIZON,
    LABEL_PEAK_THRESHOLD,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    _build_risk_profile_for_scope,
    _cache_payload,
    _format_currency,
    _format_pct,
    _format_ratio,
    _points_from_series,
)

if TYPE_CHECKING:
    from src.backtest_engine.analytics.terminal_ui.service import TerminalRuntimeContext


def _risk_cache_parameters(
    risk_scope: str,
    stress: StressMultipliers,
    *,
    risk_vol_window_days: int | None = None,
    risk_sharpe_horizon: str | None = None,
) -> Dict[str, Any]:
    """Builds cache-sensitive parameters for derived risk payloads."""
    params: Dict[str, Any] = {
        "risk_scope": risk_scope,
        "stress": {
            "volatility": float(stress.volatility),
            "slippage": float(stress.slippage),
            "commission": float(stress.commission),
        },
    }
    if risk_vol_window_days is not None:
        params["risk_vol_window_days"] = int(risk_vol_window_days)
    if risk_sharpe_horizon is not None:
        params["risk_sharpe_horizon"] = str(risk_sharpe_horizon)
    return params


def _compute_sharpe_for_horizon(daily_returns: Any, risk_free_rate: float, horizon: str) -> float:
    """Computes annualized Sharpe on selected horizon returns."""
    if daily_returns is None:
        return float("nan")
    clean = daily_returns.dropna().astype(float)
    if len(clean) < 2:
        return float("nan")

    horizon_to_rule = {"1d": "1D", "1w": "1W", "1m": "1ME"}
    horizon_to_periods = {"1d": 252.0, "1w": 52.0, "1m": 12.0}
    resolved_horizon = horizon if horizon in horizon_to_rule else DEFAULT_RISK_SHARPE_HORIZON

    if resolved_horizon != "1d":
        periodic = (1.0 + clean).resample(horizon_to_rule[resolved_horizon]).prod() - 1.0
        clean = periodic.dropna().astype(float)
        if len(clean) < 2:
            return float("nan")

    periods_per_year = horizon_to_periods[resolved_horizon]
    period_rf = (1.0 + float(risk_free_rate)) ** (1.0 / periods_per_year) - 1.0
    excess = clean - period_rf
    volatility = float(clean.std())
    if volatility <= 1e-8:
        return float("nan")
    return float(excess.mean() / volatility * np.sqrt(periods_per_year))


def _build_risk_panel_context_uncached(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds uncached risk summary context before TTL caching."""
    profile = _build_risk_profile_for_scope(
        bundle=bundle,
        runtime=runtime,
        risk_scope=risk_scope,
        stress=stress,
    )
    summary = profile.summary
    available_windows = tuple(int(window) for window in runtime.risk_config.rolling_vol_windows)
    resolved_vol_window = (
        int(DEFAULT_RISK_VOL_WINDOW_DAYS)
        if int(DEFAULT_RISK_VOL_WINDOW_DAYS) in available_windows
        else int(available_windows[0])
    )
    resolved_sharpe_horizon = (
        DEFAULT_RISK_SHARPE_HORIZON
        if DEFAULT_RISK_SHARPE_HORIZON in {"1d", "1w", "1m"}
        else "1m"
    )
    selected_vol_column = f"{resolved_vol_window}D"
    latest_vol = (
        float(profile.rolling_vol[selected_vol_column].dropna().iloc[-1])
        if not profile.rolling_vol.empty and selected_vol_column in profile.rolling_vol.columns
        and not profile.rolling_vol[selected_vol_column].dropna().empty
        else float("nan")
    )
    selected_sharpe = _compute_sharpe_for_horizon(
        profile.daily_returns,
        runtime.risk_free_rate,
        resolved_sharpe_horizon,
    )
    primary_label = int(runtime.risk_config.var_confidence_primary * 100)
    tail_label = int(runtime.risk_config.var_confidence_tail * 100)
    stress_rows = [
        {
            "Scenario": scenario.label,
            "Final PnL": _format_currency(float(scenario.metrics.get("final_pnl", float("nan")))),
            "Delta": _format_currency(float(scenario.pnl_delta)),
            f"VaR {primary_label}": _format_currency(float(scenario.metrics.get("var_primary", float("nan")))),
            "Max DD": _format_pct(float(scenario.metrics.get("max_drawdown_pct", float("nan")))),
            "Sharpe": _format_ratio(float(scenario.metrics.get("sharpe", float("nan")))),
        }
        for scenario in profile.stress_results
    ]
    return {
        "profile_label": profile.label,
        "summary_cards": [
            {"label": f"VaR {primary_label}", "value": _format_currency(float(summary.get("var_primary", float("nan"))))},
            {"label": f"ES {primary_label}", "value": _format_currency(float(summary.get("es_primary", float("nan"))))},
            {"label": f"VaR {tail_label}", "value": _format_currency(float(summary.get("var_tail", float("nan"))))},
            {"label": "Max DD", "value": _format_pct(float(summary.get("max_drawdown_pct", float("nan"))))},
            {"label": "DD 95", "value": _format_pct(float(summary.get("drawdown_95_pct", float("nan"))))},
            {"label": f"Latest Vol ({resolved_vol_window}D)", "value": _format_pct(latest_vol)},
            {"label": f"Sharpe ({resolved_sharpe_horizon.upper()})", "value": _format_ratio(selected_sharpe)},
            {"label": "Total PnL", "value": _format_currency(float(summary.get("total_pnl", float("nan"))))},
        ],
        "methodology_notice": (
            "Top ribbon Volatility/Sharpe are full-run performance metrics. "
            f"Risk cards show latest {resolved_vol_window}D rolling volatility and "
            f"{resolved_sharpe_horizon.upper()}-horizon Sharpe for the selected scope ({profile.label})."
        ),
        "stress_rows": stress_rows,
        "scenario_notice": (
            "Warning: Approximation only; does not change trade logic. "
            "Use Stress Testing for full queued reruns."
            if bundle.run_type == "portfolio"
            else ""
        ),
    }


def build_risk_panel_context(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds server-rendered context for the risk summary and stress tables."""
    return _cache_payload(
        runtime,
        bundle,
        metric_name="risk_panel_context",
        parameters=_risk_cache_parameters(
            risk_scope,
            stress,
            risk_vol_window_days=DEFAULT_RISK_VOL_WINDOW_DAYS,
            risk_sharpe_horizon=DEFAULT_RISK_SHARPE_HORIZON,
        ),
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: _build_risk_panel_context_uncached(
            bundle,
            runtime,
            risk_scope=risk_scope,
            stress=stress,
        ),
    )


def _build_risk_var_payload_uncached(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds an uncached rolling VaR / ES payload."""
    profile = _build_risk_profile_for_scope(bundle, runtime, risk_scope, stress)
    rolling = profile.rolling_var.dropna(subset=["pnl"], how="all")
    return {
        "title": f"{profile.label} Tail Risk",
        "series": [
            {
                "name": "Daily PnL",
                "color": runtime.portfolio_total_color,
                "points": _points_from_series(rolling["pnl"], runtime.max_chart_points) if "pnl" in rolling else [],
            },
            {
                "name": f"VaR {int(runtime.risk_config.var_confidence_primary * 100)}",
                "color": runtime.var_colors[0],
                "points": _points_from_series(-rolling["var_primary"], runtime.max_chart_points) if "var_primary" in rolling else [],
            },
            {
                "name": f"ES {int(runtime.risk_config.var_confidence_primary * 100)}",
                "color": runtime.var_colors[1],
                "points": _points_from_series(-rolling["es_primary"], runtime.max_chart_points) if "es_primary" in rolling else [],
            },
            {
                "name": f"VaR {int(runtime.risk_config.var_confidence_tail * 100)}",
                "color": runtime.var_colors[2],
                "points": _points_from_series(-rolling["var_tail"], runtime.max_chart_points) if "var_tail" in rolling else [],
            },
        ],
    }


def build_risk_var_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds the rolling VaR / ES payload for the risk panel."""
    return _cache_payload(
        runtime,
        bundle,
        metric_name="risk_var",
        parameters=_risk_cache_parameters(risk_scope, stress),
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: _build_risk_var_payload_uncached(bundle, runtime, risk_scope=risk_scope, stress=stress),
    )


def _build_risk_drawdown_payload_uncached(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds an uncached drawdown payload."""
    profile = _build_risk_profile_for_scope(bundle, runtime, risk_scope, stress)
    return {
        "title": f"{profile.label} Drawdown",
        "series": [
            {
                "name": "Drawdown",
                "color": runtime.drawdown_color,
                "points": _points_from_series(profile.drawdown, runtime.max_chart_points),
            }
        ],
        "thresholds": [{"value": 0.0, "label": LABEL_PEAK_THRESHOLD}],
    }


def build_risk_drawdown_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds the drawdown curve payload for the risk panel."""
    return _cache_payload(
        runtime,
        bundle,
        metric_name="risk_drawdown",
        parameters=_risk_cache_parameters(risk_scope, stress),
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: _build_risk_drawdown_payload_uncached(bundle, runtime, risk_scope=risk_scope, stress=stress),
    )


def _build_risk_volatility_payload_uncached(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds an uncached rolling-volatility payload."""
    profile = _build_risk_profile_for_scope(bundle, runtime, risk_scope, stress)
    series = []
    for index, column_name in enumerate(profile.rolling_vol.columns.tolist()):
        series.append(
            {
                "name": column_name,
                "color": runtime.rolling_vol_colors[index % len(runtime.rolling_vol_colors)],
                "points": _points_from_series(profile.rolling_vol[column_name], runtime.max_chart_points),
            }
        )
    return {
        "title": f"{profile.label} Rolling Volatility",
        "series": series,
    }


def build_risk_volatility_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds the rolling-volatility payload for the risk panel."""
    return _cache_payload(
        runtime,
        bundle,
        metric_name="risk_volatility",
        parameters=_risk_cache_parameters(risk_scope, stress),
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: _build_risk_volatility_payload_uncached(bundle, runtime, risk_scope=risk_scope, stress=stress),
    )


def _build_risk_stress_payload_uncached(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds an uncached stress-preview payload."""
    profile = _build_risk_profile_for_scope(bundle, runtime, risk_scope, stress)
    baseline_points = _points_from_series(profile.equity, runtime.max_chart_points)
    series = [
        {
            "name": profile.label,
            "color": runtime.portfolio_total_color,
            "points": baseline_points,
        }
    ]
    for index, scenario in enumerate(profile.stress_results):
        series.append(
            {
                "name": scenario.label,
                "color": runtime.stress_colors[index % len(runtime.stress_colors)],
                "points": _points_from_series(scenario.equity, runtime.max_chart_points),
            }
        )
    return {
        "title": f"{profile.label} Stress Test Preview (Approximation)",
        "series": series,
    }


def build_risk_stress_payload(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    *,
    risk_scope: str,
    stress: StressMultipliers,
) -> Dict[str, Any]:
    """Builds the stress-preview payload for the risk panel."""
    return _cache_payload(
        runtime,
        bundle,
        metric_name="risk_stress",
        parameters=_risk_cache_parameters(risk_scope, stress),
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: _build_risk_stress_payload_uncached(bundle, runtime, risk_scope=risk_scope, stress=stress),
    )
