from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from src.backtest_engine.analytics.dashboard.core.data_layer import ResultBundle
from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.constants import LABEL_PEAK_THRESHOLD
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


def _risk_cache_parameters(risk_scope: str, stress: StressMultipliers) -> Dict[str, Any]:
    """Builds cache-sensitive parameters for derived risk payloads."""
    return {
        "risk_scope": risk_scope,
        "stress": {
            "volatility": float(stress.volatility),
            "slippage": float(stress.slippage),
            "commission": float(stress.commission),
        },
    }


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
            {"label": "Latest Vol", "value": _format_pct(float(summary.get("latest_vol_pct", float("nan"))))},
            {"label": "Sharpe", "value": _format_ratio(float(summary.get("sharpe", float("nan"))))},
            {"label": "Total PnL", "value": _format_currency(float(summary.get("total_pnl", float("nan"))))},
        ],
        "stress_rows": stress_rows,
        "scenario_notice": (
            "Queue heavy scenario reruns from Operations. Simulation Analysis remains backlog-only."
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
        parameters=_risk_cache_parameters(risk_scope, stress),
        ttl_seconds=runtime.cache_service.policy.risk_ttl_seconds,
        compute_fn=lambda: _build_risk_panel_context_uncached(bundle, runtime, risk_scope=risk_scope, stress=stress),
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
        "title": f"{profile.label} Stress Preview",
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
