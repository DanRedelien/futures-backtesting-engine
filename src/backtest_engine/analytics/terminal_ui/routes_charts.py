from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.backtest_engine.analytics.dashboard.risk_analysis.models import StressMultipliers
from src.backtest_engine.analytics.terminal_ui.chart_builders import (
    build_decomposition_chart_payload,
    build_equity_chart_payload,
    build_exposure_correlation_payload,
    build_pnl_distribution_payload,
    build_rolling_sharpe_payload,
    build_strategy_correlation_payload,
)
from src.backtest_engine.analytics.terminal_ui.constants import (
    DEFAULT_CORRELATION_HORIZON,
)
from src.backtest_engine.analytics.terminal_ui.risk_builders import (
    build_risk_drawdown_payload,
    build_risk_stress_payload,
    build_risk_var_payload,
    build_risk_volatility_payload,
)
from src.backtest_engine.analytics.terminal_ui.service import (
    TerminalRuntimeContext,
    load_terminal_bundle,
)


def register_chart_routes(
    app: FastAPI,
    *,
    runtime: TerminalRuntimeContext,
    results_dir: Optional[str],
    build_stress_from_query: Callable[[Request, StressMultipliers], StressMultipliers],
) -> None:
    """Registers JSON endpoints for charts and risk payloads."""

    def _load_bundle_json() -> tuple[Optional[Any], Optional[JSONResponse]]:
        bundle = load_terminal_bundle(results_dir=results_dir)
        if bundle is None:
            return None, JSONResponse({"error": "bundle_unavailable"}, status_code=404)
        return bundle, None

    def _json_response(builder: Callable[..., Dict[str, Any]], **kwargs: Any) -> JSONResponse:
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(builder(bundle, runtime, **kwargs))

    @app.get("/api/charts/equity", response_class=JSONResponse)
    def equity_chart() -> JSONResponse:
        """Returns JSON for the primary TradingView equity chart."""
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(build_equity_chart_payload(bundle, runtime))

    @app.get("/api/charts/rolling-sharpe", response_class=JSONResponse)
    def rolling_sharpe_chart() -> JSONResponse:
        """Returns JSON for the rolling-Sharpe mini-chart."""
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(build_rolling_sharpe_payload(bundle, runtime))

    @app.get("/api/charts/pnl-distribution", response_class=JSONResponse)
    def pnl_distribution_chart() -> JSONResponse:
        """Returns JSON for the daily PnL distribution chart."""
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(build_pnl_distribution_payload(bundle))

    @app.get("/api/charts/decomposition", response_class=JSONResponse)
    def decomposition_chart() -> JSONResponse:
        """Returns JSON for the strategy decomposition bar chart."""
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(build_decomposition_chart_payload(bundle, runtime))

    @app.get("/api/charts/strategy-correlation", response_class=JSONResponse)
    def strategy_correlation_chart(horizon: str = DEFAULT_CORRELATION_HORIZON) -> JSONResponse:
        """Returns JSON for the strategy-correlation heatmap."""
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(build_strategy_correlation_payload(bundle, runtime, horizon=horizon))

    @app.get("/api/charts/exposure-correlation", response_class=JSONResponse)
    def exposure_correlation_chart(horizon: str = DEFAULT_CORRELATION_HORIZON) -> JSONResponse:
        """Returns JSON for the exposure-correlation heatmap."""
        bundle, error_response = _load_bundle_json()
        if error_response is not None:
            return error_response
        return JSONResponse(build_exposure_correlation_payload(bundle, runtime, horizon=horizon))

    @app.get("/api/charts/risk-var", response_class=JSONResponse)
    def risk_var_chart(
        request: Request,
        risk_scope: str = "portfolio",
    ) -> JSONResponse:
        """Returns JSON for the rolling VaR / ES chart."""
        stress = build_stress_from_query(request, runtime.risk_config.stress_defaults)
        return _json_response(
            build_risk_var_payload,
            risk_scope=risk_scope,
            stress=stress,
        )

    @app.get("/api/charts/risk-drawdown", response_class=JSONResponse)
    def risk_drawdown_chart(
        request: Request,
        risk_scope: str = "portfolio",
    ) -> JSONResponse:
        """Returns JSON for the drawdown chart."""
        stress = build_stress_from_query(request, runtime.risk_config.stress_defaults)
        return _json_response(
            build_risk_drawdown_payload,
            risk_scope=risk_scope,
            stress=stress,
        )

    @app.get("/api/charts/risk-volatility", response_class=JSONResponse)
    def risk_volatility_chart(
        request: Request,
        risk_scope: str = "portfolio",
    ) -> JSONResponse:
        """Returns JSON for the rolling-volatility chart."""
        stress = build_stress_from_query(request, runtime.risk_config.stress_defaults)
        return _json_response(
            build_risk_volatility_payload,
            risk_scope=risk_scope,
            stress=stress,
        )

    @app.get("/api/charts/risk-stress", response_class=JSONResponse)
    def risk_stress_chart(
        request: Request,
        risk_scope: str = "portfolio",
    ) -> JSONResponse:
        """Returns JSON for the stress-preview chart."""
        stress = build_stress_from_query(request, runtime.risk_config.stress_defaults)
        return _json_response(
            build_risk_stress_payload,
            risk_scope=risk_scope,
            stress=stress,
        )
