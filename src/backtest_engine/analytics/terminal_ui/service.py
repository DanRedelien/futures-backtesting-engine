from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.backtest_engine.analytics.dashboard.core.data_layer import (
    ArtifactLoadStatus,
    ResultBundle,
    result_bundle_service,
)
from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_risk_profile,
    build_strategy_equity_curve,
)
from src.backtest_engine.analytics.dashboard.risk_analysis.models import (
    RiskDashboardConfig,
    RiskProfile,
    StressMultipliers,
)
from src.backtest_engine.analytics.terminal_ui.cache import (
    TerminalCachePolicy,
    TerminalCacheService,
)
from src.backtest_engine.analytics.terminal_ui.jobs import TerminalQueueConfig


@dataclass(frozen=True)
class TerminalRuntimeContext:
    """Runtime settings shared by the terminal UI routes."""

    risk_config: RiskDashboardConfig
    risk_free_rate: float
    instrument_specs: Dict[str, Dict[str, float]]
    rolling_sharpe_window_days: int
    cache_service: TerminalCacheService
    queue_config: TerminalQueueConfig
    benchmark_color: str
    strategy_colors: Tuple[str, ...]
    portfolio_total_color: str
    long_color: str
    short_color: str
    drawdown_color: str
    rolling_sharpe_color: str
    rolling_vol_colors: Tuple[str, ...]
    stress_colors: Tuple[str, ...]
    var_colors: Tuple[str, ...]
    max_chart_points: int = 2000
    trade_page_size: int = 25


@dataclass(frozen=True)
class TerminalShellContext:
    """Template context for the terminal dashboard shell."""

    mode: str
    mode_label: str
    artifact_id: str
    artifact_created_at: str
    engine_version: str
    schema_version: str
    tabs: Tuple[Dict[str, str], ...]
    default_tab: str
    risk_scope_options: Tuple[Dict[str, str], ...]
    default_risk_scope: str
    exit_strategy_options: Tuple[Dict[str, str], ...]
    default_exit_strategy: str
    hidden_panels: Tuple[str, ...]
    default_correlation_horizon: str
    stress_defaults: StressMultipliers
    stress_bounds: Dict[str, float]
    report_preview: str
    scenario_notice: str


def _fallback_terminal_ui_settings() -> Any:
    """Returns a default terminal UI settings object when env-backed loading fails."""
    try:
        from src.backtest_engine.settings import TerminalUISettings

        return TerminalUISettings()
    except Exception:
        return None


def _setting_value(settings_obj: Any, attribute_name: str, fallback: Any) -> Any:
    """Reads one setting value while keeping fallback behavior explicit."""
    if settings_obj is None:
        return fallback
    return getattr(settings_obj, attribute_name, fallback)


def _build_terminal_runtime_context(
    *,
    terminal_ui_settings: Any,
    risk_free_rate: float,
    instrument_specs: Dict[str, Dict[str, float]],
    redis_url: Optional[str],
    queue_name: str,
) -> TerminalRuntimeContext:
    """Builds the shared runtime context from one resolved settings object."""
    resolved_settings = terminal_ui_settings or _fallback_terminal_ui_settings()
    return TerminalRuntimeContext(
        risk_config=RiskDashboardConfig(
            var_confidence_primary=float(
                _setting_value(resolved_settings, "dashboard_risk_var_primary_confidence", 0.95)
            ),
            var_confidence_tail=float(
                _setting_value(resolved_settings, "dashboard_risk_var_tail_confidence", 0.99)
            ),
            rolling_var_window_days=int(
                _setting_value(resolved_settings, "dashboard_risk_rolling_var_window_days", 60)
            ),
            rolling_vol_windows=(
                int(_setting_value(resolved_settings, "dashboard_risk_rolling_vol_window_short_days", 20)),
                int(_setting_value(resolved_settings, "dashboard_risk_rolling_vol_window_medium_days", 50)),
                int(_setting_value(resolved_settings, "dashboard_risk_rolling_vol_window_long_days", 100)),
            ),
            stress_slider_min=float(
                _setting_value(resolved_settings, "dashboard_stress_slider_min_multiplier", 1.0)
            ),
            stress_slider_max=float(
                _setting_value(resolved_settings, "dashboard_stress_slider_max_multiplier", 5.0)
            ),
            stress_slider_step=float(
                _setting_value(resolved_settings, "dashboard_stress_slider_step", 0.5)
            ),
            stress_defaults=StressMultipliers(
                volatility=float(
                    _setting_value(
                        resolved_settings,
                        "dashboard_stress_volatility_default_multiplier",
                        2.0,
                    )
                ),
                slippage=float(
                    _setting_value(
                        resolved_settings,
                        "dashboard_stress_slippage_default_multiplier",
                        3.0,
                    )
                ),
                commission=float(
                    _setting_value(
                        resolved_settings,
                        "dashboard_stress_commission_default_multiplier",
                        2.0,
                    )
                ),
            ),
        ),
        risk_free_rate=risk_free_rate,
        instrument_specs=instrument_specs,
        rolling_sharpe_window_days=int(
            _setting_value(resolved_settings, "rolling_sharpe_window_days", 90)
        ),
        cache_service=TerminalCacheService(
            redis_url=redis_url,
            policy=TerminalCachePolicy(
                correlation_ttl_seconds=int(
                    _setting_value(
                        resolved_settings,
                        "terminal_correlation_cache_ttl_seconds",
                        600,
                    )
                ),
                risk_ttl_seconds=int(
                    _setting_value(resolved_settings, "terminal_risk_cache_ttl_seconds", 300)
                ),
            ),
        ),
        queue_config=TerminalQueueConfig(
            redis_url=redis_url,
            queue_name=queue_name,
            timeout_seconds=int(
                _setting_value(resolved_settings, "terminal_job_timeout_seconds", 1800)
            ),
            max_retries=int(_setting_value(resolved_settings, "terminal_job_max_retries", 2)),
            sse_max_updates_per_second=float(
                _setting_value(resolved_settings, "terminal_sse_max_updates_per_second", 2.0)
            ),
        ),
        benchmark_color=str(_setting_value(resolved_settings, "terminal_benchmark_color", "#777777")),
        strategy_colors=tuple(
            _setting_value(
                resolved_settings,
                "terminal_strategy_colors",
                ["#22C55E", "#3B82F6", "#EAB308", "#F97316", "#EC4899", "#A855F7"],
            )
        ),
        portfolio_total_color=str(
            _setting_value(resolved_settings, "terminal_portfolio_total_color", "#FFFFFF")
        ),
        long_color=str(_setting_value(resolved_settings, "terminal_long_color", "#22C55E")),
        short_color=str(_setting_value(resolved_settings, "terminal_short_color", "#EF4444")),
        drawdown_color=str(_setting_value(resolved_settings, "terminal_drawdown_color", "#EF4444")),
        rolling_sharpe_color=str(
            _setting_value(resolved_settings, "terminal_rolling_sharpe_color", "#FFFFFF")
        ),
        rolling_vol_colors=tuple(
            _setting_value(
                resolved_settings,
                "terminal_rolling_vol_colors",
                ["#22C55E", "#3B82F6", "#F59E0B"],
            )
        ),
        stress_colors=tuple(
            _setting_value(
                resolved_settings,
                "terminal_stress_colors",
                ["#22C55E", "#F59E0B", "#EF4444"],
            )
        ),
        var_colors=tuple(
            _setting_value(
                resolved_settings,
                "terminal_var_colors",
                ["#F59E0B", "#FBBF24", "#EF4444"],
            )
        ),
        max_chart_points=int(_setting_value(resolved_settings, "terminal_max_chart_points", 2000)),
        trade_page_size=int(_setting_value(resolved_settings, "terminal_trade_page_size", 25)),
    )


def load_terminal_runtime_context() -> TerminalRuntimeContext:
    """
    Loads dashboard runtime settings from BacktestSettings.

    Methodology:
        The terminal UI should respect the same settings-backed configuration as
        the legacy analytics views while keeping request handlers free of UI
        framework imports.
    """
    try:
        from src.backtest_engine.settings import BacktestSettings

        settings = BacktestSettings()
        return _build_terminal_runtime_context(
            terminal_ui_settings=settings.terminal_ui,
            risk_free_rate=float(settings.risk_free_rate),
            instrument_specs=dict(settings.instrument_specs),
            redis_url=settings.terminal_redis_url,
            queue_name=settings.terminal_queue_name,
        )
    except Exception:
        return _build_terminal_runtime_context(
            terminal_ui_settings=None,
            risk_free_rate=0.0,
            instrument_specs={},
            redis_url=None,
            queue_name="terminal-scenarios",
        )


def load_terminal_bundle(results_dir: Optional[str] = None) -> Optional[ResultBundle]:
    """Loads the active artifact bundle using the shared loader contract."""
    return result_bundle_service.load_bundle(results_dir=results_dir, use_cache=False)


def inspect_terminal_bundle(results_dir: Optional[str] = None) -> ArtifactLoadStatus:
    """Inspects the active artifact bundle using the shared loader contract."""
    return result_bundle_service.inspect_bundle(results_dir=results_dir)


def _artifact_cache_identity(bundle: ResultBundle) -> Tuple[str, str]:
    """Returns the artifact identity used by cache keys."""
    metadata = bundle.artifact_metadata
    if metadata is None:
        return "unknown", "unknown"
    return metadata.artifact_id, metadata.schema_version


def _cache_payload(
    runtime: TerminalRuntimeContext,
    bundle: ResultBundle,
    *,
    metric_name: str,
    parameters: Dict[str, Any],
    ttl_seconds: int,
    compute_fn: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Caches one JSON-ready payload under the terminal cache key contract."""
    artifact_id, schema_version = _artifact_cache_identity(bundle)
    return runtime.cache_service.get_or_compute(
        metric_name=metric_name,
        artifact_id=artifact_id,
        schema_version=schema_version,
        parameters=parameters,
        ttl_seconds=ttl_seconds,
        compute_fn=compute_fn,
    )


def _downsample_series(series: pd.Series, max_points: int) -> pd.Series:
    """Returns a roughly even downsampled series for browser-friendly payloads."""
    clean = series.dropna().astype(float)
    if clean.empty or len(clean) <= max_points:
        return clean
    step = max(1, len(clean) // max_points)
    sampled = clean.iloc[::step]
    if sampled.index[-1] != clean.index[-1]:
        sampled = pd.concat([sampled, clean.iloc[[-1]]])
        sampled = sampled[~sampled.index.duplicated(keep="last")]
    return sampled


def _points_from_series(series: pd.Series, max_points: int) -> List[Dict[str, float | str]]:
    """Converts a time series into JSON-ready chart points."""
    sampled = _downsample_series(series, max_points=max_points)
    return [
        {"time": idx.isoformat(), "value": float(value)}
        for idx, value in sampled.items()
    ]


def _format_currency(value: float) -> str:
    """Formats a currency metric for terminal cards and tables."""
    if pd.isna(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(float(value)):,.0f}"


def _format_pct(value: float) -> str:
    """Formats a percentage metric for terminal cards and tables."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}%"


def _format_ratio(value: float) -> str:
    """Formats a generic float metric for terminal cards and tables."""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def _build_risk_profile_for_scope(
    bundle: ResultBundle,
    runtime: TerminalRuntimeContext,
    risk_scope: str,
    stress: StressMultipliers,
) -> RiskProfile:
    """Builds a risk profile for portfolio, single, or one selected strategy."""
    if bundle.run_type == "portfolio" and risk_scope not in {"portfolio", "single"}:
        slot_lookup = {strategy_name: slot_id for slot_id, strategy_name in (bundle.slots or {}).items()}
        if risk_scope in slot_lookup:
            slot_id = slot_lookup[risk_scope]
            strategy_equity = build_strategy_equity_curve(
                history=bundle.history,
                slot_id=str(slot_id),
                slot_weight=(
                    float(bundle.slot_weights.get(slot_id))
                    if bundle.slot_weights and slot_id in bundle.slot_weights
                    else None
                ),
                slot_count=len(bundle.slots or {}),
            )
            strategy_trades = (
                bundle.trades[bundle.trades["strategy"] == risk_scope].copy()
                if bundle.trades is not None
                and not bundle.trades.empty
                and "strategy" in bundle.trades.columns
                else pd.DataFrame()
            )
            return build_risk_profile(
                label=risk_scope,
                equity=strategy_equity,
                trades_df=strategy_trades,
                instrument_specs=runtime.instrument_specs,
                primary_confidence=runtime.risk_config.var_confidence_primary,
                tail_confidence=runtime.risk_config.var_confidence_tail,
                rolling_var_window_days=runtime.risk_config.rolling_var_window_days,
                rolling_vol_windows=runtime.risk_config.rolling_vol_windows,
                stress_multipliers=stress,
                risk_free_rate=runtime.risk_free_rate,
            )

    return build_risk_profile(
        label="Portfolio" if bundle.run_type == "portfolio" else "Single Asset",
        equity=bundle.history["total_value"],
        trades_df=bundle.trades,
        instrument_specs=runtime.instrument_specs,
        primary_confidence=runtime.risk_config.var_confidence_primary,
        tail_confidence=runtime.risk_config.var_confidence_tail,
        rolling_var_window_days=runtime.risk_config.rolling_var_window_days,
        rolling_vol_windows=runtime.risk_config.rolling_vol_windows,
        stress_multipliers=stress,
        risk_free_rate=runtime.risk_free_rate,
    )
