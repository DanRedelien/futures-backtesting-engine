"""
src/backtest_engine/analytics/dashboard/core/transforms/__init__.py
Pure computation layer for PnL Analysis dashboard blocks.
"""
from .pnl import (
    build_bar_pnl_matrix,
    resample_pnl_to_horizon,
    compute_pnl_dist_stats,
    build_strategy_equity_curve,
    derive_daily_pnl_from_equity,
)
from .correlations import (
    compute_strategy_correlation,
    compute_exposure_correlation,
)
from .risk import (
    compute_drawdown_series,
    compute_drawdown_episodes,
    compute_var_es_metrics,
    compute_rolling_var_es,
    compute_rolling_volatility,
    compute_annualised_sharpe,
    compute_rolling_sharpe,
    build_risk_profile,
)
from .stress import (
    compute_stress_scenarios,
)
from .summaries import (
    compute_strategy_decomp,
    compute_per_strategy_summary,
    compute_exit_summary,
)
from .strategy_stats import (
    TERMINAL_STRATEGY_STATS_COLUMNS,
    compute_strategy_stats,
    compute_strategy_stats_map,
)

__all__ = [
    'build_bar_pnl_matrix',
    'resample_pnl_to_horizon',
    'compute_pnl_dist_stats',
    'build_strategy_equity_curve',
    'derive_daily_pnl_from_equity',
    'compute_strategy_correlation',
    'compute_exposure_correlation',
    'compute_drawdown_series',
    'compute_drawdown_episodes',
    'compute_var_es_metrics',
    'compute_rolling_var_es',
    'compute_rolling_volatility',
    'compute_annualised_sharpe',
    'compute_rolling_sharpe',
    'build_risk_profile',
    'compute_stress_scenarios',
    'compute_strategy_decomp',
    'compute_per_strategy_summary',
    'compute_exit_summary',
    'TERMINAL_STRATEGY_STATS_COLUMNS',
    'compute_strategy_stats',
    'compute_strategy_stats_map',
]
