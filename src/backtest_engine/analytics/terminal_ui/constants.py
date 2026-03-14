from __future__ import annotations

DEFAULT_BOTTOM_TAB = "pnl-distribution"
DEFAULT_CORRELATION_HORIZON = "1d"

BASE_BOTTOM_TABS: tuple[dict[str, str], ...] = (
    {"id": "pnl-distribution", "label": "PnL Distribution"},
    {"id": "strategy-stats", "label": "Strategy Stats"},
    {"id": "risk", "label": "Risk"},
    {"id": "exit-analysis", "label": "Exit Analysis"},
    {"id": "operations", "label": "Operations"},
)
PORTFOLIO_ONLY_BOTTOM_TABS: tuple[dict[str, str], ...] = (
    {"id": "decomposition", "label": "Decomposition"},
    {"id": "correlations", "label": "Correlations"},
)

TITLE_EQUITY_CURVE = "Equity Curve"
TITLE_ROLLING_SHARPE = "Rolling Sharpe"
TITLE_PNL_DISTRIBUTION = "Daily PnL Distribution"
TITLE_STRATEGY_DECOMPOSITION = "Strategy Decomposition"
TITLE_STRATEGY_CORRELATION = "Strategy PnL Correlation"
TITLE_EXPOSURE_CORRELATION = "Exposure Correlation"

LABEL_BENCHMARK = "Benchmark"
LABEL_PORTFOLIO_TOTAL = "Portfolio Total"
LABEL_STRATEGY = "Strategy"
LABEL_LONG = "Long"
LABEL_SHORT = "Short"
LABEL_DRAWDOWN_PCT = "Drawdown %"
LABEL_ZERO_THRESHOLD = "Zero"
LABEL_PEAK_THRESHOLD = "Peak"
LABEL_VAR_95 = "VaR 95"
LABEL_CVAR_95 = "CVaR 95"
LABEL_VAR_99 = "VaR 99"
Y_AXIS_CUMULATIVE_PNL = "Cumulative PnL ($)"
