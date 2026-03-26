from __future__ import annotations

DEFAULT_BOTTOM_TAB = "pnl-distribution"
DEFAULT_CORRELATION_HORIZON = "1d"
DEFAULT_RISK_VOL_WINDOW_DAYS = 20
DEFAULT_RISK_SHARPE_HORIZON = "1m"

BASE_BOTTOM_TABS: tuple[dict[str, str], ...] = (
    {"id": "pnl-distribution", "label": "PnL Distribution"},
    {"id": "strategy-stats", "label": "Strategy Stats"},
    {"id": "risk", "label": "Risk"},
    {"id": "stress-testing", "label": "Stress Testing"},
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
LABEL_MEAN = "Mean"
Y_AXIS_CUMULATIVE_PNL = "Cumulative PnL ($)"

# PnL distribution histogram defaults.
PNL_DIST_BASE_BINS_CAP = 40
PNL_DIST_BASE_BINS_FLOOR = 10
PNL_DIST_DETAILED_BINS_CAP = 120
PNL_DIST_DETAILED_BINS_FLOOR = 30
PNL_DIST_DETAILED_MULTIPLIER = 3
PNL_DIST_SAMPLE_BIN_FLOOR = 20
PNL_DIST_SAMPLE_BIN_CAP = 160
PNL_DIST_FD_WIDTH_FACTOR = 2.0

# Decomposition chart defaults.
DECOMPOSITION_SORT_COLUMN = "PnL Contrib (%)"
DECOMPOSITION_RISK_COLUMN = "Risk Contrib (%)"
DECOMPOSITION_PNL_CONTRIB_COLUMN = "PnL Contrib (%)"

# Exit analysis chart titles.
TITLE_EXIT_MFE_MAE = "MFE vs MAE"
TITLE_EXIT_PNL_DECAY = "PnL Decay (Forward Horizon)"
TITLE_EXIT_HOLDING_TIME = "Avg PnL by Holding Time"
TITLE_EXIT_VOL_REGIME = "Avg PnL by Exit volatility"
TITLE_EXIT_REASON = "Total PnL by Exit Reason"

# Default sub-view for the exit-analysis detail workspace.
DEFAULT_EXIT_DETAIL_VIEW = "trade-log"
