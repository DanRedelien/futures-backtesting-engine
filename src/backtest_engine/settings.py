"""
Backtest engine configuration.

Engine-level settings only.  Strategy-specific parameters are defined
inside each strategy class via get_search_space() and dataclass configs.
Loaded from environment variables (prefix: QUANT_BACKTEST_) or .env file.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BacktestSettings(BaseSettings):
    """
    Central configuration for the BacktestEngine.

    Covers data, portfolio accounting, execution simulation, risk limits,
    IB Fetcher connectivity, and Walk-Forward Optimizer scheduling.
    Strategy-specific parameters must NOT be added here.
    """

    model_config = SettingsConfigDict(
        env_prefix="QUANT_BACKTEST_",
        env_file=".env",
        extra="allow",
    )

    # ── System paths ───────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).parent.parent.parent
    cache_dir: Path = Field(default=Path("data/cache"), description="Parquet cache location")
    results_dir: Path = Field(default=Path("results"), description="Output directory for reports")

    # ── Primary instrument ─────────────────────────────────────────────────────
    default_symbol: str = "ES"

    # ── Bar settings ───────────────────────────────────────────────────────────
    low_interval: str = "30m"      # Base resolution used for data loading
    bar_type: str = "time"        # Options: "time", "volume", "range", "heikin_ashi"
    bar_size: float = 0.0         # Threshold for volume / range bar types

    # ── Portfolio & execution ──────────────────────────────────────────────────
    initial_capital: float = 1_000_000.0
    risk_free_rate: float = 0.02
    commission_rate: float = 2.5      # Per contract, in dollars
    max_slippage_ticks: int = 1       # Random slippage: uniform in [0, max]
    fixed_qty: int = 1                # Default number of contracts per signal
    random_seed: int = 42             # Seed for reproducible slippage simulation

    # ── Trading hours (exchange time, HH:MM strings) ───────────────────────────
    use_trading_hours: bool = True               # Toggle to enable/disable trading session limits
    trade_start_time: Optional[str] = "06:00"    # E.g. "06:00", None = disabled if use_trading_hours is False
    trade_end_time: Optional[str] = "15:00"      # E.g. "15:00", None = disabled if use_trading_hours is False
    eod_close_time: Optional[str] = "15:30"      # Force-close time; None = disabled

    # ── Risk limits (kill switches) ────────────────────────────────────────────
    max_daily_loss: Optional[float] = None      # Halt today if daily loss exceeds value
    max_drawdown_pct: Optional[float] = None    # Permanent halt at this drawdown %
    max_account_floor: Optional[float] = None   # Permanent halt below this equity level

    # ── Statistical Filters / Numerical Protections ────────────────────────────
    # Remark: Setting any of these to None will disable the corresponding numerical protection.
    hl_lambda_min: Optional[float] = 1e-4       # Minimum mean-reverting speed (slope) to consider the series stationary
    hl_max_cap: Optional[float] = 500.0         # Hard cap limit for calculated Half-Life (preventing explosion to infinity)
    
    # ── Volatility Regime Analytics Defaults ──────────────────────────────────
    vol_regime_window_default: int = 50         # Short-term window for rolling price std calculation
    vol_history_window_default: int = 500       # Historical window for percentile ranking of volatility
    vol_min_pct_default: float = 0.20           # Lower percentile bound: below this is 'Compression'
    vol_max_pct_default: float = 0.80           # Upper percentile bound: above this is 'Panic'

    # ── IB Fetcher ─────────────────────────────────────────────────────────────
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497          # 7497 = TWS paper; 4002 = Gateway paper
    ib_client_id: int = 1
    ib_timeout: int = 30
    max_historical_years: int = 2
    delayed_data_minutes: int = 15
    ib_use_rth: bool = False
    # ── Cache Management ───────────────────────────────────────────────────────
    max_cache_staleness_days: int = Field(
        default=5,
        description="Maximum allowed cache age in days for backtest runs.",
    )

    def get_ib_request_delay(self) -> float:
        """Standard pacing delay to respect IB rate limits (~6 req/min)."""
        return 11.0

    # ── Dashboard Analytics ──────────────────────────────────────────────────
    dashboard: "DashboardSettings" = Field(default_factory=lambda: DashboardSettings())

    # ── Walk-Forward Validation (WFV) scheduling ──────────────────────────────
    wfo_n_folds: int = 4             # Number of walk-forward folds
    wfo_test_size_pct: float = 0.20  # Fraction of total data used per test fold
    wfo_n_trials: int = 220          # Optuna trials per fold
    wfo_max_parameters: int = 6      # Strict maximum limit for optimized variables

    # Pruning / quality gates
    wfo_prune_min_trades: int = 8         # Minimum trades for a trial to pass
    wfo_prune_max_dd_pct: float = 35.0    # Max drawdown % before early pruning
    wfo_prune_target_trades_mult: int = 3  # target_trades = min_trades * this

    # ── Path helpers ───────────────────────────────────────────────────────────
    def get_results_path(self) -> Path:
        """Creates and returns the results directory path."""
        path = self.base_dir / self.results_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_path(self) -> Path:
        """Creates and returns the data cache directory path."""
        path = self.base_dir / self.cache_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Instrument specifications ──────────────────────────────────────────────
    instrument_specs: dict = Field(
        default_factory=lambda: {
            "ES":  {"tick_size": 0.25,  "multiplier": 50.0},
            "NQ":  {"tick_size": 0.25,  "multiplier": 20.0},
            "CL":  {"tick_size": 0.01,  "multiplier": 1000.0},
            "GC":  {"tick_size": 0.10,  "multiplier": 100.0},
            "SI":  {"tick_size": 0.005, "multiplier": 5000.0},
            "NG":  {"tick_size": 0.001, "multiplier": 10000.0},
            "PL":  {"tick_size": 0.10,  "multiplier": 50.0},
            "YM":  {"tick_size": 1.0,   "multiplier": 5.0},
            "RTY": {"tick_size": 0.10,  "multiplier": 50.0},
            "ZC":  {"tick_size": 0.25,  "multiplier": 50.0},
            "ZB":  {"tick_size": 0.03125, "multiplier": 1000.0},
            "6E":  {"tick_size": 0.00005, "multiplier": 125000.0},
        },
        description="Per-instrument tick sizes and dollar multipliers.",
    )

    def get_instrument_spec(self, symbol: str) -> dict:
        """
        Returns tick_size and multiplier for a symbol.

        Falls back to generic defaults if the symbol is unknown, allowing
        the engine to run on unlisted instruments without crashing.

        Args:
            symbol: Futures symbol string (e.g. 'ES').

        Returns:
            Dict with 'tick_size' and 'multiplier' keys.
        """
        return self.instrument_specs.get(symbol, {"tick_size": 0.01, "multiplier": 1.0})


class DashboardSettings(BaseModel):
    """
    Settings specifically for the Streamlit analytics dashboard.
    """
    # ── 1. Rolling Metrics (Windows) ──────────────────────────────────────────
    rolling_sharpe_window_days: int = Field(
        default=90,
        description="Window in calendar days for rolling Sharpe calculation in the dashboard.",
    )
    dashboard_risk_rolling_var_window_days: int = Field(
        default=60,
        description="Trailing daily window for rolling VaR / ES curves in the dashboard risk tab.",
    )
    dashboard_risk_rolling_vol_window_short_days: int = Field(
        default=20,
        description="Short rolling volatility window in trading days for the dashboard risk tab.",
    )
    dashboard_risk_rolling_vol_window_medium_days: int = Field(
        default=50,
        description="Medium rolling volatility window in trading days for the dashboard risk tab.",
    )
    dashboard_risk_rolling_vol_window_long_days: int = Field(
        default=100,
        description="Long rolling volatility window in trading days for the dashboard risk tab.",
    )

    # ── 2. VaR & ES Calculations ──────────────────────────────────────────────
    dashboard_risk_var_primary_confidence: float = Field(
        default=0.95,
        description="Primary historical VaR / ES confidence level for the dashboard risk tab.",
    )
    dashboard_risk_var_tail_confidence: float = Field(
        default=0.99,
        description="Tail historical VaR / ES confidence level for the dashboard risk tab.",
    )
    dashboard_bars_per_day: float = Field(
        default=13.0,
        description="Trading bars per calendar day for annualisation (13 = 30-min session 06:30-13:00).",
    )

    # ── 3. Stress Testing Configuration ───────────────────────────────────────
    dashboard_stress_slider_min_multiplier: float = Field(
        default=1.0,
        description="Minimum multiplier exposed by stress-test sliders in the dashboard risk tab.",
    )
    dashboard_stress_slider_max_multiplier: float = Field(
        default=5.0,
        description="Maximum multiplier exposed by stress-test sliders in the dashboard risk tab.",
    )
    dashboard_stress_slider_step: float = Field(
        default=0.5,
        description="Step size for stress-test sliders in the dashboard risk tab.",
    )
    dashboard_stress_volatility_default_multiplier: float = Field(
        default=2.0,
        description="Default volatility shock multiplier for the dashboard risk tab.",
    )
    dashboard_stress_slippage_default_multiplier: float = Field(
        default=3.0,
        description="Default slippage shock multiplier for the dashboard risk tab.",
    )
    dashboard_stress_commission_default_multiplier: float = Field(
        default=2.0,
        description="Default commission shock multiplier for the dashboard risk tab.",
    )
