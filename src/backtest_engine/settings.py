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
    terminal_redis_url: Optional[str] = Field(
        default=None,
        description="Optional Redis URL used by terminal job queueing and cache services.",
    )
    terminal_queue_name: str = Field(
        default="terminal-scenarios",
        description="RQ queue name for terminal-driven async scenario jobs.",
    )


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
    fixed_qty: int = 1                # Default number of contracts per signal

    # ── Deterministic spread model ─────────────────────────────────────────────
    # Controls how fill-price adjustments are computed.  No random component.
    #
    # spread_mode: 'static' applies spread_ticks on every fill unchanged.
    #              'adaptive_volatility' widens/narrows spread_ticks based on
    #              realized volatility relative to a rolling baseline.
    #
    # BUY  fills: executed_price = price + spread_ticks * tick_size
    # SELL fills: executed_price = price - spread_ticks * tick_size
    spread_mode: str = "adaptive_volatility"
    spread_ticks: int = 1                      # Base tick count per fill (0 = no spread)
    spread_volatility_step_pct: float = 0.10   # Vol band width per adaptive step (10 %)
    spread_step_multiplier: float = 1.5        # Multiplier per adaptive step above/below baseline
    spread_vol_lookback: int = 20              # Bars for current realized vol (short window)
    spread_vol_baseline_lookback: int = 100    # Bars for baseline vol (long reference window)

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

    # ── Terminal / analytics UI settings ────────────────────────────────────
    terminal_ui: "TerminalUISettings" = Field(default_factory=lambda: TerminalUISettings())

    @property
    def dashboard(self) -> "TerminalUISettings":
        """Legacy alias kept while older analytics modules finish migrating."""
        return self.terminal_ui

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


class TerminalUISettings(BaseModel):
    """
    Settings shared by the Streamlit analytics views and the terminal UI shell.
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

    # ── 4. Terminal UI Payload Budgets ────────────────────────────────────────
    terminal_max_chart_points: int = Field(
        default=2000,
        description="Default maximum point budget for terminal chart payloads.",
    )
    terminal_trade_page_size: int = Field(
        default=25,
        description="Default page size for terminal trade-detail drilldowns.",
    )
    terminal_result_bundle_cache_ttl_seconds: float = Field(
        default=15.0,
        description="TTL in seconds for the small in-process result-bundle cache.",
    )
    terminal_min_correlation_samples: int = Field(
        default=5,
        description="Minimum resampled observations required before correlation views are rendered.",
    )

    # ── 5. Terminal UI Cache Policy ───────────────────────────────────────────
    terminal_correlation_cache_ttl_seconds: int = Field(
        default=600,
        description="TTL in seconds for expensive correlation-matrix payloads.",
    )
    terminal_risk_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL in seconds for rolling stats and derived risk views.",
    )

    # ── 6. Terminal UI Async Operations ───────────────────────────────────────
    terminal_job_timeout_seconds: int = Field(
        default=1800,
        description="Timeout in seconds for queued terminal scenario jobs.",
    )
    terminal_job_max_retries: int = Field(
        default=2,
        description="Maximum retry count for queued terminal scenario jobs.",
    )
    terminal_sse_max_updates_per_second: float = Field(
        default=2.0,
        description="Maximum SSE update frequency for terminal job progress streams.",
    )

    # ── 7. Terminal UI Theme ────────────────────────────────────────────────
    terminal_benchmark_color: str = Field(
        default="#777777",
        description="Line color for benchmark overlays in terminal charts.",
    )
    terminal_strategy_colors: list[str] = Field(
        default_factory=lambda: ["#22C55E", "#3B82F6", "#EAB308", "#F97316", "#EC4899", "#A855F7"],
        description="Palette used for per-strategy chart series in portfolio mode.",
    )
    terminal_portfolio_total_color: str = Field(
        default="#FFFFFF",
        description="Primary color for the aggregate portfolio or strategy line.",
    )
    terminal_long_color: str = Field(
        default="#22C55E",
        description="Color used for long-side series in single-mode charts.",
    )
    terminal_short_color: str = Field(
        default="#EF4444",
        description="Color used for short-side series in single-mode charts.",
    )
    terminal_drawdown_color: str = Field(
        default="#EF4444",
        description="Color used for drawdown overlays and drawdown charts.",
    )
    terminal_rolling_sharpe_color: str = Field(
        default="#FFFFFF",
        description="Color used for rolling Sharpe mini-charts.",
    )
    terminal_rolling_vol_colors: list[str] = Field(
        default_factory=lambda: ["#22C55E", "#3B82F6", "#F59E0B"],
        description="Palette used for rolling-volatility series.",
    )
    terminal_stress_colors: list[str] = Field(
        default_factory=lambda: ["#22C55E", "#F59E0B", "#EF4444"],
        description="Palette used for stress-preview scenarios.",
    )
    terminal_var_colors: list[str] = Field(
        default_factory=lambda: ["#F59E0B", "#FBBF24", "#EF4444"],
        description="Palette used for VaR / ES risk overlays.",
    )

    # ── 8. Terminal Loading Overlay ───────────────────────────────────────────
    terminal_loading_words: list[str] = Field(
        default_factory=lambda: [
            "Put red box into green.",
            "Oh, no! Claude, what you did!?",
            "Please, put red box into green one! Make no mistakes!",
            "It worked!",
            "Oh..., I forgot about the yellow one.",
        ],
        description="Rotating loading captions. And yes, you can change them... ;)",
    )
    terminal_loading_word_interval_ms: int = Field(
        default=1100,
        description="Interval in milliseconds for loading-word rotation.",
    )
    terminal_loading_eta_per_request_seconds: float = Field(
        default=2.2,
        description="Fallback ETA estimate (seconds) per pending chart request.",
    )
