"""
Backtest engine configuration.

Engine-level settings only.  Strategy-specific parameters are defined
inside each strategy class via get_search_space() and dataclass configs.
Loaded from environment variables (prefix: QUANT_BACKTEST_) or .env file.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
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
    default_symbol: str = "RTY"

    # ── Bar settings ───────────────────────────────────────────────────────────
    low_interval: str = "5m"      # Base resolution used for data loading
    bar_type: str = "time"        # Options: "time", "volume", "range", "heikin_ashi"
    bar_size: float = 0.0         # Threshold for volume / range bar types

    # ── Portfolio & execution ──────────────────────────────────────────────────
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.02
    commission_rate: float = 2.5      # Per contract, in dollars
    max_slippage_ticks: int = 1       # Random slippage: uniform in [0, max]
    fixed_qty: int = 1                # Default number of contracts per signal

    # ── Trading hours (exchange time, HH:MM strings) ───────────────────────────
    trade_start_time: str = "06:00"
    trade_end_time: str = "15:00"
    eod_close_time: Optional[str] = "15:30"  # Force-close time; None = disabled

    # ── Risk limits (kill switches) ────────────────────────────────────────────
    max_daily_loss: Optional[float] = None      # Halt today if daily loss exceeds value
    max_drawdown_pct: Optional[float] = None    # Permanent halt at this drawdown %
    max_account_floor: Optional[float] = None   # Permanent halt below this equity level

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

    # ── IB Fetcher ─────────────────────────────────────────────────────────────
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497          # 7497 = TWS paper; 4002 = Gateway paper
    ib_client_id: int = 1
    ib_timeout: int = 30
    max_historical_years: int = 2
    delayed_data_minutes: int = 15
    ib_use_rth: bool = False

    def get_ib_request_delay(self) -> float:
        """Standard pacing delay to respect IB rate limits (~6 req/min)."""
        return 11.0

    # ── Walk-Forward Validation (WFV) scheduling ──────────────────────────────
    wfo_n_folds: int = 5             # Number of walk-forward folds
    wfo_test_size_pct: float = 0.15  # Fraction of total data used per test fold
    wfo_n_trials: int = 500          # Optuna trials per fold
    wfo_max_parameters: int = 9      # Strict maximum limit for optimized variables

    # Pruning / quality gates
    wfo_prune_min_trades: int = 10         # Minimum trades for a trial to pass
    wfo_prune_max_dd_pct: float = 30.0    # Max drawdown % before early pruning
    wfo_prune_target_trades_mult: int = 5  # target_trades = min_trades * this

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


# ── Singleton accessor ─────────────────────────────────────────────────────────
_settings: Optional[BacktestSettings] = None


def get_settings() -> BacktestSettings:
    """Returns the singleton BacktestSettings instance (lazy initialisation)."""
    global _settings
    if _settings is None:
        _settings = BacktestSettings()
    return _settings
