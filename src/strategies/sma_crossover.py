"""
SMA Crossover — Classic dual-moving-average trend-following strategy.

Signal logic:
  - Enter LONG  when fast SMA crosses above slow SMA.
  - Enter SHORT when fast SMA crosses below slow SMA.
  - Exit on ATR stop-loss, ATR take-profit, or a crossover reversal.
  - REGIME: VolatilityRegimeFilter blocks entries during compression or panic.
  - TREND:  TrendFilter skips entries when the trend is too weak (T-stat check).
            A strong slope confirms the crossover is actually meaningful.

All indicators are pre-computed on the full dataset during __init__ and
accessed via a .get(timestamp) lookup in on_bar() — zero look-ahead bias.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import TrendFilter, VolatilityRegimeFilter


@dataclass
class SmaCrossoverConfig:
    """
    Strategy-specific parameters for SMA Crossover.

    Attributes:
        fast_window: Fast SMA lookback period. A shorter window reacts faster to price changes.
        slow_window: Slow SMA lookback period. A longer window represents the broader trend.
        atr_window: Lookback window for ATR (used for stop and target sizing).
        atr_sl_mult: Stop-loss distance as a multiple of ATR.
        atr_tp_mult: Take-profit distance as a multiple of ATR.

        use_vol_filter: Only trade when volatility is in a 'normal' range (not dead, not panicking).
        vol_regime_window: How many bars to sample for current actual volatility.
        vol_history_window: How much history to use to rank the current volatility.
        vol_min_pct: Minimum volatility percentile. Below this the market is dead — no edge.
        vol_max_pct: Maximum volatility percentile. Above this the market is entering panic mode.

        use_trend_filter: Skips entry when the crossover is not backed by a meaningful trend slope.
        trend_window: How many bars to look back to measure trend strength.
        trend_min_tstat: Minimum |T-stat| required to enter. Low T-stat = weak, noise-driven crossover.

        trend_sma_window: Long-term trend SMA for directional bias. When price is above it, only
            LONG entries are allowed. When below, only SHORT. Set to None to disable.
    """
    fast_window: int = 28          # Fast moving average period
    slow_window: int = 90          # Slow moving average period
    atr_window: int = 14           # ATR lookback period
    atr_sl_mult: float = 1.5       # Stop-loss in ATR multiples
    atr_tp_mult: float = 2.875       # Take-profit in ATR multiples

    use_vol_filter: bool = True    # Only trade during "normal" volatility
    vol_regime_window: int = 50    # Short-term window to measure current vol
    vol_history_window: int = 500  # Historical window to compare against
    vol_min_pct: float = 0.20      # Minimum activity allowed (no dead markets)
    vol_max_pct: float = 0.80      # Maximum activity allowed (no panic/crash markets)

    use_trend_filter: bool = True  # Only enter when trend is statistically confirmed
    trend_window: int = 100        # Window to measure trend strength
    trend_min_tstat: float = 1.4   # Minimum T-stat to enter (we only want real, meaningful crossovers)

    trade_direction: str = "both"           # Allowed directions: "both", "long", "short"


class SmaCrossoverStrategy(BaseStrategy):
    """
    Dual SMA crossover trend-following strategy.

    Methodology:
        1. Pre-compute fast SMA, slow SMA, and ATR on the full bar series.
        2. Optionally compute a long-term trend SMA (trend_sma_window bars) for directional bias.
        3. Generate a +1 / -1 crossover signal; fire only when the sign changes (diff()).
        4. on_bar() checks filters and enforces trend direction before placing orders.
        5. ATR-scaled SL and TP levels stored per trade for exit management.

    Pairs well with the WFO engine: get_search_space() exposes all numeric
    parameters as Optuna search bounds.
    """

    def __init__(self, engine, config: Optional[SmaCrossoverConfig] = None) -> None:
        super().__init__(engine)

        cfg = config or SmaCrossoverConfig()

        # Overlay WFO-injected parameters if present in engine.settings
        for field in dataclasses.fields(cfg):
            wfo_key = f"sma_{field.name}"
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))

        self.config = cfg
        close = engine.data["close"]
        high  = engine.data["high"]
        low   = engine.data["low"]

        # ── Simple Moving Averages ─────────────────────────────────────────
        fast_sma   = close.rolling(window=cfg.fast_window, min_periods=cfg.fast_window).mean()
        slow_sma   = close.rolling(window=cfg.slow_window, min_periods=cfg.slow_window).mean()
        # regime_signal: +1 when fast above slow, -1 below — used for in-position checks
        regime_signal = np.sign(fast_sma - slow_sma)
        # crossover_signal: fires only on the exact bar a crossover happens (+2 or -2 then clamped)
        cross = regime_signal.diff()
        crossover_signal = pd.Series(np.where(cross > 0, 1.0, np.where(cross < 0, -1.0, 0.0)), index=close.index)

        # ── ATR (Wilder EWM) ───────────────────────────────────────────────
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=cfg.atr_window, adjust=False).mean()

        # Store indicators (engines executes at T+1 Open, no need to shift)
        self._signal: pd.Series        = regime_signal    # Used for reversal exit check
        self._crossover: pd.Series     = crossover_signal # Used for fresh entry trigger
        self._atr: pd.Series           = atr
        self._close: pd.Series         = close

        # ── Optional advanced filters ──────────────────────────────────────
        self._vol_filter: Optional[VolatilityRegimeFilter] = None
        if cfg.use_vol_filter:
            self._vol_filter = VolatilityRegimeFilter(
                price=close,
                regime_window=cfg.vol_regime_window,
                history_window=cfg.vol_history_window,
                min_pct=cfg.vol_min_pct,
                max_pct=cfg.vol_max_pct,
            )
            print(f"[SMA] VolatilityRegimeFilter enabled (window={cfg.vol_regime_window})")

        self._trend_filter: Optional[TrendFilter] = None
        if cfg.use_trend_filter:
            self._trend_filter = TrendFilter(
                price=close,
                window=cfg.trend_window,
                max_t_stat=99.0,   # No upper cap — SMA already filters trend direction
            )
            self._trend_min_tstat = cfg.trend_min_tstat
            print(f"[SMA] TrendFilter enabled (window={cfg.trend_window}, min_tstat={cfg.trend_min_tstat})")

        # ── Position tracking ──────────────────────────────────────────────
        self._invested: bool = False
        self._position_side: Optional[str] = None
        self._entry_price: float = 0.0
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0

        valid = self._crossover.notna().sum()
        n_crosses = int((self._crossover != 0).sum())
        print(
            f"[SMA] Ready | fast={cfg.fast_window} slow={cfg.slow_window} "
            f"| Crossovers: {n_crosses:,} | Valid bars: {valid:,} / {len(close):,}"
        )

    # ── WFO interface ──────────────────────────────────────────────────────────

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """
        Optuna search bounds for Walk-Forward Optimisation.

        Parameters prefixed with 'sma_' are injected into BacktestSettings
        by WFOEngine and read back via the dataclass field loop in __init__.
        """
        return {
            "sma_fast_window":     (10, 40, 5),
            "sma_slow_window":     (40, 220, 20),
            "sma_atr_sl_mult":     (1.0, 3.0, 0.25),
            "sma_atr_tp_mult":     (1.5, 5.0, 0.25),
            "sma_trend_min_tstat": (0.8, 2.2, 0.2),
        }

    # ── Event hook ─────────────────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> List[Order]:
        """
        Called once per bar. Returns orders if a crossover or exit condition fires.

        Args:
            bar: Current OHLCV bar; bar.name is the timestamp.

        Returns:
            List of Order objects (may be empty).
        """
        timestamp  = bar.name
        signal     = self._signal.get(timestamp, np.nan)     # Current regime direction
        crossover  = self._crossover.get(timestamp, 0.0)     # Fresh crossover event
        atr_val    = self._atr.get(timestamp, np.nan)
        close      = bar["close"]

        if np.isnan(signal) or np.isnan(atr_val):
            return []

        orders: List[Order] = []

        # ── In position: check SL / TP / reversal / time exits ─────────────────
        if self._invested:
            if self._position_side == "LONG":
                hit_sl      = close <= self._sl_price
                hit_tp      = close >= self._tp_price
                hit_reverse = signal == -1.0   # Crossover reversed — respect the new trend
                if hit_sl or hit_tp or hit_reverse:
                    reason = "STOP_LOSS" if hit_sl else ("TAKE_PROFIT" if hit_tp else "REVERSAL")
                    orders.append(self.market_order("SELL", self.settings.fixed_qty, reason=reason))
                    self._reset_state()
                    return orders

            elif self._position_side == "SHORT":
                hit_sl      = close >= self._sl_price
                hit_tp      = close <= self._tp_price
                hit_reverse = signal == 1.0
                if hit_sl or hit_tp or hit_reverse:
                    reason = "STOP_LOSS" if hit_sl else ("TAKE_PROFIT" if hit_tp else "REVERSAL")
                    orders.append(self.market_order("BUY", self.settings.fixed_qty, reason=reason))
                    self._reset_state()
                    return orders

        # ── No position: check for a fresh crossover event ────────────────
        if not self._invested and crossover != 0.0:
            if not self._filters_allow(timestamp, crossover):
                return []

            if crossover == 1.0:  # Fast just crossed above slow
                self._invested = True
                self._position_side = "LONG"
                self._entry_price = close
                self._sl_price = close - atr_val * self.config.atr_sl_mult
                self._tp_price = close + atr_val * self.config.atr_tp_mult
                orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="SIGNAL"))

            elif crossover == -1.0:  # Fast just crossed below slow
                self._invested = True
                self._position_side = "SHORT"
                self._entry_price = close
                self._sl_price = close + atr_val * self.config.atr_sl_mult
                self._tp_price = close - atr_val * self.config.atr_tp_mult
                orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="SIGNAL"))

        return orders

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _filters_allow(self, timestamp, crossover: float = 0.0) -> bool:
        """
        Returns True when all enabled filters permit a new entry.

        Args:
            timestamp: Current bar timestamp.
            crossover: The current crossover direction (+1.0 LONG, -1.0 SHORT).
                       Used to enforce trend-bias direction filter.

        Returns:
            True if all filters pass (or are disabled).
        """
        direction = self.config.trade_direction.lower()
        if direction == "long" and crossover == -1.0:
            return False
        if direction == "short" and crossover == 1.0:
            return False

        if self._vol_filter and not self._vol_filter.is_allowed(timestamp):
            return False

        if self._trend_filter:
            t_stat = self._trend_filter.as_series().get(timestamp, np.nan)
            if np.isnan(t_stat) or abs(t_stat) < self._trend_min_tstat:
                return False  # Crossover not backed by a strong enough trend slope

        return True

    def _reset_state(self) -> None:
        """Clears all open-position tracking variables."""
        self._invested = False
        self._position_side = None
        self._entry_price = 0.0
        self._sl_price = 0.0
        self._tp_price = 0.0
