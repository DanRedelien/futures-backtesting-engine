"""
ICT Order Block Strategy — Classic Smart Money Concept (SMC).

Order Block Formation (3-candle pattern):
  Bullish OB:
    C1: Any candle.
    C2: Sweeps the low of C1 (Low[2] < Low[1]), acts as a local low
        (Low[2] < Low[3] after the fact).
    C3: Impulsive UP candle that closes above High[2] (confirms the block).
  → OB Zone: [Low[2], High[2]]  (the last bearish candle before the impulse)

  Bearish OB:
    C1: Any candle.
    C2: Sweeps the high of C1 (High[2] > High[1]), acts as a local high.
    C3: Impulsive DOWN candle that closes below Low[2].
  → OB Zone: [Low[2], High[2]]  (the last bullish candle before the impulse)

Entry Logic (any bar AFTER the OB forms):
  Bull entry: Price dips into OB zone (Low ≤ OB_High) and current candle
              closes bullish (Close > Open) above OB_Low → BUY.
  Bear entry: Price rallies into OB zone (High ≥ OB_Low) and current candle
              closes bearish (Close < Open) below OB_High → SELL.
  A block is invalidated (discarded) when the impulse closes through it.

Stop Loss:
  Bull: OB_Low - sl_offset_ticks * tick_size
  Bear: OB_High + sl_offset_ticks * tick_size

Take Profit:  max(ATR TP, Fixed RR TP)
  If the ATR-derived TP gives a better reward than min_rr_ratio:1, use ATR.
  Otherwise fall back to the fixed minimum RR.

Filters:
  - Trade Direction  : "both" / "long" / "short"
  - Trend Bias SMA   : Don't take LONG below SMA, don't take SHORT above SMA.
  - Trend T-Stat     : Skip if slope T-stat < trend_min_tstat (flat market).
  - Volatility Regime: Block entries in compression or panic regimes.

All indicators pre-computed on the full dataset, shifted 1 bar for zero
look-ahead bias.  on_bar() does only O(1) dict lookups.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import TrendFilter, VolatilityRegimeFilter


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IctOrderBlockConfig:
    """
    All tunable parameters for the ICT Order Block strategy.

    Order Block Detection
    ─────────────────────
    ob_impulse_atr_mult : Minimum body-size of the impulse candle (C3) as a
                          multiple of ATR.  Prevents weak, noise-driven blocks.
    ob_max_age_bars     : Discard an OB after this many bars if untested.

    Risk Management
    ───────────────
    atr_window          : ATR lookback (Wilder EWM).
    atr_tp_mult         : Take-profit distance as ATR multiples from entry.
    min_rr_ratio        : Minimum risk-reward ratio enforced.  If the ATR TP
                          would yield a worse RR, a static TP is used instead.
    sl_offset_ticks     : Extra ticks pushed behind the OB boundary for the SL.

    Direction Filter
    ────────────────
    trade_direction     : "both" | "long" | "short"

    Trend SMA Bias
    ──────────────
    trend_sma_window    : Long-term SMA length for directional bias.
                          None = disabled.

    Trend Filter
    ────────────
    use_trend_filter    : Enable/disable T-stat trend gate.
    trend_window        : Rolling OLS regression window.
    trend_min_tstat     : Minimum |T-stat|; below this = flat, skip entry.

    Volatility Regime Filter
    ────────────────────────
    use_vol_filter      : Enable/disable regime gate.
    vol_regime_window   : Short-term window for current vol measurement.
    vol_history_window  : Historical window for percentile ranking.
    vol_min_pct         : Lower percentile bound (dead market gate).
    vol_max_pct         : Upper percentile bound (panic market gate).
    """
    # ── OB detection ───────────────────────────────────────────────────────────
    ob_impulse_atr_mult: float = 0.8   # Impulse body must be >= this * ATR
    ob_max_age_bars: int = 50          # Discard untested OB after N bars

    # ── Risk management ────────────────────────────────────────────────────────
    atr_window: int = 14
    atr_tp_mult: float = 3.0           # Primary TP target in ATR
    min_rr_ratio: float = 3.0          # Fallback minimum RR if ATR TP is tiny
    sl_offset_ticks: int = 0           # Extra ticks outside the OB for the SL

    # ── Direction ──────────────────────────────────────────────────────────────
    trade_direction: str = "both"      # "both" | "long" | "short"

    # ── Trend SMA bias ─────────────────────────────────────────────────────────
    trend_sma_window: Optional[int] = 1000  # None = disabled

    # ── T-stat trend filter ────────────────────────────────────────────────────
    use_trend_filter: bool = True
    trend_window: int = 100
    trend_min_tstat: float = 1.0       # Below this = flat, no entry

    # ── Volatility regime filter ───────────────────────────────────────────────
    use_vol_filter: bool = True
    vol_regime_window: int = 50
    vol_history_window: int = 500
    vol_min_pct: float = 0.20
    vol_max_pct: float = 0.87


# Small container for a pending order block
class _OrderBlock(NamedTuple):
    direction: str    # "bull" or "bear"
    high: float       # Top of the OB zone
    low: float        # Bottom of the OB zone
    sl_base: float    # Raw SL price (before tick offset)
    formed_bar: int   # Bar index at which the OB was confirmed


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy
# ═══════════════════════════════════════════════════════════════════════════════

class IctOrderBlockStrategy(BaseStrategy):
    """
    ICT Order Block trend/reversal strategy.

    Pre-computes all indicators in __init__ then handles each bar in O(1)
    via Series.get() lookups.  Compatible with the WFO engine via
    get_search_space().
    """

    def __init__(self, engine, config: Optional[IctOrderBlockConfig] = None) -> None:
        super().__init__(engine)

        cfg = config or IctOrderBlockConfig()

        # WFO parameter injection: engine.settings may carry ob_* / ict_* keys
        for field in dataclasses.fields(cfg):
            wfo_key = f"ict_{field.name}"
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))

        self.config = cfg

        close  = engine.data["close"]
        high   = engine.data["high"]
        low    = engine.data["low"]
        open_  = engine.data["open"]

        # ── ATR (Wilder EWM) ───────────────────────────────────────────────────
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=cfg.atr_window, adjust=False).mean()

        # Shift by 1 → strategy sees previous bar's indicators only
        self._atr:   pd.Series = atr.shift(1)
        self._open:  pd.Series = open_.shift(1)
        self._high:  pd.Series = high.shift(1)
        self._low:   pd.Series = low.shift(1)
        self._close: pd.Series = close.shift(1)

        # Raw (unshifted) OHLC arrays used to detect OB formation on the
        # *previous* two bars.  We shift by 1 for the "current" bar and by 2/3
        # for the look-back into recent history — still no look-ahead.
        self._open_raw  = open_
        self._high_raw  = high
        self._low_raw   = low
        self._close_raw = close
        self._atr_raw   = atr

        # Pre-compute bar indices for O(1) lookup
        self._bar_index: pd.Series = pd.Series(
            np.arange(len(close)), index=close.index
        )

        # ── Trend SMA ──────────────────────────────────────────────────────────
        self._trend_sma: Optional[pd.Series] = None
        if cfg.trend_sma_window is not None:
            ts = close.rolling(
                window=cfg.trend_sma_window,
                min_periods=cfg.trend_sma_window
            ).mean()
            self._trend_sma = ts.shift(1)
            print(
                f"[ICT_OB] TrendBias SMA enabled "
                f"(window={cfg.trend_sma_window}) — LONG above, SHORT below"
            )

        # ── Volatility regime filter ───────────────────────────────────────────
        self._vol_filter: Optional[VolatilityRegimeFilter] = None
        if cfg.use_vol_filter:
            self._vol_filter = VolatilityRegimeFilter(
                price=close,
                regime_window=cfg.vol_regime_window,
                history_window=cfg.vol_history_window,
                min_pct=cfg.vol_min_pct,
                max_pct=cfg.vol_max_pct,
            )
            print(
                f"[ICT_OB] VolFilter enabled "
                f"(window={cfg.vol_regime_window})"
            )

        # ── T-stat trend filter ────────────────────────────────────────────────
        self._trend_filter: Optional[TrendFilter] = None
        if cfg.use_trend_filter:
            self._trend_filter = TrendFilter(
                price=close,
                window=cfg.trend_window,
                max_t_stat=99.0,    # No upper cap; strategy doesn't cap trend strength
            )
            self._trend_min_tstat = cfg.trend_min_tstat
            print(
                f"[ICT_OB] TrendFilter enabled "
                f"(window={cfg.trend_window}, min_tstat={cfg.trend_min_tstat})"
            )

        # ── State ──────────────────────────────────────────────────────────────
        self._invested: bool = False
        self._position_side: Optional[str] = None
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0

        # The most recent valid (pending) order block waiting to be tested
        self._pending_ob: Optional[_OrderBlock] = None

        print(
            f"[ICT_OB] Ready | atr={cfg.atr_window} "
            f"| direction={cfg.trade_direction} "
            f"| min_rr={cfg.min_rr_ratio} | sl_offset={cfg.sl_offset_ticks}t"
        )

    # ── WFO interface ──────────────────────────────────────────────────────────

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """
        Optuna search bounds for Walk-Forward Optimisation.

        All keys prefixed with 'ict_' are injected into BacktestSettings by
        WFOEngine and read back via the dataclass field loop in __init__.
        """
        return {
            "ict_ob_impulse_atr_mult": (0.3, 2.0, 0.1),
            "ict_ob_max_age_bars":     (10,  100,  5),
            "ict_atr_tp_mult":         (1.5,  8.0, 0.5),
            "ict_min_rr_ratio":        (1.5,  6.0, 0.5),
            "ict_sl_offset_ticks":     (0,    10,   1),
            "ict_trend_min_tstat":     (0.5,  3.0, 0.25),
            "ict_trend_sma_window":    (200, 3000,  100),
            "ict_vol_min_pct":         (0.10, 0.60, 0.05),
            "ict_vol_max_pct":         (0.60, 1.00, 0.05),
        }

    # ── Main event hook ────────────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> List[Order]:
        """
        Called once per bar.  All data accesses use the shifted (lag-1) series
        so the strategy observes only information available at bar open.

        Args:
            bar: Current OHLCV bar; bar.name is the timestamp.

        Returns:
            List of Order objects (may be empty).
        """
        timestamp = bar.name
        atr_val   = self._atr.get(timestamp, np.nan)
        c_open    = self._open.get(timestamp, np.nan)
        c_high    = self._high.get(timestamp, np.nan)
        c_low     = self._low.get(timestamp, np.nan)
        c_close   = self._close.get(timestamp, np.nan)
        bar_idx   = self._bar_index.get(timestamp, -1)

        if any(np.isnan(v) for v in [atr_val, c_open, c_high, c_low, c_close]):
            return []
        if bar_idx < 3:      # Need at least 3 previous bars for OB detection
            return []

        orders: List[Order] = []
        tick_size = self.settings.get_instrument_spec(
            self.settings.default_symbol
        )["tick_size"]

        # ── In position: manage SL / TP exits ─────────────────────────────────
        if self._invested:
            if self._position_side == "LONG":
                if c_low <= self._sl_price or c_high >= self._tp_price:
                    reason = "STOP_LOSS" if c_low <= self._sl_price else "TAKE_PROFIT"
                    orders.append(
                        self.market_order("SELL", self.settings.fixed_qty, reason=reason)
                    )
                    self._reset_state()
                    return orders

            elif self._position_side == "SHORT":
                if c_high >= self._sl_price or c_low <= self._tp_price:
                    reason = "STOP_LOSS" if c_high >= self._sl_price else "TAKE_PROFIT"
                    orders.append(
                        self.market_order("BUY", self.settings.fixed_qty, reason=reason)
                    )
                    self._reset_state()
                    return orders

        # ── Detect new order blocks on previous bars ───────────────────────────
        # We look at bars [idx-3, idx-2, idx-1] (all shifted by 1 already in
        # the raw series so bar_idx-1 == the candle that just closed).
        self._try_detect_ob(bar_idx, atr_val)

        # Expire stale order blocks
        if self._pending_ob is not None:
            age = bar_idx - self._pending_ob.formed_bar
            if age > self.config.ob_max_age_bars:
                self._pending_ob = None

        # ── No position: check for an entry at the pending OB ─────────────────
        if not self._invested and self._pending_ob is not None:
            ob = self._pending_ob
            entry_triggered = False

            if ob.direction == "bull":
                # Price tagged the OB zone and closed bullish inside or above
                touched   = c_low  <= ob.high
                bullish_c = c_close > c_open
                not_blown = c_close >= ob.low   # Closed above OB bottom → OB still valid
                entry_triggered = touched and bullish_c and not_blown

                if entry_triggered and self._filters_allow(timestamp, 1.0):
                    sl_price = ob.sl_base - self.config.sl_offset_ticks * tick_size
                    sl_dist  = c_close - sl_price
                    if sl_dist <= 0:
                        return []   # Degenerate SL — skip
                    atr_tp_dist   = atr_val * self.config.atr_tp_mult
                    fixed_tp_dist = sl_dist * self.config.min_rr_ratio
                    tp_dist  = max(atr_tp_dist, fixed_tp_dist)
                    tp_price = c_close + tp_dist

                    self._invested       = True
                    self._position_side  = "LONG"
                    self._sl_price       = sl_price
                    self._tp_price       = tp_price
                    self._pending_ob     = None     # Consumed
                    orders.append(
                        self.market_order("BUY", self.settings.fixed_qty, reason="OB_LONG")
                    )

            elif ob.direction == "bear":
                touched   = c_high >= ob.low
                bearish_c = c_close < c_open
                not_blown = c_close <= ob.high
                entry_triggered = touched and bearish_c and not_blown

                if entry_triggered and self._filters_allow(timestamp, -1.0):
                    sl_price = ob.sl_base + self.config.sl_offset_ticks * tick_size
                    sl_dist  = sl_price - c_close
                    if sl_dist <= 0:
                        return []
                    atr_tp_dist   = atr_val * self.config.atr_tp_mult
                    fixed_tp_dist = sl_dist * self.config.min_rr_ratio
                    tp_dist  = max(atr_tp_dist, fixed_tp_dist)
                    tp_price = c_close - tp_dist

                    self._invested       = True
                    self._position_side  = "SHORT"
                    self._sl_price       = sl_price
                    self._tp_price       = tp_price
                    self._pending_ob     = None
                    orders.append(
                        self.market_order("SELL", self.settings.fixed_qty, reason="OB_SHORT")
                    )

            # Invalidate OB if blown through without a valid entry
            if not entry_triggered and self._pending_ob is not None:
                ob = self._pending_ob
                blown_bull = ob.direction == "bull" and c_close < ob.low
                blown_bear = ob.direction == "bear" and c_close > ob.high
                if blown_bull or blown_bear:
                    self._pending_ob = None

        return orders

    # ── Order block detection ──────────────────────────────────────────────────

    def _try_detect_ob(self, bar_idx: int, atr_val: float) -> None:
        """
        Inspects the three most recent *fully closed* bars (lag 1, 2, 3)
        to decide if a new order block has formed.

        All accesses use .iloc on the shifted raw series so there is no
        look-ahead — bar_idx in the raw series equals the bar that JUST
        closed (C3 is bar_idx-1, C2 is bar_idx-2, C1 is bar_idx-3).

        A new OB overwrites a pending one only if it is in the same
        direction or there is no pending OB yet.
        """
        cfg = self.config

        # Pull OHLC for C1 / C2 / C3 from the unshifted (raw) series
        # We use iloc on the raw series; the raw series has no shift applied.
        close_raw = self._close_raw
        open_raw  = self._open_raw
        high_raw  = self._high_raw
        low_raw   = self._low_raw

        # bar_idx corresponds to the bar that on_bar() is currently processing.
        # bar_idx - 1 == last closed bar (C3), -2 == C2, -3 == C1.
        c3_idx = bar_idx - 1
        c2_idx = bar_idx - 2
        c1_idx = bar_idx - 3

        if c1_idx < 0:
            return

        # Use iloc for guaranteed positional access
        c1_high  = high_raw.iloc[c1_idx]
        c1_low   = low_raw.iloc[c1_idx]

        c2_open  = open_raw.iloc[c2_idx]
        c2_high  = high_raw.iloc[c2_idx]
        c2_low   = low_raw.iloc[c2_idx]
        c2_close = close_raw.iloc[c2_idx]

        c3_open  = open_raw.iloc[c3_idx]
        c3_high  = high_raw.iloc[c3_idx]
        c3_low   = low_raw.iloc[c3_idx]
        c3_close = close_raw.iloc[c3_idx]

        c3_body = abs(c3_close - c3_open)
        min_impulse = atr_val * cfg.ob_impulse_atr_mult

        # ── Bullish Order Block ────────────────────────────────────────────────
        #   C2 sweeps below C1's low (liquidity grab) and below C3's low.
        #   C3 is a strong bullish impulse closing above C2's high.
        c2_sweeps_low   = c2_low < c1_low
        c2_is_lcl_low   = c2_low < c3_low           # C2 low is the lowest of 3
        c3_bull_impulse = c3_close > c2_high and c3_body >= min_impulse
        if c2_sweeps_low and c2_is_lcl_low and c3_bull_impulse:
            # SL base = bottom of C2 (the last bearish candle before impulse)
            ob = _OrderBlock(
                direction="bull",
                high=c2_high,
                low=c2_low,
                sl_base=c2_low,
                formed_bar=bar_idx,
            )
            if self._pending_ob is None or self._pending_ob.direction == "bull":
                self._pending_ob = ob

        # ── Bearish Order Block ────────────────────────────────────────────────
        #   C2 sweeps above C1's high and above C3's high.
        #   C3 is a strong bearish impulse closing below C2's low.
        c2_sweeps_high  = c2_high > c1_high
        c2_is_lcl_high  = c2_high > c3_high
        c3_bear_impulse = c3_close < c2_low and c3_body >= min_impulse
        if c2_sweeps_high and c2_is_lcl_high and c3_bear_impulse:
            ob = _OrderBlock(
                direction="bear",
                high=c2_high,
                low=c2_low,
                sl_base=c2_high,
                formed_bar=bar_idx,
            )
            if self._pending_ob is None or self._pending_ob.direction == "bear":
                self._pending_ob = ob

    # ── Filters ────────────────────────────────────────────────────────────────

    def _filters_allow(self, timestamp, crossover: float) -> bool:
        """
        Returns True only when all enabled filters permit an entry.

        Args:
            timestamp: Current bar timestamp.
            crossover: +1.0 for a LONG signal, -1.0 for a SHORT signal.
        """
        # ── Direction filter ──────────────────────────────────────────────────
        direction = self.config.trade_direction.lower()
        if direction == "long"  and crossover == -1.0:
            return False
        if direction == "short" and crossover == 1.0:
            return False

        # ── Volatility regime ─────────────────────────────────────────────────
        if self._vol_filter and not self._vol_filter.is_allowed(timestamp):
            return False

        # ── Trend T-stat (flat market gate) ───────────────────────────────────
        if self._trend_filter:
            t_stat = self._trend_filter.as_series().get(timestamp, np.nan)
            if np.isnan(t_stat) or abs(t_stat) < self._trend_min_tstat:
                return False

        # ── Directional bias via long-term SMA ───────────────────────────────
        if self._trend_sma is not None:
            sma_val    = self._trend_sma.get(timestamp, np.nan)
            cur_close  = self._close.get(timestamp, np.nan)
            if not (np.isnan(sma_val) or np.isnan(cur_close)):
                above_sma = cur_close > sma_val
                if crossover == 1.0 and not above_sma:
                    return False   # LONG signal but price below long-term SMA
                if crossover == -1.0 and above_sma:
                    return False   # SHORT signal but price above long-term SMA

        return True

    # ── State reset ────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Clears all open-position tracking variables."""
        self._invested       = False
        self._position_side  = None
        self._sl_price       = 0.0
        self._tp_price       = 0.0
