"""
Statistical Level Strategy.

Category: Statistical Edge
Description:
  Identifies an anomalous momentum bar (e.g., body size in the 90th percentile
  over a rolling window). The bar immediately preceding this anomaly is evaluated
  as the "Base". If the Base's body is average or small, its entire range (Low to High
  including wicks) becomes a strong Support (Demand) or Resistance (Supply) level.

Execution:
  - We maintain a list of the last N active (unmitigated) levels.
  - A level is touched when price enters its [Low, High] range.
  - Upon touch, if the touching bar exhibits high relative volume (or True Range),
    we enter a reversal trade (Long at Demand, Short at Supply).
  - The level is then wiped (used only once).
  - Stop Loss: Placed exactly 1 tick behind the level.
  - Take Profit: ATR multiple.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import VolatilityRegimeFilter


@dataclass
class StatLevelConfig:
    """Tunable parameters for Statistical Level Strategy."""

    # ── Anomaly detection ──────────────────────────────────────────────────────
    lookback_window: int = 120           # Window to calculate percentiles
    impulse_pct_threshold: float = 0.90  # Impulse must be >= 90th percentile of body size
    base_max_pct_threshold: float = 0.50 # The base candle before impulse must be <= 50th percentile

    # ── Level Management ───────────────────────────────────────────────────────
    max_active_levels: int = 50          # How many unmitigated levels to store in memory

    # ── Entry confirmation ─────────────────────────────────────────────────────
    volatility_window: int = 20          # Window for Volume/TR moving average
    volatility_mult: float = 1.25        # Touching bar must have Volume/TR > MA * mult to validate entry

    # ── Risk management ────────────────────────────────────────────────────────
    atr_window: int = 14
    atr_tp_mult: float = 4.0             # Take Profit distance in ATRs
    sl_offset_ticks: int = 1             # Strict tick offset completely behind the level (e.g., 1 tick)

    # ── Volatility regime filter ───────────────────────────────────────────────
    use_vol_filter: bool = False
    vol_regime_window: int = 50
    vol_history_window: int = 500
    vol_min_pct: float = 0.20
    vol_max_pct: float = 0.80


class _Level(NamedTuple):
    direction: str     # "bull" (Demand) or "bear" (Supply)
    high: float        # Top of the base candle wick
    low: float         # Bottom of the base candle wick
    formed_bar: int    # Bar index where the impulse closed


class StatisticalLevelStrategy(BaseStrategy):
    """
    Statistical Edge strategy capturing pullbacks to the origin of anomalous momentum.
    """

    def __init__(self, engine, config: Optional[StatLevelConfig] = None) -> None:
        super().__init__(engine)
        
        cfg = config or StatLevelConfig()

        # Inject from settings for WFO
        for field in dataclasses.fields(cfg):
            wfo_key = f"statl_{field.name}"
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))

        self.config = cfg

        close = engine.data["close"]
        open_ = engine.data["open"]
        high  = engine.data["high"]
        low   = engine.data["low"]

        # Calculate True Range for ATR and fallback Volatility
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=cfg.atr_window, adjust=False).mean()
        
        # ── Pre-compute anomalies ─────────────────────────────────────────────
        body_abs = (close - open_).abs()
        
        # We need a rolling percentile for the body size
        # Pandas rolling quantile is slow but required for accurate historical percentiles.
        rolling_body = body_abs.rolling(window=cfg.lookback_window, min_periods=cfg.lookback_window)
        impulse_threshold = rolling_body.quantile(cfg.impulse_pct_threshold)
        base_threshold    = rolling_body.quantile(cfg.base_max_pct_threshold)

        # ── Entry Filter (Volume or True Range) ──────────────────────────────
        if "volume" in engine.data and engine.data["volume"].sum() > 0:
            vol_metric = engine.data["volume"]
            self._use_volume = True
        else:
            vol_metric = tr
            self._use_volume = False
            
        vol_sma = vol_metric.rolling(window=cfg.volatility_window, min_periods=cfg.volatility_window).mean()
        self._vol_metric = vol_metric
        self._vol_sma = vol_sma

        # Expose Series to fast dict lookups
        self._atr = atr
        self._close = close
        self._open = open_
        self._high = high
        self._low = low
        
        self._body_abs = body_abs
        self._impulse_threshold = impulse_threshold
        self._base_threshold = base_threshold

        self._bar_index = pd.Series(np.arange(len(close)), index=close.index)
        
        # Caching raw arrays for anomaly checking without lookahead
        self._close_raw = close
        self._open_raw = open_
        self._high_raw = high
        self._low_raw = low

        # ── State ──────────────────────────────────────────────────────────────
        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
        self._tp_price = 0.0

        self._active_levels: List[_Level] = []

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
            print(f"[STAT_LEVEL] VolFilter enabled (window={cfg.vol_regime_window})")

        vol_type_str = "Volume" if self._use_volume else "TrueRange"
        print(
            f"[STAT_LEVEL] Ready | Lookback={cfg.lookback_window} | "
            f"Impulse={cfg.impulse_pct_threshold:.0%} | Base<={cfg.base_max_pct_threshold:.0%} | "
            f"Filter={vol_type_str}_x{cfg.volatility_mult}"
        )

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        return {
            "statl_impulse_pct_threshold":  (0.88, 0.98, 0.02),
            "statl_base_max_pct_threshold": (0.30, 0.60, 0.05),
            "statl_volatility_mult":        (1.10, 2.00, 0.10),
            "statl_atr_tp_mult":            (2.0, 6.0, 0.5),
            "statl_lookback_window":        (80, 180, 20),
        }

    def on_bar(self, bar: pd.Series) -> List[Order]:
        timestamp = bar.name
        atr_val = self._atr.get(timestamp, np.nan)
        c_open = self._open.get(timestamp, np.nan)
        c_high = self._high.get(timestamp, np.nan)
        c_low = self._low.get(timestamp, np.nan)
        c_close = self._close.get(timestamp, np.nan)
        
        c_vol_metric = self._vol_metric.get(timestamp, np.nan)
        c_vol_sma = self._vol_sma.get(timestamp, np.nan)
        
        bar_idx = self._bar_index.get(timestamp, -1)

        if any(np.isnan(v) for v in [atr_val, c_close, c_vol_sma]):
            return []

        if bar_idx < 3:
            return []

        orders: List[Order] = []
        tick_size = self.settings.get_instrument_spec(self.settings.default_symbol)["tick_size"]

        # ── Position Management ────────────────────────────────────────────────
        if self._invested:
            if self._position_side == "LONG":
                if c_low <= self._sl_price or c_high >= self._tp_price:
                    reason = "STOP_LOSS" if c_low <= self._sl_price else "TAKE_PROFIT"
                    orders.append(self.market_order("SELL", self.settings.fixed_qty, reason=reason))
                    self._reset_state()
                    return orders
            elif self._position_side == "SHORT":
                if c_high >= self._sl_price or c_low <= self._tp_price:
                    reason = "STOP_LOSS" if c_high >= self._sl_price else "TAKE_PROFIT"
                    orders.append(self.market_order("BUY", self.settings.fixed_qty, reason=reason))
                    self._reset_state()
                    return orders

        # ── Detect New Levels ──────────────────────────────────────────────────
        # We look at bar_idx (just closed) and bar_idx-1 (the base)
        self._try_detect_level(bar_idx)

        # ── Check for Entries (Mitigation) ─────────────────────────────────────
        if not self._invested and self._active_levels:
            # Volatility filter check
            is_vol_allowed = self._vol_filter.is_allowed(timestamp) if self._vol_filter else True

            # Reversal confirmation filter: Was this touch accompanied by high relative volume/volatility?
            is_valid_touch = (c_vol_metric > (c_vol_sma * self.config.volatility_mult)) and is_vol_allowed
            
            # Find the first level we touched. We process from newest to oldest.
            mitigated_idx = -1
            
            for i in range(len(self._active_levels) - 1, -1, -1):
                lvl = self._active_levels[i]
                
                if lvl.direction == "bull":
                    # Touch Demand Level: Low dips into the zone
                    if c_low <= lvl.high and c_close >= lvl.low:
                        if is_valid_touch:
                            # Reversal Long
                            # Stop loss: exactly 1 tick below the base's low extreme.
                            self._sl_price = lvl.low - (self.config.sl_offset_ticks * tick_size)
                            # Failsafe
                            if c_close - self._sl_price <= 0:
                                continue
                                
                            self._tp_price = c_close + (atr_val * self.config.atr_tp_mult)
                            
                            self._invested = True
                            self._position_side = "LONG"
                            mitigated_idx = i
                            orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="STAT_LVL_LONG"))
                            break
                        else:
                            # If we blew through it or touched without volume, we STILL invalidate it.
                            # "Может быть использован всего 1 раз."
                            mitigated_idx = i
                            break
                            
                elif lvl.direction == "bear":
                    # Touch Supply Level: High rallies into the zone
                    if c_high >= lvl.low and c_close <= lvl.high:
                        if is_valid_touch:
                            # Reversal Short
                            # Stop loss: exactly 1 tick above the base's high extreme.
                            self._sl_price = lvl.high + (self.config.sl_offset_ticks * tick_size)
                            if self._sl_price - c_close <= 0:
                                continue
                            
                            self._tp_price = c_close - (atr_val * self.config.atr_tp_mult)
                            
                            self._invested = True
                            self._position_side = "SHORT"
                            mitigated_idx = i
                            orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="STAT_LVL_SHORT"))
                            break
                        else:
                            # Mitigated but no trade because no setup confirm
                            mitigated_idx = i
                            break
                            
            # Remove the utilized level
            if mitigated_idx != -1:
                self._active_levels.pop(mitigated_idx)

        # Trim active levels to max_active_levels limit
        if len(self._active_levels) > self.config.max_active_levels:
            # Discard oldest
            self._active_levels = self._active_levels[-self.config.max_active_levels:]

        return orders

    def _try_detect_level(self, bar_idx: int) -> None:
        """
        Evaluate if bar_idx is an anomalous impulse, and if bar_idx-1 is a valid base.
        """
        c1_idx = bar_idx - 1 # Base
        c2_idx = bar_idx     # Impulse
        
        # Access pre-computed series
        ts_c1 = self._close_raw.index[c1_idx]
        ts_c2 = self._close_raw.index[c2_idx]
        
        c2_body = self._body_abs.get(ts_c2, np.nan)
        c2_thresh = self._impulse_threshold.get(ts_c2, np.nan)
        
        c1_body = self._body_abs.get(ts_c1, np.nan)
        c1_thresh = self._base_threshold.get(ts_c1, np.nan)
        
        if any(np.isnan(v) for v in [c2_body, c2_thresh, c1_body, c1_thresh]):
            return
            
        # 1. Check if the current bar is a huge anomaly
        if c2_body >= c2_thresh:
            # 2. Check if the prior bar is an "average" resting bar
            if c1_body <= c1_thresh:
                
                c2_open = self._open_raw.iloc[c2_idx]
                c2_close = self._close_raw.iloc[c2_idx]
                
                c1_high = self._high_raw.iloc[c1_idx]
                c1_low = self._low_raw.iloc[c1_idx]
                
                # Was the impulse Bullish or Bearish?
                if c2_close > c2_open:
                    # Bullish impulse -> Demand Level
                    lvl = _Level(
                        direction="bull",
                        high=c1_high,
                        low=c1_low,
                        formed_bar=bar_idx
                    )
                    self._active_levels.append(lvl)
                elif c2_close < c2_open:
                    # Bearish impulse -> Supply Level
                    lvl = _Level(
                        direction="bear",
                        high=c1_high,
                        low=c1_low,
                        formed_bar=bar_idx
                    )
                    self._active_levels.append(lvl)

    def _reset_state(self) -> None:
        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
        self._tp_price = 0.0
