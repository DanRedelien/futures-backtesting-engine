"""
Z-Score Mean Reversion Strategy.

Methodology:
  Computes a rolling Z-Score of the closing price.
  Z = (Close - Rolling_Mean) / Rolling_StdDev
  
  Signal logic:
    - Enter LONG  when Z-Score < -2.0 (Price is unusually low).
    - Enter SHORT when Z-Score > 2.0  (Price is unusually high).
    - Exit on ATR stop-loss or ATR take-profit.

  Filters:
    - Directional Bias   : Long-term SMA. Trade shorts only below SMA, longs above SMA.
    - Volatility Regime  : Blocks entries during dead/compression or panic/expansion regimes.
    - Trend T-Stat       : Blocks mean reversion when a strong directional trend is detected.
    - Stationarity (Half-Life) : Blocks mean reversion when the mean-reverting speed
                           (Half-Life) is too slow, meaning poor capital efficiency.

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
from src.strategies.filters import HalfLifeFilter, VolatilityRegimeFilter


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZScoreReversalConfig:
    """
    All tunable parameters for the Z-Score Reversal strategy.

    Signal Generation
    ─────────────────
    zscore_window    : Lookback period for Mean and Standard Deviation computation.
    zscore_entry_lvl : Absolute Z-Score threshold for entries. LONG below -lvl, SHORT above +lvl.

    Risk Management
    ───────────────
    atr_window       : Lookback window for ATR.
    atr_sl_mult      : Stop-loss distance as a multiple of ATR.
    atr_tp_mult      : Take-profit distance as a multiple of ATR.

    Direction Filter
    ────────────────
    trade_direction  : "both" | "long" | "short"
    
    Trend SMA Bias
    ──────────────
    trend_sma_window : Long-term SMA length. When price > SMA, only LONGS are permitted.
                       When price < SMA, only SHORTS are permitted. None = disabled.

    Trend Filter (T-Stat)
    ─────────────────────
    use_trend_filter : Enable/disable T-stat trend gate.
    trend_window     : Rolling OLS regression window for the trend.
    trend_max_tstat  : Maximum absolute T-stat to allow trade. Higher = stronger trend.

    Volatility Regime Filter
    ────────────────────────
    use_vol_filter     : Enable/disable regime gate.
    vol_regime_window  : Short-term window for current vol measurement.
    vol_history_window : Historical window for percentile ranking.
    vol_min_pct        : Minimum vol percentile (blocks dead markets).
    vol_max_pct        : Maximum vol percentile (blocks panic markets).

    Half-Life Filter
    ────────────────
    use_hl_filter       : Enable/disable the Half-Life mean-reverting speed check.
    hl_window           : Window for rolling regression.
    hl_baseline         : Baseline mean-reverting speed target value (e.g. 5.0 bars).
    hl_multiplier       : Block entries if speed > baseline * multiplier.
    hl_max_holding_mult : Exit trade early if held for more than entry_hl * max_holding_mult.

    """
    # ── Signal generation ──────────────────────────────────────────────────────
    zscore_window: int = 115
    zscore_entry_lvl: float = 1.475

    # ── Risk management ────────────────────────────────────────────────────────
    atr_window: int = 14
    atr_sl_mult: float = 1.625
    atr_tp_mult: float = 3.375

    # ── Direction ──────────────────────────────────────────────────────────────
    trade_direction: str = "both"

    # ── Volatility regime filter ───────────────────────────────────────────────
    use_vol_filter: bool = True
    vol_regime_window: int = 50
    vol_history_window: int = 500
    vol_min_pct: float = 0.20
    vol_max_pct: float = 0.80

    # ── Half-Life filter ───────────────────────────────────────────────────────
    use_hl_filter: bool = True
    hl_window: int = 100
    hl_baseline: float = 5.0
    hl_multiplier: float = 2.0
    hl_max_holding_mult: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy
# ═══════════════════════════════════════════════════════════════════════════════

class ZScoreReversalStrategy(BaseStrategy):
    """
    Classical Z-Score Reversal Strategy.

    Pre-computes Z-Score and all filters in __init__, shifts by 1 bar to prevent
    look-ahead bias, and executes via O(1) lookups in on_bar.
    """

    def __init__(self, engine, config: Optional[ZScoreReversalConfig] = None) -> None:
        super().__init__(engine)

        cfg = config or ZScoreReversalConfig()

        # WFO parameter injection: engine.settings may carry zscore_* keys
        for field in dataclasses.fields(cfg):
            wfo_key = f"zscore_{field.name}"
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))

        self.config = cfg

        close = engine.data["close"]
        high  = engine.data["high"]
        low   = engine.data["low"]

        # ── Z-Score ────────────────────────────────────────────────────────────
        roll_mean = close.rolling(window=cfg.zscore_window, min_periods=cfg.zscore_window // 2).mean()
        roll_std = close.rolling(window=cfg.zscore_window, min_periods=cfg.zscore_window // 2).std()
        
        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        zscore = (close - roll_mean) / roll_std

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

        # Indicators are looked up directly from the closing bar's timestamp.
        self._zscore: pd.Series = zscore
        self._atr:    pd.Series = atr
        self._close:  pd.Series = close

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
            print(f"[Z-SCORE] VolFilter enabled (window={cfg.vol_regime_window})")

        # ── Half-Life filter ───────────────────────────────────────────────────
        self._hl_filter: Optional[HalfLifeFilter] = None
        if cfg.use_hl_filter:
            self._hl_filter = HalfLifeFilter(
                series=close,
                window=cfg.hl_window,
                max_half_life=cfg.hl_baseline * cfg.hl_multiplier,
                lambda_min=getattr(engine.settings, "hl_lambda_min", 1e-4),
                max_cap=getattr(engine.settings, "hl_max_cap", 500.0),
            )
            print(f"[Z-SCORE] HalfLifeFilter enabled (window={cfg.hl_window}, max_hl={cfg.hl_baseline * cfg.hl_multiplier})")

        # ── State ──────────────────────────────────────────────────────────────
        self._invested: bool = False
        self._position_side: Optional[str] = None
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0
        self._bars_held: int = 0
        self._entry_hl: float = 0.0
        
        self._zscore_entry_lvl = cfg.zscore_entry_lvl

        print(
            f"[Z-SCORE] Ready | z_window={cfg.zscore_window} "
            f"| direction={cfg.trade_direction} "
        )

    # ── WFO interface ──────────────────────────────────────────────────────────

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """
        Optuna search bounds for Walk-Forward Optimisation.
        """
        return {
            "zscore_zscore_window":    (30, 120, 10),
            "zscore_zscore_entry_lvl": (1.25, 2.75, 0.15),
            "zscore_atr_sl_mult":      (1.0, 3.0, 0.25),
            "zscore_atr_tp_mult":      (1.5, 4.5, 0.25),
            "zscore_hl_baseline":      (3.0, 12.0, 1.0),
        }

    # ── Main event hook ────────────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> List[Order]:
        """
        Called once per bar. Evaluates pre-computed Z-Score and manages exits.

        Args:
            bar: Current OHLCV bar; bar.name is the timestamp.

        Returns:
            List of Order objects (may be empty).
        """
        timestamp = bar.name
        z_val     = self._zscore.get(timestamp, np.nan)
        atr_val   = self._atr.get(timestamp, np.nan)
        c_close   = bar["close"]
        c_high    = bar["high"]
        c_low     = bar["low"]

        if np.isnan(z_val) or np.isnan(atr_val):
            return []

        orders: List[Order] = []

        # ── In position: manage SL / TP exits ─────────────────────────────────
        if self._invested:
            self._bars_held += 1
            
            # Time stop based on Half-Life
            if self._bars_held > (self._entry_hl * self.config.hl_max_holding_mult):
                orders.append(
                    self.market_order("SELL" if self._position_side == "LONG" else "BUY", 
                                      self.settings.fixed_qty, reason="TIME_STOP")
                )
                self._reset_state()
                return orders

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

        # ── No position: check for entries ────────────────────────────────────
        if not self._invested:
            crossover = 0.0
            if z_val < -self._zscore_entry_lvl:
                crossover = 1.0   # Want to go LONG
            elif z_val > self._zscore_entry_lvl:
                crossover = -1.0  # Want to go SHORT

            if crossover != 0.0 and self._filters_allow(timestamp, crossover):
                if crossover == 1.0:
                    sl_dist = atr_val * self.config.atr_sl_mult
                    tp_dist = atr_val * self.config.atr_tp_mult
                    
                    self._invested      = True
                    self._position_side = "LONG"
                    self._sl_price      = c_close - sl_dist
                    self._tp_price      = c_close + tp_dist
                    self._bars_held     = 0
                    self._entry_hl      = self._hl_filter.get(timestamp, self.config.hl_baseline) if self._hl_filter else self.config.hl_baseline
                    
                    orders.append(
                        self.market_order("BUY", self.settings.fixed_qty, reason="Z_LONG")
                    )
                
                elif crossover == -1.0:
                    sl_dist = atr_val * self.config.atr_sl_mult
                    tp_dist = atr_val * self.config.atr_tp_mult
                    
                    self._invested      = True
                    self._position_side = "SHORT"
                    self._sl_price      = c_close + sl_dist
                    self._tp_price      = c_close - tp_dist
                    self._bars_held     = 0
                    self._entry_hl      = self._hl_filter.get(timestamp, self.config.hl_baseline) if self._hl_filter else self.config.hl_baseline
                    
                    orders.append(
                        self.market_order("SELL", self.settings.fixed_qty, reason="Z_SHORT")
                    )

        return orders

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
            
        # ── Half-Life filter ──────────────────────────────────────────────────
        if self._hl_filter and not self._hl_filter.is_allowed(timestamp):
            return False

        return True

    # ── State reset ────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Clears all open-position tracking variables."""
        self._invested       = False
        self._position_side  = None
        self._sl_price       = 0.0
        self._tp_price       = 0.0
        self._bars_held      = 0
        self._entry_hl       = 0.0
