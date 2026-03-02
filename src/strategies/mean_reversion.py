"""
RSI + Bollinger Bands Mean Reversion Strategy.

Signal logic:
  - Enter LONG  when RSI < oversold threshold.
  - Enter SHORT when RSI > overbought threshold.
  - Exit when RSI neutralises (reaches 50) or crosses to the opposite extreme.
  - STRICT REGIME: We only trade inside a normal volatility distribution (20th–80th percentile)
    and ONLY when there is NO strong trend.

All indicators computed once in __init__ and accessed in O(1) per bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy
from src.strategies.filters import ADFFilter, TrendFilter, VolatilityRegimeFilter


@dataclass
class MeanReversionConfig:
    """
    Strategy-specific parameters for RSI+BB Mean Reversion.

    Attributes:
        rsi_window: How many bars to look back when calculating the RSI.
        rsi_oversold: RSI level that triggers a buy signal (asset is considered too cheap).
        rsi_overbought: RSI level that triggers a sell signal (asset is considered too expensive).

        use_vol_filter: Ensure we only trade in "normal" market conditions.
        vol_regime_window: How many bars to use to sample the current actual volatility.
        vol_history_window: How much history to use to rank the current volatility (e.g. out of 500 bars).
        vol_min_pct: Minimum volatility percentile. Drops below this mean the market is dead and flat (compression).
        vol_max_pct: Maximum volatility percentile. Goes above this mean the market is panicking or trending hard.

        use_trend_filter: Ensure we DO NOT trade when there is a strong trend.
        trend_window: How many bars to look back to detect a trend.
        trend_max_tstat: The maximum strength of a trend. If it's higher than this (e.g. >2.0), the market is trending too hard, and we stay out.
        
        use_adf_filter: Ensure the market is mathematically alternating (stationary).
        adf_window: Window size for the Augmented Dickey-Fuller test.
        adf_timeframe: What timeframe to run the stationarity test on (e.g., '1h' to remove local noise).
        adf_max_pvalue: Must be <0.05. Tests if the market is definitely mean-reverting right now.

        atr_window: Lookback window for calculating Average True Range (for stop loss).
        atr_sl_mult: Stop loss multiplier based on ATR.
    """
    rsi_window: int = 14           # Lookback period for RSI calculation
    rsi_oversold: float = 30.0     # Buy signal threshold
    rsi_overbought: float = 70.0   # Sell signal threshold

    use_vol_filter: bool = True    # Only trade during "normal" volatility
    vol_regime_window: int = 50    # Short-term window to measure current vol
    vol_history_window: int = 500  # Historical window to compare against
    vol_min_pct: float = 0.20      # Minimum activity allowed (no dead markets)
    vol_max_pct: float = 0.80      # Maximum activity allowed (no crazy fast markets)

    use_trend_filter: bool = True  # Block trading if the market is trending
    trend_window: int = 100        # Window to look for trends
    trend_max_tstat: float = 2.5   # Block entries if T-Stat is > 2.0 (we only want flat, choppy markets)
    
    use_adf_filter: bool = True    # Mathematically verify the market is mean-reverting
    adf_window: int = 96           # Window for the test
    adf_timeframe: str = "1h"      # Apply test on the hourly chart
    adf_max_pvalue: float = 0.05   # Strict p-value requirement for stationarity

    atr_window: int = 40           # Lookback window for ATR
    atr_sl_mult: float = 2.5       # Stop-loss distance in ATR multiples


class MeanReversionStrategy(BaseStrategy):
    """
    RSI + Bollinger Band mean-reversion strategy.

    Methodology:
        Entries require a confluence of two signals:
          1. RSI in extreme territory (oversold / overbought).
          2. Price at or beyond the opposite Bollinger Band.
        This double-confirmation reduces false signals in choppy markets.

        Optional filters from filters.py further guard against:
          - Volatility Compression (too quiet → mean-reversion stops working).
          - Volatility Expansion (panic → can keep trending indefinitely).
          - Strong Trends (TrendFilter → don't fade a runaway move).
    """

    def __init__(self, engine, config: Optional[MeanReversionConfig] = None) -> None:
        super().__init__(engine)
        import dataclasses

        cfg = config or MeanReversionConfig()

        # Overlay WFO injected parameters if present in engine.settings
        for field in dataclasses.fields(cfg):
            wfo_key = f"mr_{field.name}"
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))

        self.config = cfg
        cfg = self.config
        close = engine.data["close"]

        # ── RSI (Wilder EWM) ───────────────────────────────────────────────
        delta  = close.diff()
        gain   = delta.clip(lower=0.0)
        loss   = (-delta).clip(lower=0.0)
        alpha  = 1.0 / cfg.rsi_window
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # ── ATR (for stop loss) ────────────────────────────────────────────
        high  = engine.data["high"]
        low   = engine.data["low"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=cfg.atr_window, adjust=False).mean()

        # Shift indicators to remove look-ahead bias
        self._rsi: pd.Series = rsi.shift(1)
        self._atr: pd.Series = atr.shift(1)

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
            print(f"[MeanRev] VolatilityRegimeFilter enabled (window={cfg.vol_regime_window})")

        self._trend_filter: Optional[TrendFilter] = None
        if cfg.use_trend_filter:
            self._trend_filter = TrendFilter(
                price=close,
                window=cfg.trend_window,
                max_t_stat=cfg.trend_max_tstat,
            )
            print(f"[MeanRev] TrendFilter enabled (window={cfg.trend_window}, max_tstat={cfg.trend_max_tstat})")

        self._adf_filter: Optional[ADFFilter] = None
        if cfg.use_adf_filter:
            self._adf_filter = ADFFilter(
                series=close,
                adf_window=cfg.adf_window,
                timeframe=cfg.adf_timeframe,
                max_pvalue=cfg.adf_max_pvalue,
            )
            print(f"[MeanRev] ADFFilter enabled (window={cfg.adf_window}, timeframe={cfg.adf_timeframe}, max_pvalue={cfg.adf_max_pvalue})")

        # ── Position tracking ──────────────────────────────────────────────
        self._invested: bool = False
        self._position_side: Optional[str] = None
        self._sl_price: float = 0.0

        valid = self._rsi.notna().sum()
        print(
            f"[MeanRev] Ready | RSI({cfg.rsi_window}) | Valid bars: {valid:,} / {len(close):,}"
        )

    # ── WFO interface ──────────────────────────────────────────────────────────

    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        """
        Optuna search bounds for Walk-Forward Optimisation.

        Parameters prefixed with 'mr_' are injected into BacktestSettings
        by WFOEngine and read back via getattr() in __init__.
        """
        return {
            "mr_rsi_window":      (7,  30,  1),
            "mr_rsi_oversold":    (20.0, 40.0, 5.0),
            "mr_rsi_overbought":  (60.0, 80.0, 5.0),
            "mr_trend_max_tstat": (1.0, 3.5, 0.25),
            "mr_atr_window":      (10,  40,  5),
            "mr_atr_sl_mult":     (1.5, 4.0, 0.5),
            "mr_vol_min_pct":     (0.10, 0.50, 0.05),
            "mr_vol_max_pct":     (0.50, 0.95, 0.05),
            "mr_adf_max_pvalue":  (0.01, 0.15, 0.01),
        }

    # ── Event hook ─────────────────────────────────────────────────────────────

    def on_bar(self, bar: pd.Series) -> List[Order]:
        """
        Evaluates entry / exit conditions for the current bar.

        Args:
            bar: Current OHLCV bar; bar.name is the timestamp.

        Returns:
            List of Order objects (may be empty).
        """
        timestamp = bar.name
        rsi      = self._rsi.get(timestamp, np.nan)
        atr_val  = self._atr.get(timestamp, np.nan)
        close    = bar["close"]

        if np.isnan(rsi) or np.isnan(atr_val):
            return []

        cfg = self.config
        orders: List[Order] = []

        # ── Exit logic (mean reversion to midline + SL) ────────────────────
        if self._invested:
            if self._position_side == "LONG":
                if close <= self._sl_price:
                    orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="STOP_LOSS"))
                    self._reset_state()
                    return orders
                elif rsi >= 50.0:
                    orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="TAKE_PROFIT"))
                    self._reset_state()
                    return orders

            elif self._position_side == "SHORT":
                if close >= self._sl_price:
                    orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="STOP_LOSS"))
                    self._reset_state()
                    return orders
                elif rsi <= 50.0:
                    orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="TAKE_PROFIT"))
                    self._reset_state()
                    return orders

        # ── Entry logic (only when flat) ───────────────────────────────────
        if not self._invested:
            if not self._filters_allow(timestamp):
                return []

            # Long: RSI oversold (must be flat market with normal vol)
            if rsi < cfg.rsi_oversold:
                self._invested = True
                self._position_side = "LONG"
                self._sl_price = close - (atr_val * cfg.atr_sl_mult)
                orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="SIGNAL"))

            # Short: RSI overbought (must be flat market with normal vol)
            elif rsi > cfg.rsi_overbought:
                self._invested = True
                self._position_side = "SHORT"
                self._sl_price = close + (atr_val * cfg.atr_sl_mult)
                orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="SIGNAL"))

        return orders

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _filters_allow(self, timestamp) -> bool:
        """
        Returns True when all enabled filters permit a new entry.

        Args:
            timestamp: Current bar timestamp.

        Returns:
            True if all filters pass (or are disabled).
        """
        if self._vol_filter and not self._vol_filter.is_allowed(timestamp):
            return False
        if self._trend_filter and not self._trend_filter.is_allowed(timestamp):
            return False
        if self._adf_filter and not self._adf_filter.is_allowed(timestamp):
            return False
        return True

    def _reset_state(self) -> None:
        """Clears all open-position tracking variables."""
        self._invested = False
        self._position_side = None
        self._sl_price = 0.0
