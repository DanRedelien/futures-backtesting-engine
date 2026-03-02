"""
Strategies package.

Available strategies:
  SmaCrossoverStrategy  — Dual SMA trend following with ATR-scaled SL/TP.
  MeanReversionStrategy — RSI + Bollinger Bands with optional regime filters.
  IctOrderBlockStrategy — 3-candle SMC order block logic with Trend/Vol filters.
  ZScoreReversalStrategy — Z-Score based mean-reversion with stationarity filters.

Reusable components in filters.py:
  VolatilityRegimeFilter, TrendFilter, ADFFilter, KalmanBeta.
"""

from src.strategies.base import BaseStrategy
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.ict_order_block import IctOrderBlockStrategy
from src.strategies.zscore_reversal import ZScoreReversalStrategy

__all__ = [
    "BaseStrategy",
    "SmaCrossoverStrategy",
    "MeanReversionStrategy",
    "IctOrderBlockStrategy",
    "ZScoreReversalStrategy",
]
