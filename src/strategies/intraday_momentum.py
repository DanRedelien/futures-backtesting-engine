"""
Intraday Momentum — M30 Statistical Breakout.

Logic:
    1. For each time slot (e.g. 09:30, 10:00) compute |Close - Open| over trailing LOOKBACK_DAYS.
    2. Build a per-slot distribution of absolute moves.
    3. Signal fires when current bar's move exceeds the PERCENTILE_THRESHOLD of its OWN slot's history.
    4. Trade WITH the impulse (Close > Open -> LONG).
    5. Hold exactly 1 bar.
"""


# HERE IS NO HMM, THAT"S WHY, THIS STRATEGY MOSTLY LOSES. LATER I WILL ADD HMM


from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine.execution import Order
from src.strategies.base import BaseStrategy

@dataclass
class IntradayMomentumConfig:
    lookback_days: int = 60
    percentile_threshold: float = 0.6
    cooldown_bars: int = 0


class IntradayMomentumStrategy(BaseStrategy):
    """
    Intraday Momentum M30 Statistical Breakout.
    """
    
    def __init__(self, engine, config: Optional[IntradayMomentumConfig] = None) -> None:
        super().__init__(engine)
        cfg = config or IntradayMomentumConfig()
        
        # Overlay WFO-injected parameters
        for field in dataclasses.fields(cfg):
            wfo_key = f"intraday_{field.name}"
            if hasattr(engine.settings, wfo_key):
                setattr(cfg, field.name, getattr(engine.settings, wfo_key))
                
        self.config = cfg
        close = engine.data["close"]
        open_price = engine.data["open"]
        
        # 1. Per-bar absolute move
        abs_move = (close - open_price).abs()
        
        # 2. Time-of-day slot key (HH:MM)
        slots = close.index.strftime("%H:%M")
        
        # 3. Rolling per-slot percentile
        threshold = pd.Series(np.nan, index=close.index)
        
        for s, grp in abs_move.groupby(slots):
            past_moves = grp.shift(1)
            rolling_q = past_moves.rolling(
                window=cfg.lookback_days, 
                min_periods=max(20, cfg.lookback_days // 4)
            ).quantile(cfg.percentile_threshold)
            threshold.loc[rolling_q.index] = rolling_q.values
            
        # 4. Signal: move > slot threshold
        is_breakout = abs_move > threshold
        direction = np.sign(close - open_price)
        
        entries = pd.Series(0, index=close.index)
        entries[is_breakout] = direction[is_breakout].astype(int)
        
        # 4.5. Cooldown after profitable trade
        if cfg.cooldown_bars > 0:
            entries_arr = entries.to_numpy(copy=True)
            close_arr = close.to_numpy()
            n_bars = len(entries_arr)
            cooldown_until = -1
            
            for i in range(n_bars):
                if i <= cooldown_until:
                    entries_arr[i] = 0
                
                if entries_arr[i] != 0:
                    if i + 1 < n_bars:
                        pnl = entries_arr[i] * (close_arr[i+1] - close_arr[i])
                        if pnl > 0:
                            cooldown_until = max(cooldown_until, i + cfg.cooldown_bars)
                            
            entries = pd.Series(entries_arr, index=close.index)
            
        # The engine naturally executes orders at open[t+1], so there is no
        # look-ahead bias if we pass the signal derived from close[t].
        # Shifting here would create a double-delay (2 bars late)!
        self._entries = entries
        
        # ── State ──────────────────────────────────────────────────────────────
        self._invested = False
        self._position_side = None
        
        valid = self._entries.notna().sum()
        n_trades = int((self._entries != 0).sum())
        print(
            f"[Intraday Momentum] Ready | Lookback={cfg.lookback_days} Threshold={cfg.percentile_threshold} "
            f"| Signals generated: {n_trades:,} | Valid bars: {valid:,} / {len(close):,}"
        )
        

    def on_bar(self, bar: pd.Series) -> List[Order]:
        timestamp = bar.name
        
        try:
            signal = self._entries.at[timestamp]
        except KeyError:
            return []
            
        if np.isnan(signal):
            return []
            
        orders: List[Order] = []
        
        # 1-bar hold exit: If invested, exit immediately
        if self._invested:
            side = "SELL" if self._position_side == "LONG" else "BUY"
            orders.append(self.market_order(side, self.settings.fixed_qty, reason="TIME_STOP_1BAR"))
            self._invested = False
            self._position_side = None
            
        # Entry logic
        if not self._invested and signal != 0:
            if signal == 1:
                orders.append(self.market_order("BUY", self.settings.fixed_qty, reason="BREAKOUT_LONG"))
                self._invested = True
                self._position_side = "LONG"
            elif signal == -1:
                orders.append(self.market_order("SELL", self.settings.fixed_qty, reason="BREAKOUT_SHORT"))
                self._invested = True
                self._position_side = "SHORT"
                
        return orders
        
    @classmethod
    def get_search_space(cls) -> Dict[str, Any]:
        return {
            "intraday_lookback_days": (60, 250, 10),
            "intraday_percentile_threshold": (0.80, 0.97, 0.01),
            "intraday_cooldown_bars": (0, 3, 1),
        }
