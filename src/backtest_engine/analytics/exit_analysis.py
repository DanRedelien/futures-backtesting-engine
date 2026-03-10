"""
src/backtest_engine/analytics/exit_analysis.py

Data enrichment layer for Exit Analysis.
Computes MFE, MAE, Holding Time, and PnL Decay by slicing OHLCV data.
This runs once at the end of the backtest to offload dashboard computation.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def enrich_trades_with_exit_analytics(
    trades_df: pd.DataFrame, 
    data_map: Dict[Any, pd.DataFrame],
    regime_window: Optional[int] = None,
    history_window: Optional[int] = None,
    vol_min_pct: Optional[float] = None,
    vol_max_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Enriches a trades DataFrame with exit analytics:
      - holding_time (Timedelta)
      - mfe (Maximum Favorable Excursion, in $)
      - mae (Maximum Adverse Excursion, in $)
      - pnl_decay_5m, 15m, 30m, 60m (Hypothetical PnL if exited at T+N)
      - entry_volatility (14-period standard deviation of returns at entry)

    Args:
        trades_df: Basic trades dataframe containing entry_time, exit_time, symbol, direction, entry_price.
        data_map: In single mode, dict of {symbol: df}. 
                  In portfolio mode, dict of {(slot_id, symbol): df}.
                  
    Returns:
        Enriched DataFrame.
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    multiplier_cache: Dict[str, float] = {}

    try:
        from src.backtest_engine.settings import BacktestSettings
        settings = BacktestSettings()
    except Exception:
        settings = None
    
    # Resolve parameters from settings if not passed explicitly
    _rw = regime_window if regime_window is not None else (settings.vol_regime_window_default if settings else 50)
    _hw = history_window if history_window is not None else (settings.vol_history_window_default if settings else 500)
    _min_p = vol_min_pct if vol_min_pct is not None else (settings.vol_min_pct_default if settings else 0.20)
    _max_p = vol_max_pct if vol_max_pct is not None else (settings.vol_max_pct_default if settings else 0.80)
    
    # Pre-allocate columns with proper dtypes
    df["holding_time"] = pd.Series(dtype='timedelta64[ns]')
    df["mfe"] = np.nan
    df["mae"] = np.nan
    
    horizons = [5, 15, 30, 60, 120, 240, 480, 720, 1440]
    for h in horizons:
        df[f"pnl_decay_{h}m"] = np.nan
    df["entry_volatility"] = np.nan
    df["vol_min_pct"] = _min_p
    df["vol_max_pct"] = _max_p

    # Group by symbol to pre-calculate volatility and handle lookups efficiently
    for (slot_id, symbol), group in df.groupby(["slot_id", "symbol"], dropna=False):
        # Determine data map key
        df_sym = data_map.get((slot_id, symbol)) if slot_id is not None else None
        if df_sym is None or df_sym.empty:
            df_sym = data_map.get(symbol)
            
        if df_sym is None or df_sym.empty:
            continue
            
        # Ensure monotonic index for searchsorted
        if not df_sym.index.is_monotonic_increasing:
            df_sym = df_sym.sort_index()

        # Pre-calculate rolling volatility (matching VolatilityRegimeFilter logic)
        # 1. Price Standard Deviation (Window=_rw)
        # 2. Percentile Rank (Window=_hw)
        try:
            rolling_std = df_sym["close"].rolling(window=_rw, min_periods=_rw // 2).std()
            rolling_vol = rolling_std.rolling(window=_hw, min_periods=_hw // 2).rank(pct=True)
        except Exception:
            rolling_vol = pd.Series(np.nan, index=df_sym.index)

        # Pre-cache index for searchsorted
        idx_array = df_sym.index.values

        for idx in group.index:
            row = df.loc[idx]
            entry = row.get("entry_time")
            exit_ = row.get("exit_time")
            entry_price = row.get("entry_price")
            direction = row.get("direction", "LONG")
            sign = 1.0 if direction == "LONG" else -1.0
            qty = abs(float(row.get("quantity", 1.0)))
            multiplier = multiplier_cache.get(symbol, 1.0)
            
            if pd.isna(entry) or pd.isna(exit_) or pd.isna(entry_price):
                continue
                
            df.at[idx, "holding_time"] = exit_ - entry
            
            # Round trip costs
            comm = row.get("commission", 0.0)
            slip = row.get("slippage", 0.0)
            costs = (0.0 if pd.isna(comm) else float(comm)) + (0.0 if pd.isna(slip) else float(slip))
            
            # MFE / MAE (Slicing is generally fast enough if done efficiently)
            try:
                trade_bars = df_sym.loc[entry:exit_]
                if not trade_bars.empty:
                    max_p = trade_bars["high"].max()
                    min_p = trade_bars["low"].min()
                    
                    if direction == "LONG":
                        mfe = (max_p - entry_price) * qty * multiplier
                        mae = (min_p - entry_price) * qty * multiplier
                    else:
                        mfe = (entry_price - min_p) * qty * multiplier
                        mae = (entry_price - max_p) * qty * multiplier
                        
                    df.at[idx, "mfe"] = float(mfe if mfe > 0 else 0.0)
                    df.at[idx, "mae"] = float(mae if mae < 0 else 0.0)
            except Exception:
                pass

            # Entry Volatility (Calculated from pre-cached rolling_vol)
            try:
                # searchsorted with side='right' and minus 1 mimics method='pad'
                target_dt = np.datetime64(entry)
                pos = np.searchsorted(idx_array, target_dt, side='right') - 1
                if pos >= 0:
                    val = rolling_vol.iloc[pos]
                    df.at[idx, "entry_volatility"] = float(val)
            except Exception:
                pass

            # PnL Decay (Forward PnL)
            for minutes in horizons:
                target_time = entry + pd.Timedelta(minutes=minutes)
                col_name = f"pnl_decay_{minutes}m"
                try:
                    pos = np.searchsorted(idx_array, target_time, side='right') - 1
                    if pos >= 0:
                        hypo_price = df_sym.iloc[pos]["close"]
                        hypo_gross = sign * (hypo_price - entry_price) * qty * multiplier
                        df.at[idx, col_name] = float(hypo_gross - costs)
                except Exception:
                    pass
    
    return df
