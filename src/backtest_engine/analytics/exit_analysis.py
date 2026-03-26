"""
src/backtest_engine/analytics/exit_analysis.py

Data enrichment layer for Exit Analysis.
Computes MFE, MAE, Holding Time, and PnL Decay by slicing OHLCV data.
This runs once at the end of the backtest to offload dashboard computation.
"""

from typing import Any, Dict, Optional

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

    def _norm_slot_id(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    def _to_np_dt64(value: Any) -> np.datetime64:
        # np.searchsorted on datetime64 arrays does not accept pd.Timestamp directly.
        # Cast everything to numpy datetime64 for stable comparisons.
        # Normalize to naive UTC to avoid timezone mismatch with data index.
        ts = pd.Timestamp(value)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return np.datetime64(ts.to_datetime64())

    def _resolve_data_for_symbol(sym: Any, slot: Any) -> Optional[pd.DataFrame]:
        """Resolve OHLCV DataFrame for symbol from data_map (single or portfolio keying)."""
        sym_str = str(sym) if sym is not None else ""
        # Portfolio mode: (slot_id, symbol)
        if slot is not None:
            out = data_map.get((slot, sym))
            if out is not None and not out.empty:
                return out
            out = data_map.get((slot, sym_str))
            if out is not None and not out.empty:
                return out
        # Single-asset mode: symbol-only key
        out = data_map.get(sym)
        if out is not None and not out.empty:
            return out
        out = data_map.get(sym_str)
        if out is not None and not out.empty:
            return out
        # Fallback: scan for any key containing this symbol
        for k, v in data_map.items():
            if v is None or v.empty:
                continue
            if k == sym or k == sym_str:
                return v
            if isinstance(k, tuple) and len(k) >= 2 and (k[1] == sym or k[1] == sym_str):
                return v
        return None

    group_cols = ["symbol"] if "slot_id" not in df.columns else ["slot_id", "symbol"]

    # Group by symbol (and slot_id in portfolio mode) to pre-calculate volatility and handle lookups efficiently
    for keys, group in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 2:
            slot_id, symbol = keys
        else:
            # Single column: keys is a 1-tuple e.g. ('ES',), not the raw value
            slot_id = None
            symbol = keys[0] if isinstance(keys, tuple) else keys

        slot_id = _norm_slot_id(slot_id)
        df_sym = _resolve_data_for_symbol(symbol, slot_id)

        if df_sym is None or df_sym.empty:
            continue

        # Normalize index to naive UTC for searchsorted compatibility with trade timestamps
        idx = df_sym.index
        if hasattr(idx, "tz") and idx.tz is not None:
            df_sym = df_sym.copy()
            df_sym.index = idx.tz_convert("UTC").tz_localize(None)

        # Ensure monotonic index for searchsorted
        if not df_sym.index.is_monotonic_increasing:
            df_sym = df_sym.sort_index()

        # Pre-calculate rolling volatility (matching VolatilityRegimeFilter logic)
        # 1. Price Standard Deviation (Window=_rw)
        # 2. Percentile Rank (Window=_hw)
        # 3. Shift by one bar so volatility at entry never includes the
        #    entry bar close (entry is executed at open).
        try:
            rolling_std = df_sym["close"].rolling(window=_rw, min_periods=_rw // 2).std()
            rolling_vol = rolling_std.rolling(window=_hw, min_periods=_hw // 2).rank(pct=True).shift(1)
        except Exception:
            rolling_vol = pd.Series(np.nan, index=df_sym.index)

        # Pre-cache index for searchsorted
        idx_array = df_sym.index.values

        for row_idx in group.index:
            row = df.loc[row_idx]
            entry = row.get("entry_time")
            exit_ = row.get("exit_time")
            entry_price = row.get("entry_price")
            direction = row.get("direction", "LONG")
            sign = 1.0 if direction == "LONG" else -1.0
            qty = abs(float(row.get("quantity", 1.0)))
            if symbol not in multiplier_cache:
                if settings is not None:
                    multiplier_cache[symbol] = float(settings.get_instrument_spec(str(symbol)).get("multiplier", 1.0))
                else:
                    multiplier_cache[symbol] = 1.0
            multiplier = multiplier_cache.get(symbol, 1.0)
            
            if pd.isna(entry) or pd.isna(exit_) or pd.isna(entry_price):
                continue
                
            df.at[row_idx, "holding_time"] = exit_ - entry
            
            # Round trip costs
            comm = row.get("commission", 0.0)
            slip = row.get("slippage", 0.0)
            costs = (0.0 if pd.isna(comm) else float(comm)) + (0.0 if pd.isna(slip) else float(slip))
            
            # MFE / MAE includes the entry bar because the position is live from
            # entry open onward.
            try:
                entry_dt = _to_np_dt64(entry)
                exit_dt = _to_np_dt64(exit_)
                # Include the entry bar in excursion analysis. The trade is live
                # from entry open, so excluding the full entry bar introduces an
                # artificial one-bar delay in MFE/MAE.
                start_pos = np.searchsorted(idx_array, entry_dt, side='left')
                end_pos = np.searchsorted(idx_array, exit_dt, side='right')
                trade_bars = df_sym.iloc[start_pos:end_pos]
                if not trade_bars.empty:
                    max_p = trade_bars["high"].max()
                    min_p = trade_bars["low"].min()
                    
                    if direction == "LONG":
                        mfe = (max_p - entry_price) * qty * multiplier
                        mae = (min_p - entry_price) * qty * multiplier
                    else:
                        mfe = (entry_price - min_p) * qty * multiplier
                        mae = (entry_price - max_p) * qty * multiplier
                        
                    df.at[row_idx, "mfe"] = float(mfe if mfe > 0 else 0.0)
                    df.at[row_idx, "mae"] = float(mae if mae < 0 else 0.0)
            except Exception:
                pass

            # Entry Volatility (Calculated from pre-cached rolling_vol)
            try:
                # searchsorted with side='right' and minus 1 mimics method='pad'
                target_dt = _to_np_dt64(entry)
                pos = np.searchsorted(idx_array, target_dt, side='right') - 1
                if pos >= 0:
                    val = rolling_vol.iloc[pos]
                    df.at[row_idx, "entry_volatility"] = float(val)
            except Exception:
                pass

            # PnL Decay (Forward PnL)
            for minutes in horizons:
                target_time = entry + pd.Timedelta(minutes=minutes)
                col_name = f"pnl_decay_{minutes}m"
                try:
                    target_dt = _to_np_dt64(target_time)
                    pos = np.searchsorted(idx_array, target_dt, side='right') - 1
                    if pos >= 0:
                        hypo_price = df_sym.iloc[pos]["close"]
                        hypo_gross = sign * (hypo_price - entry_price) * qty * multiplier
                        df.at[row_idx, col_name] = float(hypo_gross - costs)
                except Exception:
                    pass
    
    return df
