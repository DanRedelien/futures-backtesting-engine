"""
Data lake module for Parquet-based caching.

Supports multiple timeframes (M5, H1) with smart caching.
For downloading new data, use IBFetcher directly.
"""

from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

from ..backtest_engine.settings import get_settings, BacktestSettings as Settings


class DataLake:
    """
    Parquet-based data caching layer with multi-timeframe support.
    
    Rule: For backtesting, load cached data only.
    For downloading, use IBFetcher directly.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize data lake.
        
        Args:
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
    
    def _get_cache_file(self, symbol: str, timeframe: str = "5m") -> Path:
        """Get path to cache file for symbol and timeframe."""
        cache_dir = self.settings.get_cache_path()
        return cache_dir / f"{symbol}_{timeframe}.parquet"
    
    def load(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Load cached data for a specific timeframe.
        
        Args:
            symbol: Futures symbol (e.g., 'ES').
            timeframe: '5m' or '1h'.
            
        Returns:
            DataFrame with OHLCV data from cache.
        """
        cache_file = self._get_cache_file(symbol, timeframe)
        
        if not cache_file.exists():
            print(f"[WARNING] No cached {timeframe} data for {symbol}")
            print(f"  Run: python run.py --download {symbol}")
            return pd.DataFrame()
        
        df = pd.read_parquet(cache_file)
        
        if df.empty:
            print(f"[WARNING] Cache file corrupted for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        
        # Normalize timezone
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        
        df = df.sort_index()
        
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
            
        if df.empty:
            print(f"[WARNING] Data for {symbol} is empty after applying date filters.")
            return df
            
        print(f"[INFO] Loaded {len(df):,} {timeframe} bars for {symbol}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def load_m1(self, symbol: str) -> pd.DataFrame:
        """Load M1 (1-minute) data for symbol."""
        return self.load(symbol, "1m")
        
    def load_m5(self, symbol: str) -> pd.DataFrame:
        """Load M5 (5-minute) data for symbol."""
        return self.load(symbol, "5m")
    
    def load_m30(self, symbol: str) -> pd.DataFrame:
        """Load M30 (30-minute) data for symbol."""
        return self.load(symbol, "30m")
    
    def load_h1(self, symbol: str) -> pd.DataFrame:
        """Load H1 (1-hour) data for symbol."""
        return self.load(symbol, "1h")
    
    def load_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Load both M5 and H1 data for a symbol.
        
        Args:
            symbol: Futures symbol.
            
        Returns:
            Dict with '1m', 'm5' and 'h1' DataFrames.
        """
        return {
            "1m": self.load_m1(symbol),
            "m5": self.load_m5(symbol),
            "m30": self.load_m30(symbol),
            "h1": self.load_h1(symbol),
        }
    
    def save(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        timeframe: str = "5m"
    ) -> None:
        """
        Save DataFrame to cache.
        
        Args:
            symbol: Futures symbol.
            df: DataFrame to cache.
            timeframe: Data timeframe.
        """
        if df.empty:
            print(f"[WARNING] Empty DataFrame, not saving")
            return
        
        cache_file = self._get_cache_file(symbol, timeframe)
        df.to_parquet(cache_file)
        print(f"[INFO] Saved {len(df):,} {timeframe} bars to {cache_file}")
    
    def list_cached_symbols(self) -> List[str]:
        """
        List all symbols with cached data.
        
        Returns:
            List of unique symbol names.
        """
        cache_dir = self.settings.get_cache_path()
        files = list(cache_dir.glob("*_*.parquet"))
        
        # Extract unique symbols
        symbols = set()
        for f in files:
            # Format: SYMBOL_timeframe.parquet
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                symbols.add(parts[0])
        
        return sorted(list(symbols))
    
    def get_cache_info(self, symbol: str) -> Dict[str, dict]:
        """
        Get info about cached data for all timeframes.
        
        Args:
            symbol: Futures symbol.
            
        Returns:
            Dict with metadata for each timeframe.
        """
        result = {}
        
        for timeframe in ["1m", "5m", "30m", "1h"]:
            cache_file = self._get_cache_file(symbol, timeframe)
            
            if not cache_file.exists():
                result[timeframe] = {"exists": False}
                continue
            
            df = pd.read_parquet(cache_file)
            
            result[timeframe] = {
                "exists": True,
                "bars": len(df),
                "start_date": str(df.index[0]) if not df.empty else None,
                "end_date": str(df.index[-1]) if not df.empty else None,
                "file_size_mb": cache_file.stat().st_size / (1024 * 1024),
            }
        
        return result
