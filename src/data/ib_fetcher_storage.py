"""
Cache and checkpoint helpers for IBFetcher.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .ib_timeframes import Timeframe


class IBFetcherStorageMixin:
    """Cache and checkpoint operations shared by the public IBFetcher."""

    def _get_checkpoint_file(self, symbol: str, timeframe: Timeframe) -> Path:
        """Returns the checkpoint path for one symbol and timeframe."""
        cache_dir = self.settings.get_cache_path()
        return cache_dir / f"{symbol}_{timeframe.file_suffix}_checkpoint.json"

    def _load_checkpoint(self, symbol: str, timeframe: Timeframe) -> Optional[dict]:
        """Loads a persisted historical-download checkpoint when present."""
        checkpoint_file = self._get_checkpoint_file(symbol, timeframe)
        if checkpoint_file.exists():
            return json.loads(checkpoint_file.read_text(encoding="utf-8"))
        return None

    def _save_checkpoint(
        self,
        symbol: str,
        timeframe: Timeframe,
        last_date: str,
        total_bars: int,
    ) -> None:
        """Persists one resumable historical-download checkpoint."""
        checkpoint_file = self._get_checkpoint_file(symbol, timeframe)
        checkpoint = {
            "symbol": symbol,
            "timeframe": timeframe.file_suffix,
            "last_date": last_date,
            "total_bars": total_bars,
            "updated_at": datetime.now().isoformat(),
        }
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    def _clear_checkpoint(self, symbol: str, timeframe: Timeframe) -> None:
        """Deletes the checkpoint after a successful history extension."""
        checkpoint_file = self._get_checkpoint_file(symbol, timeframe)
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    def _load_cache_safe(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        """Loads cached parquet data with consistent datetime-index handling."""
        cache_dir = self.settings.get_cache_path()
        cache_file = cache_dir / f"{symbol}_{timeframe.file_suffix}.parquet"

        if not cache_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                return df
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            return df.sort_index()
        except Exception as exc:
            print(f"[WARNING] Failed to load cache for {symbol}: {exc}")
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> None:
        """Writes cached parquet data after deduplicating and sorting the index."""
        if df.empty:
            return

        cache_dir = self.settings.get_cache_path()
        cache_file = cache_dir / f"{symbol}_{timeframe.file_suffix}.parquet"
        clean_df = df[~df.index.duplicated(keep="last")].sort_index()
        clean_df.to_parquet(cache_file)
