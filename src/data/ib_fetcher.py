"""
Interactive Brokers data fetcher for dual timeframes.

CRITICAL: Both M5 and H1 bars are fetched DIRECTLY from IB API.
NO RESAMPLING M5 -> H1 is allowed!

Features:
    - Fetches M5 (5-minute) bars directly
    - Fetches H1 (1-hour) bars directly
    - Resumable downloads with checkpoints
    - Respects IB pacing limits
    - Automatic contract rolling for futures
"""

import json
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from ib_insync import IB, Future, util

from ..backtest_engine.settings import get_settings, BacktestSettings as Settings


class Timeframe(Enum):
    """
    Supported timeframes for data fetching.
    
    Each timeframe has its own IB bar size setting.
    """
    M1 = ("1m", "1 min")
    M5 = ("5m", "5 mins")
    M30 = ("30m", "30 mins")
    H1 = ("1h", "1 hour")
    
    def __init__(self, file_suffix: str, ib_bar_size: str):
        self.file_suffix = file_suffix
        self.ib_bar_size = ib_bar_size


class IBFetcher:
    """
    IB API data fetcher with dual-timeframe support.
    
    Downloads M5 and H1 bars DIRECTLY from IB (no resampling).
    Saves checkpoints for resumable downloads.
    
    All pacing limits and parameters come from settings.py.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize IB fetcher.
        
        Args:
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self.ib = IB()
        self._connected = False
        self._request_count = 0
        self._last_request_time = 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # CONNECTION
    # ═══════════════════════════════════════════════════════════════════
    
    def connect(self) -> bool:
        """
        Connect to TWS/Gateway.
        
        Returns:
            True if connected successfully.
        """
        if self._connected:
            return True
        
        try:
            self.ib.connect(
                host=self.settings.ib_host,
                port=self.settings.ib_port,
                clientId=self.settings.ib_client_id,
                timeout=self.settings.ib_timeout,
            )
            self._connected = True
            print(f"[INFO] Connected to IB on {self.settings.ib_host}:{self.settings.ib_port}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to IB: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from IB."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            print("[INFO] Disconnected from IB")
    
    # ═══════════════════════════════════════════════════════════════════
    # CONTRACT HANDLING
    # ═══════════════════════════════════════════════════════════════════
    
    def _get_exchange(self, symbol: str) -> str:
        """Get exchange for symbol."""
        exchange_map = {
            "ES": "CME",
            "NQ": "CME",
            "RTY": "CME",
            "YM": "CBOT",
            "GC": "COMEX",
            "SI": "COMEX",
            "CL": "NYMEX",
            "NG": "NYMEX",
            "PL": "NYMEX",
            "ZC": "CBOT",
            "6E": "CME",
        }
        return exchange_map.get(symbol, "CME")
    
    def _get_all_contracts(self, symbol: str) -> list:
        """
        Get all available quarterly contracts for a symbol.
        
        Includes EXPIRED contracts for historical data.
        
        Args:
            symbol: Futures symbol (ES, NQ, etc.).
            
        Returns:
            List of Future contracts sorted by expiry (oldest first).
        """
        exchange = self._get_exchange(symbol)
        
        contract = Future(symbol, exchange=exchange)
        contract.includeExpired = True
        
        details = self.ib.reqContractDetails(contract)
        
        if not details:
            raise ValueError(f"Could not find contracts for {symbol}")
        
        # Filter for quarterly contracts (H=Mar, M=Jun, U=Sep, Z=Dec)
        quarterly_codes = ["H", "M", "U", "Z"]
        
        quarterly_contracts = [
            d.contract for d in details 
            if len(d.contract.localSymbol) >= 3 
            and d.contract.localSymbol[2] in quarterly_codes
        ]
        
        if not quarterly_contracts:
            quarterly_contracts = [d.contract for d in details]
        
        quarterly_contracts.sort(key=lambda c: c.lastTradeDateOrContractMonth)
        
        print(f"[INFO] Found {len(quarterly_contracts)} quarterly contracts for {symbol}")
        return quarterly_contracts
    
    def _get_contract_for_date(
        self, 
        all_contracts: list, 
        target_date: datetime
    ) -> Optional[Future]:
        """
        Get the appropriate contract for a specific date.
        
        Args:
            all_contracts: List of contracts sorted by expiry.
            target_date: Date for which to find active contract.
            
        Returns:
            The active contract for that date.
        """
        for contract in all_contracts:
            expiry_str = contract.lastTradeDateOrContractMonth
            if len(expiry_str) == 8:
                expiry = datetime.strptime(expiry_str, "%Y%m%d")
            else:
                expiry = datetime.strptime(expiry_str + "15", "%Y%m%d")
            
            if expiry > target_date:
                return contract
        
        return all_contracts[-1] if all_contracts else None
    
    # ═══════════════════════════════════════════════════════════════════
    # PACING
    # ═══════════════════════════════════════════════════════════════════
    
    def _wait_for_pacing(self) -> None:
        """Wait to respect IB pacing limits."""
        elapsed = time.time() - self._last_request_time
        delay = self.settings.get_ib_request_delay()
        
        if elapsed < delay:
            sleep_time = delay - elapsed
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECKPOINTING
    # ═══════════════════════════════════════════════════════════════════
    
    def _get_checkpoint_file(self, symbol: str, timeframe: Timeframe) -> Path:
        """Get path to checkpoint file."""
        cache_dir = self.settings.get_cache_path()
        return cache_dir / f"{symbol}_{timeframe.file_suffix}_checkpoint.json"
    
    def _load_checkpoint(self, symbol: str, timeframe: Timeframe) -> Optional[dict]:
        """Load download checkpoint if exists."""
        checkpoint_file = self._get_checkpoint_file(symbol, timeframe)
        
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        
        return None
    
    def _save_checkpoint(
        self, 
        symbol: str, 
        timeframe: Timeframe, 
        last_date: str, 
        total_bars: int
    ) -> None:
        """Save download checkpoint."""
        checkpoint_file = self._get_checkpoint_file(symbol, timeframe)
        
        checkpoint = {
            "symbol": symbol,
            "timeframe": timeframe.file_suffix,
            "last_date": last_date,
            "total_bars": total_bars,
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    def _clear_checkpoint(self, symbol: str, timeframe: Timeframe) -> None:
        """Remove checkpoint file after successful completion."""
        checkpoint_file = self._get_checkpoint_file(symbol, timeframe)
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    # ═══════════════════════════════════════════════════════════════════
    # DATA FETCHING
    # ═══════════════════════════════════════════════════════════════════
    
    def fetch_chunk(
        self,
        contract: Future,
        end_date: datetime,
        timeframe: Timeframe,
        duration: str = "1 W",
    ) -> pd.DataFrame:
        """
        Fetch a chunk of historical bars.
        
        Args:
            contract: IB Future contract.
            end_date: End date for the request.
            timeframe: Timeframe enum (M5 or H1).
            duration: IB duration string.
            
        Returns:
            DataFrame with OHLCV data.
        """
        self._wait_for_pacing()
        
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                durationStr=duration,
                barSizeSetting=timeframe.ib_bar_size,
                whatToShow="TRADES",
                useRTH=self.settings.ib_use_rth,
                formatDate=1,
            )
            
            if bars:
                df = util.df(bars)
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch {timeframe.file_suffix} {duration} chunk ending {end_date.date()}: {e}")
            return pd.DataFrame()
    
    def fetch_m1(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """
        Fetch M1 (1-minute) bars from IB.
        
        Args:
            symbol: Futures symbol (ES, NQ, etc.).
            force_restart: If True, ignore checkpoint and start fresh.
            
        Returns:
            DataFrame with M1 OHLCV data.
        """
        return self._fetch_timeframe(symbol.upper(), Timeframe.M1, force_restart)

    def fetch_m5(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """
        Fetch M5 (5-minute) bars from IB.
        
        Args:
            symbol: Futures symbol (ES, NQ, etc.).
            force_restart: If True, ignore checkpoint and start fresh.
            
        Returns:
            DataFrame with M5 OHLCV data.
        """
        return self._fetch_timeframe(symbol.upper(), Timeframe.M5, force_restart)
    
    def fetch_m30(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """
        Fetch M30 (30-minute) bars from IB.
        
        Args:
            symbol: Futures symbol (ES, NQ, etc.).
            force_restart: If True, ignore checkpoint and start fresh.
            
        Returns:
            DataFrame with M30 OHLCV data.
        """
        return self._fetch_timeframe(symbol.upper(), Timeframe.M30, force_restart)
    
    def fetch_h1(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """
        Fetch H1 (1-hour) bars from IB.
        
        CRITICAL: This fetches H1 bars DIRECTLY from IB.
        NO resampling from M5 is performed!
        
        Args:
            symbol: Futures symbol (ES, NQ, etc.).
            force_restart: If True, ignore checkpoint and start fresh.
            
        Returns:
            DataFrame with H1 OHLCV data.
        """
        return self._fetch_timeframe(symbol.upper(), Timeframe.H1, force_restart)
    
    def fetch_all_timeframes(self, symbol: str, force_restart: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch both M1, M5 and H1 bars for a symbol.
        
        Args:
            symbol: Futures symbol.
            force_restart: If True, restart downloads.
            
        Returns:
            Dict with '1m', 'm5' and 'h1' DataFrames.
        """
        return {
            "1m": self.fetch_m1(symbol, force_restart),
            "m5": self.fetch_m5(symbol, force_restart),
            "m30": self.fetch_m30(symbol, force_restart),
            "h1": self.fetch_h1(symbol, force_restart),
        }
    
    def _load_cache_safe(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        """
        Load cache with robust index handling (fixes 1970 issue).
        """
        cache_dir = self.settings.get_cache_path()
        cache_file = cache_dir / f"{symbol}_{timeframe.file_suffix}.parquet"
        
        if not cache_file.exists():
            return pd.DataFrame()
            
        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                return df
                
            # 1. Prefer 'date' column if it exists (Source of Truth)
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            
            # 2. Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 3. Sanity Check: If index is still likely garbage (e.g. 1970 epoch), warn or drop?
            # For now, we assume step 1 fixed the main cause.
            
            # Ensure native timezone for comparison
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
                
            return df.sort_index()
            
        except Exception as e:
            print(f"[WARNING] Failed to load cache for {symbol}: {e}")
            return pd.DataFrame()

    def _save_cache(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> None:
        """Save dataframe to cache with safeguards."""
        if df.empty:
            return
            
        cache_dir = self.settings.get_cache_path()
        cache_file = cache_dir / f"{symbol}_{timeframe.file_suffix}.parquet"
        
        # Ensure unique index and sorted
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        df.to_parquet(cache_file)
        # print(f"[INFO] Saved {len(df):,} bars to cache")

    def _backfill_loop(
        self, 
        symbol: str, 
        timeframe: Timeframe, 
        start_date: datetime, 
        stop_date: datetime,
        all_contracts: list,
        checkpoint_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download loop: starts at `start_date` and goes BACKWARDS to `stop_date`.
        Returns collected DataFrame.
        """
        collected_data = []
        current_date = start_date
        total_fetched = 0
        chunk_days = 7
        
        # Back-Adjustment State
        current_contract = None
        next_chunk_first_open = None # The 'Open' of the chunk immediately newer than current
        cumulative_adj = 0.0          # Panama Canal Shift

        
        print(f"[FETCH] {timeframe.file_suffix}: {start_date.date()} -> {stop_date.date()}")
        
        try:
            while current_date > stop_date:
                # 1. Get Contract
                contract = self._get_contract_for_date(all_contracts, current_date)
                if not contract:
                    current_date -= timedelta(days=chunk_days)
                    continue
                    
                # Log contract switches
                if current_contract is None or contract.localSymbol != current_contract.localSymbol:
                    # print(f" [ROLL] {contract.localSymbol}")
                    current_contract = contract

                print(f"[{timeframe.file_suffix}] {current_date.date()} | Fetched: {total_fetched:,}", end="\r")

                # 2. Fetch Chunk
                df_chunk = self.fetch_chunk(contract, current_date, timeframe, duration="1 W")
                
                if not df_chunk.empty:
                    # ─── BACK ADJUSTMENT (Panama Canal) ─────────────────────────
                    # If we switched contracts, calculate the price gap
                    if current_contract and contract.localSymbol != current_contract.localSymbol:
                        if next_chunk_first_open is not None:
                             # Gap = Price(Newer) - Price(Older)
                             # We want Older + Adj = Newer => Adj = Newer - Older
                             # We use the boundary: Open of Newer Chunk vs Close of Older Chunk
                             # Note: This assumes intrinsic gap is small compared to Roll Gap.
                             old_close = df_chunk['close'].iloc[-1]
                             gap = next_chunk_first_open - old_close
                             
                             cumulative_adj += gap
                             print(f" [ADJUST] Roll {contract.localSymbol}->{current_contract.localSymbol} | Gap: {gap:.2f} | CumAdj: {cumulative_adj:.2f}")

                    # Apply Adjustment to Histoy
                    if cumulative_adj != 0.0:
                        df_chunk['open'] += cumulative_adj
                        df_chunk['high'] += cumulative_adj
                        df_chunk['low'] += cumulative_adj
                        df_chunk['close'] += cumulative_adj
                    
                    # Update State for next iteration (which is older)
                    current_contract = contract
                    next_chunk_first_open = df_chunk['open'].iloc[0]
                    # ────────────────────────────────────────────────────────────
                    df_chunk["contract"] = contract.localSymbol
                    
                    # Ensure index is standard
                    if df_chunk.index.tz is not None:
                         df_chunk.index = df_chunk.index.tz_convert("UTC").tz_localize(None)
                         
                    collected_data.append(df_chunk)
                    total_fetched += len(df_chunk)
                
                # 3. Checkpointing (Only if key provided)
                # We save periodic checkpoints if this is a long history download
                if checkpoint_key and total_fetched % 50000 == 0 and total_fetched > 0:
                     self._save_checkpoint(symbol, timeframe, current_date.isoformat(), total_fetched)
                     # Partial Save? We rely on standard memory aggregation for now to avoid complexity of partial merges.
                     # But passing data back to caller or saving intermediate file is safer.
                     # For simplicity/cleanliness, we just save the checkpoint marker.
                
                # 4. Step Back
                current_date -= timedelta(days=chunk_days)
                
        except KeyboardInterrupt:
            print(f"\n[STOP] Interrupted at {current_date.date()}")
            if checkpoint_key:
                self._save_checkpoint(symbol, timeframe, current_date.isoformat(), total_fetched)
        except Exception as e:
            print(f"\n[ERROR] Loop error: {e}")
            
        print() # Newline
        
        if not collected_data:
            return pd.DataFrame()
            
        return pd.concat(collected_data)

    def _fetch_timeframe(
        self,
        symbol: str,
        timeframe: Timeframe,
        force_restart: bool = False,
    ) -> pd.DataFrame:
        """
        Smart fetch: Updates recent data (forward fill) AND extends history (backfill).
        """
        if not self.connect():
            return pd.DataFrame()
        
        # 1. Setup Ranges
        delayed_minutes = self.settings.delayed_data_minutes
        max_years = self.settings.max_historical_years
        
        now_date = datetime.now() - timedelta(minutes=delayed_minutes)
        min_history_date = now_date - timedelta(days=max_years * 365)
        
        all_contracts = self._get_all_contracts(symbol)
        if not all_contracts: return pd.DataFrame()

        # 2. Load Existing
        existing_df = pd.DataFrame()
        if not force_restart:
            existing_df = self._load_cache_safe(symbol, timeframe)
            
        # 3. Determine Gaps
        new_head = pd.DataFrame()
        new_tail = pd.DataFrame()
        
        if not existing_df.empty:
            cache_max = existing_df.index.max()
            cache_min = existing_df.index.min()
            
            print(f"[INFO] Existing {timeframe.file_suffix}: {cache_min.date()} to {cache_max.date()} ({len(existing_df):,} bars)")
            
            # GAP A: HEAD (Update recent)
            # If cache_max is older than Now - 2 days (tolerance)
            if cache_max < (now_date - timedelta(days=2)):
                print(f"[UPDATE] Creating new data from {now_date.date()} down to {cache_max.date()}")
                new_head = self._backfill_loop(
                    symbol, timeframe, 
                    start_date=now_date, 
                    stop_date=cache_max, 
                    all_contracts=all_contracts,
                    checkpoint_key=None # Minimal risk, no checkpoint
                )

            # GAP B: TAIL (Extend history)
            # If cache_min is newer than target limit
            if cache_min > (min_history_date + timedelta(days=7)):
                print(f"[EXTEND] Filling history from {cache_min.date()} down to {min_history_date.date()}")
                # Check for checkpoint to resume
                checkpoint = self._load_checkpoint(symbol, timeframe)
                start_resume = cache_min
                if checkpoint:
                     cp_date = datetime.fromisoformat(checkpoint["last_date"])
                     if cp_date < start_resume:
                         start_resume = cp_date
                         print(f"  (Resuming from {start_resume.date()})")

                new_tail = self._backfill_loop(
                    symbol, timeframe, 
                    start_date=start_resume, 
                    stop_date=min_history_date, 
                    all_contracts=all_contracts,
                    checkpoint_key="history"
                )
                
                # Clear checkpoint if done
                if not new_tail.empty and new_tail.index.min() <= (min_history_date + timedelta(days=30)):
                    self._clear_checkpoint(symbol, timeframe)

        else:
            # NO CACHE: Full Download
            print(f"[INIT] Fresh download: {now_date.date()} -> {min_history_date.date()}")
            checkpoint = self._load_checkpoint(symbol, timeframe)
            start_init = now_date
            if checkpoint:
                start_init = datetime.fromisoformat(checkpoint["last_date"])
                print(f"  (Resuming from {start_init.date()})")
                
            new_tail = self._backfill_loop(
                 symbol, timeframe,
                 start_date=start_init,
                 stop_date=min_history_date,
                 all_contracts=all_contracts,
                 checkpoint_key="init"
            )
            # If successful full download, clear cp
            # Logic roughly: if we got close to target, clear it.
        
        # 4. Merge
        dfs_to_merge = []
        if not new_head.empty: dfs_to_merge.append(new_head)
        if not existing_df.empty: dfs_to_merge.append(existing_df)
        if not new_tail.empty: dfs_to_merge.append(new_tail)
        
        if not dfs_to_merge:
            return pd.DataFrame()
            
        final_df = pd.concat(dfs_to_merge)
        
        # 5. Save
        self._save_cache(final_df, symbol, timeframe)
        
        print(f"[SUCCESS] {symbol} {timeframe.file_suffix}: Total {len(final_df):,} bars. Range: {final_df.index.min()} - {final_df.index.max()}")
        self.disconnect()
        return final_df
    
    def test_connection(self) -> bool:
        """
        Test IB connection.
        
        Returns:
            True if connection successful.
        """
        if self.connect():
            print(f"[SUCCESS] Connected to IB")
            print(f"[INFO] Account: {self.ib.managedAccounts()}")
            self.disconnect()
            return True
        return False
