"""
Historical backfill orchestration for IBFetcher.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from ib_insync import Future

from .ib_timeframes import Timeframe


class IBFetcherHistoryMixin:
    """Historical download workflows shared by the public IBFetcher."""

    def fetch_m1(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """Fetches 1-minute bars directly from IB."""
        return self._fetch_timeframe(symbol.upper(), Timeframe.M1, force_restart)

    def fetch_m5(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """Fetches 5-minute bars directly from IB."""
        return self._fetch_timeframe(symbol.upper(), Timeframe.M5, force_restart)

    def fetch_m30(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """Fetches 30-minute bars directly from IB."""
        return self._fetch_timeframe(symbol.upper(), Timeframe.M30, force_restart)

    def fetch_h1(self, symbol: str, force_restart: bool = False) -> pd.DataFrame:
        """Fetches 1-hour bars directly from IB."""
        return self._fetch_timeframe(symbol.upper(), Timeframe.H1, force_restart)

    def fetch_all_timeframes(
        self,
        symbol: str,
        force_restart: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Fetches the full supported timeframe set for one symbol."""
        return {
            "1m": self.fetch_m1(symbol, force_restart),
            "m5": self.fetch_m5(symbol, force_restart),
            "m30": self.fetch_m30(symbol, force_restart),
            "h1": self.fetch_h1(symbol, force_restart),
        }

    def _backfill_loop(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: datetime,
        stop_date: datetime,
        all_contracts: list[Future],
        checkpoint_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """Downloads one historical range backwards, applying roll adjustments."""
        collected_data: list[pd.DataFrame] = []
        current_date = start_date
        total_fetched = 0
        chunk_days = 7
        current_contract: Optional[Future] = None
        next_chunk_first_open: Optional[float] = None
        cumulative_adj = 0.0

        print(f"[FETCH] {timeframe.file_suffix}: {start_date.date()} -> {stop_date.date()}")

        try:
            while current_date > stop_date:
                contract = self._get_contract_for_date(all_contracts, current_date)
                if not contract:
                    current_date -= timedelta(days=chunk_days)
                    continue

                previous_contract = current_contract
                if previous_contract is None or contract.localSymbol != previous_contract.localSymbol:
                    current_contract = contract

                print(
                    f"[{timeframe.file_suffix}] {current_date.date()} | Fetched: {total_fetched:,}",
                    end="\r",
                )

                df_chunk = self.fetch_chunk(contract, current_date, timeframe, duration="1 W")

                if not df_chunk.empty:
                    if (
                        previous_contract is not None
                        and contract.localSymbol != previous_contract.localSymbol
                        and next_chunk_first_open is not None
                    ):
                        old_close = df_chunk["close"].iloc[-1]
                        gap = next_chunk_first_open - old_close
                        cumulative_adj += gap
                        print(
                            f" [ADJUST] Roll {contract.localSymbol}->{previous_contract.localSymbol} "
                            f"| Gap: {gap:.2f} | CumAdj: {cumulative_adj:.2f}"
                        )

                    if cumulative_adj != 0.0:
                        for column in ["open", "high", "low", "close"]:
                            df_chunk[column] += cumulative_adj

                    current_contract = contract
                    next_chunk_first_open = float(df_chunk["open"].iloc[0])
                    df_chunk["contract"] = contract.localSymbol

                    if df_chunk.index.tz is not None:
                        df_chunk.index = df_chunk.index.tz_convert("UTC").tz_localize(None)

                    collected_data.append(df_chunk)
                    total_fetched += len(df_chunk)

                if checkpoint_key and total_fetched % 50000 == 0 and total_fetched > 0:
                    self._save_checkpoint(symbol, timeframe, current_date.isoformat(), total_fetched)

                current_date -= timedelta(days=chunk_days)
        except KeyboardInterrupt:
            print(f"\n[STOP] Interrupted at {current_date.date()}")
            if checkpoint_key:
                self._save_checkpoint(symbol, timeframe, current_date.isoformat(), total_fetched)
        except Exception as exc:
            print(f"\n[ERROR] Loop error: {exc}")

        print()

        if not collected_data:
            return pd.DataFrame()

        return pd.concat(collected_data)

    def _fetch_timeframe(
        self,
        symbol: str,
        timeframe: Timeframe,
        force_restart: bool = False,
    ) -> pd.DataFrame:
        """Extends recent data and historical tail for one symbol and timeframe."""
        if not self.connect():
            return pd.DataFrame()

        delayed_minutes = self.settings.delayed_data_minutes
        max_years = self.settings.max_historical_years

        now_date = datetime.now() - timedelta(minutes=delayed_minutes)
        min_history_date = now_date - timedelta(days=max_years * 365)

        all_contracts = self._get_all_contracts(symbol)
        if not all_contracts:
            return pd.DataFrame()

        existing_df = pd.DataFrame()
        if not force_restart:
            existing_df = self._load_cache_safe(symbol, timeframe)

        new_head = pd.DataFrame()
        new_tail = pd.DataFrame()

        if not existing_df.empty:
            cache_max = existing_df.index.max()
            cache_min = existing_df.index.min()
            print(
                f"[INFO] Existing {timeframe.file_suffix}: {cache_min.date()} "
                f"to {cache_max.date()} ({len(existing_df):,} bars)"
            )

            if cache_max < (now_date - timedelta(days=2)):
                print(f"[UPDATE] Creating new data from {now_date.date()} down to {cache_max.date()}")
                new_head = self._backfill_loop(
                    symbol,
                    timeframe,
                    start_date=now_date,
                    stop_date=cache_max,
                    all_contracts=all_contracts,
                    checkpoint_key=None,
                )

            if cache_min > (min_history_date + timedelta(days=7)):
                print(f"[EXTEND] Filling history from {cache_min.date()} down to {min_history_date.date()}")
                checkpoint = self._load_checkpoint(symbol, timeframe)
                start_resume = cache_min
                if checkpoint:
                    checkpoint_date = datetime.fromisoformat(checkpoint["last_date"])
                    if checkpoint_date < start_resume:
                        start_resume = checkpoint_date
                        print(f"  (Resuming from {start_resume.date()})")

                new_tail = self._backfill_loop(
                    symbol,
                    timeframe,
                    start_date=start_resume,
                    stop_date=min_history_date,
                    all_contracts=all_contracts,
                    checkpoint_key="history",
                )

                if not new_tail.empty and new_tail.index.min() <= (min_history_date + timedelta(days=30)):
                    self._clear_checkpoint(symbol, timeframe)
        else:
            print(f"[INIT] Fresh download: {now_date.date()} -> {min_history_date.date()}")
            checkpoint = self._load_checkpoint(symbol, timeframe)
            start_init = now_date
            if checkpoint:
                start_init = datetime.fromisoformat(checkpoint["last_date"])
                print(f"  (Resuming from {start_init.date()})")

            new_tail = self._backfill_loop(
                symbol,
                timeframe,
                start_date=start_init,
                stop_date=min_history_date,
                all_contracts=all_contracts,
                checkpoint_key="init",
            )

        dfs_to_merge: list[pd.DataFrame] = []
        if not new_head.empty:
            dfs_to_merge.append(new_head)
        if not existing_df.empty:
            dfs_to_merge.append(existing_df)
        if not new_tail.empty:
            dfs_to_merge.append(new_tail)

        if not dfs_to_merge:
            return pd.DataFrame()

        final_df = pd.concat(dfs_to_merge)
        self._save_cache(final_df, symbol, timeframe)

        print(
            f"[SUCCESS] {symbol} {timeframe.file_suffix}: Total {len(final_df):,} bars. "
            f"Range: {final_df.index.min()} - {final_df.index.max()}"
        )
        self.disconnect()
        return final_df
