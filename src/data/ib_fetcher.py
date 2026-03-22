"""
Interactive Brokers data fetcher for direct futures timeframes.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
from ib_insync import IB, Future, util

from ..backtest_engine.settings import BacktestSettings as Settings
from .ib_fetcher_contracts import IBFetcherContractsMixin
from .ib_fetcher_history import IBFetcherHistoryMixin
from .ib_fetcher_storage import IBFetcherStorageMixin
from .ib_timeframes import Timeframe


class IBFetcher(
    IBFetcherContractsMixin,
    IBFetcherStorageMixin,
    IBFetcherHistoryMixin,
):
    """
    IB API data fetcher with direct multi-timeframe support.

    Methodology:
        M1, M5, M30, and H1 bars are fetched directly from IB. The public
        class remains the stable entry point while contract discovery, storage,
        and historical backfill logic live in focused helper modules.
    """

    def __init__(self, settings: Settings):
        """Initializes the fetcher with the shared runtime settings."""
        self.settings = settings
        self.ib = IB()
        self._connected = False
        self._request_count = 0
        self._last_request_time = 0.0

    def connect(self) -> bool:
        """Connects to TWS or IB Gateway."""
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
        except Exception as exc:
            print(f"[ERROR] Failed to connect to IB: {exc}")
            return False

    def disconnect(self) -> None:
        """Disconnects from IB when a live session is open."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            print("[INFO] Disconnected from IB")

    def _wait_for_pacing(self) -> None:
        """Sleeps as needed to respect the configured historical pacing policy."""
        elapsed = time.time() - self._last_request_time
        delay = self.settings.get_ib_request_delay()

        if elapsed < delay:
            time.sleep(delay - elapsed)

        self._last_request_time = time.time()
        self._request_count += 1

    def fetch_chunk(
        self,
        contract: Future,
        end_date: datetime,
        timeframe: Timeframe,
        duration: str = "1 W",
    ) -> pd.DataFrame:
        """
        Fetches one historical chunk directly from IB.

        Args:
            contract: Qualified IB futures contract.
            end_date: Chunk end timestamp.
            timeframe: Target timeframe enum.
            duration: IB historical duration string.
        """
        self._wait_for_pacing()
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting=timeframe.ib_bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
        )
        if not bars:
            return pd.DataFrame()

        df = util.df(bars)
        if df.empty:
            return df

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df.sort_index()

    def test_connection(self) -> bool:
        """Tests connectivity and prints the visible account list when successful."""
        if self.connect():
            print("[SUCCESS] Connected to IB")
            print(f"[INFO] Account: {self.ib.managedAccounts()}")
            self.disconnect()
            return True
        return False


__all__ = ["IBFetcher", "Timeframe"]
