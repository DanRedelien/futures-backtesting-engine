"""
Interactive Brokers fetcher timeframe definitions.
"""

from __future__ import annotations

from enum import Enum


class Timeframe(Enum):
    """Supported timeframes for data fetching."""

    M1 = ("1m", "1 min")
    M5 = ("5m", "5 mins")
    M30 = ("30m", "30 mins")
    H1 = ("1h", "1 hour")

    def __init__(self, file_suffix: str, ib_bar_size: str):
        self.file_suffix = file_suffix
        self.ib_bar_size = ib_bar_size
