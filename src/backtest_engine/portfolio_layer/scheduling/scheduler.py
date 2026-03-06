"""
src/backtest_engine/portfolio_layer/scheduling/scheduler.py

Rebalance gate: decides whether the Allocator should compute new targets on a
given bar.  Three concrete implementations are provided.

Methodology:
    IntrabarScheduler: always returns True (rebalance on every bar).
    DailyScheduler: returns True only once per calendar day — on the first bar
        the engine processes for that date.  All subsequent intraday bars for
        the same date return False (hold).
    WeeklyScheduler: returns True only once per ISO week — on the first bar
        the engine processes for that week.  All subsequent bars in the same
        week return False (hold).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseScheduler(ABC):
    """Abstract rebalance gate."""

    @abstractmethod
    def should_rebalance(self, timestamp: Any) -> bool:
        """
        Returns True if the Allocator should run on this bar.

        Args:
            timestamp: Current bar timestamp (pd.Timestamp or datetime).

        Returns:
            True to rebalance, False to hold current targets.
        """

    def reset(self) -> None:
        """Optional reset hook called before the bar loop starts."""


class IntrabarScheduler(BaseScheduler):
    """
    Rebalances on every bar.

    Use this when strategies continuously adjust positions intraday based on
    each bar's signal.
    """

    def should_rebalance(self, timestamp: Any) -> bool:
        return True


class DailyScheduler(BaseScheduler):
    """
    Rebalances once per calendar day at the first available bar.

    Use this when strategies produce a morning signal that should be held
    constant throughout the trading day.  The scheduler tracks the last date
    it fired and skips all subsequent bars with the same date.
    """

    def __init__(self) -> None:
        self._last_date: Optional[str] = None

    def should_rebalance(self, timestamp: Any) -> bool:
        """
        Returns True only for the first bar of each new calendar day.

        Methodology:
            Converts timestamp to a YYYY-MM-DD string for comparison.
            On the first call of a new date, records it and returns True.
            All subsequent bars on the same date return False.
        """
        date_str = str(timestamp)[:10]  # "YYYY-MM-DD" prefix
        if date_str != self._last_date:
            self._last_date = date_str
            return True
        return False

    def reset(self) -> None:
        self._last_date = None


class WeeklyScheduler(BaseScheduler):
    """
    Rebalances once per ISO week at the first available bar.

    Use this when position sizing should be recalculated weekly rather than
    daily, reducing turnover while still adapting to changing volatility regimes.
    The ISO week number (1–53) combined with the year is used as the key, so
    the year boundary is handled correctly.
    """

    def __init__(self) -> None:
        self._last_week: Optional[str] = None

    def should_rebalance(self, timestamp: Any) -> bool:
        """
        Returns True only for the first bar of each new ISO week.

        Methodology:
            Uses timestamp.isocalendar() to get (year, week, day).
            Fires on the first bar whose (year, week) pair differs from the
            last recorded pair.
        """
        iso = timestamp.isocalendar()
        week_str = f"{iso[0]}-W{iso[1]:02d}"
        if week_str != self._last_week:
            self._last_week = week_str
            return True
        return False

    def reset(self) -> None:
        self._last_week = None


def make_scheduler(frequency: str) -> BaseScheduler:
    """
    Factory: converts a rebalance_frequency string to the correct scheduler.

    Args:
        frequency: 'intrabar', 'daily', or 'weekly'.

    Returns:
        A concrete BaseScheduler instance.

    Raises:
        ValueError: If the frequency string is not recognised.
    """
    mapping = {
        "intrabar": IntrabarScheduler,
        "daily":    DailyScheduler,
        "weekly":   WeeklyScheduler,
    }
    cls = mapping.get(frequency)
    if cls is None:
        raise ValueError(
            f"Unknown rebalance_frequency '{frequency}'. "
            f"Valid options: {list(mapping.keys())}"
        )
    return cls()
