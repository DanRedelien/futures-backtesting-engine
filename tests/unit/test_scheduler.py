"""tests/unit/test_scheduler.py — DailyScheduler, IntrabarScheduler, and WeeklyScheduler unit tests."""

import pytest
import pandas as pd

from src.backtest_engine.portfolio_layer.scheduling.scheduler import (
    IntrabarScheduler,
    DailyScheduler,
    WeeklyScheduler,
    make_scheduler,
)


class TestIntrabarScheduler:
    def test_fires_every_bar(self):
        s = IntrabarScheduler()
        for ts in pd.date_range("2023-01-02", periods=10, freq="30min"):
            assert s.should_rebalance(ts) is True


class TestDailyScheduler:
    def test_fires_once_per_day(self):
        s = DailyScheduler()
        day1 = pd.date_range("2023-01-02 09:30", periods=4, freq="30min")
        day2 = pd.date_range("2023-01-03 09:30", periods=4, freq="30min")
        timestamps = list(day1) + list(day2)

        results = [s.should_rebalance(ts) for ts in timestamps]
        fired_count = sum(results)
        assert fired_count == 2, (
            f"Expected 2 fires (one per day), got {fired_count}"
        )

    def test_first_bar_each_day_fires(self):
        s = DailyScheduler()
        day1_bar1 = pd.Timestamp("2023-01-02 09:30")
        day1_bar2 = pd.Timestamp("2023-01-02 10:00")
        day2_bar1 = pd.Timestamp("2023-01-03 09:30")

        assert s.should_rebalance(day1_bar1) is True
        assert s.should_rebalance(day1_bar2) is False
        assert s.should_rebalance(day2_bar1) is True

    def test_reset_allows_refiring(self):
        s = DailyScheduler()
        ts = pd.Timestamp("2023-01-02 09:30")
        assert s.should_rebalance(ts) is True
        assert s.should_rebalance(ts) is False
        s.reset()
        assert s.should_rebalance(ts) is True


class TestWeeklyScheduler:
    def test_fires_once_per_iso_week(self):
        s = WeeklyScheduler()
        # 2023-01-02 (Mon, week 1) and 2023-01-09 (Mon, week 2)
        week1 = pd.date_range("2023-01-02 09:30", periods=5, freq="30min")
        week2 = pd.date_range("2023-01-09 09:30", periods=5, freq="30min")
        timestamps = list(week1) + list(week2)

        results = [s.should_rebalance(ts) for ts in timestamps]
        fired_count = sum(results)
        assert fired_count == 2, (
            f"Expected 2 fires (one per week), got {fired_count}"
        )

    def test_first_bar_of_each_week_fires(self):
        s = WeeklyScheduler()
        week1_bar1 = pd.Timestamp("2023-01-02 09:30")
        week1_bar2 = pd.Timestamp("2023-01-02 10:00")
        week2_bar1 = pd.Timestamp("2023-01-09 09:30")

        assert s.should_rebalance(week1_bar1) is True
        assert s.should_rebalance(week1_bar2) is False
        assert s.should_rebalance(week2_bar1) is True

    def test_reset_allows_refiring(self):
        s = WeeklyScheduler()
        ts = pd.Timestamp("2023-01-02 09:30")
        assert s.should_rebalance(ts) is True
        assert s.should_rebalance(ts) is False
        s.reset()
        assert s.should_rebalance(ts) is True

    def test_year_boundary_handled_correctly(self):
        """Ensure year-week key includes year to avoid collisions at year boundary."""
        s = WeeklyScheduler()
        # Last week of 2022 and first week of 2023 differ by year
        last_week_2022 = pd.Timestamp("2022-12-26 09:30")
        first_week_2023 = pd.Timestamp("2023-01-02 09:30")

        assert s.should_rebalance(last_week_2022) is True
        assert s.should_rebalance(first_week_2023) is True  # new year-week key


class TestMakeScheduler:
    def test_intrabar_factory(self):
        s = make_scheduler("intrabar")
        assert isinstance(s, IntrabarScheduler)

    def test_daily_factory(self):
        s = make_scheduler("daily")
        assert isinstance(s, DailyScheduler)

    def test_weekly_factory(self):
        s = make_scheduler("weekly")
        assert isinstance(s, WeeklyScheduler)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown rebalance_frequency"):
            make_scheduler("quarterly")
