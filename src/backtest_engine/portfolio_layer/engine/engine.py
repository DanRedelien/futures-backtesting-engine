"""
src/backtest_engine/portfolio_layer/engine/engine.py

PortfolioBacktestEngine — multi-strategy, multi-asset event loop.

No-lookahead contract (same as BacktestEngine):
    1. Data loaded for all (slot, symbol) pairs.
    2. Iterate bar-by-bar on the union timeline.
    3. Strategies see bar[t] close.
    4. Signals from bar[t] -> orders fill at open[t+1].
    5. MtM and equity recomputed after fills.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from src.backtest_engine.settings import BacktestSettings
from src.backtest_engine.execution import Order
from src.backtest_engine.spread_model import compute_spread_ticks
from src.data.data_lake import DataLake

from ..domain.contracts import PortfolioConfig
from ..domain.signals import StrategySignal, TargetPosition
from ..execution.portfolio_book import PortfolioBook
from ..allocation.allocator import Allocator
from ..execution.strategy_runner import StrategyRunner
from ..reporting.results import save_portfolio_results
from ..scheduling.scheduler import make_scheduler


class PortfolioBacktestEngine:
    """
    Event-driven portfolio backtest engine.

    Methodology:
        Builds a unified timeline (union of all loaded bar timestamps).
        At each bar:
          A. Fill pending orders at open[t] (strict no-lookahead).
             Orders are CARRIED FORWARD through gap bars — never silently dropped.
          B. Mark-to-market all positions (using last-known price on gap bars).
          C. Risk-limit check (daily loss, drawdown, account floor).
          D. Scheduler decides whether to rebalance this bar.
          E. If rebalancing and not halted, call StrategyRunner.collect_signals().
          F. Call Allocator.compute_targets() -> List[TargetPosition].
          G. Compute order deltas vs current positions.
          H. Queue delta orders for execution at open[t+1].
          I. EOD forced close (weekend-safe).
          J. Record portfolio snapshot.

        Shared-capital invariant:
            All positions share a single cash pool.  No slot can run more
            capital than its weight allocation of total_equity.

        Kill-switches (from BacktestSettings):
            max_daily_loss      — daily PnL drawdown halt (temporary, resets next day).
            max_drawdown_pct    — peak-to-trough drawdown permanent halt.
            max_account_floor   — absolute equity level permanent halt.

        EOD weekend guard:
            A Friday-to-Monday date transition is treated as an EOD boundary,
            preventing positions from being carried into the weekend.
    """

    def __init__(
        self,
        config: PortfolioConfig,
        settings: Optional[BacktestSettings] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Args:
            config: Validated PortfolioConfig defining slots and weights.
            settings: Optional BacktestSettings override.
            start_date: Optional backtest start filter.
            end_date: Optional backtest end filter.
        """
        config.validate()
        self.config     = config
        if settings is None:
            raise ValueError("BacktestSettings must be provided via Dependency Injection.")
        self.settings   = settings
        self.start_date = start_date
        self.end_date   = end_date

        self.book      = PortfolioBook(config.initial_capital)
        self.allocator = Allocator(config)
        self.data_lake = DataLake(self.settings)
        self.scheduler = make_scheduler(config.rebalance_frequency)

        from src.backtest_engine.execution import ExecutionHandler

        # One ExecutionHandler per slot — commission/slippage from settings.py
        self._execution_handlers: Dict[int, Any] = {
            i: ExecutionHandler(self.settings) for i in range(len(config.slots))
        }
        # Live reference to each handler's trade list
        self._slot_trades: Dict[int, List[Any]] = {
            i: self._execution_handlers[i].trades for i in range(len(config.slots))
        }

        # Loaded data: (slot_id, symbol) -> DataFrame
        self._data_map: Dict[Tuple[int, str], pd.DataFrame] = {}
        # Symbol -> last-available bar lookup: updated as bars are visited
        self._last_bars: Dict[Tuple[int, str], pd.Series] = {}

        # Computed once from real data after _load_data(); passed to Allocator.
        self._bars_per_year: int = 3276  # Conservative default (30m, 252d x 13 bars)

        # Daily risk state (mirrors BacktestEngine._check_risk_limits)
        self._last_date: Optional[date] = None
        self._equity_at_day_start: float = config.initial_capital
        self._peak_equity: float = config.initial_capital
        self.trading_halted_today: bool = False
        self.trading_halted_permanently: bool = False

    # ── Risk management ────────────────────────────────────────────────────────

    def _check_risk_limits(self, current_date: date) -> None:
        """
        Evaluates daily loss, max drawdown, and account floor limits.

        Methodology:
            Checks are performed once per day boundary and after each bar.
            Breached limits set halt flags which cause the loop to skip
            strategy logic and force liquidation.

        Args:
            current_date: Current bar's calendar date.
        """
        if self.trading_halted_permanently:
            return

        # Reset daily tracking on new day
        if self._last_date != current_date:
            self._last_date = current_date
            self._equity_at_day_start = self.book.total_equity
            self.trading_halted_today = False

        daily_pnl = self.book.total_equity - self._equity_at_day_start

        if self.settings.max_daily_loss is not None:
            if daily_pnl < -self.settings.max_daily_loss:
                if not self.trading_halted_today:
                    print(
                        f"[Portfolio Risk] Daily loss limit hit "
                        f"({daily_pnl:.0f} < -{self.settings.max_daily_loss}). "
                        f"Halting today."
                    )
                    self.trading_halted_today = True

        if self.settings.max_drawdown_pct is not None:
            drawdown_pct = (
                (self._peak_equity - self.book.total_equity) / self._peak_equity
                if self._peak_equity > 0 else 0.0
            )
            if drawdown_pct > self.settings.max_drawdown_pct:
                print(
                    f"[Portfolio Risk] Max drawdown hit "
                    f"({drawdown_pct:.1%} > {self.settings.max_drawdown_pct:.1%}). "
                    f"Permanent halt."
                )
                self.trading_halted_permanently = True

        if self.settings.max_account_floor is not None:
            if self.book.total_equity <= self.settings.max_account_floor:
                print(
                    f"[Portfolio Risk] Account floor hit "
                    f"({self.book.total_equity:.0f}). Permanent halt."
                )
                self.trading_halted_permanently = True

        self._peak_equity = max(self._peak_equity, self.book.total_equity)

    def _liquidate_slot(
        self,
        slot_id: int,
        symbol: str,
        qty: float,
        bar: "pd.Series",
        execute_at_close: bool = False,
        reason: str = "RISK_LIQ",
        effective_spread_ticks: Optional[int] = None,
    ) -> None:
        """
        Liquidates a single (slot, symbol) position at the given bar.

        Args:
            slot_id: Slot index.
            symbol: Instrument ticker.
            qty: Current signed quantity (non-zero guaranteed by caller).
            bar: Current bar pd.Series (for fill simulation).
            execute_at_close: Fill at close price if True, else open.
            reason: Tag written to the order.
            effective_spread_ticks: Pre-computed tick count from the spread model.
        """
        side = "SELL" if qty > 0 else "BUY"
        self._execute_order(
            slot_id, symbol, side, abs(qty), bar,
            execute_at_close=execute_at_close,
            reason=reason,
            effective_spread_ticks=effective_spread_ticks,
        )

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """
        Loads OHLCV data for every (slot_id, symbol) pair.

        Also computes bars_per_year from the actual data span so the
        Allocator's vol annualisation is derived from real bar frequency
        rather than a hardcoded constant.

        Raises:
            RuntimeError: If any required data is missing from the data lake.
        """
        for slot_id, slot in enumerate(self.config.slots):
            for symbol in slot.symbols:
                df = self.data_lake.load(
                    symbol,
                    timeframe=slot.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )
                if df.empty:
                    raise RuntimeError(
                        f"[Portfolio] No data for {symbol} @ {slot.timeframe}. "
                        f"Run: python run.py --download {symbol}"
                    )
                self._data_map[(slot_id, symbol)] = df

        # Estimate bars_per_year from the first available DataFrame.
        # Formula: total_bars / total_calendar_years (using date span).
        first_df = next(iter(self._data_map.values()))
        if len(first_df) >= 2:
            span_days = (first_df.index[-1] - first_df.index[0]).total_seconds() / 86400
            span_years = max(span_days / 365.25, 1e-6)
            self._bars_per_year = max(1, round(len(first_df) / span_years))
        print(f"[Portfolio] bars_per_year estimate: {self._bars_per_year:,}")

    # ── Spread helpers ─────────────────────────────────────────────────────────

    def _effective_spread_ticks(
        self,
        slot_id: int,
        symbol: str,
        price_history: Dict[Tuple[int, str], "pd.Series"],
        up_to_ts: Any,
    ) -> Optional[int]:
        """
        Computes the deterministic spread tick count for a symbol at the current bar.

        Methodology:
            For static mode, returns None so ExecutionHandler reads
            settings.spread_ticks directly.
            For adaptive_volatility mode, slices close history strictly before
            up_to_ts (no-lookahead: execution happens at open[ts], so close[ts]
            is not yet observable).  Mirrors the single engine which uses
            closes[:bar_index] to exclude the current bar.

        Args:
            slot_id: Slot index for strategy-local market data context.
            symbol: Instrument ticker.
            price_history: Mapping of (slot_id, symbol) to full close-price Series.
            up_to_ts: Current bar timestamp.  Close at this timestamp is excluded.

        Returns:
            Integer tick count for adaptive mode, or None for static mode.
        """
        if self.settings.spread_mode != "adaptive_volatility":
            return None

        series = price_history.get((slot_id, symbol))
        if series is None:
            return self.settings.spread_ticks

        # Exclude the close at up_to_ts: that bar's close is not yet available
        # when the order fills at open[ts].  This mirrors closes[:bar_index] in
        # the single engine which is a Python slice (exclusive upper bound).
        closes = series.loc[:up_to_ts]
        if up_to_ts in series.index:
            closes = closes.iloc[:-1]

        return compute_spread_ticks(
            mode=self.settings.spread_mode,
            base_ticks=self.settings.spread_ticks,
            closes=closes,
            vol_step_pct=self.settings.spread_volatility_step_pct,
            step_multiplier=self.settings.spread_step_multiplier,
            vol_lookback=self.settings.spread_vol_lookback,
            vol_baseline_lookback=self.settings.spread_vol_baseline_lookback,
        )

    # ── Execution simulation ───────────────────────────────────────────────────

    def _execute_order(
        self,
        slot_id: int,
        symbol: str,
        side: str,
        quantity: float,
        bar: "pd.Series",
        execute_at_close: bool = False,
        reason: str = "PORTFOLIO_SYNC",
        effective_spread_ticks: Optional[int] = None,
    ) -> None:
        """
        Simulates a fill via ExecutionHandler and applies it to PortfolioBook.

        Delegates to the slot's ExecutionHandler to generate a proper Trade
        object (same FIFO matching as the single-asset engine), then writes
        the fill into the shared PortfolioBook.

        Args:
            slot_id: Slot index.
            symbol: Instrument ticker.
            side: 'BUY' or 'SELL'.
            quantity: Absolute quantity to trade.
            bar: Current bar pd.Series for price reference.
            execute_at_close: Fill at close price if True, else open.
            reason: Tag written to the order.
            effective_spread_ticks: Pre-computed tick count from the spread model.
                                    If None, ExecutionHandler reads settings.spread_ticks.
        """
        spec = self.settings.get_instrument_spec(symbol)
        multiplier = spec["multiplier"]

        order = Order(
            symbol=symbol,
            quantity=abs(quantity),
            side=side,
            order_type="MARKET",
            reason=reason,
            timestamp=bar.name if hasattr(bar, "name") else None,
        )

        handler = self._execution_handlers[slot_id]
        fill = handler.execute_order(
            order, bar,
            execute_at_close=execute_at_close,
            effective_spread_ticks=effective_spread_ticks,
        )

        if fill:
            signed_qty = quantity if side == "BUY" else -quantity
            self.book.apply_fill(
                slot_id=slot_id,
                symbol=symbol,
                fill_price=fill.fill_price,
                quantity=signed_qty,
                commission=fill.commission,
                multiplier=multiplier,
                timestamp=fill.timestamp,
            )

    # ── Order delta computation ────────────────────────────────────────────────

    def _compute_orders(
        self,
        targets: List[TargetPosition],
        pending_orders: List[Tuple[int, str, str, float, str]],
    ) -> List[Tuple[int, str, str, float, str]]:
        """
        Computes order deltas from target positions vs current holdings + pending orders.

        Returns:
            List of (slot_id, symbol, side, abs_quantity, reason) tuples.
        """
        pending_qty: Dict[Tuple[int, str], float] = {}
        for (sid, sym, side, qty, reason) in pending_orders:
            signed_qty = qty if side == "BUY" else -qty
            pending_qty[(sid, sym)] = pending_qty.get((sid, sym), 0.0) + signed_qty

        orders = []
        for target in targets:
            current = self.book.get_position(target.slot_id, target.symbol)
            pending = pending_qty.get((target.slot_id, target.symbol), 0.0)
            delta = target.target_qty - (current + pending)

            if abs(delta) < 0.5:   # ignore sub-contract deltas
                continue

            side = "BUY" if delta > 0 else "SELL"
            orders.append((target.slot_id, target.symbol, side, abs(delta), target.reason))

        return orders

    # ── EOD helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_eod_boundary(current_date: date, next_date: Optional[date]) -> bool:
        """
        Returns True if the current bar is the last bar of the trading day.

        Weekend guard: a Friday -> Monday transition (weekday 4 -> 0) is also
        treated as an EOD boundary so positions are never carried into the
        weekend across the gap.

        Args:
            current_date: Calendar date of the current bar.
            next_date: Calendar date of the next bar (None at end of data).

        Returns:
            True if this is the effective end-of-day bar.
        """
        if next_date is None:
            return True
        if next_date != current_date:
            # Any date change counts, including Friday -> Monday gap.
            return True
        return False

    def _liquidate_all_eod(
        self,
        ts: Any,
        price_history: Dict[Tuple[int, str], "pd.Series"],
    ) -> None:
        """
        Force-closes all open positions using each symbol's last-available bar.

        Methodology:
            The union timeline may have a timestamp ts that is NOT present in
            every symbol's DataFrame (gap bars).  Using ts directly would cause
            the price-lookup to silently skip the symbol, leaving the position
            open overnight.  Instead this helper uses self._last_bars — a dict
            kept current by the main loop — which always holds the most recent
            bar that was actually available for each (slot, symbol) pair.

        Args:
            ts: Current union-timeline timestamp (used only as order tag).
            price_history: (slot_id, symbol) -> close price Series for spread computation.
        """
        for (slot_id, symbol), qty in list(self.book.positions.items()):
            if qty == 0:
                continue
            bar = self._last_bars.get((slot_id, symbol))
            if bar is None:
                continue
            side = "SELL" if qty > 0 else "BUY"
            spread_ticks = self._effective_spread_ticks(slot_id, symbol, price_history, ts)
            self._execute_order(
                slot_id, symbol, side, abs(qty), bar,
                execute_at_close=True, reason="EOD_CLOSE",
                effective_spread_ticks=spread_ticks,
            )

    # ── Main event loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Runs the portfolio backtest bar-by-bar on the union timeline.

        No-lookahead enforcement:
            Signals generated at close[t] -> orders filled at open[t+1].
        Gap-bar safety:
            Pending orders for symbols with no bar at ts are carried forward
            to the next available bar — never silently discarded.
        """
        if not self._data_map:
            print("[Portfolio] Loading data...")
            self._load_data()

        specs = {
            symbol: self.settings.get_instrument_spec(symbol)
            for slot in self.config.slots
            for symbol in slot.symbols
        }

        # Build slot-local close series. The same structure is used by both:
        # 1) spread estimation at execution time and 2) per-slot sizing volatility.
        slot_price_history: Dict[Tuple[int, str], pd.Series] = {
            (sid, symbol): df["close"]
            for (sid, symbol), df in self._data_map.items()
        }

        all_timestamps = pd.DatetimeIndex([])
        for df in self._data_map.values():
            all_timestamps = all_timestamps.union(df.index)
        all_timestamps = all_timestamps.sort_values()
        timestamps_arr = all_timestamps.to_pydatetime()
        all_dates      = [ts.date() for ts in timestamps_arr]
        data_len       = len(all_timestamps)

        runner = StrategyRunner(self.config, self._data_map, self.settings)
        self.scheduler.reset()

        pending_orders: List[Tuple[int, str, str, float, str]] = []
        current_targets: Dict[Tuple[int, str], TargetPosition] = {}
        self._last_bars: Dict[Tuple[int, str], pd.Series] = {}

        # Snapshot equity for sizing (updated by scheduler gate)
        self._allocation_equity = self.book.total_equity

        print(f"[Portfolio] Starting event loop: {data_len:,} bars")

        for i, ts in enumerate(all_timestamps):
            # Break early only when permanently halted and no pending orders remain
            if self.trading_halted_permanently and not pending_orders:
                break

            current_date = all_dates[i]
            next_date    = all_dates[i + 1] if i + 1 < data_len else None

            # ── A. Fill pending orders (carry forward through gap bars) ──────────
            # Spread ticks are computed per symbol from history available at ts.
            still_pending: List[Tuple[int, str, str, float, str]] = []
            for (slot_id, symbol, side, qty, reason) in pending_orders:
                df = self._data_map.get((slot_id, symbol))
                if df is None or ts not in df.index:
                    still_pending.append((slot_id, symbol, side, qty, reason))
                    continue
                bar = df.loc[ts]
                self._last_bars[(slot_id, symbol)] = bar  # keep cache current
                spread_ticks = self._effective_spread_ticks(slot_id, symbol, slot_price_history, ts)
                self._execute_order(
                    slot_id, symbol, side, qty, bar,
                    reason=reason,
                    effective_spread_ticks=spread_ticks,
                )
            pending_orders = still_pending

            # ── B. Mark-to-market ────────────────────────────────────────────────
            prices = {
                (sid, symbol): self._data_map[(sid, symbol)].loc[ts, "close"]
                for (sid, symbol), df in self._data_map.items()
                if ts in df.index
            }
            # Update last-bar cache with any symbol that has a bar at ts
            for (sid, symbol), df in self._data_map.items():
                if ts in df.index:
                    self._last_bars[(sid, symbol)] = df.loc[ts]
            self.book.mark_to_market(prices, specs)

            # ── C. Risk-limit check ──────────────────────────────────────────────
            self._check_risk_limits(current_date)

            # ── D. Halt handling: liquidate and skip strategy logic ───────────────
            if self.trading_halted_today or self.trading_halted_permanently:
                for (slot_id, symbol), qty in list(self.book.positions.items()):
                    if qty != 0:
                        df = self._data_map.get((slot_id, symbol))
                        if df is not None and ts in df.index:
                            bar = df.loc[ts]
                            spread_ticks = self._effective_spread_ticks(
                                slot_id, symbol, slot_price_history, ts
                            )
                            self._liquidate_slot(
                                slot_id, symbol, qty, bar,
                                reason="RISK_LIQ",
                                effective_spread_ticks=spread_ticks,
                            )
                self.book.mark_to_market(prices, specs)
                self.book.record_snapshot(ts, instrument_specs=specs)
                continue

            # ── E. Scheduler gate: snapshot equity for sizing ────────────────────
            if self.scheduler.should_rebalance(ts):
                self._allocation_equity = self.book.total_equity

            # ── F. Collect intraday signals ──────────────────────────────────────
            bar_map: Dict[Tuple[int, str], pd.Series] = {
                (sid, sym): df.loc[ts]
                for (sid, sym), df in self._data_map.items()
                if ts in df.index
            }
            signals = runner.collect_signals(bar_map, ts)

            # ── G. Compute target positions via vol-targeting ────────────────────
            if signals:
                # Build a truncated price history up to and including bar[t]
                # so the vol estimate never looks ahead.
                history_to_t: Dict[Tuple[int, str], pd.Series] = {
                    key: series.loc[:ts]
                    for key, series in slot_price_history.items()
                }
                # Allocator.compute_targets expects current_prices keyed by symbol
                # (str), not by (slot_id, symbol) tuple used by mark_to_market.
                # Flatten here; all slots share the same instrument price.
                symbol_prices: Dict[str, float] = {
                    sym: p for (_, sym), p in prices.items()
                }
                new_targets = self.allocator.compute_targets(
                    signals,
                    self._allocation_equity,
                    symbol_prices,
                    specs,
                    history_to_t,
                    bars_per_year=self._bars_per_year,
                )
                for t in new_targets:
                    current_targets[(t.slot_id, t.symbol)] = t

            # Build full target list from maintained state
            target_list = list(current_targets.values())

            # ── H. Compute order deltas -> queue for t+1 ─────────────────────────
            orders = self._compute_orders(target_list, pending_orders)
            pending_orders.extend(orders)

            # ── I. End-of-day forced close (weekend-safe) ─────────────────────────
            is_eod = self._is_eod_boundary(current_date, next_date)

            if is_eod and self.settings.eod_close_time:
                # 1. Execute outstanding pending orders at market-on-close
                still_pending = []
                for (slot_id, symbol, side, qty, reason) in pending_orders:
                    bar = self._last_bars.get((slot_id, symbol))
                    if bar is not None:
                        spread_ticks = self._effective_spread_ticks(
                            slot_id, symbol, slot_price_history, ts
                        )
                        self._execute_order(
                            slot_id, symbol, side, qty, bar,
                            execute_at_close=True, reason=reason,
                            effective_spread_ticks=spread_ticks,
                        )
                    else:
                        still_pending.append((slot_id, symbol, side, qty, reason))
                pending_orders = still_pending

                # 2. Force-liquidate ALL open positions using last-available bar
                #    (not ts — prevents silent skip on gap-bar symbols)
                self._liquidate_all_eod(ts, slot_price_history)

                # 3. Re-mark after all closings
                self.book.mark_to_market(prices, specs)

                # 4. Clear targets so allocator does not reopen at bar[t+1] open
                current_targets.clear()

                # 5. Reset strategy invested-state flags
                for instance in runner._instances.values():
                    if hasattr(instance, "_invested"):
                        instance._invested = False
                    if hasattr(instance, "_position_side"):
                        instance._position_side = None

            # ── J. Snapshot ───────────────────────────────────────────────────────
            self.book.record_snapshot(ts, instrument_specs=specs)

        print("[Portfolio] Backtest complete.")

    # ── Results ────────────────────────────────────────────────────────────────

    def show_results(
        self,
        benchmark: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        manifest_metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Computes metrics, prints the full report, and saves all result artifacts.

        Args:
            benchmark: Optional DataFrame with 'close' column for buy-and-hold comparison.
            output_dir: Optional alternate artifact directory for scenario runs.
            manifest_metadata: Optional extra metadata persisted into manifest.json.
        """
        from src.backtest_engine.analytics import PerformanceMetrics

        history = self.book.get_history_df()
        if history.empty:
            print("[Portfolio] No history to display.")
            return

        all_trades = [t for trades in self._slot_trades.values() for t in trades]

        analytics = PerformanceMetrics(risk_free_rate=self.settings.risk_free_rate)
        metrics   = analytics.calculate_metrics(history, all_trades)
        report    = analytics.get_full_report_str(metrics, all_trades)
        print(report)

        def _optional_int(value: Any) -> Optional[int]:
            """Safely coerces optional numeric config values to int."""
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return int(value)

        def _optional_float(value: Any) -> Optional[float]:
            """Safely coerces optional numeric config values to float."""
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            return float(value)

        save_portfolio_results(
            history=history,
            exposure_df=self.book.get_exposure_df(),
            slot_trades=self._slot_trades,
            report_str=report,
            metrics=metrics,
            slot_names={
                i: f"{slot.strategy_class.__name__}({'_'.join(slot.symbols)}+{slot.timeframe.upper()})"
                for i, slot in enumerate(self.config.slots)
            },
            benchmark=benchmark,
            data_map=self._data_map,
            slot_weights={
                i: slot.weight for i, slot in enumerate(self.config.slots)
            },
            slot_vol_params={
                i: {
                    "regime_window": _optional_int(slot.params.get("vol_regime_window")),
                    "history_window": _optional_int(slot.params.get("vol_history_window")),
                    "vol_min_pct": _optional_float(slot.params.get("vol_min_pct")),
                    "vol_max_pct": _optional_float(slot.params.get("vol_max_pct")),
                }
                for i, slot in enumerate(self.config.slots)
            },
            instrument_specs=self.settings.instrument_specs,
            output_dir=output_dir,
            manifest_metadata=manifest_metadata,
        )
