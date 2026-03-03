"""
Event-driven single-asset backtest engine.

No look-ahead bias contract:
  1. Load data for the primary symbol.
  2. Iterate bar-by-bar.
  3. Strategy sees bar[t].
  4. Any returned orders execute at open[t+1] (next bar).
  5. Risk checks run *after* execution, before strategy logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Type

import pandas as pd
import numpy as np

from .execution import ExecutionHandler, Fill, Order, Trade
from .analytics import PerformanceMetrics
from .settings import BacktestSettings, get_settings
from .visualizer import Visualizer
from src.data.data_lake import DataLake
from src.data.bar_builder import BarBuilder

class FastBar:
    __slots__ = ['name', 'open', 'high', 'low', 'close', 'volume', '_dict']
    def __init__(self, name, o, h, l, c, v):
        self.name = name
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self._dict = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}

    def __getitem__(self, key):
        return self._dict[key]

    def get(self, key, default=None):
        return self._dict.get(key, default)


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio
# ═══════════════════════════════════════════════════════════════════════════════


class Portfolio:
    """
    Tracks cash, open positions, and portfolio value history.

    Methodology:
        Cash accounting uses full notional: buying a contract deducts
        price * qty * multiplier from cash.  This is conservative (no
        margin leverage) and makes capital usage transparent.
    """

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.current_cash: float = initial_capital
        self.positions: Dict[str, float] = {}   # Symbol → signed quantity
        self.holdings_value: float = 0.0
        self.total_value: float = initial_capital
        self.history: List[Dict] = []

    def update(self, fill: Optional[Fill], current_prices: Dict[str, float]) -> None:
        """
        Updates cash, positions, and total portfolio value.

        Args:
            fill: Newly executed fill; None for a mark-to-market only update.
            current_prices: Latest close prices keyed by symbol.
        """
        if fill is not None:
            symbol = fill.order.symbol
            spec = get_settings().get_instrument_spec(symbol)
            multiplier = spec["multiplier"]
            notional = fill.fill_price * fill.order.quantity * multiplier

            if fill.order.side == "BUY":
                self.current_cash -= notional + fill.commission
                self.positions[symbol] = self.positions.get(symbol, 0.0) + fill.order.quantity
            else:  # SELL
                self.current_cash += notional - fill.commission
                self.positions[symbol] = self.positions.get(symbol, 0.0) - fill.order.quantity

        # Mark-to-market open positions
        self.holdings_value = 0.0
        for sym, qty in self.positions.items():
            if qty != 0 and sym in current_prices:
                spec = get_settings().get_instrument_spec(sym)
                self.holdings_value += qty * current_prices[sym] * spec["multiplier"]

        self.total_value = self.current_cash + self.holdings_value

    def record_snapshot(self, timestamp: datetime) -> None:
        """Appends the current portfolio state to the history log."""
        self.history.append(
            {
                "timestamp": timestamp,
                "cash": self.current_cash,
                "holdings": self.holdings_value,
                "total_value": self.total_value,
            }
        )

    def get_history_df(self) -> pd.DataFrame:
        """Returns portfolio history as a DataFrame indexed by timestamp."""
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        df.set_index("timestamp", inplace=True)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ═══════════════════════════════════════════════════════════════════════════════


class BacktestEngine:
    """
    Bar-by-bar event loop: load → iterate → execute → risk-check → strategy.

    Supports any strategy implementing BaseStrategy.  Pairs-specific logic
    has been fully removed; the engine is single-asset by design.
    """

    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        settings: Optional[BacktestSettings] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.execution = ExecutionHandler(self.settings)
        self.analytics = PerformanceMetrics(self.settings.risk_free_rate)
        self.visualizer = Visualizer()
        self.data_lake = DataLake()

        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = Portfolio(self.settings.initial_capital)
        self.strategy = None
        
        # Inject pre-sliced data if provided (e.g., from WFO)
        self.data: pd.DataFrame = data if data is not None else pd.DataFrame()

        # Daily risk state
        self._daily_pnl: float = 0.0
        self._last_date = None
        self._equity_at_day_start: float = self.settings.initial_capital
        self._peak_equity: float = self.settings.initial_capital
        self.trading_halted_today: bool = False
        self.trading_halted_permanently: bool = False

    # ── Risk management ────────────────────────────────────────────────────────

    def _check_risk_limits(self, date) -> None:
        """
        Evaluates daily loss, max drawdown, and account floor limits.

        Methodology:
            Checks are performed once per day boundary and after each bar.
            If limits are breached, the engine sets halt flags which
            cause the loop to skip strategy logic and force liquidation.

        Args:
            date: Current bar's calendar date (datetime.date).
        """
        if self.trading_halted_permanently:
            return

        # Reset daily tracking on new day
        if self._last_date != date:
            self._last_date = date
            self._equity_at_day_start = self.portfolio.total_value
            self.trading_halted_today = False

        daily_pnl = self.portfolio.total_value - self._equity_at_day_start

        if self.settings.max_daily_loss is not None:
            if daily_pnl < -self.settings.max_daily_loss:
                if not self.trading_halted_today:
                    print(
                        f"[Risk] Daily loss limit hit "
                        f"({daily_pnl:.0f} < -{self.settings.max_daily_loss}). "
                        f"Halting today."
                    )
                    self.trading_halted_today = True

        if self.settings.max_drawdown_pct is not None:
            drawdown_pct = (
                (self._peak_equity - self.portfolio.total_value) / self._peak_equity
                if self._peak_equity > 0
                else 0.0
            )
            if drawdown_pct > self.settings.max_drawdown_pct:
                print(
                    f"[Risk] Max drawdown hit "
                    f"({drawdown_pct:.1%} > {self.settings.max_drawdown_pct:.1%}). "
                    f"Permanent halt."
                )
                self.trading_halted_permanently = True

        if self.settings.max_account_floor is not None:
            if self.portfolio.total_value <= self.settings.max_account_floor:
                print(
                    f"[Risk] Account floor hit ({self.portfolio.total_value:.0f}). "
                    f"Permanent halt."
                )
                self.trading_halted_permanently = True

        self._peak_equity = max(self._peak_equity, self.portfolio.total_value)

    def _liquidate_all(self, timestamp, reason: str = "RISK_LIQ") -> List[Order]:
        """
        Generates market orders to flatten all open positions.

        Args:
            timestamp: Current bar's timestamp for order tagging.
            reason: Exit reason tag.

        Returns:
            List of liquidation Orders.
        """
        orders = []
        for sym, qty in self.portfolio.positions.items():
            if qty != 0:
                orders.append(
                    Order(
                        symbol=sym,
                        quantity=abs(qty),
                        side="SELL" if qty > 0 else "BUY",
                        order_type="MARKET",
                        reason=reason,
                        timestamp=timestamp,
                    )
                )
        # Reset any strategy invested state if it exposes one
        if self.strategy is not None and hasattr(self.strategy, "_invested"):
            self.strategy._invested = False
            self.strategy._position_side = None
        return orders

    # ── Main event loop ────────────────────────────────────────────────────────

    def run(
        self,
        strategy_class: Type,
        step_callback=None,
    ) -> None:
        """
        Runs the full bar-by-bar backtest.

        Args:
            strategy_class: Any class implementing BaseStrategy.
            step_callback: Optional callable(engine, date, step, total) for
                           WFO intermediate reporting / pruning.
        """
        print("[Engine] Initialising backtest...")

        symbol = self.settings.default_symbol
        timeframe = self.settings.low_interval

        # Load data only if not injected via __init__
        if self.data.empty:
            print(f"[Engine] Loading {symbol} @ {timeframe}...")
            data = self.data_lake.load(
                symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
            )

            if data.empty and timeframe != "1h":
                print("[Engine] No data found; falling back to 1h.")
                data = self.data_lake.load(
                    symbol, timeframe="1h",
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

            if data.empty:
                print("[Engine] No data found. Aborting.")
                return

            # Optional bar-type conversion (volume / range / heikin-ashi)
            bar_type = self.settings.bar_type
            bar_size = self.settings.bar_size
            if bar_type != "time":
                spec = self.settings.get_instrument_spec(symbol)
                data = BarBuilder.build(data, bar_type, bar_size, spec["tick_size"])
                print(f"[Engine] Converted to {bar_type.upper()} bars: {len(data):,} bars")

            self.data = data
        else:
            # Data was injected
            data = self.data
        print(
            f"[Engine] {len(data):,} bars loaded "
            f"({data.index[0].date()} -> {data.index[-1].date()})."
        )
        
        # Instantiate strategy (triggers indicator pre-computation in __init__)
        self.strategy = strategy_class(self)
        if hasattr(self.strategy, "on_start"):
            self.strategy.on_start()

        # Reset daily risk state
        self._equity_at_day_start = self.settings.initial_capital
        self._peak_equity = self.settings.initial_capital
        self._last_date = None

        pending_orders: List[Order] = []
        print("[Engine] Starting event loop...")

        # Pre-extract all data to Numpy arrays for ~70x speedup over iloc
        timestamps = data.index
        dates = [ts.date() for ts in data.index]
        
        # We assume standard OHLCV columns exist
        opens = data["open"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        volumes = data["volume"].to_numpy() if "volume" in data else np.zeros(len(data))
        
        data_len = len(data)

        for i in range(data_len):
            # Break only if we have no pending orders (like liquidation orders)
            if self.trading_halted_permanently and not pending_orders:
                break

            timestamp = timestamps[i]
            current_date = dates[i]
            
            # FastBar guarantees compatibility with strategy.on_bar(bar)
            # while avoiding pd.Series overhead.
            bar = FastBar(
                name=timestamp,
                o=opens[i],
                h=highs[i],
                l=lows[i],
                c=closes[i],
                v=volumes[i]
            )
            
            c_close = closes[i]
            current_prices = {symbol: c_close}

            # A. Execute pending orders at open of this bar
            risk_orders = [o for o in pending_orders if "RISK" in o.reason]
            normal_orders = [
                o for o in pending_orders if "RISK" not in o.reason
            ]
            orders_to_execute = risk_orders
            if not self.trading_halted_today:
                orders_to_execute += normal_orders

            for order in orders_to_execute:
                fill = self.execution.execute_order(order, bar)
                if fill:
                    self.portfolio.update(fill, current_prices)

            pending_orders = []

            # B. Mark-to-market + risk checks
            self.portfolio.update(None, current_prices)
            self._check_risk_limits(current_date)

            # C. WFO pruning hook
            if step_callback:
                step_callback(self, current_date, i, data_len)

            # D. Halt handling: liquidate and skip strategy
            if self.trading_halted_today or self.trading_halted_permanently:
                liq = self._liquidate_all(timestamp)
                pending_orders.extend(liq)
                self.portfolio.record_snapshot(timestamp)
                continue

            # E. Strategy logic (signal at close of bar t → order fills at open of bar t+1)
            new_orders = self.strategy.on_bar(bar)
            if new_orders:
                pending_orders.extend(new_orders)

            # F. End-of-day forced close (if enabled)
            is_last_bar = i == data_len - 1
            is_eod = is_last_bar or dates[i + 1] != current_date

            if is_eod and pending_orders:
                for order in pending_orders:
                    fill = self.execution.execute_order(order, bar, execute_at_close=True)
                    if fill:
                        self.portfolio.update(fill, current_prices)
                pending_orders = []

            # Force-close open positions at EOD close time if configured
            eod_close = self.settings.eod_close_time
            if is_eod and eod_close:
                liq = self._liquidate_all(timestamp, reason="EOD_CLOSE")
                for order in liq:
                    fill = self.execution.execute_order(order, bar, execute_at_close=True)
                    if fill:
                        self.portfolio.update(fill, current_prices)

            self.portfolio.record_snapshot(timestamp)

        print("[Engine] Backtest complete.")

    # ── Results ────────────────────────────────────────────────────────────────

    def show_results(self) -> None:
        """
        Computes performance metrics and renders the dashboard.

        Benchmark: Buy-and-hold of the primary symbol (self.data close prices).
        """
        history = self.portfolio.get_history_df()
        if history.empty:
            print("[Engine] No portfolio history to display.")
            return

        trades = self.execution.trades
        metrics = self.analytics.calculate_metrics(history, trades)
        self.analytics.print_full_report(metrics, trades)

        benchmark = self.data["close"] if not self.data.empty else None
        self.visualizer.plot_dashboard(history, trades, benchmark=benchmark)
