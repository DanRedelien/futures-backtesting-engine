"""
src/backtest_engine/portfolio_layer/execution/portfolio_book.py

Shared-capital portfolio ledger for multi-asset, multi-strategy backtesting.

Responsibility: Maintains cash, per-(slot, symbol) positions, mark-to-market
valuations, and snapshots.  Enforces the shared-capital invariant:
    total_equity = cash + Σ(MtM of all open positions)

Per-slot PnL is tracked as:
    slot_realized_pnl:   Cumulative closed-trade PnL for this slot.
    slot_unrealized_pnl: Current open MtM PnL for this slot.
    slot_total_pnl:      realized + unrealized (recomputed each bar).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class PortfolioBook:
    """
    Central ledger for the portfolio backtest.

    Methodology:
        Uses full notional cash accounting (no margin/leverage).
        Each position is tracked as a signed quantity keyed by (slot_id, symbol).
        Mark-to-market runs after every bar using the latest close prices.
        The shared-capital invariant (cash + MtM == total_equity) must hold
        at every snapshot.

    The _last_prices cache ensures gap bars (union-timeline holes) do not
    temporarily zero-value open positions.
    """

    def __init__(self, initial_capital: float) -> None:
        """
        Args:
            initial_capital: Starting cash in dollars.
        """
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital

        # (slot_id, symbol) → signed quantity
        self.positions: Dict[Tuple[int, str], float] = {}

        # Average entry price per (slot_id, symbol)
        self.avg_prices: Dict[Tuple[int, str], float] = {}

        self.total_equity: float = initial_capital
        self.holdings_value: float = 0.0

        # Per-slot accounting: cumulative realized PnL (closed trades)
        self.slot_realized_pnl: Dict[int, float] = {}

        # Per-slot unrealized PnL, recomputed on each mark-to-market
        self.slot_unrealized_pnl: Dict[int, float] = {}

        # Snapshot history: list of dicts → converted to DataFrame at end
        self.history: List[Dict[str, Any]] = []

        # Exposure snapshot history: list of dicts → used for get_exposure_df()
        self._exposure_history: List[Dict[str, Any]] = []

        # Last-known close price per symbol (prevents gap-bar zero-valuation)
        self._last_prices: Dict[str, float] = {}

    # ── Position updates ───────────────────────────────────────────────────────

    def apply_fill(
        self,
        slot_id: int,
        symbol: str,
        fill_price: float,
        quantity: float,   # signed: positive = bought, negative = sold
        commission: float,
        multiplier: float,
        timestamp: Any,
    ) -> None:
        """
        Applies an executed fill to cash and position records.

        Also updates slot-level realized PnL when a position is reduced or closed.

        Methodology:
            Cash delta = -(signed_quantity * fill_price * multiplier) - commission
            When reducing a position, realized PnL is computed against avg_price.
            Commission is always debited from realized PnL.

        Args:
            slot_id: Index of the originating StrategySlot.
            symbol: Ticker filled.
            fill_price: Actual execution price (with slippage).
            quantity: Signed contracts filled (+buy / -sell).
            commission: Dollar commission for this fill.
            multiplier: Instrument dollar multiplier.
            timestamp: Bar timestamp for PnL attribution.
        """
        notional = quantity * fill_price * multiplier
        self.cash -= notional + commission

        key = (slot_id, symbol)
        prev_qty = self.positions.get(key, 0.0)
        new_qty = prev_qty + quantity
        self.positions[key] = new_qty

        # ── Realized PnL accounting ────────────────────────────────────────────
        realized = self.slot_realized_pnl.get(slot_id, 0.0)

        if prev_qty != 0.0:
            avg = self.avg_prices.get(key, fill_price)
            # Closed or partially reduced portion
            closed_qty = min(abs(quantity), abs(prev_qty))
            if quantity < 0 and prev_qty > 0:
                # Reducing a long
                realized += closed_qty * (fill_price - avg) * multiplier
            elif quantity > 0 and prev_qty < 0:
                # Covering a short
                realized += closed_qty * (avg - fill_price) * multiplier

        realized -= commission
        self.slot_realized_pnl[slot_id] = realized

        # ── Avg price maintenance ──────────────────────────────────────────────
        if abs(new_qty) < 1e-9:
            self.avg_prices.pop(key, None)
            self.positions.pop(key, None)
        elif (prev_qty > 0) != (new_qty > 0):
            # Direction flip through zero
            self.avg_prices[key] = fill_price
        elif prev_qty == 0:
            self.avg_prices[key] = fill_price
        elif (quantity > 0) == (prev_qty > 0):
            old_price = self.avg_prices.get(key, fill_price)
            self.avg_prices[key] = (
                (abs(prev_qty) * old_price + abs(quantity) * fill_price)
                / abs(new_qty)
            )

    # ── Mark-to-market ─────────────────────────────────────────────────────────

    def mark_to_market(
        self,
        current_prices: Dict[str, float],
        instrument_specs: Dict[str, Dict],
    ) -> None:
        """
        Recomputes holdings_value, total_equity, and per-slot unrealized PnL.

        Uses _last_prices cache so gap bars do not zero-value open positions.

        Args:
            current_prices: Symbol → latest close price (current bar only).
            instrument_specs: Symbol → {multiplier, tick_size}.
        """
        self.holdings_value = 0.0
        new_unrealized: Dict[int, float] = {}

        for (slot_id, symbol), qty in self.positions.items():
            if abs(qty) < 1e-9:
                continue
            if symbol in current_prices:
                self._last_prices[symbol] = current_prices[symbol]
            price = self._last_prices.get(symbol, 0.0)
            spec = instrument_specs.get(symbol, {"multiplier": 1.0})
            multiplier = spec["multiplier"]

            avg = self.avg_prices.get((slot_id, symbol), price)
            position_value = qty * price * multiplier
            self.holdings_value += position_value

            # Unrealized PnL = current MtM - cost basis
            unrealized = qty * (price - avg) * multiplier
            new_unrealized[slot_id] = new_unrealized.get(slot_id, 0.0) + unrealized

        self.slot_unrealized_pnl = new_unrealized
        self.total_equity = self.cash + self.holdings_value

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def record_snapshot(
        self,
        timestamp: Any,
        instrument_specs: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Appends a portfolio state snapshot to the history log.

        Per-slot total_pnl (realized + unrealized) is embedded as slot_N_pnl columns.
        An exposure snapshot recording qty and notional per (slot, symbol) is also saved.

        Args:
            timestamp: Current bar timestamp.
            instrument_specs: Instrument specs for notional calculation in exposure.
        """
        row: Dict[str, Any] = {
            "timestamp":   timestamp,
            "cash":        self.cash,
            "holdings":    self.holdings_value,
            "total_value": self.total_equity,
        }

        # Embed per-slot total PnL = realized + unrealized
        all_slot_ids = set(self.slot_realized_pnl) | set(self.slot_unrealized_pnl)
        for slot_id in all_slot_ids:
            realized   = self.slot_realized_pnl.get(slot_id, 0.0)
            unrealized = self.slot_unrealized_pnl.get(slot_id, 0.0)
            row[f"slot_{slot_id}_pnl"] = realized + unrealized

        self.history.append(row)

        # Exposure snapshot: qty + notional per (slot, symbol)
        exp_row: Dict[str, Any] = {"timestamp": timestamp}
        for (slot_id, symbol), qty in self.positions.items():
            price = self._last_prices.get(symbol, 0.0)
            multiplier = 1.0
            if instrument_specs:
                multiplier = instrument_specs.get(symbol, {}).get("multiplier", 1.0)
            notional = qty * price * multiplier
            exp_row[f"slot_{slot_id}_{symbol}_qty"]      = qty
            exp_row[f"slot_{slot_id}_{symbol}_notional"] = notional

        self._exposure_history.append(exp_row)

    def get_history_df(self) -> pd.DataFrame:
        """
        Returns snapshot history as a DataFrame indexed by timestamp.

        Returns:
            DataFrame with columns: cash, holdings, total_value, slot_N_pnl...
        """
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        df.set_index("timestamp", inplace=True)
        return df

    def get_exposure_df(self) -> pd.DataFrame:
        """
        Returns the bar-by-bar exposure matrix indexed by timestamp.

        Columns follow the pattern:
            slot_{slot_id}_{symbol}_qty      — Signed quantity held.
            slot_{slot_id}_{symbol}_notional — Dollar notional (qty * price * multiplier).

        Returns:
            DataFrame with exposure columns, or empty DataFrame.
        """
        if not self._exposure_history:
            return pd.DataFrame()
        df = pd.DataFrame(self._exposure_history).fillna(0.0)
        df.set_index("timestamp", inplace=True)
        return df

    # ── Convenience ────────────────────────────────────────────────────────────

    def get_position(self, slot_id: int, symbol: str) -> float:
        """Returns the signed quantity for (slot_id, symbol), or 0."""
        return self.positions.get((slot_id, symbol), 0.0)

    def get_symbol_net_position(self, symbol: str) -> float:
        """Returns the net signed quantity across all slots for a symbol."""
        return sum(
            qty for (sid, sym), qty in self.positions.items() if sym == symbol
        )
