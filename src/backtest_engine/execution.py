from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, replace
from datetime import datetime
import pandas as pd

@dataclass
class Order:
    """
    Represents an intent to trade at a specific time.
    """
    symbol: str
    quantity: float
    side: str # 'BUY' or 'SELL'
    order_type: str = 'MARKET'
    reason: str = 'SIGNAL' # e.g., 'SIGNAL', 'SL', 'TP', 'TIME'
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Fill:
    """
    Represents a finalized trade execution against the market.

    Methodology:
    `slippage` is stored in price units per contract so the fill record preserves
    the actual execution-price adjustment applied by the simulator.
    """
    order: Order
    fill_price: float
    commission: float
    slippage: float
    cost: float
    timestamp: datetime
    
@dataclass
class Trade:
    """
    Represents a completed round-trip trade (Entry + Exit) for analytics scoring.

    Methodology:
    `commission` and `slippage` are stored as positive dollar cost magnitudes for
    the completed round trip so exporters and dashboard analytics do not need to
    infer units from the execution layer.
    """
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    direction: str # 'LONG' or 'SHORT'
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    slippage: float = 0.0
    exit_reason: str = 'SIGNAL'
    entry_signal_time: Optional[datetime] = None

class ExecutionHandler:
    """
    Handles order execution tracking and simulates fills into Round-trip Trades.

    Methodology:
    Fill prices are adjusted by a deterministic spread model.  The number of
    spread ticks applied per fill is supplied by the calling engine via the
    `effective_spread_ticks` parameter, which is computed from
    `spread_model.compute_spread_ticks()` before each execution.  When
    `effective_spread_ticks` is not provided the handler reads
    `settings.spread_ticks` directly (appropriate for static mode and tests).
    No random number generator is used anywhere in this class.
    """
    
    def __init__(self, settings: Any):
        """
        Initializes the ExecutionHandler.
        
        Args:
            settings: Configuration object containing trading specifications and fee models.
        """
        self.settings = settings
        self.fills: List[Fill] = []
        self.trades: List[Trade] = []
        
        # Position tracking for Trade matching (FIFO basis)
        self._positions: Dict[str, List[Fill]] = {} 

    def execute_order(
        self,
        order: Order,
        data_bar: pd.Series,
        execute_at_close: bool = False,
        effective_spread_ticks: Optional[int] = None,
    ) -> Optional[Fill]:
        """
        Simulates order execution with deterministic spread and commission constraints.
        
        Methodology:
        Derives an execution price adjusted by a deterministic spread model.
        The spread tick count is either supplied externally by the engine (adaptive
        mode) or read from settings.spread_ticks (static mode).  No random sampling
        is performed; the same inputs always produce the same output.

        Fill-price convention:
            BUY  fill: price + spread_ticks * tick_size
            SELL fill: price - spread_ticks * tick_size
        
        Args:
            order: The Order object to execute.
            data_bar: The current OHLCV bar representing the market state at execution.
            execute_at_close: Evaluates execution against close prices if True,
                              open prices otherwise.
            effective_spread_ticks: Pre-computed tick count from the spread model.
                                    If None, falls back to settings.spread_ticks.
            
        Returns:
            The executed Fill object, or None if execution fails.
        """
        price = data_bar['close'] if execute_at_close else data_bar['open']
        
        spec = self.settings.get_instrument_spec(order.symbol)

        ticks = (
            effective_spread_ticks
            if effective_spread_ticks is not None
            else getattr(self.settings, 'spread_ticks', 1)
        )
        slippage = ticks * spec["tick_size"]
        
        executed_price = price + slippage if order.side == 'BUY' else price - slippage
        commission = abs(order.quantity) * self.settings.commission_rate
        cost = (executed_price * order.quantity) if order.side == 'BUY' else -(executed_price * order.quantity)
        
        fill = Fill(
            order=order,
            fill_price=executed_price,
            commission=commission,
            slippage=slippage,
            cost=cost,
            timestamp=data_bar.name if isinstance(data_bar.name, datetime) else order.timestamp
        )
        self.fills.append(fill)
        self._process_trades(fill)
        return fill

    def _process_trades(self, fill: Fill):
        """
        Reconciles fills into completed Trades (Round-trips) for analytics.
        
        Methodology:
        Applies FIFO matching logic against open positions. 
        Safely calculates continuous multi-fill Net PnL distributions, accounting 
        for proportionate commissions and asset multipliers.
        
        Args:
            fill: The recently executed Fill to match against existing Open tracking metrics.
        """
        symbol = fill.order.symbol
        if symbol not in self._positions:
            self._positions[symbol] = []
            
        fill_qty = fill.order.quantity
        fill_price = fill.fill_price
        fill_comm = fill.commission
        fill_time = fill.timestamp
        remaining_qty = fill.order.quantity
        
        side = 1 if fill.order.side == 'BUY' else -1
        new_open_positions = []
        
        for open_fill in self._positions[symbol]:
            if remaining_qty == 0:
                new_open_positions.append(open_fill)
                continue
                
            open_qty = open_fill.order.quantity
            open_side = 1 if open_fill.order.side == 'BUY' else -1
            
            if side == open_side:
                new_open_positions.append(open_fill)
                continue
                
            match_qty = min(abs(remaining_qty), abs(open_qty))
            entry_price = open_fill.fill_price
            spec = self.settings.get_instrument_spec(symbol)
            multiplier = spec["multiplier"]
            
            # Calculate Base PnL honoring whether the trade entered as Long or Short
            if open_side == 1:
                pnl = (fill_price - entry_price) * match_qty * multiplier
                direction = 'LONG'
            else:
                pnl = (entry_price - fill_price) * match_qty * multiplier
                direction = 'SHORT'
            
            # Approximate proportional commission for the matched chunk
            entry_comm_per_share = open_fill.commission / abs(open_fill.order.quantity) if open_fill.order.quantity != 0 else 0
            exit_comm_per_share = fill_comm / abs(fill_qty) if fill_qty != 0 else 0
            
            trade_comm = (entry_comm_per_share + exit_comm_per_share) * match_qty
            entry_slippage_per_unit = abs(open_fill.slippage) * multiplier
            exit_slippage_per_unit = abs(fill.slippage) * multiplier
            trade_slippage = (entry_slippage_per_unit + exit_slippage_per_unit) * match_qty
            net_pnl = pnl - trade_comm - trade_slippage
            
            self.trades.append(Trade(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=fill_price,
                quantity=match_qty,
                direction=direction,
                entry_time=open_fill.timestamp,
                exit_time=fill_time,
                pnl=net_pnl,
                commission=trade_comm,
                slippage=trade_slippage,
                exit_reason=fill.order.reason,
                entry_signal_time=open_fill.order.timestamp
            ))
            
            # Reduce matched quantities from ongoing open position trackers
            if abs(remaining_qty) >= abs(open_qty):
                remaining_qty = (abs(remaining_qty) - abs(open_qty)) * side
            else:
                residue = (abs(open_qty) - abs(remaining_qty)) * open_side
                residue_fill = self._clone_fill_with_quantity(open_fill, residue)
                new_open_positions.append(residue_fill)
                remaining_qty = 0
        
        # Track any remaining unmatched execution quantities as new open positions
        if remaining_qty != 0:
            new_fill_tracker = self._clone_fill_with_quantity(fill, remaining_qty)
            new_open_positions.append(new_fill_tracker)
            
        self._positions[symbol] = new_open_positions

    def _clone_fill_with_quantity(self, fill: Fill, quantity: float) -> Fill:
        """
        Clones a fill tracker with proportional total costs for a new quantity.

        Methodology:
        Open-position residue objects are accounting trackers, not new market
        executions. They preserve the original fill timestamp and execution
        price. `commission` and signed cash `cost` are scaled to the remaining
        quantity because they are aggregate fill totals. `slippage` is
        intentionally left unchanged because Fill stores it as a per-contract
        price-unit adjustment; scaling it here would understate residue slippage
        on later matched trade fragments.

        Args:
            fill: Original fill object.
            quantity: Signed quantity to keep on the cloned tracker.

        Returns:
            A cloned Fill with scaled aggregate fields for the requested quantity.
        """
        original_qty = abs(fill.order.quantity)
        scaled_fill = replace(fill)
        scaled_fill.order = replace(fill.order)
        scaled_fill.order.quantity = quantity

        if original_qty <= 0:
            scaled_fill.commission = 0.0
            scaled_fill.cost = 0.0
            return scaled_fill

        qty_ratio = abs(quantity) / original_qty
        scaled_fill.commission = float(fill.commission) * qty_ratio
        scaled_fill.cost = float(fill.cost) * qty_ratio
        return scaled_fill
