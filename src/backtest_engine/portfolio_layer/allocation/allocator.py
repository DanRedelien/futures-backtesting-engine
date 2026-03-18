"""
src/backtest_engine/portfolio_layer/allocation/allocator.py

Capital allocation and contract sizing for the portfolio backtester.

Responsibility: Given total equity, strategy weights, signals, and recent
price history, compute TargetPosition quantities for each (slot, symbol) pair
using volatility-targeting methodology.

Methodology (per slot, per symbol):
    1.  slot_equity        = total_equity * slot.weight
    2.  instrument_vol     = annualised rolling stddev of close-to-close returns
                             over the last vol_lookback_bars bars.
    3.  annual_dollar_risk = slot_equity * target_portfolio_vol
    4.  contract_dollar_vol = instrument_vol * price * multiplier
    5.  raw_contracts      = annual_dollar_risk / contract_dollar_vol
    6.  contracts          = round(raw_contracts)
    7.  contracts          = min(contracts, max_contracts_per_slot)
    8.  signed_qty         = contracts * signal.direction

The key denominator is the annualised dollar volatility of one full contract:
`annualised_vol * price * multiplier`. Tick size is relevant for execution and
slippage simulation, but it is not the correct denominator for volatility-
targeted contract sizing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..domain.contracts import PortfolioConfig
from ..domain.signals import StrategySignal, TargetPosition


# Fallback annualisation factor used when the engine doesn't supply bars_per_year.
# Overridden at runtime by PortfolioBacktestEngine with a value derived from
# the actual data span (total_bars / calendar_years).
_BARS_PER_YEAR_FALLBACK: int = 252 * 13   # conservative 30-min futures assumption

class Allocator:
    """
    Converts strategy signals into sized target contract quantities.

    Uses volatility-targeting: each slot's equity is scaled by the ratio of
    the target annualised vol to the instrument's realised vol, so the
    expected portfolio volatility matches target_portfolio_vol regardless of
    regime.

    A hard cap of max_contracts_per_slot prevents runaway sizing in
    low-volatility regimes.
    """

    def __init__(self, config: PortfolioConfig) -> None:
        """
        Args:
            config: Validated PortfolioConfig holding vol-targeting parameters.
        """
        self._config = config

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_targets(
        self,
        signals: List[StrategySignal],
        total_equity: float,
        current_prices: Dict[str, float],
        instrument_specs: Dict[str, Dict],
        price_history: Dict[Tuple[int, str], pd.Series],
        bars_per_year: int = _BARS_PER_YEAR_FALLBACK,
    ) -> List[TargetPosition]:
        """
        Computes desired target positions for all active signals.

        Methodology:
            See module docstring for the full five-step formula.
            If instrument vol cannot be estimated (insufficient history or
            zero variance), falls back to 1 contract to avoid zero allocation.
            A zero-direction signal always produces target_qty = 0 (flat).

        Args:
            signals: List of StrategySignals for this bar.
            total_equity: Current portfolio equity (cash + MtM).
            current_prices: Symbol -> latest close price.
            instrument_specs: Symbol -> {multiplier, tick_size}.
            price_history: (slot_id, symbol) -> pd.Series of recent close prices
                           (at least vol_lookback_bars entries).

        Returns:
            List of TargetPosition objects (one per signal).
        """
        targets: List[TargetPosition] = []

        for sig in signals:
            slot      = self._config.slots[sig.slot_id]
            n_tickers = len(slot.symbols)

            # Each ticker in the slot gets an equal share of the slot equity.
            slot_equity        = total_equity * slot.weight
            equity_per_ticker  = slot_equity / n_tickers if n_tickers > 0 else 0.0

            price     = current_prices.get(sig.symbol, 0.0)
            spec      = instrument_specs.get(sig.symbol, {"multiplier": 1.0, "tick_size": 0.01})
            multiplier = float(spec.get("multiplier", 1.0))

            if price > 0 and multiplier > 0 and sig.direction != 0:
                vol = self._estimate_vol(sig.slot_id, sig.symbol, price_history, bars_per_year)
                contracts = self._size_contracts(equity_per_ticker, price, multiplier, vol)
            else:
                contracts = 0

            targets.append(TargetPosition(
                slot_id=sig.slot_id,
                symbol=sig.symbol,
                target_qty=contracts * sig.direction,
                reason=sig.reason,
            ))

        return targets

    # ── Private helpers ────────────────────────────────────────────────────────

    def _estimate_vol(
        self,
        slot_id: int,
        symbol: str,
        price_history: Dict[Tuple[int, str], pd.Series],
        bars_per_year: int = _BARS_PER_YEAR_FALLBACK,
    ) -> float:
        """
        Estimates annualised realised volatility from recent close prices.

        Methodology:
            Computes close-to-close log returns over the last vol_lookback_bars
            bars, takes their standard deviation, and annualises by multiplying
            by sqrt(bars_per_year).  Returns a conservative fallback of 1.0
            (100 % vol, resulting in very small sizing) when data is insufficient.

        Args:
            slot_id: Strategy slot index for series isolation.
            symbol: Instrument ticker.
            price_history: (slot_id, symbol) -> pd.Series of close prices.
            bars_per_year: Annualisation factor from the engine (actual data frequency).

        Returns:
            Annualised volatility estimate (e.g. 0.15 = 15 %).
        """
        series = price_history.get((slot_id, symbol))
        lookback = self._config.vol_lookback_bars

        if series is None or len(series) < lookback + 1:
            return 1.0  # Conservative fallback: will produce minimal sizing.

        closes   = series.iloc[-(lookback + 1):]
        log_rets = np.log(closes / closes.shift(1)).dropna()

        if len(log_rets) < 2:
            return 1.0

        bar_vol = float(np.std(log_rets, ddof=1))
        if bar_vol <= 0.0:
            return 1.0

        return bar_vol * math.sqrt(bars_per_year)

    def _compute_raw_contracts(
        self,
        equity: float,
        price: float,
        multiplier: float,
        annualised_vol: float,
    ) -> float:
        """
        Computes the continuous vol-target contract quantity before rounding.

        Methodology:
            `annualised_vol` is expected to already be annualized by
            `_estimate_vol()`. The annualized dollar risk budget is
            `equity * target_portfolio_vol`, and one contract contributes
            `annualised_vol * price * multiplier` of annualized dollar
            volatility. Rounding and hard caps are applied only after this raw
            quantity is computed so the mathematical target remains testable.

        Args:
            equity: Dollars allocated to this (slot, ticker).
            price: Current close price of the instrument.
            multiplier: Contract multiplier used to convert price moves to dollars.
            annualised_vol: Estimated annualized volatility (e.g. 0.15).

        Returns:
            Continuous contract quantity before integer conversion.
        """
        if equity <= 0.0 or price <= 0.0 or multiplier <= 0.0 or annualised_vol <= 0.0:
            return 0.0

        annual_dollar_risk = equity * self._config.target_portfolio_vol
        contract_dollar_vol = annualised_vol * price * multiplier
        if contract_dollar_vol <= 0.0:
            return 0.0

        return max(0.0, annual_dollar_risk / contract_dollar_vol)

    def _size_contracts(
        self,
        equity: float,
        price: float,
        multiplier: float,
        annualised_vol: float,
    ) -> int:
        """
        Computes integer contract count using vol-targeting.

        Methodology:
            First compute the continuous vol-target contract quantity from the
            annualized risk budget and the annualized dollar volatility of one
            contract. Integer rounding happens after the raw quantity is
            computed, and the hard cap is applied last.

        Args:
            equity: Dollars allocated to this (slot, ticker).
            price: Current close price of the instrument.
            multiplier: Contract multiplier used to convert price moves to dollars.
            annualised_vol: Estimated annualized volatility (e.g. 0.15).

        Returns:
            Integer contract count (>= 0), rounded then capped at
            max_contracts_per_slot.
        """
        raw_contracts = self._compute_raw_contracts(
            equity=equity,
            price=price,
            multiplier=multiplier,
            annualised_vol=annualised_vol,
        )
        contracts = max(0, round(raw_contracts))
        return min(contracts, self._config.max_contracts_per_slot)
