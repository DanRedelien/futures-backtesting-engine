"""tests/unit/test_portfolio_book.py — PortfolioBook unit tests."""

import pytest
import pandas as pd

from src.backtest_engine.portfolio_layer.execution.portfolio_book import PortfolioBook

SPECS = {"ES": {"multiplier": 50.0, "tick_size": 0.25}}


def _book(capital: float = 100_000.0) -> PortfolioBook:
    return PortfolioBook(capital)


class TestApplyFill:
    def test_buy_debits_cash(self):
        b = _book()
        b.apply_fill(0, "ES", fill_price=4000.0, quantity=1, commission=2.5,
                     multiplier=50.0, timestamp=None)
        assert b.cash == pytest.approx(100_000.0 - 4000.0 * 50.0 - 2.5)

    def test_sell_to_flat_restores_cash(self):
        b = _book()
        b.apply_fill(0, "ES", 4000.0, 1, 2.5, 50.0, None)
        b.apply_fill(0, "ES", 4010.0, -1, 2.5, 50.0, None)
        assert (0, "ES") not in b.positions
        assert (0, "ES") not in b.avg_prices

    def test_avg_price_unchanged_on_partial_close(self):
        """Reducing a long position must not change avg_price."""
        b = _book()
        b.apply_fill(0, "ES", 4000.0, 2, 0.0, 50.0, None)
        entry_avg = b.avg_prices[(0, "ES")]
        b.apply_fill(0, "ES", 4050.0, -1, 0.0, 50.0, None)  # reduce, not close
        assert b.avg_prices[(0, "ES")] == pytest.approx(entry_avg)

    def test_avg_price_updates_on_add(self):
        b = _book(1_000_000.0)
        b.apply_fill(0, "ES", 4000.0, 1, 0.0, 50.0, None)
        b.apply_fill(0, "ES", 4100.0, 1, 0.0, 50.0, None)
        expected = (4000.0 + 4100.0) / 2
        assert b.avg_prices[(0, "ES")] == pytest.approx(expected)

    def test_direction_flip_resets_avg_price(self):
        b = _book(1_000_000.0)
        b.apply_fill(0, "ES", 4000.0, 2, 0.0, 50.0, None)   # long 2
        b.apply_fill(0, "ES", 4050.0, -4, 0.0, 50.0, None)   # flip to short 2
        assert b.positions[(0, "ES")] == pytest.approx(-2.0)
        assert b.avg_prices[(0, "ES")] == pytest.approx(4050.0)


class TestMarkToMarket:
    def test_invariant_cash_plus_holdings_equals_equity(self):
        b = _book()
        b.apply_fill(0, "ES", 4000.0, 1, 2.5, 50.0, None)
        b.mark_to_market({"ES": 4050.0}, SPECS)
        assert b.total_equity == pytest.approx(b.cash + b.holdings_value)

    def test_gap_bar_uses_last_known_price(self):
        """MtM on a bar with no price update must NOT zero-value the position."""
        b = _book(1_000_000.0)
        b.apply_fill(0, "ES", 4000.0, 1, 0.0, 50.0, None)
        b.mark_to_market({"ES": 4100.0}, SPECS)  # first bar, price known
        equity_before = b.total_equity
        b.mark_to_market({}, SPECS)               # gap bar — ES price missing
        assert b.total_equity == pytest.approx(equity_before)  # must hold, not 0

    def test_shared_capital_invariant(self):
        b = _book()
        b.apply_fill(0, "ES", 4000.0, 1, 0.0, 50.0, None)
        b.apply_fill(1, "NQ", 15000.0, 1, 0.0, 20.0, None)
        b.mark_to_market({"ES": 4050.0, "NQ": 15100.0},
                         {"ES": {"multiplier": 50.0}, "NQ": {"multiplier": 20.0}})
        assert b.total_equity == pytest.approx(b.cash + b.holdings_value)
