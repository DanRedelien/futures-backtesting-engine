"""
Unit tests for dashboard/transforms.py.

Invariants tested:
    1. bar_pnl_matrix contains incremental (not cumulative) PnL.
    2. PnL contribution percentages sum to ~100% when total PnL is positive.
    3. Strategy correlation diagonal = 1, off-diagonal < 1 for decorrelated series.
    4. VaR 95% is <= 0 for a mixed PnL series with losses.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest_engine.analytics.dashboard.core.transforms import (
    build_bar_pnl_matrix,
    compute_strategy_decomp,
    compute_strategy_correlation,
    compute_exposure_correlation,
    compute_pnl_dist_stats,
    compute_rolling_sharpe,
    compute_per_strategy_summary,
    compute_strategy_stats,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_history(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic portfolio history DataFrame with two slot PnL columns.

    The cumulative slot_0_pnl and slot_1_pnl columns are deliberately
    constructed from different random walks so they are decorrelated at the
    bar level.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="30min")

    s0_bar = rng.normal(10, 80, n)
    s1_bar = rng.normal(-5, 60, n)

    df = pd.DataFrame({
        "total_value": 100_000 + np.cumsum(s0_bar + s1_bar),
        "slot_0_pnl":  np.cumsum(s0_bar),
        "slot_1_pnl":  np.cumsum(s1_bar),
    }, index=idx)
    return df


def _make_slots() -> dict:
    return {"0": "StrategyA", "1": "StrategyB"}


def _make_trades(slots: dict) -> pd.DataFrame:
    """Generates minimal trades DataFrame with strategy, pnl, gross_pnl columns."""
    rng = np.random.default_rng(0)
    n = 60
    strategies = [list(slots.values())[i % 2] for i in range(n)]
    pnls = rng.normal(50, 200, n)
    return pd.DataFrame({
        "strategy":  strategies,
        "pnl":       pnls,
        "gross_pnl": pnls + rng.uniform(1, 10, n),  # gross > net (fees)
        "direction": ["LONG"] * n,
    })


# ── Test 1: bar PnL matrix is incremental (not cumulative) ────────────────────

def test_bar_pnl_matrix_not_cumulative() -> None:
    """
    Ensures build_bar_pnl_matrix() returns *incremental* bar PnL, not the
    raw cumulative equity columns.

    A cumulative series is strictly monotonically increasing for a profitable
    strategy — the incremental series must have both positive and negative bars.
    """
    history = _make_history()
    slots   = _make_slots()
    matrix  = build_bar_pnl_matrix(history, slots)

    assert not matrix.empty, "Bar PnL matrix must not be empty."
    assert "StrategyA" in matrix.columns
    assert "StrategyB" in matrix.columns

    for col in matrix.columns:
        series = matrix[col].dropna()
        # A cumulative series has std(diff) ≈ 0; incremental has both signs
        assert (series > 0).any(), f"{col}: no positive bars found — may be cumulative?"
        assert (series < 0).any(), f"{col}: no negative bars found — may be cumulative?"
        # Additional invariant: NOT monotonically non-decreasing
        is_monotone = (series.diff().dropna() >= 0).all()
        assert not is_monotone, f"{col}: series is monotonically non-decreasing (cumulative bug)."


# ── Test 2: PnL contribution absolute sum is ~100% ────────────────────────────

def test_decomp_contribution_abs_sum_is_100_pct() -> None:
    """
    Ensures that the sum of absolute PnL Contribution %
    values across all strategies equals ~100%.
    """
    history = _make_history(seed=1)
    slots   = _make_slots()
    trades  = _make_trades(slots)

    decomp = compute_strategy_decomp(trades, history, slots)
    assert not decomp.empty, "Decomp table must not be empty."

    # Using the new robust denominator sum(|pnl|), the absolute sum of contributions
    # should be exactly 100.0.
    total_abs_contribution = decomp["PnL Contrib (%)"].abs().sum()
    assert math.isclose(total_abs_contribution, 100.0, abs_tol=0.11), (
        f"Absolute PnL contributions sum to {total_abs_contribution:.3f}%, expected ~100%."
    )
    assert "MTM PnL ($)" not in decomp.columns, "MTM PnL should not be rendered in decomposition."


# ── Test 3: Correlation diagonal = 1, off-diagonal < 1 ───────────────────────

def test_strategy_correlation_diagonal_is_one() -> None:
    """
    Ensures that the strategy correlation matrix has diagonal = 1.0 and that
    decorrelated random series produce off-diagonal < 1.
    """
    history = _make_history(seed=7)
    slots   = _make_slots()
    bar_pnl = build_bar_pnl_matrix(history, slots)

    corr = compute_strategy_correlation(bar_pnl, horizon="1d")

    assert not corr.empty, "Correlation matrix must not be empty."
    # Diagonal must be exactly 1 (self-correlation)
    for col in corr.columns:
        assert math.isclose(corr.loc[col, col], 1.0, abs_tol=1e-9), (
            f"Diagonal element [{col},{col}] = {corr.loc[col, col]:.6f}, expected 1.0"
        )

    # Off-diagonal must be < 1 for decorrelated normal random walks
    for r in corr.index:
        for c in corr.columns:
            if r != c:
                assert abs(corr.loc[r, c]) < 1.0, (
                    f"Off-diagonal [{r},{c}] = {corr.loc[r, c]:.4f} is not < 1."
                )


# ── Test 4: VaR 95% is a positive loss magnitude ─────────────────────────────

def test_pnl_dist_stats_var_is_positive_loss_magnitude() -> None:
    """
    Ensures that the historical VaR 95% is reported as a positive loss
    magnitude when the PnL series contains losses.

    VaR is derived from the lower-tail percentile of daily PnL, but the
    dashboard-facing convention is to expose loss magnitude as a positive value.
    """
    rng = np.random.default_rng(42)
    daily = pd.Series(rng.normal(0, 100, 250))  # plenty of negative values

    dist_stats = compute_pnl_dist_stats(daily)

    assert "var_95" in dist_stats
    assert dist_stats["var_95"] >= 0, (
        f"VaR 95% = {dist_stats['var_95']:.2f} — expected positive loss magnitude."
    )
    # CVaR (Expected Shortfall) must be at least as large as VaR in loss terms.
    if not np.isnan(dist_stats.get("cvar_95", float("nan"))):
        assert dist_stats["cvar_95"] >= dist_stats["var_95"], (
            "CVaR must be >= VaR 95% when both are expressed as positive loss magnitudes."
        )


# ── Test 5: Rolling Sharpe has finite values after warm-up ────────────────────

def test_rolling_sharpe_finite_after_warmup() -> None:
    """
    Ensures rolling Sharpe series has finite (non-NaN) values after the
    initial warm-up window, confirming the calculation is not degenerate.

    The new implementation resamples to daily frequency internally, so the
    history fixture must span well more than `window_days` calendar days.
    _make_history(n=500) creates 500 30-min bars ≈ 38 trading days — not
    enough for a 90-day window. Use n=2000 bars ≈ 153 trading days.
    """
    history = _make_history(n=2000)     # ~153 trading days @ 13 bars/day
    sharpe  = compute_rolling_sharpe(history, window_days=45)  # 45-day window

    assert isinstance(sharpe, pd.Series)
    finite_values = sharpe.dropna()
    assert len(finite_values) > 0, "No finite Sharpe values after warm-up window."


# ── Test 6: Per-strategy summary returns trade count ─────────────────────────

def test_per_strategy_summary_trade_counts() -> None:
    """Ensures per-strategy summary correctly counts trades per strategy."""
    slots  = _make_slots()
    trades = _make_trades(slots)

    summary = compute_per_strategy_summary(trades, slots)

    for strat_name in slots.values():
        assert strat_name in summary, f"Missing strategy {strat_name} in summary."
        expected_count = int((trades["strategy"] == strat_name).sum())
        assert summary[strat_name]["trade_count"] == expected_count, (
            f"Trade count mismatch for {strat_name}: "
            f"got {summary[strat_name]['trade_count']}, expected {expected_count}."
        )


def test_strategy_stats_contract_uses_trade_stat_semantics() -> None:
    """Strategy Stats should stay aligned with the shared trade-stat semantics."""
    slots = {"0": "StrategyA"}
    trades = pd.DataFrame(
        {
            "strategy": ["StrategyA"] * 4,
            "pnl": [100.0, -40.0, 20.0, -10.0],
        }
    )

    stats_frame = compute_strategy_stats(trades, slots)

    assert list(stats_frame.columns) == [
        "Strategy",
        "Total Trades",
        "Win Rate %",
        "Avg Trade ($)",
        "Max Loss ($)",
        "T-Stat",
        "P-Value",
    ]
    assert len(stats_frame) == 1

    row = stats_frame.iloc[0]
    assert row["Strategy"] == "StrategyA"
    assert row["Total Trades"] == 4
    assert math.isclose(float(row["Win Rate %"]), 50.0, rel_tol=1e-9)
    assert math.isclose(float(row["Avg Trade ($)"]), 17.5, rel_tol=1e-9)
    assert math.isclose(float(row["Max Loss ($)"]), -40.0, rel_tol=1e-9)


# ── Test 7: Empty inputs return empty outputs gracefully ──────────────────────

def test_transforms_handle_empty_inputs() -> None:
    """All transform functions must return empty, not raise, on empty inputs."""
    empty_df   = pd.DataFrame()
    empty_hist = pd.DataFrame(columns=["total_value"])

    assert build_bar_pnl_matrix(empty_hist, {}).empty
    assert compute_strategy_decomp(empty_df, empty_hist, {}).empty
    assert compute_strategy_correlation(empty_df).empty
    assert compute_exposure_correlation(empty_df)[0].empty

    stats = compute_pnl_dist_stats(pd.Series([], dtype=float))
    assert all(np.isnan(v) for v in stats.values())

    sharpe = compute_rolling_sharpe(empty_hist)
    assert isinstance(sharpe, pd.Series)


# ── Test 8: Per-strategy summary includes Alpha, Beta, T-stat, etc. ────────────

def test_per_strategy_summary_statistics() -> None:
    """Ensures per-strategy summary computes the new statistics correctly."""
    slots = {"0": "StrategyA"}
    rng = np.random.default_rng(42)
    n_days = 60
    
    idx = pd.date_range("2024-01-01", periods=n_days, freq="1D")
    
    # Mock history
    history = pd.DataFrame({
        "total_value": np.full(n_days, 100000.0),
        "slot_0_pnl": np.cumsum(rng.normal(10, 50, n_days))
    }, index=idx)
    
    # Mock instrument closes
    instrument_closes = pd.DataFrame({
        "TEST_SYM": np.cumsum(rng.normal(0, 2, n_days)) + 100
    }, index=idx)
    
    # Mock trades
    trades = pd.DataFrame({
        "strategy": ["StrategyA"] * 10,
        "symbol": ["TEST_SYM"] * 10,
        "pnl": rng.normal(50, 20, 10),
        "direction": ["LONG"] * 10
    })
    
    summary = compute_per_strategy_summary(trades, slots, history=history, instrument_closes=instrument_closes)
    
    strat_stats = summary["StrategyA"]
    
    # Verifying presence and validity of new keys
    for key in ["tstat", "pvalue", "alpha", "alpha_p", "beta", "beta_p"]:
        assert key in strat_stats, f"Missing key {key} in strategy summary"
        assert not np.isnan(strat_stats[key]), f"Key {key} should not be NaN"
        
    # Check that t-stat computation matches scipy's output directly
    from scipy import stats
    expected_t, _ = stats.ttest_1samp(trades["pnl"], 0.0)
    assert math.isclose(strat_stats["tstat"], float(expected_t), rel_tol=1e-5)


def test_per_strategy_summary_alpha_beta_use_daily_returns() -> None:
    """
    Ensures alpha/beta regression is performed on daily strategy returns,
    producing stable estimates on a synthetic linear factor model.
    """
    slots = {"0": "StrategyA"}
    n_days = 120
    beta_true = 1.5
    alpha_daily = 0.0004
    initial_capital = 100_000.0

    idx = pd.date_range("2024-01-01", periods=n_days, freq="1D")
    rng = np.random.default_rng(123)
    instrument_returns = pd.Series(rng.normal(0.0005, 0.01, n_days), index=idx)
    strategy_returns = alpha_daily + beta_true * instrument_returns

    instrument_closes = pd.DataFrame({
        "TEST_SYM": 100.0 * (1.0 + instrument_returns).cumprod()
    }, index=idx)

    strategy_equity = initial_capital * (1.0 + strategy_returns).cumprod()
    history = pd.DataFrame({
        "total_value": strategy_equity,
        "slot_0_pnl": strategy_equity - initial_capital,
    }, index=idx)

    trades = pd.DataFrame({
        "strategy": ["StrategyA"] * 12,
        "symbol": ["TEST_SYM"] * 12,
        "pnl": np.linspace(10.0, 120.0, 12),
        "direction": ["LONG"] * 12,
    })

    summary = compute_per_strategy_summary(
        trades,
        slots,
        history=history,
        instrument_closes=instrument_closes,
        slot_weights={"0": 1.0},
    )

    strat_stats = summary["StrategyA"]

    assert math.isclose(strat_stats["beta"], beta_true, rel_tol=0.03), (
        f"Expected beta near {beta_true}, got {strat_stats['beta']:.4f}"
    )
    expected_alpha_annual_pct = alpha_daily * 252.0 * 100.0
    assert math.isclose(strat_stats["alpha"], expected_alpha_annual_pct, rel_tol=0.08), (
        f"Expected annual alpha near {expected_alpha_annual_pct:.4f}, got {strat_stats['alpha']:.4f}"
    )
    assert 0.0 <= strat_stats["beta_p"] <= 1.0
    assert 0.0 <= strat_stats["alpha_p"] <= 1.0
