from __future__ import annotations
from functools import lru_cache
from typing import Dict, List
import pandas as pd
from .pnl import resample_pnl_to_horizon, _HORIZON_RULE


@lru_cache(maxsize=1)
def _min_corr_samples() -> int:
    """Loads the minimum sample count required for correlation views."""
    try:
        from src.backtest_engine.settings import BacktestSettings

        return int(BacktestSettings().terminal_ui.terminal_min_correlation_samples)
    except Exception:
        return 5

def compute_strategy_correlation(
    bar_pnl_matrix: pd.DataFrame,
    horizon: str = "1d",
) -> pd.DataFrame:
    """
    Computes a Pearson correlation matrix of per-strategy incremental PnL.

    Methodology:
        Correlation is computed on *bar/daily PnL series*, not cumulative equity.
        Cumulative equity correlations are artificially inflated by shared drift
        and are meaningless for risk decomposition.

        Horizon resampling lets the user detect correlations that appear only
        at longer time scales (e.g. independent intrabar but correlated daily).

    Args:
        bar_pnl_matrix: Output of build_bar_pnl_matrix() — incremental PnL.
        horizon: Resampling horizon ('1d', '1w', '1m').

    Returns:
        Correlation DataFrame (strategies x strategies). Empty if < 2 strategies
        or too few observations after resampling.
    """
    if bar_pnl_matrix.empty or bar_pnl_matrix.shape[1] < 2:
        return pd.DataFrame()

    resampled = resample_pnl_to_horizon(bar_pnl_matrix, horizon)
    if len(resampled) < _min_corr_samples():
        return pd.DataFrame()
    return resampled.corr(method="pearson")

def compute_exposure_correlation(
    exposure_df: pd.DataFrame,
    horizon: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Computes correlation between per-instrument absolute exposures.

    Methodology:
        Step 1: Keep only `*_notional` columns.
        Step 2: Aggregate by symbol — sum slot_N_SYM_notional across all slots.
        Step 3: Resample using `mean()` (average active exposure per period). 
                We use the raw exposure, NOT the ratio to total gross_exposure. 
                Using the ratio crushes the variance and leads to 0.00 correlation
                for larger horizons when one strategy's exposure dominates.
        Step 4: Compute Pearson correlation on the resampled absolute levels.

    Args:
        exposure_df: Raw bar-level exposure DataFrame with columns like
                     `slot_0_NQ_notional`.
        horizon: Resampling horizon '1d', '1w', '1m' (default: '1d').

    Returns:
        tuple[pd.DataFrame, list[str]]: Correlation matrix (symbols x symbols) 
        and a list of dropped symbols due to insufficient data.
    """
    if exposure_df is None or exposure_df.empty:
        return pd.DataFrame(), []

    # Step 1: keep only notional columns
    notional_cols = [c for c in exposure_df.columns if c.endswith("_notional")]
    if not notional_cols:
        notional_cols = exposure_df.columns.tolist()

    df = exposure_df[notional_cols].copy()

    # Step 2: extract symbol name (slot_N_SYM_notional -> SYM)
    symbol_map: dict = {}
    for col in notional_cols:
        parts = col.split("_")
        if len(parts) >= 4 and parts[0] == "slot" and parts[-1] == "notional":
            sym = "_".join(parts[2:-1])
        else:
            sym = col
        symbol_map[col] = sym

    # Step 3: sum per-symbol notional across all slots
    symbol_notional: dict = {}
    for col, sym in symbol_map.items():
        if sym not in symbol_notional:
            symbol_notional[sym] = df[col].copy()
        else:
            symbol_notional[sym] = symbol_notional[sym].add(df[col], fill_value=0.0)

    sym_df = pd.DataFrame(symbol_notional, index=df.index)

    if sym_df.shape[1] < 2:
        return pd.DataFrame(), []

    # Step 4: resample using MEAN (average absolute exposure level)
    # This provides a more robust correlation measure for longer intervals.
    rule = _HORIZON_RULE.get(horizon, "1D")
    resampled = sym_df.abs().resample(rule).mean()

    # We want to identify symbols with sufficient sample size.
    active_cols = []
    dropped_cols = []

    for c in resampled.columns:
        valid_samples = resampled[c].notna().sum()
        if valid_samples < _min_corr_samples():
            dropped_cols.append(c)
        elif resampled[c].std() <= 1e-8:
            dropped_cols.append(c)
        else:
            active_cols.append(c)

    if len(active_cols) < 2:
        return pd.DataFrame(), dropped_cols

    return resampled[active_cols].corr(method="pearson"), dropped_cols

