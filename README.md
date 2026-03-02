# Single Asset Backtester

A high-performance, event-driven backtesting engine designed for rapid hypothesis testing and rigorous statistical validation of alpha strategies.
The pipeline specifically addresses the problem of hidden look-ahead bias in algorithmic trading and provides strict risk assessment through Out-of-Sample (OOS) Walk-Forward Validation.

## Problem Statement

Standard backtesting frameworks frequently suffer from data leakage during parameter optimization and hidden look-ahead bias when generating trading signals. This inevitably leads to in-sample (IS) curve fitting and catastrophic drawdowns in live trading.
The primary challenge is to construct a pipeline that structurally separates IS optimization from OOS validation while remaining computationally efficient for exhaustive parameter search spaces.

## Methodology

### 1. Model
The engine is built on a hybrid architecture. It leverages fully vectorized pre-computation of indicators via `pandas`/`numpy` to achieve O(1) bar lookups, whilst maintaining a precise event-driven posture for position management. 
Several regime filters are applied natively: Augmented Dickey-Fuller (ADF) for stationarity, percentile-based volatility filters, and T-Stat trend estimation.

### 2. Calculation Algorithm
To structurally prevent any data leakage, all signal matrices are strictly shifted by 1 bar:
$$ Signal_t = F(Price_{0..t-1}) $$
Positions are evaluated and executed explicitly at $t$, factoring in standard slippage models.

### 3. Pipeline Architecture
Raw OHLCV minute data is fetched via `IBFetcher` and cached locally as Parquet files. The strategy defines its parameter search space (using Optuna bounds), which is then processed by the `WalkForwardOptimizer` to perform rigorous statistical parameter selection.

## Risk Controls / Validation

| Control | Implementation |
|---------|---------------|
| Walk-Forward | 5-Fold Rolling Window (IS/OOS) |
| No Look-Ahead | Strict `.shift(1)` enforced within Strategy `__init__` |
| Stats Validation | T-Statistic, P-Value, and Deflated Sharpe Ratio (DSR) tracking |
| Parameter Stability| Algorithmic penalties for inconsistent outcomes and Alpha Decay across OOS folds |

The pipeline strictly isolates optimization folds. The objective function actively penalizes low trade counts and evaluates parameter robustness through metrics such as the Calmar Ratio, Sortino Ratio, and Deflated Sharpe Ratio, while proactively tracking strategy degradation (Alpha Decay).

### Data Integrity
Working with minute-level OHLCV requires robust cleaning:
*   **Survivorship Bias**: Currently limited (acts on active continuous futures/single assets).
*   **Corporate Actions**: Adjusted natively upstream (IBKR fetched data).
*   **Missing Bar Handling**: Forward-fills close prices to prevent indicator corruption, forces $0 volume for inactive periods.
*   **Outlier Detection**: IQR-based spike filtering during the initial Parquet build process.

## Assumptions & Limitations

- **Single Asset Focus:** The engine does not currently calculate portfolio-level correlations or cross-margining requirements.
- **Slippage Model:** Uses a fixed slippage assumption, which does not account for order book depth sparsity during extreme volatility.
- **Execution:** Assumes immediate market order execution without modeling microstructural queue positions.

## Outputs & Diagnostics

<img width="1591" height="796" alt="image" src="https://github.com/user-attachments/assets/e5e783b3-ce9d-470e-8185-620e1cedb90d" />

> Displays the overall PnL curve, max drawdown underwater plots, and the core statistical metrics table. Analyzed: (`sma_crossover.py`)


<img width="816" height="384" alt="image" src="https://github.com/user-attachments/assets/4fca7414-0392-42ae-91ee-57c8d781944f" />

> Displays the strategy's configuration (`sma_crossover.py`), highlighting the parameter search bounds and active filters.


<img width="650" height="205" alt="image" src="https://github.com/user-attachments/assets/971925c3-c893-4e14-91f6-e0913cacff43" />

> Walk-Forward Validation parameters (`settings.py`) designed to prevent curve-fitting:
> *   **Walk-Forward Folds (`wfo_n_folds`)**: Number of IS/OOS segments.
> *   **OOS Test Size (`wfo_test_size_pct`)**: Data fraction reserved for out-of-sample testing.
> *   **Degrees of Freedom (`wfo_max_parameters`)**: Hard cap on optimizable variables to penalize complexity.
> *   **Search Depth (`wfo_n_trials`)**: Optuna iterations per fold.
> *   **Significance Gate (`wfo_prune_min_trades`)**: Minimum trades required; ensures metrics are statistically valid.
> *   **Activity Penalty (`wfo_prune_target_trades_mult`)**: Penalizes strategies that deviate from the expected trade frequency.
> *   **Risk Pruning (`wfo_prune_max_dd_pct`)**: Kills trials instantly if drawdown exceeds the limit.


*(Placeholder: Add screenshot of WFO Terminal Output)*
> Shows the rolling-window fold progression, out-of-sample parameter selection stability, and applications of the stability penalty. Optimized: (`sma_crossover.py`)


## Computational Profile

*Benchmark evaluated on Intel Core i7 / 8GB RAM / Python 3.11*

Optimization on minute-level data over 2-year periods is heavily vectorized:
*   **Optuna Budget**: 500 parameter trials per Fold
*   **Single Trial Speed**: ~ 0.90 seconds
*   **Average Fold Runtime**: ~ 7 minutes
*   **Total WFV Execution**: ~ 35 minutes

## Project Structure

```bash
├── run.py
├── requirements.txt
├── src/
│   ├── backtest_engine/
│   │   ├── engine.py
│   │   └── optimization/
│   │       └── wfv_optimizer.py
│   └── strategies/
│       ├── base.py
│       ├── filters.py
│       └── sma_crossover.py
```

## Usage

```bash
pip install -r requirements.txt

# Run a single standard backtest:
python run.py --backtest --strategy sma

# Run Walk-Forward Validation (WFO) optimization:
python run.py --wfo --strategy mean_rev
```

## Future Improvements

- [ ] Tick-level orderbook replay integration.
- [ ] Portfolio-level margin and correlation analysis.
- [ ] Probability of bankruptcy estimation via Monte Carlo simulations.
