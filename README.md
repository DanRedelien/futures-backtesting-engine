# Backtesting Platform

A futures-focused backtesting platform for single-strategy research, portfolio backtests, walk-forward validation, and terminal-style analytics review.

## Quick Start

```bash
git clone <repo-url>
cd <repo-folder>

pip install -r requirements.txt   # install all dependencies

pytest tests/                     # verify the installation (should be all green)

python run.py --portfolio-backtest --dashboard   # run a backtest and open the dashboard
# Dashboard → http://127.0.0.1:8000
```

> Python 3.11+ recommended. No manual `PYTHONPATH` export needed — `pyproject.toml` configures it automatically for pytest, and `run.py` is always launched from the repo root.

## Problem Statement

Standard backtesting frameworks frequently suffer from data leakage during parameter optimization and hidden look-ahead bias when generating trading signals. This inevitably leads to in-sample (IS) curve fitting and catastrophic drawdowns in live trading.
The primary challenge is to construct a pipeline that structurally separates IS optimization from OOS validation while remaining computationally efficient for exhaustive parameter search spaces.

## Methodology

### 1. Model
The engine is built on a hybrid architecture. It leverages fully vectorized pre-computation of indicators via `pandas`/`numpy` to achieve O(1) bar lookups, whilst maintaining a precise event-driven posture for position management. 
Several regime filters are applied natively: Half-Life mean-reversion speed estimation, Augmented Dickey-Fuller (ADF) for macro stationarity, percentile-based volatility filters, and T-Stat trend estimation.

### 2. FastBar Architecture
To solve the computational overhead of iterative row-by-row `pandas.iloc` lookups, the engine's core loop was entirely refactored to use pre-extracted $C$-level `numpy` arrays. A lightweight `FastBar` python class acts as a proxy bridge, mapping array indices back to `pandas.Series` conventions (e.g., `bar["open"]`) for strategy compatibility, achieving **~5x faster event loops**.

### 3. Calculation Algorithm
To structurally prevent any data leakage, all signal matrices are strictly shifted by 1 bar:
$$ Signal_t = F(Price_{0..t-1}) $$
Positions are evaluated and executed explicitly at $t$, factoring in standard slippage models.

### 4. Pipeline Architecture
Raw OHLCV minute data is fetched via `IBFetcher` and cached locally as Parquet files. The strategy defines its parameter search space (using Optuna bounds), which is then processed by the `WalkForwardOptimizer` to perform rigorous statistical parameter selection.

## Risk Controls / Validation

| Control | Implementation |
|---------|---------------|
| Walk-Forward | 5-Fold Rolling Window (IS/OOS) |
| No Look-Ahead | Engine automatically executes at **$OPEN_{T+1}$** for signals evaluated at **$CLOSE_T$** |
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


<img width="622" height="540" alt="image" src="https://github.com/user-attachments/assets/33cfaf5c-bb29-4502-a650-e3ad0fb034ee" />

> Shows the rolling-window fold progression, out-of-sample parameter selection stability, and applications of the stability penalty. Optimized: (`sma_crossover.py`)


## Computational Profile

*Benchmark evaluated on Intel Core i7 / 8GB RAM / Python 3.11*

Optimization on minute-level data over 2-year periods is heavily vectorized:
*   **Optuna Budget**: 500 parameter trials per Fold
*   **Single Trial Speed**: ~ 0.24 seconds
*   **Average Fold Runtime**: ~ 124 seconds
*   **Total WFV Execution**: ~ 10 minutes

## Project Structure

```bash
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODULE_MAP.md
│   └── agents.md
├── cli/
│   ├── single.py
│   ├── portfolio.py
│   └── wfo.py
├── run.py
├── requirements.txt
├── tests/
│   ├── README.md
│   ├── regression/
│   ├── unit/
│   ├── test_engine_integration.py
│   ├── test_invariants.py
│   ├── test_kalman.py
│   └── test_beta.py
├── src/
│   ├── backtest_engine/
│   │   ├── settings.py
│   │   ├── engine.py
│   │   ├── execution.py
│   │   ├── optimization/
│   │   ├── analytics/
│   │   │   ├── dashboard/
│   │   │   └── terminal_ui/
│   │   └── portfolio_layer/
│   ├── strategies/
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── *.py
│   └── data/
│       └── data_lake.py
└── TODO.md
```

## Usage

```bash
pip install -r requirements.txt

# Run a single standard backtest
python run.py --backtest --strategy sma

# Run a portfolio backtest
python run.py --portfolio-backtest --portfolio-config portfolio_config.yaml

# Launch the terminal dashboard for the latest artifacts
python run.py --dashboard

# Run Walk-Forward Validation (WFO) optimization
python run.py --wfo --strategy mean_rev
```

## Outputs

- Single runs write artifacts to `results/`.
- Portfolio runs write artifacts to `results/portfolio/`.
- Scenario reruns write namespaced artifacts under `results/scenarios/`.
- The active web dashboard is the FastAPI terminal UI in `src/backtest_engine/analytics/terminal_ui/`.

## Future Improvements

- [ ] Move to NautilusTrader for Ultra-High performance.
- [ ] Tick-level orderbook replay integration.
- [ ] Portfolio-level margin and correlation analysis.
- [ ] Probability of bankruptcy estimation via Monte Carlo simulations.
