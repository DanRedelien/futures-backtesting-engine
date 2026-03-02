# STRATEGY OPTIMIZATION MCP (Model Context Protocol)

**Role**: You are a Quantitative Engineer.
**Objective**: When verifying, building, or refactoring the "Optimization Engine", use THIS specification. This is the source of truth for parameter tuning logic.

---

## 1. Problem Statement

Naive optimization (grid search on full history) leads to **Overfitting**. Finding parameters that worked "perfectly" in the past does not guarantee future performance; in fact, it often guarantees failure (fitting noise).

**Solution**:
1.  **Strict Data Separation**: We NEVER optimize on the Test set.
2.  **Bayesian Optimization (Optuna)**: Efficiently explores high-dimensional space without brute force.
3.  **Composite Objective**: We do not optimize for "Net Profit" (which favors high risk). We optimize for a risk-adjusted score (Sharpe/Sortino × Stability).
4.  **Verification**: We automatically verify "best" parameters on Out-of-Sample data.

---

## 2. Core Methodology

### 2.1. The Optimizer Engine (`OptunaOptimizer`)

**Class**: `src.multi_strategy_backtest.optimization.optimizer.OptunaOptimizer`

**Data Split Strategy**:
-   **input**: 100% Data
-   **Train**: First 70% (Used for `study.optimize`)
-   **Test**: Last 30% (Used ONLY for final verification)

**Code Pattern**:
```python
class OptunaOptimizer:
    MIN_TRADES_PER_TRIAL: int = 10      # Hard Floor (Prune if < 10)
    TARGET_TRADES_PER_TRIAL: int = 50   # Soft Target (Ramp penalty if < 50)

    def __init__(self, data_m5, data_h1, test_size_pct=0.3):
        # Strict Slicing
        split_idx = int(len(data_m5) * (1 - test_size_pct))
        self.train_m5 = data_m5.iloc[:split_idx]
        self.test_m5 = data_m5.iloc[split_idx:]
### 2.1. The Optimizer Engine (`OptunaOptimizer`)

**Class**: `src.multi_strategy_backtest.optimization.optimizer.OptunaOptimizer`

**Optimization Workflow**:
1.  **Strict Data Separation**:
    -   `Train`: Used for `study.optimize`.
    -   `Test`: Used ONLY for final verification.
2.  **Pruning**:
    -   Trials with `< MIN_TRADES_PER_TRIAL` (10) are pruned immediately (Score = -1.0).
    -   *Note*: If a strategy is unprofitable on all trials, the IS Score will be `0.00` (clamped). This is Valid.

### 2.2. Walk-Forward Validation (`WalkForwardOptimizer`)

**Class**: `src.multi_strategy_backtest.optimization.wfv_optimizer.WalkForwardOptimizer`

**Methodology**:
We use **Purged K-Fold Cross Validation** to simulate realistic re-optimization cycles.
-   **Purging**: Small gap between Train and Test to prevent label leakage.
-   **Embargo**: Additional gap after Test to prevent correlation leakage.

**Workflow**:
for fold in folds:
    1. Optimize on `Train[fold]`.
    2. Select Best Params.
    3. Evaluate on `Test[fold]` (Out-of-Sample).
    4. Calculate **Performance Decay** (IS vs OOS).

### 2.2. The Objective Function

**Module**: `src.multi_strategy_backtest.optimization.objective`

We use a **Composite Score** to guide the optimizer.

**Formula**:
$$ Score = BaseScore \times ActivityPenalty \times StabilityPenalty $$

**Unit Standards** (CRITICAL):
-   **Drawdown Input**: Must be a **Percentage** (e.g., `-25.0` or `25.0`).
-   **Drawdown Logic**: Converted internally to **Fraction** (`max_dd_frac = abs(dd) / 100.0`).
-   **Trades**:
    -   `min_trades` (Hard Floor): Absolute minimum to considered valid.
    -   `target_trades` (Soft Target): Level where `ActivityPenalty` reaches 1.0 (no penalty).

**Logic**:
1.  **Blended Risk-Adjusted Return**: Sharpe (70%) + Sortino (30%).
2.  **Activity Penalty**: Smooth ramp from 0→1 as trades approach `target_trades`.
    -   `penalty = min(1.0, trades / target_trades)`
    -   *Logic*: A strategy with 10 trades is "allowed" (passed hard floor) but heavily penalized (0.2 factor) vs one with 50 trades (1.0 factor).
3.  **Stability Penalty**: Soft quadratic decay as MaxDD approaches `max_dd_limit` (0.25).
    -   `penalty = 1.0 - (max_dd_frac / max_dd_limit)²`

**Implementation**:
```python
def objective_score(
    stats: dict, 
    min_trades=10, 
    target_trades=50, 
    max_dd_limit=0.25
) -> float:
    # 1. Hard Rejection (Floor)
    if stats['total_trades'] < min_trades:
    # 1. Hard Rejection (Floor)
    if stats['total_trades'] < min_trades:
        return -1.0  # Immediate kill
    
    # 2. Base Score (Blended)
    base_score = 0.7 * stats['sharpe_ratio'] + 0.3 * stats['sortino_ratio']
    
    # CRITICAL: If strategy loses money (Sharpe < 0), base_score is negative.
    # The final clamp `max(0.0, score)` means unprofitable strategies get Score 0.0.
    
    # 3. Activity Penalty (Ramp to Target)
    # Returns 1.0 if trades >= target_trades, else linear reduction
    activity_penalty = min(1.0, stats['total_trades'] / target_trades)
    
    # 4. Stability Penalty (Soft Barrier)
    max_dd_frac = abs(stats['max_drawdown']) / 100.0 # Standardize to 0.0-1.0
    
    if max_dd_frac > max_dd_limit:
        stability_penalty = 0.0 # Hard floor
    else:
        # Quadratic decay: 1.0 at DD=0, drops to 0.0 at DD=limit
        stability_penalty = 1.0 - (max_dd_frac / max_dd_limit) ** 2
    
    return max(0.0, base_score * activity_penalty * stability_penalty)
```

---

## 3. Strict Engineering Constraints (`Validation`)

**Module**: `src.multi_strategy_backtest.optimization.validation`

**Rules**:
1.  **Low Dimensionality**: Max **6 optimization parameters**.
2.  **Logic Only**: Optimization of Risk Management (Stop Loss, Leverage) is **FORBIDDEN**. 

---

## 4. Execution Workflow

When running `python run.py --optimize`:

1.  **Optimization Loop** (Train Data - 70%):
    -   Run Trial -> Get `stats` (with DD as %)
    -   Check `MIN_TRADES` (Hard Prune)
    -   Calculate `objective_score(stats, min_trades=10, target_trades=50)`
2.  **Verification** (Test Data - 30%):
    -   Run Best Params -> Get `test_stats`
    -   Calculate `test_score` using **SAME** objective function and constants!
    -   Compare `Train` vs `Test`.

---

## 5. Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| `n_trials` | 50-100 | Optimization budget. |
| `test_size_pct` | 0.3 (30%) | Holdout for verification. |
| `max_dd_limit` | 0.25 | 25% Drawdown limit (Soft Barrier). |
| `min_trades` | 10 | **Hard Floor**. Trials < 10 are Pruned/-1.0. |
| `target_trades` | 50 | **Soft Target**. Trials < 50 are allowed but penalized. |

---

## 6. Known Limitations

| Issue | Mitigation |
|---|---|
| **Single Split** | We use a single Train/Test split. This is a "Sanity Check", NOT a full robustness test (Walk-Forward). |
| **Market Regime** | Training data might be "Bull", Test might be "Bear". This is a feature (robustness test). |
| **Speed** | Serial execution is slow. |

---

## 7. Reference Implementation

See:
-   `src/multi_strategy_backtest/optimization/optimizer.py`
-   `src/multi_strategy_backtest/optimization/wfv_optimizer.py` (Walk-Forward)
-   `src/multi_strategy_backtest/optimization/fold_generator.py` (Purged K-Fold)
-   `src/multi_strategy_backtest/optimization/objective.py`
-   `src/multi_strategy_backtest/optimization/objective.py`
