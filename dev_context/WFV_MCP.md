# WFV (Walk-Forward Validation) MCP: The Robustness Protocol

**Role**: You are a Senior Quantitative Researcher.
**Objective**: When validating ANY trading strategy or model, use THIS specification. This is the "Litmus Test" for production viability.

---

## 1. Philosophy: The "Skeptic" Mindset

Standard Backtesting (training and testing on the same data, or a single split) is **statistically invalid** for financial time-series due to:
1.  **Overfitting**: Learning noise instead of signal.
2.  **Look-Ahead Bias**: Using future information (e.g., "knowing" the crash happens in 2020).
3.  **Cherry-Picking**: Selecting the one lucky path out of 1000 trials.

**WFV Principle**:
> "A strategy is only valid if it can survive repeated re-optimization on past data and perform on future unseen data, maintaining its characteristics."

---

## 2. Core Methodology: Purged K-Fold CV

We use **Purged & Embargoed Walk-Forward Validation**. This is the Gold Standard (Lopez de Prado).

### 2.1. The Splitting Logic (`PurgedFoldGenerator`)

Dividing time-series is dangerous. We must ensure **Zero Leakage**.

**Structure**:
```text
[ TRAIN SEGMENT ...... ] [PURGE] [EMBARGO] [ TEST SEGMENT ]
<--------------------->  <-----> <-------> <------------>
      Optimization        Gap 1    Gap 2     Evaluation
```

1.  **Train**: Used by `Optuna` to find best parameters.
2.  **Purge**: A gap (e.g., 1 week) to prevent "Label Leakage" (e.g., entry at end of Train, exit in Test).
3.  **Embargo**: A gap after Purge to prevent "Correlation Leakage" (decay of serial correlation).
4.  **Test**: Strictly unseen data for verification.

### 2.2. Rolling vs. Anchored

*   **Rolling (Standard)**: Train window moves (e.g., "Last 2 Years"). Adapts to regime changes.
*   **Anchored**: Train window grows (Start is fixed). Good for "Infinite Memory" models.

---

## 3. Metrics: How to Judge "Success"

We do NOT look at "Total Profit". We look at **Robustness**.

### 3.1. Performance Degradation (`degradation`)
The drop in performance when moving from In-Sample (IS) to Out-of-Sample (OOS).

$$ Degradation = \frac{OOS\_Score - IS\_Score}{IS\_Score} $$

*   **Passing**: > -50% (Score drops by less than half).
*   **Warning**: -50% to -75%.
*   **failure**: < -75% (Model completely broke down).

### 3.2. Deflated Sharpe Ratio (DSR)
Adjusts the Sharpe Ratio for the "Multiple Testing Nightmare". If you try 1000 parameters, one will look good by chance.

*   **Logic**: Calculates the probability that the strategy is NOT a false positive, given the variance of all trials.
*   **Correction Note**: True DSR requires `Total Trials` = All backtests ever run (Global Count). Using only the current optimization's trial count underestimates the risk of overfitting (Bonferroni correction).
*   **Threshold**: > 95% (Ideal), > 75% (Acceptable for research).

---

## 4. Implementation Guidelines

### 4.1. The Orchestrator (`WalkForwardOptimizer`)

**Class**: `src.multi_strategy_backtest.optimization.wfv_optimizer.WalkForwardOptimizer`

**Workflow**:
1.  Generate `N` Folds (default 5).
2.  For each Fold:
    *   **Optimize** on `Train` (using `OptunaOptimizer`).
    *   **Evaluate** on `Test` (using pure Forward Test).
    *   **Store** IS Score, OOS Score, and Params.
3.  **Aggregate** results into `WFVReport`.

### 4.2. Handling "Zero" IS Scores (Strict Mode)

In our strict `objective_score`:
*   Negative Sharpe (Loss) = **0.00 Score**.
*   Low Trade Count (< 10) = **-1.0 Score** (Pruned).

**Scenario**: Strategy 05 shows `IS Score: 0.00` on 5 folds.
**Interpretation**: The strategy, across *all* tested parameter combinations, failed to generate a positive risk-adjusted return on the training data.
**Action**: **FAIL**. Do not deploy. The strategy logic itself is likely flawed or the market does not suit it.

---

## 5. Universal Rules & Pitfalls

### Rule 1: Never Leak Test Data
*   **Bad**: Calculating `global_mean = df['close'].mean()` before splitting.
*   **Good**: Calculate mean ONLY on `df.iloc[train_idx]`.

### Rule 2: Respect the Gap (Purge)
*   If your strategy holds trades for up to 3 days, `purge_bars` MUST cover at least 3 days.
*   Otherwise, a trade opened in Train could close in Test, leaking the outcome.

### Rule 3: The "Parameter Stability" Check
*   If Fold 1 Best Params = `{'ema': 10}`
*   And Fold 2 Best Params = `{'ema': 200}`
*   **Warning**: The parameter surface is unstable/chaotic.

---

## 6. Report Example (Human-Readable)

```text
WFA REPORT: Strategy_MeanReversion @ BTCUSDT
---------------------------------------------------------------
Fold | Period       | IS Score | OOS Score | Decay   | Verdict
1    | 2023-01...   | 2.50     | 2.10      | -16%    | OK
2    | 2023-04...   | 2.40     | 0.50      | -79%    | FAIL
3    | 2023-07...   | 2.60     | 2.80      | +7%     | GREAT
...
Median Decay: -25% (PASS)
DSR Confidence: 88% (ACCEPTABLE)
```

## 7. Code Reference (Universal Pattern)

```python
def walk_forward_validation(model, data, n_folds=5):
    splitter = PurgedFoldGenerator(n_folds)
    results = []
    
    for train_idx, test_idx in splitter.split(data):
        # 1. Train
        train_data = data.iloc[train_idx]
        model.fit(train_data)
        is_score = model.score(train_data)
        
        # 2. Test
        test_data = data.iloc[test_idx]
        oos_score = model.score(test_data)
        
        results.append({
            "is": is_score, 
            "oos": oos_score,
            "decay": (oos_score - is_score) / is_score
        })
        
    return aggregate_results(results)
```
