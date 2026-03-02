# VALIDATION PROTOCOL: QA & Risk Assessment

**Objective**: Ensure the Risk Model tells the truth, not what we want to hear.
**Status**: MANDATORY for all Model Releases.

## 1. Statistical Validation Tests
Every VaR model MUST pass these two tests before deployment.

### A. Kupiec POF Test (Unconditional Coverage)
**Purpose**: Checks if the *number* of breaches is consistent with the confidence level.
**Null Hypothesis ($H_0$)**: $ObservedBreachRate == TargetBreachRate$.
*   **Result**: p-value > 0.05 $\rightarrow$ **PASS**.
*   **Result**: p-value < 0.05 $\rightarrow$ **FAIL** (Model is either too risky or too conservative).

### B. Christoffersen Test (Independence)
**Purpose**: Checks if breaches are *clustered* (e.g., failing 3 days in a row).
**Null Hypothesis ($H_0$)**: Breaches are independent.
*   **Result**: p-value > 0.05 $\rightarrow$ **PASS**.
*   **Failure**: Implies the model does not adapt fast enough to volatility clusters.

**Implementation**: See `src/hmm_var/analytics.py::_christoffersen_test`.

---

## 2. Backtesting Metrics
Report these metrics for every horizon ($H=1, 5, 21$).

| Metric | Acceptable Range (95% VaR) | Explanation |
|:-------|:---------------------------|:------------|
| **Breach Rate** | 4.0% - 6.0% | Ideal is 5.0%. <4% is Capital Inefficient. >6% is Dangerous. |
| **Avg Quantile** | N/A | Monitor average VaR $ value to track "cost of insurance". |
| **Recovery** | < 1.5x Market | How fast does VaR normalize after a crash? |

---

## 3. Side-by-Side Comparison (Regression Testing)
**Rule**: Never replace a model without running it against the Legacy version.

**Procedure**:
1.  Run `Legacy` (Old Code) on Dataset X.
2.  Run `Candidate` (New Code) on Dataset X.
3.  Compare:
    *   Did Breach Rate improve (move closer to target)?
    *   Did Volatility of VaR increase? (Ideally, we want stable VaR, not jumpy VaR).

**Sanity Checks**:
*   If `New_VaR` is consistently **50% lower** than `Old_VaR`, assume ERROR. Markets haven't changed, your math has.
*   If `Kupiec p-value` drops from 0.30 to 0.01, REJECT the change.

---

## 4. Operational Guardrails
**Data Quality**:
*   **Gap Check**: If `Time(t) - Time(t-1) > 2 * Timeframe`, flag WARNING.
*   **Zeros**: If `Price == 0` or `NaN`, Drop or Forward Fill. DO NOT compute Returns.

**Code Example**:
```python
# src/hmm_var/data_fetcher.py
if df.isnull().values.any():
    logger.warning("NaN values detected. Forward filling...")
    df.fillna(method='ffill', inplace=True)
```
