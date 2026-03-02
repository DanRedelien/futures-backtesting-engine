# MATH SPEC: Financial Mathematics Standards

**Status**: IMMUTABLE.
**Enforcement**: Code reviews must reject any PR violating these formulas.

## 1. Return Calculations
The most common error in quantitative finance is mixing up Logarithmic and Simple returns.

### A. Log Returns (Time-Series / statistical Modeling)
**Use Case**: Accumulating returns over time, Distribution analysis, HMM/Regime Detection.
**Formula**: $r_t = \ln(P_t / P_{t-1})$
**Properties**: Time-additive ($r_{0 \to T} = \sum r_t$).
**Code**:
```python
# CORRECT
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
```

### B. Simple Returns (Cross-Sectional / Portfolio)
**Use Case**: Portfolio aggregation of assets.
**Formula**: $R_t = \frac{P_t}{P_{t-1}} - 1$
**Properties**: Asset-additive ($R_{port} = \sum w_i R_i$).
**WARNING**: $\ln(\sum w_i e^{r_i}) \neq \sum w_i r_i$. You CANNOT sum log returns across assets.

**Real Project Example (`src/hmm_var/var_model.py`)**:
```python
# 1. Calculate Simple Returns for each asset
asset_ret[symbol] = (df['close'] / df['close'].shift(1)) - 1

# 2. Weighted Sum (Linear)
portfolio_simple_returns += aligned_simple_returns[symbol] * weight

# 3. Convert back to Log Returns for VaR Engine
portfolio_log_returns = np.log(1 + portfolio_simple_returns)
```

---

## 2. Annualization Constants
**Rule**: Be explicit. Do not assume.

*   **Crypto**: `365` days. Crypto never sleeps.
*   **TradFi**: `252` days.
*   **Intraday**: Do not annualize unless necessary. Use "per-interval" volatility.

**Code**:
```python
# settings.py
ANNUALIZATION_FACTOR = 365 if asset_class == 'CRYPTO' else 252
vol_ann = vol_daily * np.sqrt(ANNUALIZATION_FACTOR)
```

---

## 3. Weighted Historical Simulation (WHS)
**Methodology**: Variant A (Similarity Weighting).

**Formula**:
1.  Calculate Stress Probability $P(S_t)$ for today.
2.  For each historical day $k$, calculate Similarity: $Sim_k = 1 - |P(S_t) - P(S_k)|$.
3.  Normalize weights: $w_k = \frac{Sim_k}{\sum Sim}$.
4.  VaR = Weighted Quantile of historical returns using $w_k$.

**Edge Case**:
*   If **Effective Sample Size (ESS)** $< 10$, the weights are too concentrated. Fallback to standard HS or warn.
*   **ESS Formula**: $ESS = \frac{1}{\sum w_i^2}$.

---

## 4. Rebalancing Logic
**Drift**: Weights change as prices move.
$w_{i,t} = \frac{n_i P_{i,t}}{V_t}$ where $n_i$ is number of units.

**Transaction Fees**:
Cost is incurred on the **Turnover** (change in weight).
$Cost = V_t \times \sum |w_{new} - w_{old}| \times FeeRate$
$V_{post} = V_{pre} - Cost$

**Implementation Note**:
Apply fee impact to the *Closing Value* of the rebalance day to capture the drag immediately.
