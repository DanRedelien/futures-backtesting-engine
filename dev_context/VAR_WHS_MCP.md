# VAR + WHS MCP: Adaptive Value-at-Risk Engine

**Role**: You are a Risk Quant.
**Objective**: When asked to "Build a VaR model" or "Calculate portfolio risk" — use THIS specification. This is the production-tested reference for Regime-Adaptive VaR with Weighted Historical Simulation.

---

## 1. Problem Statement

Traditional Historical Simulation VaR treats all historical observations equally:
$$VaR_{0.95} = \text{Percentile}_5(r_{t-W}, ..., r_{t-1})$$

**Problem**: A crisis from 200 days ago has the same weight as a calm day from yesterday.

**Solution**: Weight historical observations by their **Regime Similarity** to the current market state:
$$w_k = 1 - |P(Stress_t) - P(Stress_{t-k})|$$

This is **Weighted Historical Simulation (WHS)** with Regime Similarity.

---

## 2. Core Methodology

### 2.1. WHS Algorithm (Step-by-Step)

```
INPUT:
  - df: Price DataFrame with 'close' column
  - regime_probs: P(Stress) series from HMM
  - horizon: Holding period (e.g., 1 day)
  - window: Lookback (e.g., 252 days)
  - confidence: VaR confidence (e.g., 0.95)

FOR each day t in [window, T]:
    1. current_prob ← P(Stress_t)
    
    2. historical_returns ← returns from [t-window, t-1]
       historical_probs ← P(Stress) from [t-window, t-1]
    
    3. FOR each historical observation k:
         similarity_k ← 1 - |current_prob - historical_probs[k]|
       
    4. weights ← normalize(similarities) so Σweights = 1
    
    5. VaR_t ← Weighted Percentile(historical_returns, weights, q=0.05)
       CVaR_t ← Weighted Mean(returns where return <= VaR_t)
    
    6. ESS_t ← 1 / Σ(weights²)  # Effective Sample Size
       IF ESS_t < threshold: LOG WARNING

OUTPUT: VaR series, CVaR series, ESS series
```

### 2.2. Mathematical Formulas

**Similarity Weight**:
$$w_k = \frac{1 - |P_t^{stress} - P_{t-k}^{stress}|}{\sum_{j=1}^{W}(1 - |P_t^{stress} - P_{t-j}^{stress}|)}$$

**Effective Sample Size (ESS)**:
$$ESS = \frac{1}{\sum_{k=1}^{W} w_k^2}$$

If $ESS \ll W$, weights are concentrated on a few observations → unstable VaR.

**Weighted Percentile** (for VaR):
1. Sort returns ascending: $r_{(1)} \le r_{(2)} \le ... \le r_{(W)}$.
2. Sort weights correspondingly: $w_{(1)}, w_{(2)}, ..., w_{(W)}$.
3. Cumulative weights: $C_k = \sum_{i=1}^{k} w_{(i)}$.
4. Find $k^*$ where $C_{k^*} \ge \alpha$ (e.g., 0.05).
5. Interpolate: $VaR = r_{(k^*-1)} + \frac{\alpha - C_{k^*-1}}{C_{k^*} - C_{k^*-1}}(r_{(k^*)} - r_{(k^*-1)})$.

**Weighted CVaR (Expected Shortfall)**:
$$CVaR = \frac{\sum_{k: r_k \le VaR} w_k \cdot r_k}{\sum_{k: r_k \le VaR} w_k}$$

---

## 3. Implementation (from `var_model.py`)

### 3.1. Core Functions

| Function | Purpose |
|----------|---------|
| `_calc_similarity_weights` | Computes normalized weights from regime similarity |
| `_calc_effective_sample_size` | Monitors weight concentration |
| `_calc_weighted_quantile` | Weighted VaR and CVaR calculation |
| `_calc_whs_var` | Full WHS VaR for one time step |
| `calc_adaptive_var` | Main entry point (loop over all dates) |

### 3.2. Key Code Snippets

**Similarity Weights Calculation**:
```python
def _calc_similarity_weights(self, current_prob: float, historical_probs: np.ndarray) -> np.ndarray:
    """
    Methodology (Variant A - Similarity):
    - Similarity_k = 1 - |P(Stress_t) - P(Stress_{t-k})|
    - Weight w_k ∝ Similarity_k (normalized to sum to 1)
    """
    similarities = 1.0 - np.abs(current_prob - historical_probs)
    similarities = np.maximum(similarities, 0.0)  # Non-negative
    
    total = similarities.sum()
    if total > 0:
        return similarities / total
    else:
        return np.ones(len(similarities)) / len(similarities)  # Fallback: equal weights
```

**Weighted Quantile Calculation**:
```python
def _calc_weighted_quantile(self, returns: np.ndarray, weights: np.ndarray, quantile: float):
    """
    Linear interpolation for weighted percentile.
    """
    sorted_indices = np.argsort(returns)
    sorted_returns = returns[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cumsum_weights = np.cumsum(sorted_weights)
    var_idx = np.searchsorted(cumsum_weights, quantile)
    
    # Interpolation for exact quantile
    if var_idx == 0:
        var_value = sorted_returns[0]
    else:
        w0, w1 = cumsum_weights[var_idx - 1], cumsum_weights[var_idx]
        t = (quantile - w0) / (w1 - w0)
        var_value = (1 - t) * sorted_returns[var_idx - 1] + t * sorted_returns[var_idx]
    
    # CVaR: weighted mean of tail
    tail_mask = sorted_returns <= var_value
    cvar_value = np.sum(sorted_returns[tail_mask] * sorted_weights[tail_mask]) / sorted_weights[tail_mask].sum()
    
    return var_value, cvar_value
```

### 3.3. Fallback: Legacy Regime Blending

If WHS fails (not enough regime data), fall back to probability-weighted blending:
```python
# Legacy Mode: Simple Blend
VaR_blended = P(Calm) * VaR_calm_window + P(Stress) * VaR_stress_window
```

Where:
*   `VaR_calm_window` = standard HS VaR over `window_calm` (252 days).
*   `VaR_stress_window` = standard HS VaR over `window_stress` (90 days).

---

## 4. Configuration Parameters (from `settings.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `var_confidence` | 0.95 | VaR confidence level (0.95 = 95% VaR) |
| `window_calm` | 252 | Basel-standard lookback for Calm regime HS |
| `window_stress` | 90 | Reactive lookback for Stress regime HS |
| `enable_whs` | True | Enable Regime-Similarity WHS (else use Legacy Blending) |
| `ess_threshold` | 10 | Minimum ESS before warning (weight concentration) |
| `horizons` | [1, 5, 21] | Holding periods to analyze (days) |
| `capital` | 100,000 | Portfolio capital for USD conversion |

---

## 5. Portfolio Construction

### 5.1. CRITICAL Math Note

**You CANNOT sum weighted log returns directly!**
$$\ln(1 + \sum_i w_i R_i) \neq \sum_i w_i \ln(1 + R_i)$$

This approximation error **explodes** during stress (-20% moves).

**Correct Procedure**:
1. Use **Simple Returns** for aggregation: $R_{port} = \sum_i w_i \cdot R_i^{simple}$.
2. Convert to **Log Returns** only after aggregation: $r_{port} = \ln(1 + R_{port})$.

**Code** (from `var_model.py`):
```python
# Correct: Aggregate simple returns, then convert to log
portfolio_simple_returns = sum(weight * simple_return[asset] for asset in portfolio)
portfolio_log_returns = np.log(1 + portfolio_simple_returns.clip(lower=-0.9999))
```

### 5.2. Rebalancing Modes

| Mode | Setting | Behavior |
|------|---------|----------|
| **Standard VaR** | `enable_rebalancing=False` | Daily reset to target weights. No drift, no fees. |
| **Realistic Backtest** | `enable_rebalancing=True` | Weights drift between rebalance dates. Fees deducted. |

**Rebalance Logic**:
```python
turnover = sum(|current_weight - target_weight|)
fee = portfolio_value * turnover * fee_rate_bps / 10000
portfolio_value -= fee
```

---

## 6. Output DataFrame Schema

`calc_adaptive_var()` returns a DataFrame with columns:

| Column | Description |
|--------|-------------|
| `realized_return` | Actual forward return (for backtest) |
| `var_forecast` | WHS VaR at time t |
| `cvar_forecast` | WHS CVaR (Expected Shortfall) |
| `var_calm` | Legacy Calm-window VaR (reference) |
| `var_stress` | Legacy Stress-window VaR (reference) |
| `regime_prob` | P(Stress) from HMM |
| `ess` | Effective Sample Size |
| `var_usd` | VaR in USD: `capital * (exp(var) - 1)` |
| `realized_usd` | Realized PnL in USD |

---

## 7. Validation & Backtesting

### 7.1. Breach Rate Analysis

**Target**: At 95% VaR, expected breach rate = 5%.

```python
breaches = df[df['realized_return'] < df['var_forecast']]
breach_pct = len(breaches) / len(df) * 100
```

**Interpretation**:
*   `breach_pct > 1.5 * target`: Model is **too aggressive** (underestimates risk).
*   `breach_pct < 0.5 * target`: Model is **too conservative** (wastes capital).

### 7.2. Statistical Tests (from `analytics.py`)

**Kupiec POF Test** (Unconditional Coverage):
*   H0: Observed breach rate = Expected breach rate.
*   p-value > 0.05 → PASS.

**Christoffersen Test** (Clustering/Independence):
*   H0: Breaches are independent (no clustering).
*   p-value > 0.05 → PASS.

---

## 8. Full Class Reference

```python
class RiskEngine:
    """
    Core Risk Calculation Engine.
    
    Handles:
    1. Adaptive VaR (WHS or Legacy Blending).
    2. Portfolio Construction (Static or Dynamic Rebalancing).
    3. Correlation Analysis.
    4. Performance Metrics.
    """
    
    def __init__(self, settings: Settings) -> None: ...
    
    def construct_synthetic_portfolio(
        self, assets_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
    
    def calc_adaptive_var(
        self, df: pd.DataFrame, regime_probs: pd.Series, horizon: int
    ) -> pd.DataFrame: ...
    
    def get_performance_summary(self, data: pd.DataFrame) -> Dict: ...
```

---

## 9. Production Checklist

- [ ] **Log Returns**: Use `np.log(P_t / P_{t-1})`, NOT simple returns for VaR.
- [ ] **Horizon Scaling**: For multi-day VaR, use overlapping returns or sqrt-scaling (approximation).
- [ ] **ESS Monitoring**: Log warning if ESS < threshold.
- [ ] **UTC Timestamps**: Ensure all DataFrames use UTC DatetimeIndex.
- [ ] **Capital Sync**: `settings.capital` must match actual portfolio value.
- [ ] **Backtest Validation**: Run Kupiec + Christoffersen before production.

---

## 10. Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Aggressive WHS** | Breach rate ~10% vs 5% target | Tune similarity decay, add time-decay component |
| **Long-Only Bias** | VaR focuses on downside | Use separate model for shorts |
| **Low ESS** | Unstable VaR swings | Widen similarity threshold or fallback to Legacy |
| **Overlapping Returns** | Autocorrelation in multi-day VaR | Use block bootstrap or non-overlapping windows |

---

## 11. Credits & References

*   **Boudoukh, Richardson, Whitelaw (1998)**: WHS original paper.
*   **Basel III**: Stressed VaR window requirements.
*   **Project Implementation**: `src/hmm_var/var_model.py`
