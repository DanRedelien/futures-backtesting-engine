# STRATEGY PROTOCOL MCP (Model Context Protocol)

**Role**: You are a Quantitative Engineer.
**Objective**: When creating, refactoring, or reviewing Trading Strategies, use THIS specification. This is the source of truth to prevent Logic Errors, Look-Ahead Bias, and Unoptimizable Code.

---

## 1. Philosophy: Strict Engineering

A strategy is NOT a script. It is a **Software Component** that must:
1.  **Be Optimizable**: Expose a strict parameter space (`get_opt_space`).
2.  **Be Vectorized**: Use pandas/numpy operations, not `for` loops (unless strictly necessary for stateful logic).
3.  **Be Safe**: Explicitly prevent Look-Ahead Bias, especially in multi-timeframe logic.

---

## 2. The Interface Contract (`BaseStrategy`)

Every strategy **MUST** inherit from `src.multi_strategy_backtest.strategies.base_strategy.BaseStrategy` and implement these three methods:

### 2.1. `get_name(self) -> str`
Return a unique, snake_case identifier.
*   *Bad*: `"RSI Strategy"`
*   *Good*: `"strategy_04_rsi_reversal"`

### 2.2. `generate_signals(self, data_m5, data_h1) -> pd.DataFrame`
The core logic engine.
*   **Inputs**: Receives both M5 (5-minute) and H1 (1-hour) dataframes.
*   **Outputs**: A DataFrame with a `signal` column (1, -1, 0) aligned to the **M5 index**.

### 2.3. `get_opt_space(self, trial) -> Dict`
**MANDATORY**. Defines the Bayessian Search Space.
*   Strategies without this cannot be tuned and are considered "Drafts".

### 2.4. Strict Parameter Naming (CRITICAL)
The keys returned by `get_opt_space` **MUST** match the Class Attribute names (case-insensitive).
*   *Why?* The Optimizer uses `setattr(strategy, key, value)` to inject trial parameters.
*   *Example*:
    ```python
    class MyStrategy(BaseStrategy):
        RSI_PERIOD = 14  # Class Attribute
        
        def get_opt_space(self, trial):
            # Key "RSI_PERIOD" matches attribute "RSI_PERIOD"
            return {"RSI_PERIOD": trial.suggest_int("rsi", 10, 20)}
    ```

---

## 3. Data Handling & Look-Ahead Bias (CRITICAL)

### 3.1. Vectorization Rule
Do NOT iterate over rows (`iterrows`) to calculate indicators. Use `pandas` or `ta-lib` (via `shared/indicators.py`).

*   *Bad*:
    ```python
    for i in range(len(df)):
        if df.close[i] > df.close[i-1]: ...
    ```
*   *Good*:
    ```python
    df['up'] = df['close'] > df['close'].shift(1)
    ```

### 3.2. Multi-Timeframe Merging (The Danger Zone)
When mixing H1 (Trend) with M5 (Entry), you risk using future data.

**The Problem**:
H1 bar at `10:00` covers `10:00` to `11:00`.
The timestamp is `10:00`.
If you merge this to M5 bar at `10:05`, you are using High/Low/Close that **hasn't happened yet** (it happens at 10:59).

**The Solution**:
1.  **Shift H1 Signals**: Move decision data forward by 1 bar.
2.  **Merge Backward**: Use `pd.merge_asof` with `direction='backward'`.

**Correct Pattern**:
```python
# 1. Calculate H1 Indicators
h1['ema'] = h1['close'].ewm(span=20).mean()

# 2. Shift signals/levels to PREVENT Look-Ahead
# We can only know the 10:00 bar closed at 11:00.
# So the info becomes available at 11:00.
h1_shifted = h1.shift(1) 

# 3. Merge to M5
merged = pd.merge_asof(
    data_m5,
    h1_shifted[['ema']],
    left_index=True,
    right_index=True,
    direction='backward', # Find latest PAST H1 bar
    tolerance=pd.Timedelta(hours=2)
)
```

---

## 4. Risk & Execution

Strategies generally **DO NOT** decide position size. They output **Signals** and **Stop Levels**.

### 4.1. Stop Loss & Take Profit
If the strategy has logical exit points (e.g., support/resistance), calculate them and return them.
If not, the `RiskManager` will apply default Volatility-based SL/TP.

**Pattern**:
```python
entries = pd.Series(0, index=m5.index)
stop_losses = pd.Series(0.0, index=m5.index)

entries[long_cond] = 1
stop_losses[long_cond] = m5['low'] - atr * 2.0
```

### 4.2. Delegation
Always use `src.multi_strategy_backtest.shared.risk_manager.RiskManager` to finalize the signal dataframe.

```python
rm = RiskManager(risk_per_trade=0.01)
result = rm.apply_risk_management(
    prices=m5,
    entries=entries,
    stop_losses=stop_losses
)
return result
```

---

---

## 5. Regime Awareness (HMM Integration)

Strategies should be Regime-Aware using the HMM filter.
The `BaseStrategy.generate_signals` method receives `regime_probs` (P(Stress) series).

### 5.1. Tuning Parameter
Always expose `regime_filter` in `get_opt_space`:
-   `0`: Calm Only (Trade only when P(Stress) < 0.5)
-   `1`: Stress Only (Trade only when P(Stress) >= 0.5)
-   `2`: All Regimes (Ignore filter)

### 5.2. Implementation Pattern
Filter logic MUST happen at the end of signal generation.

```python
# ... calculate entries (1/-1) ...

# Apply Regime Filter
# 0=Calm, 1=Stress, 2=All
regime_param = self.params.get('regime_filter', 2)

if regime_param == 0:  # Calm Only
    entries[regime_probs >= 0.5] = 0
elif regime_param == 1: # Stress Only
    entries[regime_probs < 0.5] = 0
    
# ... pass filtered entries to RiskManager ...
```

---

## 6. Optimization Protocol

When defining `get_opt_space`:
1.  **Be Strict**: Do not allow infinite ranges. Use `trial.suggest_int` or `suggest_float` limits.
2.  **Dependencies**: Enforce logical relationships (e.g., `slow_ma > fast_ma`).

```python
def get_opt_space(self, trial):
    fast = trial.suggest_int("fast_ma", 10, 50)
    slow = trial.suggest_int("slow_ma", fast + 10, 200) # Valid dependency
    return {"FAST": fast, "SLOW": slow}
```

---

## 6. Implementation Checklist

Before submitting a strategy:

- [ ] **Inheritance**: Inherits `BaseStrategy`?
- [ ] **Interface**: Implements `get_opt_space`?
- [ ] **Naming**: Do `get_opt_space` keys match Class field names?
- [ ] **Vectorization**: Uses `pandas` / `numpy` for core logic?
- [ ] **Safety**: If using H1 data, is it `shift(1)` before merge?
- [ ] **NaN Handling**: Are `fillna` or `dropna` used? (Strategies must return aligned signals).
- [ ] **Risk**: Does it use `RiskManager` to format final output?

---

## 7. Example References

-   **Simple**: `src/multi_strategy_backtest/strategies/strategy_01` (Single TF).
-   **Complex**: `src/multi_strategy_backtest/strategies/strategy_05` (Multi-TF, Pattern Recognition).
-   **Optimization**: `src/multi_strategy_backtest/optimization` (The consumer of these strategies).
