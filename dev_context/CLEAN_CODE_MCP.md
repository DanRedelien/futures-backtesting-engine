# CLEAN CODE MCP: The Codex

**Objective**: To maintain Institutional-Grade quality in a multi-agent environment.
**Status**: MANDATORY.

## 1. The Golden Rules (Clean Code)

### A. Language & Tone
*   **Rule**: **ENGLISH ONLY**. No Russian/Cyrillic comments.
*   **Reasoning**: Code must be auditable by international teams.
*   **Example**:
    ```python
    # BAD
    # считаем среднее скользящее
    
    # GOOD
    # Calculate rolling mean for volatility estimation
    ```

### B. Type Safety
*   **Rule**: All function signatures MUST have type hints (`List`, `Dict`, `Optional`, `pd.DataFrame`).
*   **Rule**: Use `pydantic` for complex data structures instead of raw dicts where possible.
*   **Example (`src/hmm_var/var_model.py`)**:
    ```python
    def calculate_var(
        self, 
        returns: pd.Series,              # Explicit type
        confidence: float = 0.95         # Default value
    ) -> Tuple[float, float]:            # Explicit return type
        ...
    ```

### C. No Magic Numbers
*   **Rule**: Never hardcode parameters (windows, thresholds) in logic files.
*   **Solution**: Use `settings.py`.
*   **Example**:
    ```python
    # BAD
    window = 180 
    
    # GOOD (from src/hmm_var/var_model.py)
    window = self.settings.window_stress 
    ```

### D. Resilient Error Handling
*   **Research/Backtest**: For non-critical failures, **Log & Continue**.
*   **Live Trading (Circuit Breaker)**: Steps must FAIL if errors > Threshold.
    *   *Rule*: If SVD fails 5 times in a row -> **STOP TRADING**. Do not trade on stale data.

*   **Example (`src/multi_strategy_backtest/analytics/regime.py`)**:
    ```python
    # BAD
    print(f"SVD Failed at {t}") # Prints 500 times
    
    # GOOD
    failures += 1
    # ... after loop ...
    if failures > 0:
        print(f"[HMM] Fit failed: SVD did not converge [{failures}]")
    ```

---

## 2. Documentation Standards

### A. Docstrings (Google Style)
Every class and public method needs a docstring explaining **Why**, not just **What**.

**Template**:
```python
def method_name(args):
    """
    [One-line summary].
    
    [Methodology/Financial Logic]: Explain the math or business reason.
    
    Args:
        arg_name: Description.
        
    Returns:
        Description.
    """
```

**Real Project Example (`src/hmm_var/var_model.py`)**:
```python
def construct_synthetic_portfolio(self, assets_data: Dict[str, pd.DataFrame]):
    """
    Constructs a Synthetic Portfolio History based on current weights.
    
    Methodology:
    1. If enable_rebalancing=False: Weights are reset daily (Standard VaR).
    2. If enable_rebalancing=True: Weights drift; fees are deducted (Realistic).
    
    CRITICAL MATH NOTE:
    ln(1 + Σw_i*R_i) ≠ Σw_i*ln(1+R_i) - We must use simple returns for aggregation!
    """
```

---

## 3. Project Vocabulary (Glossary)

*   **Calm Regime**: Low volatility state. Uses `window_calm` (default 365d, Basel standard).
*   **Stress Regime**: High volatility/clustering state. Uses `window_stress` (default 180d, Reactive).
*   **WHS (Weighted Historical Simulation)**: A VaR method where historical observations are weighted by their "Regime Similarity" to today, rather than equally.
*   **Breakout**: A risk failure event where `Realized Loss > VaR Forecast`.
