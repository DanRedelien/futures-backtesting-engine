# Tests

## Structure
- `unit/` - isolated logic and UI-contract tests.
- `regression/` - scenario and behavior invariants that should stay stable over time.
- Root-level files - broader integration and invariant coverage.

## Running
```bash
pytest tests/
pytest tests/unit/
pytest tests/unit/test_metrics.py
```

## Notes
- Set `PYTHONPATH=.` when running `pytest` directly in environments that do not auto-resolve the `src/` package.
- Shared artifact writers and job builders live in `tests/conftest.py`.

## Test Inventory
| File | Purpose |
| --- | --- |
| `test_artifact_loading.py` | Validates artifact loading, integrity states, and rerun metadata rules. |
| `test_terminal_ui_app.py` | Covers terminal shell rendering, partial responses, and chart payload contracts. |
| `test_terminal_ui_operations.py` | Covers cache-key format, job metadata persistence, and operations endpoints. |
| `test_pnl_transforms.py` | Validates canonical PnL and risk transform behavior. |
| `test_engine_regressions.py` | Protects engine and artifact regressions that must stay stable after refactors. |
| `test_portfolio_book.py` | Verifies portfolio book accounting and position tracking behavior. |
| `test_scheduler.py` | Verifies scheduling and rebalance timing logic. |
| `test_metrics.py` | Validates core performance metric calculations. |
| `test_allocator.py` | Covers portfolio capital allocation rules. |
| `test_risk_transforms.py` | Covers risk transform outputs and derived payload semantics. |
| `test_invariants.py` | Checks cross-module invariants and correctness contracts. |
| `test_engine_integration.py` | Exercises broader engine integration behavior. |
| `test_beta.py` | Covers beta-related analytics behavior. |
| `test_kalman.py` | Covers Kalman filter analytics behavior. |
| `regression/test_exit_signals.py` | Verifies regression coverage for exit signal generation. |
