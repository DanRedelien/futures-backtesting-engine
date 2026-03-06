# Architecture Overview

This document describes the top-level structure of the backtesting platform
and how its major components interact.

---

## Module Map

```
run.py                              Thin CLI entrypoint (argparse + dispatch only)

cli/
├── single.py                       --backtest handler
├── wfo.py                          --wfo handler
└── portfolio.py                    --portfolio-backtest handler

src/
├── backtest_engine/
│   ├── settings.py                 BacktestSettings (pydantic-settings, env + .env)
│   ├── engine.py                   Single-asset BacktestEngine
│   ├── execution.py                ExecutionHandler + Trade + Order
│   │
│   ├── portfolio_layer/            Multi-asset / multi-strategy engine
│   │   ├── __init__.py             Public API re-exports
│   │   │
│   │   ├── domain/                 ── Pure data structures (no I/O) ──────────────
│   │   │   ├── contracts.py        PortfolioConfig, StrategySlot
│   │   │   ├── signals.py          StrategySignal, TargetPosition
│   │   │   └── policies.py         RebalancePolicy (enum), ExecutionPolicy
│   │   │
│   │   ├── adapters/               ── Legacy strategy bridge ───────────────────
│   │   │   └── legacy_strategy_adapter.py   _MockEngine, LegacyStrategyAdapter
│   │   │                           See LIMITATIONS in that module's docstring.
│   │   │
│   │   ├── allocation/             ── Capital sizing ────────────────────────────
│   │   │   └── allocator.py        Allocator.compute_targets()
│   │   │
│   │   ├── scheduling/             ── Rebalance gate ────────────────────────────
│   │   │   └── scheduler.py        IntrabarScheduler, DailyScheduler, make_scheduler
│   │   │
│   │   ├── execution/              ── Fill + ledger ─────────────────────────────
│   │   │   ├── portfolio_book.py   PortfolioBook (cash + MtM accounting)
│   │   │   └── strategy_runner.py  StrategyRunner (signal collection)
│   │   │
│   │   ├── engine/                 ── Event loop ────────────────────────────────
│   │   │   └── engine.py           PortfolioBacktestEngine
│   │   │
│   │   └── reporting/              ── Result serialisation ──────────────────────
│   │       └── results.py          save_portfolio_results() → 5 artifacts
│   │
│   ├── analytics/                  Dashboard, charts, performance metrics
│   └── optimization/               Walk-Forward Optimizer (WFO)
│
├── strategies/                     Single-asset strategy implementations
│   ├── base.py                     BaseStrategy contract
│   └── *.py                        SmaCrossover, ZScore, ICT-OB, ...
│
└── data/
    └── data_lake.py                DataLake.load() → OHLCV DataFrame

tests/
├── unit/                           Isolated logic tests (no engine)
├── integration/                    Engine with realistic data scenarios
└── regression/                     No-lookahead + exit-signal correctness
```

---

## Data Flow — Portfolio Backtest

```
YAML config
    ↓
cli/portfolio.py          parses YAML → PortfolioConfig
    ↓
PortfolioBacktestEngine   loads data via DataLake
    ↓
Union-timeline bar loop
  ├── [Open t]   ExecutionHandler fills pending orders → PortfolioBook.apply_fill()
  ├── [Close t]  PortfolioBook.mark_to_market()
  ├── [Gate]     Scheduler.should_rebalance(ts)?
  │   └── Yes → StrategyRunner.collect_signals() → List[StrategySignal]
  │           → Allocator.compute_targets()     → List[TargetPosition]
  │           → _compute_orders(deltas)
  └── [t+1]  Orders queued for next bar
    ↓
reporting/results.py      writes 5 artifacts to results/portfolio/
```

---

## No-Lookahead Contract

Signal generated at **close[t]** → order fills at **open[t+1]**.  
This is identical to the single-asset `BacktestEngine` (`engine.py`).  
Gap bars (symbols with no data at union-timeline step) do NOT cause order
loss — pending orders are carried forward to the next available bar.

---

## Shared-Capital Invariant

```
total_equity == cash + Σ(qty × last_known_price × multiplier)
```

This holds at every snapshot step.  Validated by
`tests/unit/test_portfolio_book.py::test_shared_capital_accounting`.

---

## Settings Layering

```
.env  →  BacktestSettings (pydantic-settings, prefix QUANT_BACKTEST_)
             │
portfolio_config.yaml → PortfolioConfig (overrides per-run)
             │
_PortfolioSettingsAdapter (inside engine.py) bridges the two
into ExecutionHandler's expected interface
```
