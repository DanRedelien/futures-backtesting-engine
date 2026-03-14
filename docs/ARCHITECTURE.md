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
└── portfolio.py                    --portfolio-backtest handler + scenario metadata wiring

src/
├── backtest_engine/
│   ├── settings.py                 BacktestSettings + TerminalUISettings
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
│   ├── analytics/                  Post-execution metrics, reports, exporters, and UI adapters
│   │   ├── exporter.py             Writes artifact bundles to results/
│   │   ├── report.py               Terminal report formatter
│   │   ├── exit_analysis.py        MFE/MAE and trade-path enrichment
│   │   ├── artifact_contract.py    Artifact identity and schema helpers
│   │   ├── dashboard/              Legacy analytics views and canonical transform layer
│   │   │   ├── core/               Result loading, path helpers, transforms
│   │   │   ├── pnl_analysis/       Legacy PnL chart helpers
│   │   │   ├── risk_analysis/      Canonical risk models and views
│   │   │   └── simulation_analysis/ Reserved placeholder, backlog only
│   │   └── terminal_ui/            FastAPI + HTMX terminal dashboard
│   │       ├── app.py              App factory and route registration
│   │       ├── service.py          Runtime context and artifact-loading helpers
│   │       ├── chart_builders.py   JSON chart payload builders
│   │       ├── risk_builders.py    Risk panel and risk chart payload builders
│   │       ├── table_builders.py   Shell context and table payload builders
│   │       ├── routes_*.py         Partial, chart, and operations route modules
│   │       ├── cache.py            Redis/local TTL cache wrapper
│   │       ├── jobs.py             Async scenario job metadata and queue service
│   │       ├── templates/          Jinja shell and HTMX partials
│   │       └── static/             Terminal JS/CSS assets
│   │
│   └── optimization/               Walk-Forward Optimizer (WFO) & Validation
│
├── strategies/                     Single-asset strategy implementations
│   ├── base.py                     BaseStrategy contract
│   └── *.py                        SmaCrossover, ZScore, ICT-OB, ...
│
└── data/
    └── data_lake.py                DataLake.load() → OHLCV DataFrame

tests/
├── unit/                           Isolated logic, artifact, and UI-contract tests
├── regression/                     No-lookahead + exit-signal correctness
└── *.py                            Broader integration and invariant coverage
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
reporting/results.py      writes artifact bundle to results/portfolio/
    ↓
dashboard/core/data_layer.py or terminal_ui/service.py
    ↓
terminal_ui/app.py        serves HTML partials + JSON chart payloads
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
`tests/unit/test_portfolio_book.py::test_shared_capital_invariant`.

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

---

## Terminal UI Notes

- The terminal UI is the active web dashboard surface.
- The `dashboard/` package still holds canonical loaders, transforms, and some legacy views, but new UI work should land under `analytics/terminal_ui/`.
- The visible dashboard version string lives in `terminal_ui/templates/dashboard.html`.
- Significant terminal UI updates should bump the visible version by `+0.1`:
  - `3.0 -> 3.1`
  - `3.1 -> 3.2`
