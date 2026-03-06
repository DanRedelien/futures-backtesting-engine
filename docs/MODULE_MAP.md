# Module Map — Portfolio Layer

One-screen reference per subpackage: inputs, outputs, and dependencies.

---

## `domain/`
**Purpose:** Pure data structures — no I/O, no computation.

| Module | Class | In | Out |
|---|---|---|---|
| `contracts.py` | `PortfolioConfig` | YAML (via cli/) | config object |
|  | `StrategySlot` | YAML slot entries | slot object |
| `signals.py` | `StrategySignal` | StrategyRunner | direction + metadata |
|  | `TargetPosition` | Allocator | target qty |
| `policies.py` | `RebalancePolicy` | config string | enum value |
|  | `ExecutionPolicy` | PortfolioConfig | commission + slippage params |

**Dependencies:** `dataclasses`, `enum`, `BacktestSettings` (lazy import for default)

---

## `adapters/`
**Purpose:** Bridge legacy single-asset `BaseStrategy` to portfolio engine.

| Module | Class | In | Out |
|---|---|---|---|
| `legacy_strategy_adapter.py` | `LegacyStrategyAdapter.build()` | strategy_class, data, symbol, settings, params | instantiated strategy |

**Key limitation:** `strategy.engine.portfolio.positions` is a per-instance
mock dict, not the real `PortfolioBook`.  See module docstring for full limitations.

**Dependencies:** `_MockEngine`, `_MockPortfolio`, `_PatchedSettings` (all internal)

---

## `scheduling/`
**Purpose:** Rebalance gate — decides whether Allocator runs on a given bar.

| Class | Fires on | Use case |
|---|---|---|
| `IntrabarScheduler` | Every bar | Intraday strategies |
| `DailyScheduler` | First bar of each calendar day | Daily signal strategies |

**Factory:** `make_scheduler(frequency: str) → BaseScheduler`

**Dependencies:** stdlib only

---

## `allocation/`
**Purpose:** Weight-based capital sizing.

| Method | In | Out |
|---|---|---|
| `Allocator.compute_targets()` | `List[StrategySignal]`, total_equity, prices, specs | `List[TargetPosition]` |

**Formula:** `floor(equity × weight × leverage / (price × multiplier)) × direction`

**Dependencies:** `domain/contracts.py`, `domain/signals.py`

---

## `execution/`
**Purpose:** Fills and portfolio ledger.

| Module | Class | Key method | Effect |
|---|---|---|---|
| `portfolio_book.py` | `PortfolioBook` | `apply_fill()` | Updates cash + positions |
|  | | `mark_to_market()` | Recomputes equity |
|  | | `record_snapshot()` | Appends to history |
| `strategy_runner.py` | `StrategyRunner` | `collect_signals()` | Calls on_bar(), returns signals |

**Dependencies:** `adapters/legacy_strategy_adapter.py`, `domain/`, `src.backtest_engine.execution.Order`

---

## `engine/`
**Purpose:** Event loop orchestration.

| Method | Steps |
|---|---|
| `PortfolioBacktestEngine.run()` | Load data → union timeline → A/B/C/D/E/F steps per bar |
| `show_results()` | PerformanceMetrics + save_portfolio_results() |

**Dependencies:** all other subpackages, `DataLake`, `ExecutionHandler`, `BacktestSettings`

---

## `reporting/`
**Purpose:** Serialize 5 artifacts to `results/portfolio/`.

| Artifact | Description |
|---|---|
| `history.parquet` | Bar-by-bar equity curve |
| `exposure.parquet` | Gross / net holdings per bar |
| `strategy_pnl_daily.parquet` | Per-slot daily PnL |
| `trades.parquet` | All completed round-trip trades |
| `metrics.json` | Scalar performance metrics |
| `report.txt` | Human-readable terminal output |

**Dependencies:** `pandas`, `json`, stdlib

---

## `cli/`
**Purpose:** CLI command handlers (one file per mode).

| Module | Flag | Imports |
|---|---|---|
| `cli/single.py` | `--backtest` | `BacktestEngine` |
| `cli/wfo.py` | `--wfo` | `WalkForwardOptimizer` |
| `cli/portfolio.py` | `--portfolio-backtest` | `PortfolioBacktestEngine`, `PortfolioConfig` |

**Each module has its own strategy registry** — intentionally decoupled so adding
a new strategy to one mode doesn't require touching other modes.
