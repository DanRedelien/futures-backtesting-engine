# Backtesting Platform

Futures-focused research platform for:

- single-strategy backtests
- walk-forward optimization
- multi-strategy portfolio backtests
- terminal-style analytics review via FastAPI

The codebase is optimized for research workflows where no-lookahead execution, artifact reproducibility, and post-run analytics matter more than framework breadth.

## Screenshots

Add real image links after you capture them. Keep the images reasonably wide and focused on one workflow each.

### Terminal UI Overview

<img width="1593" height="734" alt="image" src="https://github.com/user-attachments/assets/a1a4cc0b-1a42-43bb-837f-b039d2e12b3f" />


```md
![Terminal UI overview](docs/screenshots/terminal-ui-overview.png)
```

### Portfolio Analytics

Portfolio backtest results with shared-capital behavior visible. Prefer a screen that shows allocation, equity curve, or portfolio-level tables.

```md
![Portfolio analytics](docs/screenshots/portfolio-analytics.png)
```

### Exit Analysis

One detailed analytics screen that demonstrates post-run inspection depth. Good candidates are exit breakdown, execution quality, or trade log detail.

```md
![Exit analysis](docs/screenshots/exit-analysis.png)
```

### Stress Testing / Scenario Queue

If you want to highlight the more advanced workflow, show the stress-testing tab or scenario job flow. Skip this screenshot if you do not want Redis-backed features in the first impression.

```md
![Stress testing workflow](docs/screenshots/stress-testing.png)
```

### CLI Workflow

Terminal output from a real run. Use a clean example that shows the project is usable without the UI as well.

```md
![CLI workflow](docs/screenshots/cli-workflow.png)
```

## What This Project Does

- Runs single-asset event-driven backtests on cached OHLCV futures data.
- Runs walk-forward validation through Optuna-based parameter search.
- Runs portfolio backtests with shared capital, rebalancing, allocation, and slot-level execution.
- Persists canonical artifacts for later inspection in the terminal UI.
- Supports scenario reruns and portfolio-level analytics from saved artifacts.

## Core Guarantees

- No-lookahead contract: signal at `close[t]`, execution at `open[t+1]`.
- Shared-capital accounting in the portfolio engine.
- Artifact-first workflow: runs write reusable bundles to disk for later analysis.
- Clear layer boundaries: CLI -> services -> engines/runtime.

## Quick Start

```bash
git clone https://github.com/DanRedelien/futures-backtesting-engine.git
cd futures-backtesting-engine
pip install -r requirements.txt
pytest tests/
```

Python `3.11+` is recommended.

`pyproject.toml` already configures pytest `pythonpath`, so no manual `PYTHONPATH` export is needed.

## Common Commands

```bash
# Download cached market data
python run.py --download ES NQ YM RTY CL GC SI

# Single backtest
python run.py --backtest --strategy sma --symbol ES --tf 1h

# Walk-forward optimization
python run.py --wfo --strategy zscore --symbol ES --tf 1h

# Portfolio backtest
python run.py --portfolio-backtest
python run.py --portfolio-backtest --portfolio-config src/backtest_engine/portfolio_layer/portfolio_config_example.yaml

# Lightweight batch backtests with one combined Matplotlib popup
python run.py batch --strategies sma zscore --symbol ES NQ --tf 1h 30m

# Lightweight WFO batch sweep with verdict heatmap and candidate exports
python run.py wfo-batch --strategies sma zscore --symbol ES --tf 1h

# Launch terminal UI for the latest artifacts
python run.py --dashboard

# Run a backtest and open the UI after completion
python run.py --portfolio-backtest --dashboard
```

## Redis Requirement

The Stress Testing tab in the terminal UI uses Redis/RQ. The rest of the platform does not require Redis.

Install Redis only if you want scenario queueing from the dashboard.

Windows:

```bash
winget install Redis.Redis
```

macOS:

```bash
brew install redis
```

Ubuntu/Debian:

```bash
sudo apt install redis-server
```

## Project Layout

```text
.
|-- README.md
|-- CONTRIBUTING.md
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- MODULE_MAP.md
|   `-- agents.md
|-- cli/
|   |-- single.py
|   |-- wfo.py
|   |-- portfolio.py
|   |-- batch.py
|   `-- wfo_batch.py
|-- run.py
|-- tests/
|   |-- README.md
|   |-- unit/
|   `-- regression/
`-- src/
    |-- data/
    |-- strategies/
    `-- backtest_engine/
        |-- engine.py
        |-- execution.py
        |-- analytics/
        |-- optimization/
        |-- services/
        |-- runtime/terminal_ui/
        `-- portfolio_layer/
```

## Architecture At A Glance

- `run.py` parses CLI args and dispatches to `cli/`.
- `cli/` is intentionally thin and delegates orchestration to `src/backtest_engine/services/`.
- `src/backtest_engine/engine.py` is the single-asset execution engine.
- `src/backtest_engine/portfolio_layer/engine/engine.py` is the portfolio event loop with shared capital and multi-slot execution.
- `src/backtest_engine/runtime/terminal_ui/` is the active FastAPI analytics UI.
- `src/backtest_engine/analytics/` contains artifact builders, reports, metrics, and shared analytics transforms.
- `src/backtest_engine/services/scenario_job_service.py` remains the public scenario-queue entry point, with adjacent helper modules for metadata storage and worker execution.
- `src/data/ib_fetcher.py` remains the public IB data entry point, with adjacent helper modules for contract resolution, cache/checkpoint storage, and historical backfill orchestration.

## Results And Artifacts

- Single runs write to `results/`.
- Portfolio runs write to `results/portfolio/`.
- Scenario reruns write to `results/scenarios/<scenario_id>/...`.
- The terminal UI reads saved artifacts rather than requiring a rerun.

## Documentation Map

- [Architecture](docs/ARCHITECTURE.md) - layer boundaries, engine roles, artifact flow.
- [Module Map](docs/MODULE_MAP.md) - quick reference for portfolio layer and entry points.
- [Agent Context](docs/agents.md) - compact project context for LLMs and automation agents.
- [Contributing Guide](CONTRIBUTING.md) - setup, workflow, tests, strategy additions, PR expectations.
- [Analytics README](src/backtest_engine/analytics/README.md)
- [Optimization README](src/backtest_engine/optimization/README.md)
- [Portfolio Layer README](src/backtest_engine/portfolio_layer/README.md)
- [Terminal UI README](src/backtest_engine/runtime/terminal_ui/README.md)
- [Strategies README](src/strategies/README.md)
- [Tests README](tests/README.md)

## Strategy Registry

Strategies are exposed through `src/strategies/registry.py`.

Current canonical IDs:

- `sma`
- `mean_rev`
- `zscore`
- `sma_pullback`
- `intraday_momentum`
- `stat_level`
- `ict_ob`

Aliases such as `sma_crossover` and `mean_reversion` are also accepted by the CLI.
