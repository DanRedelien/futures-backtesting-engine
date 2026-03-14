# Analytics Module

This directory contains the post-execution analytics, reporting, artifact export, canonical transform logic, and dashboard delivery layers for the backtesting engine.

## File Breakdown

- **`core.py`**: Central `PerformanceMetrics` orchestrator. Coordinates calculations from `metrics.py` and `trades.py`, and formatting from `report.py`, providing a stable public API for the main engine and optimizers to call without knowing internal math details.
- **`metrics.py`**: Pure, stateless math functions for equity-curve-level performance metrics (CAGR, Sharpe, Sortino, Volatility, Max Drawdown, Calmar).
- **`trades.py`**: Trade-level statistical analysis. Computes closed-trade KPIs (Win Rate, Profit Factor, Averages, and T-Statistics/P-Values/Alpha/Beta).
- **`exit_analysis.py`**: Data enrichment layer. Computes Maximum Favorable Excursion (MFE), Maximum Adverse Excursion (MAE), holding times, entry volatility, and PnL decay. Runs once at the end of the backtest.
- **`report.py`**: Text report formatter. Converts metrics and trade data into a human-readable ASCII table for terminal output. Contains purely presentation logic.
- **`exporter.py`**: Backtest results exporter. Persists artifacts (`history.parquet`, `trades.parquet`, `metrics.json`, `report.txt`) to the `results/` folder for downstream analytics UIs.
- **`artifact_contract.py`**: Shared artifact identity and compatibility metadata helpers.

## Subdirectories

- **`dashboard/`**: Legacy analytics UI package plus shared canonical transform code.
  - **`core/`**: Artifact loading, path helpers, transform modules, and shared data access.
  - **`pnl_analysis/`**: Legacy PnL view helpers.
  - **`risk_analysis/`**: Canonical risk models and risk visualizations.
  - **`simulation_analysis/`**: Reserved placeholder for future simulation work.
- **`terminal_ui/`**: FastAPI + HTMX terminal dashboard.
  - **`app.py`**: App factory and route composition.
  - **`routes_*.py`**: Route registration modules for partials, charts, and operations.
  - **`service.py`**: Runtime context and artifact-loading helpers.
  - **`chart_builders.py` / `risk_builders.py` / `table_builders.py`**: JSON and table payload builders.
  - **`templates/`**: Jinja shell and partial templates.
  - **`static/`**: Terminal CSS and split JavaScript modules.

## Notes

- `dashboard/` still contains canonical loaders and transforms used by both legacy views and the terminal UI.
- New UI-facing work should go into `terminal_ui/`.
- The visible terminal UI version string lives in `terminal_ui/templates/dashboard.html` and should be bumped by `+0.1` for significant UI releases.
