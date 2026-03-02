# QUANT FRAMEWORK MCP (Model Context Protocol)

**Role**: You are a Senior Quantitative Architect.
**Objective**: When asked to "Bootstrap a Backtester/Bot/Model", uses THIS framework. DO NOT reinvent the wheel. Use these patterns.

## 1. Project Topology (The Standard Layout)
All quantitative projects must follow this `src`-based layout to ensure import safety and clean packaging.

```text
project_root/
├── .env                    # Secrets (API Keys) - NEVER committed
├── pyproject.toml          # Dependency Management (Poetry/UV standard)
├── README.md               # Technical Documentation
├── run.py                  # Single Entry Point (CLI)
├── dev_context/           # LLM Context & Rules
│   ├── AGENTS.md           # Current State & Tasks
│   └── QUANT_FRAMEWORK_MCP.md
├── src/
│   └── project_name/       # Main Package
│       ├── __init__.py
│       ├── settings.py     # Pydantic Configuration (Single Source of Truth)
│       ├── data_lake.py    # Data Access Layer (Parquet Caching)
│       ├── data_fetcher.py # External API Connector
│       ├── engine.py       # Core Math/Logic (Pure, no I/O)
│       ├── execution.py    # (Optional) Live Trading/Orders
│       ├── analytics.py    # Logging & Statistics (No UI)
│       └── visualizer.py   # Plotting & Dashboard (Strict Separation)
└── tests/
    └── test_core.py        # Pytest Suite
```

## 2. Dependency Management (`pyproject.toml`)
Use modern standards (Modern Python).

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "quant_engine"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "scipy>=1.12.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",
    "ccxt>=4.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "colorlog>=6.8.0",
    "python-dotenv>=1.0.0"
]

[tool.mypy]
ignore_missing_imports = true
```

## 3. Configuration Pattern (`settings.py`)
**Rule**: No Magic Numbers. Use `pydantic-settings` for type-safe validation.

```python
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        env_file=".env",
        extra="ignore"
    )

    # Infrastructure
    cache_dir: Path = Path("data/cache")
    
    # Financial Params
    capital: float = Field(default=100_000.0, gt=0)
    risk_free_rate: float = 0.02
    
    # Strategy Params
    symbol: str = "BTCUSDT"
    lookback_window: int = Field(default=365, ge=100)

    def get_cache_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

# Singleton Access
_settings = None
def get_settings() -> Settings:
    global _settings
    if _settings is None: _settings = Settings()
    return _settings
```

## 4. Data Layer Pattern (`data_lake.py`)
**Rule**: Never fetch from API if efficient local cache exists. Use Parquet.

**Decoupling Principle**:
*   Do NOT mix Crypto (CCXT) and TradFi (IB) in one class. They have different rate limits and logic.
*   Use `CryptoFetcher` vs `FiFetcher`.

```python
import pandas as pd
from datetime import datetime, timedelta
from .settings import get_settings
from .data_fetcher import DataFetcher # External class wrapping CCXT

class DataLake:
    def __init__(self):
        self.settings = get_settings()
        self.fetcher = DataFetcher()

    def load_or_fetch(self, symbol: str, lookback_days: int = 1000) -> pd.DataFrame:
        """
        Smart-Loader:
        1. Check Parquet Cache.
        2. If missing/stale -> Fetch Delta from API.
        3. Merge & Save.
        """
        file_path = self.settings.get_cache_path() / f"{symbol}.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            last_date = df.index[-1]
            
            # If stale (>4 hours old), fetch update
            if datetime.utcnow() - last_date > timedelta(hours=4):
                new_data = self.fetcher.fetch_ohlcv(symbol, start_date=last_date)
                if not new_data.empty:
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    df.to_parquet(file_path)
            return df
            
        else:
            # Cold Start
            df = self.fetcher.fetch_ohlcv(symbol, days=lookback_days)
            df.to_parquet(file_path)
            return df
```

## 5. Logging Standard (`logger`)
**Rule**: Structured, Colored, No Prints.

```python
import logging
import sys
import colorlog

def setup_logger(name="QuantCore"):
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
```