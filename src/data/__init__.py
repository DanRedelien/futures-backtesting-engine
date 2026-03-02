"""
Data fetching and caching module.

Components:
    - IBFetcher: Fetches M5 and H1 bars from Interactive Brokers
    - DataLake: Parquet-based caching layer
    - DataValidator: Data quality checks
"""

from .ib_fetcher import IBFetcher
from .data_lake import DataLake
from .data_validator import DataValidator, ValidationResult

__all__ = ["IBFetcher", "DataLake", "DataValidator", "ValidationResult"]
