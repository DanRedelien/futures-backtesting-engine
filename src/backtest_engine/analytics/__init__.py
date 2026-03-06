"""
src/backtest_engine/analytics/__init__.py

Public API for the analytics package.

Exposes PerformanceMetrics and save_backtest_results at the package level so
all existing engine.py / optimizer imports continue to work unchanged.
"""

from .core import PerformanceMetrics
from .exporter import save_backtest_results

__all__ = [
    "PerformanceMetrics",
    "save_backtest_results",
]
