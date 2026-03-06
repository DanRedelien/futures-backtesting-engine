"""
src/backtest_engine/analytics/dashboard/__init__.py

Dashboard sub-package for the analytics layer.

Exposes the Streamlit entry point so callers can do:
    from src.backtest_engine.analytics.dashboard import main
"""

from .app import main

__all__ = ["main"]
