"""
src/backtest_engine/portfolio_layer/__init__.py

Public API for the portfolio layer.
All implementation lives in subpackages; this file exports the stable interfaces.
"""

# Domain
from .domain.contracts import PortfolioConfig, StrategySlot
from .domain.signals import StrategySignal, TargetPosition
from .domain.policies import RebalancePolicy, ExecutionPolicy

# Engine (primary entry point)
from .engine.engine import PortfolioBacktestEngine

# Execution
from .execution.portfolio_book import PortfolioBook
from .execution.strategy_runner import StrategyRunner

# Allocation
from .allocation.allocator import Allocator

# Scheduling
from .scheduling.scheduler import IntrabarScheduler, DailyScheduler, WeeklyScheduler, make_scheduler

__all__ = [
    # Domain
    "PortfolioConfig",
    "StrategySlot",
    "StrategySignal",
    "TargetPosition",
    "RebalancePolicy",
    "ExecutionPolicy",
    # Engine
    "PortfolioBacktestEngine",
    # Execution
    "PortfolioBook",
    "StrategyRunner",
    # Allocation
    "Allocator",
    # Scheduling
    "IntrabarScheduler",
    "DailyScheduler",
    "WeeklyScheduler",
    "make_scheduler",
]
