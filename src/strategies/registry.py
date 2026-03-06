"""
src/strategies/registry.py

Centralized registry for all available trading strategies.
"""

from typing import Any, Dict, List
import importlib

# Central registry of strategy metadata
# Keys are the short names (IDs) used in CLI and YAML configs.
# Values are dictionaries with metadata.
STRATEGIES = {
    "sma": {
        "class_path": "src.strategies.sma_crossover:SmaCrossoverStrategy",
        "name": "SmaCrossoverStrategy",
        "description": "Trend Following",
    },
    "mean_rev": {
        "class_path": "src.strategies.mean_reversion:MeanReversionStrategy",
        "name": "MeanReversionStrategy",
        "description": "Mean Reversion",
    },
    "ict_ob": {
        "class_path": "src.strategies.ict_order_block:IctOrderBlockStrategy",
        "name": "IctOrderBlockStrategy",
        "description": "Popular Media / ICT",
    },
    "zscore": {
        "class_path": "src.strategies.zscore_reversal:ZScoreReversalStrategy",
        "name": "ZScoreReversalStrategy",
        "description": "Mean Reversion",
    },
    "sma_pullback": {
        "class_path": "src.strategies.sma_pullback:SmaPullbackStrategy",
        "name": "SmaPullbackStrategy",
        "description": "Trend Following",
    },
    "intraday_momentum": {
        "class_path": "src.strategies.intraday_momentum:IntradayMomentumStrategy",
        "name": "IntradayMomentumStrategy",
        "description": "Momentum",
    },
    "stat_level": {
        "class_path": "src.strategies.statistical_level:StatisticalLevelStrategy",
        "name": "StatisticalLevelStrategy",
        "description": "Statistical Edge",
    },
}

def get_strategy_ids() -> List[str]:
    """Returns a list of all registered strategy IDs."""
    return list(STRATEGIES.keys())

def get_strategy_metadata(strategy_id: str) -> Dict[str, str]:
    """Returns metadata for a given strategy ID."""
    return STRATEGIES.get(strategy_id, {})

def load_strategy_by_id(strategy_id: str) -> Any:
    """
    Returns the strategy class for the given short name.

    Args:
        strategy_id: Strategy identifier ('sma', 'mean_rev', 'ict_ob', etc.).

    Returns:
        Strategy class (subclass of BaseStrategy).
    """
    if strategy_id not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {list(STRATEGIES.keys())}")
    
    module_path, class_name = STRATEGIES[strategy_id]["class_path"].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_strategy_class_by_name(class_name: str) -> Any:
    """
    Finds and loads a strategy by its class name (e.g. 'SmaCrossoverStrategy').
    Useful for portfolio YAML configs that still use the class name.
    """
    for strategy_id, metadata in STRATEGIES.items():
        if metadata["name"] == class_name:
            return load_strategy_by_id(strategy_id)
            
    # If not found by name, try to load by ID directly as a fallback
    try:
        return load_strategy_by_id(class_name)
    except ValueError:
        raise ValueError(f"Unknown strategy class/id: '{class_name}'")
