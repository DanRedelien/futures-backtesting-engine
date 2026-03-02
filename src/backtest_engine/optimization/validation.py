"""
Pre-flight validation for optimization trials.
Strict engineering rules enforced here.
"""

from typing import Dict, Any, List
from ..settings import get_settings

class ValidationException(Exception):
    pass

class Validator:
    """
    Enforces Strict Engineering Guidelines for Optimization.
    """
    
    FORBIDDEN_KEYWORDS = [
        "risk", "lot", "size", "leverage", "capital", "money"
    ]
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any], strategy_name: str) -> None:
        """
        Validate trial parameters before backtest.
        
        Rules:
        1. Max 6 parameters.
        2. No forbidden keywords (risk/money management).
        """
        # Rule 1: Dimensionality Control
        max_params = get_settings().wfo_max_parameters
        if len(params) > max_params:
            raise ValidationException(
                f"Strategy {strategy_name} has {len(params)} optimized parameters. "
                f"MAX allowed is {max_params}."
            )
            
        # Rule 2: Eligibility Rules
        for param in params.keys():
            param_lower = param.lower()
            for forbidden in cls.FORBIDDEN_KEYWORDS:
                if forbidden in param_lower:
                    raise ValidationException(
                        f"Parameter '{param}' in {strategy_name} seems to be related to '{forbidden}'. "
                        "Optimization of risk/money management parameters is FORBIDDEN."
                    )
