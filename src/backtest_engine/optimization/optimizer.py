"""
Optuna Optimizer Module — Event-Driven Adaptation.

Runs BacktestEngine per trial for full fidelity scoring.
Uses per-symbol cost model integrated via engine settings.
"""

import os
import sys
import optuna
import numpy as np
from typing import Dict, Any, Optional, Type

from ..engine import BacktestEngine
from ..settings import BacktestSettings, get_settings
from .validation import Validator, ValidationException
from .objective import objective_score


# ── Utilities ──────────────────────────────────────────────────────────────────

class _HiddenPrints:
    """Suppresses stdout during optimisation iterations."""

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_):
        sys.stdout.close()
        sys.stdout = self._orig


class OptunaOptimizer:
    """
    Bayesian Optimization Engine for Event-Driven Strategies.

    Uses Optuna TPE sampler to find optimal strategy parameters
    by running BacktestEngine per trial on in-sample data,
    then validates on out-of-sample.
    Enforces strict engineering rules via Validator.
    """

    # Thresholds are read from settings.wfo_prune_* — no magic numbers here.

    def __init__(
        self,
        settings: Optional[BacktestSettings] = None,
    ) -> None:
        """
        Initialize optimizer with settings.

        Args:
            settings: Optional settings override; defaults to singleton.
        """
        self.settings = settings or get_settings()

    # ── Search space application ───────────────────────────────────────────────

    def _apply_params(
        self, trial: optuna.Trial, search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Samples parameter values from the Optuna trial using the strategy's
        get_search_space() bounds.

        Supports three bound formats:
            - (start, stop, step): int or float range with step.
            - (start, stop): continuous range without step.
            - [v1, v2, ...]: categorical choice.

        Returns:
            Dict of sampled parameter names → values.
        """
        params = {}
        for param, bounds in search_space.items():
            if isinstance(bounds, list):
                val = trial.suggest_categorical(param, bounds)
            elif isinstance(bounds, tuple):
                if len(bounds) == 3:
                    start, stop, step = bounds
                    if all(isinstance(v, int) for v in (start, stop, step)):
                        val = trial.suggest_int(param, start, stop, step=step)
                    else:
                        val = trial.suggest_float(
                            param, float(start), float(stop), step=float(step)
                        )
                elif len(bounds) == 2:
                    start, stop = bounds
                    if isinstance(start, int) and isinstance(stop, int):
                        val = trial.suggest_int(param, start, stop)
                    else:
                        val = trial.suggest_float(
                            param, float(start), float(stop)
                        )
                else:
                    raise ValueError(
                        f"[OPT] Invalid bounds for param '{param}': {bounds}"
                    )
            else:
                continue
            params[param] = val
        return params

    # ── Run a single strategy backtest ─────────────────────────────────────────

    def _run_strategy(
        self,
        strategy_class: Type,
        params: Dict[str, Any],
        start_date=None,
        end_date=None,
    ) -> Dict[str, Any]:
        """
        Execute a single strategy backtest for optimisation scoring.

        Creates a BacktestSettings copy with injected params, runs the
        full BacktestEngine, and extracts metrics.

        Args:
            strategy_class: Strategy class inheriting BaseStrategy.
            params: Parameter dict to inject into settings via setattr.
            start_date: Optional IS start bound.
            end_date: Optional IS end bound.

        Returns:
            Dict with 'stats' sub-dict and 'engine' reference.
        """
        s = get_settings()
        for k, v in params.items():
            setattr(s, k, v)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            settings=s,
        )

        with _HiddenPrints():
            engine.run(strategy_class)

        # Extract metrics
        history = engine.portfolio.get_history_df()
        metrics = engine.analytics.calculate_metrics(
            history, engine.execution.trades
        )

        if not metrics:
            return {
                "stats": {
                    "total_trades": 0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                },
                "engine": engine,
            }

        # Normalize metric keys to match objective.py expectations
        stats = {
            "total_trades": metrics.get("Total Trades", 0),
            "sharpe_ratio": metrics.get("Sharpe Ratio", 0.0),
            "sortino_ratio": metrics.get("Sortino Ratio", 0.0),
            "calmar_ratio": metrics.get("Calmar Ratio", 0.0),
            "max_drawdown": metrics.get("Max Drawdown", 0.0) * 100,  # fraction → %
        }

        return {"stats": stats, "engine": engine}

    # ── Full Optimise flow ─────────────────────────────────────────────────────

    def optimize(
        self,
        strategy_class: Type,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run optimisation for a specific strategy class.

        Args:
            strategy_class: Any class implementing BaseStrategy + get_search_space().
            n_trials: Number of Optuna trials.

        Returns:
            Dict of best parameter values.
        """
        n_trials = n_trials or self.settings.wfo_n_trials
        
        search_space = strategy_class.get_search_space()
        if not search_space:
            print(
                f"[OPT] Warning: {strategy_class.__name__}.get_search_space() "
                f"returned empty dict. Nothing to optimise."
            )
            return {}

        # Pre-flight validation of parameter count & names
        try:
            Validator.validate_params(
                {k: None for k in search_space}, strategy_class.__name__
            )
        except ValidationException as e:
            print(f"[OPT] Validation failed: {e}")
            return {}

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            study_name=f"opt_{strategy_class.__name__}",
        )

        print(f"\n[OPT] Optimizing {strategy_class.__name__} for {n_trials} trials...")

        def objective(trial):
            params = self._apply_params(trial, search_space)
            result = self._run_strategy(strategy_class, params)
            stats = result["stats"]

            min_trades = self.settings.wfo_prune_min_trades
            max_dd_pct = self.settings.wfo_prune_max_dd_pct

            if stats["total_trades"] < min_trades:
                raise optuna.TrialPruned("Insufficient trades")

            if abs(stats["max_drawdown"]) > max_dd_pct:
                raise optuna.TrialPruned("Excessive Drawdown")

            return objective_score(
                stats,
                min_trades=min_trades,
                target_trades=min_trades * self.settings.wfo_prune_target_trades_mult,  # see settings.py
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        optuna.logging.set_verbosity(optuna.logging.INFO)

        print(f"\n{'=' * 60}")
        print(f"OPTIMIZATION RESULTS: {strategy_class.__name__}")
        print(f"{'=' * 60}")

        if not study.best_trials:
            print("[OPT] No trials completed successfully.")
            return {}

        print(f"Best Score:  {study.best_trial.value:.4f}")
        print(f"Best Params: {study.best_params}")
        print(f"{'=' * 60}\n")

        return study.best_params

    # ═══════════════════════════════════════════════════════════════════
    # WFV INTERFACE: Methods for external fold management
    # ═══════════════════════════════════════════════════════════════════

    def optimize_on_slice(
        self,
        strategy_class: Type,
        start_date,
        end_date,
        n_trials: Optional[int] = None,
        fold_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Run optimisation on a date-bounded slice (for WFV usage).

        Args:
            strategy_class: Strategy class to optimize.
            start_date: IS start date.
            end_date: IS end date.
            n_trials: Number of Optuna trials.
            fold_id: Fold identifier for study naming.

        Returns:
            Dict with best_params, best_score, n_trials, trial_std.
        """
        n_trials = n_trials or self.settings.wfo_n_trials
        
        search_space = strategy_class.get_search_space()

        study = optuna.create_study(
            study_name=f"wfv_{strategy_class.__name__}_fold{fold_id}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + fold_id),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        def objective(trial):
            params = self._apply_params(trial, search_space)

            # Pre-flight validation
            try:
                Validator.validate_params(params, strategy_class.__name__)
            except ValidationException as e:
                raise optuna.TrialPruned(str(e))

            result = self._run_strategy(
                strategy_class, params,
                start_date=start_date,
                end_date=end_date,
            )
            stats = result["stats"]

            min_trades = self.settings.wfo_prune_min_trades

            if stats["total_trades"] < min_trades:
                raise optuna.TrialPruned("Insufficient trades")

            return objective_score(
                stats,
                min_trades=min_trades,
                target_trades=min_trades * self.settings.wfo_prune_target_trades_mult,
            )

        # Suppress Optuna logging during WFV
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        optuna.logging.set_verbosity(optuna.logging.INFO)

        if not study.best_trials:
            return {"best_params": {}, "best_score": -1.0, "n_trials": 0, "trial_std": 0.0}

        # Capture trial variance for DSR calculation
        trial_values = [
            t.value
            for t in study.trials
            if t.value is not None
            and t.state == optuna.trial.TrialState.COMPLETE
        ]
        trial_std = float(np.std(trial_values)) if len(trial_values) > 1 else 0.0

        return {
            "best_params": study.best_params,
            "best_score": study.best_trial.value,
            "n_trials": len(trial_values),
            "trial_std": trial_std,
        }

    def evaluate_on_slice(
        self,
        strategy_class: Type,
        params: Dict[str, Any],
        start_date,
        end_date,
    ) -> Dict[str, Any]:
        """
        Evaluate a strategy with fixed params on a date-bounded OOS slice.

        Args:
            strategy_class: Strategy class.
            params: Fixed parameters to inject.
            start_date: OOS start date.
            end_date: OOS end date.

        Returns:
            Dict with stats and score.
        """
        result = self._run_strategy(
            strategy_class, params,
            start_date=start_date,
            end_date=end_date,
        )
        stats = result["stats"]

        min_trades = self.settings.wfo_prune_min_trades

        score = objective_score(
            stats,
            min_trades=min_trades,
            target_trades=min_trades * self.settings.wfo_prune_target_trades_mult,
        )

        return {"stats": stats, "score": score}
