"""
Research-Grade Walk-Forward Validation Optimizer.

Adapted for event-driven BacktestEngine strategies.

Includes 'Skeptic' analysis tools:
1. Deflated Sharpe Ratio (DSR) estimation.
2. Performance Degradation Analysis.
3. Regime-Specific Stability checks.
"""
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from collections import Counter
import time
import pandas as pd
import numpy as np
import math
import optuna
import optuna.logging

from ..settings import BacktestSettings, get_settings
from .fold_generator import PurgedFoldGenerator
from .optimizer import OptunaOptimizer
from src.data.data_lake import DataLake


# Suppress Optuna logging to warnings only
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════
# HELPER: MATH FOR SKEPTICS
# ═══════════════════════════════════════════════════════════════════

def estimated_dsr(
    sharpe: float,
    n_trials: int,
    trial_std: float,
) -> float:
    """
    Estimate Probabilistic Sharpe Ratio adjusted for Multiple Testing (DSR proxy).

    Based on Bailey, Lopez de Prado (2014).

    Args:
        sharpe: The best IS Sharpe Ratio found.
        n_trials: Number of optimization trials run.
        trial_std: Standard deviation of scores across trials.

    Returns:
        Probability (0.0 - 1.0) that the strategy is NOT a false positive.
    """
    if n_trials < 2 or trial_std <= 1e-6:
        return 0.5

    expected_max_sr = trial_std * math.sqrt(2 * math.log(n_trials))

    if expected_max_sr == 0:
        return 0.0

    z_score = (sharpe - expected_max_sr) / trial_std

    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Results from a single WFV fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]

    # Scores
    is_score: float
    oos_score: float

    # DSR inputs
    n_trials: int
    trial_std: float

    # Metrics
    oos_stats: Dict[str, Any]

    @property
    def degradation(self) -> float:
        """Percentage drop from IS to OOS."""
        if self.is_score <= 1e-6:
            return 0.0
        return (self.oos_score - self.is_score) / self.is_score

    @property
    def dsr_probability(self) -> float:
        """Probability this fold's success isn't luck."""
        return estimated_dsr(self.is_score, self.n_trials, self.trial_std)


@dataclass
class WFVReport:
    """Aggregated results with Skeptic Analysis."""

    symbol: str
    strategy_name: str
    n_folds: int
    fold_results: List[FoldResult]

    # Aggregates
    median_oos_score: float = 0.0
    median_degradation: float = 0.0
    avg_dsr: float = 0.0

    # Candidate Selection
    candidate_params: Dict[str, Any] = field(default_factory=dict)
    verdict: str = "FAIL"
    warnings: List[str] = field(default_factory=list)

    # Computational Profiling
    total_wfo_time_sec: float = 0.0
    avg_fold_time_sec: float = 0.0
    avg_trial_time_sec: float = 0.0

    def compute(self) -> None:
        if not self.fold_results:
            return

        oos_scores = [f.oos_score for f in self.fold_results]
        degradations = [f.degradation for f in self.fold_results]
        dsrs = [f.dsr_probability for f in self.fold_results]

        self.median_oos_score = float(np.median(oos_scores))
        self.median_degradation = float(np.median(degradations))
        self.avg_dsr = float(np.mean(dsrs))

        self._analyze_robustness()
        self._select_candidate_params()

    def _analyze_robustness(self):
        """Apply Senior Quant logic to determine verdict."""
        n_profitable = sum(1 for f in self.fold_results if f.oos_score > 0)

        if self.median_degradation < -0.50:
            self.warnings.append(
                f"Severe Overfitting: Median decay is {self.median_degradation:.0%}"
            )

        if self.avg_dsr < 0.5:
            self.warnings.append(
                f"Low Significance: DSR {self.avg_dsr:.2f} implies results "
                f"indistinguishable from noise."
            )

        if (
            n_profitable >= 3
            and self.median_degradation > -0.40
            and self.avg_dsr > 0.6
        ):
            self.verdict = "PASS"
        elif n_profitable >= 2 and self.median_degradation > -0.60:
            self.verdict = "WARNING"
        else:
            self.verdict = "FAIL"

    def _select_candidate_params(self) -> None:
        """Median/Mode of robust folds."""
        robust_folds = [f for f in self.fold_results if f.oos_score > 0]
        if not robust_folds:
            return

        source = robust_folds if len(robust_folds) >= 2 else self.fold_results

        consensus = {}
        first_params = source[0].best_params

        for key, val in first_params.items():
            all_vals = [f.best_params.get(key) for f in source]
            all_vals = [v for v in all_vals if v is not None]

            if not all_vals:
                continue

            if isinstance(val, (int, float)) and not isinstance(val, bool):
                median_val = np.median(all_vals)
                consensus[key] = (
                    int(round(median_val))
                    if isinstance(val, int)
                    else float(median_val)
                )
            else:
                consensus[key] = Counter(all_vals).most_common(1)[0][0]

        self.candidate_params = consensus


class WalkForwardOptimizer:
    """
    Orchestrator for Research-Grade Walk-Forward Validation.

    Uses date-bounded BacktestEngine runs for full compatibility
    with event-driven strategies (on_bar loop).
    """

    def __init__(
        self,
        settings: Optional[BacktestSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_optimizer = OptunaOptimizer(settings=self.settings)
        self.data_lake = DataLake()

    def run(
        self,
        strategy_class: Type,
        n_folds: Optional[int] = None,
        test_size_pct: Optional[float] = None,
        n_trials: Optional[int] = None,
        purge_bars: int = 0,
        embargo_bars: int = 0,
        anchored: bool = False,
    ) -> WFVReport:
        """
        Run Walk-Forward Validation for a strategy.

        Loads the full dataset, generates date-based folds, then for each
        fold runs IS optimization and OOS evaluation via BacktestEngine.

        Args:
            strategy_class: Strategy class implementing BaseStrategy.
            n_folds: Number of walk-forward folds.
            test_size_pct: Fraction of data for each test fold.
            n_trials: Number of Optuna trials per fold.
            purge_bars: Gap between Train and Test to remove overlap.
            embargo_bars: Additional gap to prevent correlation leakage.
            anchored: True for expanding window, False for rolling window.

        Returns:
            WFVReport with fold results and skeptic analysis.
        """
        n_folds = n_folds or self.settings.wfo_n_folds
        test_size_pct = test_size_pct or self.settings.wfo_test_size_pct
        n_trials = n_trials or self.settings.wfo_n_trials
        
        symbol = self.settings.default_symbol
        timeframe = self.settings.low_interval

        # Load full dataset to determine date boundaries
        print(f"\n[WFV] Loading {symbol} @ {timeframe} for fold generation...")
        data = self.data_lake.load(symbol, timeframe=timeframe)

        if data.empty:
            print("[WFV] No data found. Aborting.")
            return WFVReport(symbol, strategy_class.__name__, 0, [])

        print(
            f"[WFV] Data range: {data.index[0].date()} -> {data.index[-1].date()} "
            f"| {len(data):,} bars"
        )

        # Generate index-based folds
        splitter = PurgedFoldGenerator(
            n_folds=n_folds,
            test_size=test_size_pct,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            anchored=anchored,
        )
        folds = list(splitter.split(data))

        fold_results = []
        total_trials = 0
        wfo_start_time = time.time()

        # Silence Optuna's trial-by-trial logs for a cleaner console
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n[WFV] Starting {n_folds}-Fold Walk-Forward on {symbol}...")

        for i, (train_idx, test_idx) in enumerate(folds):
            # Convert indices to date boundaries AND slice data
            train_start = data.index[train_idx[0]]
            train_end = data.index[train_idx[-1]]
            test_start = data.index[test_idx[0]]
            test_end = data.index[test_idx[-1]]
            
            # Extract DataFrame slices once per fold
            train_slice = data.iloc[train_idx]
            test_slice = data.iloc[test_idx]

            print(
                f"\n  Fold {i + 1}/{len(folds)}: "
                f"IS {train_start.date()} -> {train_end.date()} | "
                f"OOS {test_start.date()} -> {test_end.date()}"
            )

            # 1. Optimize (IS) — dataset injected
            fold_start_time = time.time()
            opt_result = self.base_optimizer.optimize_on_slice(
                strategy_class=strategy_class,
                start_date=train_start,
                end_date=train_end,
                data=train_slice,
                n_trials=n_trials,
                fold_id=i,
            )
            fold_end_time = time.time()

            n_trials_actual = opt_result.get("n_trials", n_trials)
            total_trials += n_trials_actual
            trial_std = opt_result.get("trial_std", 0.1)

            # 2. Evaluate (OOS) — dataset injected
            eval_result = self.base_optimizer.evaluate_on_slice(
                strategy_class=strategy_class,
                params=opt_result["best_params"],
                start_date=test_start,
                end_date=test_end,
                data=test_slice,
            )

            fold_results.append(
                FoldResult(
                    fold_id=i + 1,
                    train_start=str(train_start.date()),
                    train_end=str(train_end.date()),
                    test_start=str(test_start.date()),
                    test_end=str(test_end.date()),
                    best_params=opt_result["best_params"],
                    is_score=opt_result["best_score"],
                    oos_score=eval_result["score"],
                    n_trials=n_trials_actual,
                    trial_std=trial_std,
                    oos_stats=eval_result["stats"],
                )
            )

            print(
                f"  Fold {i + 1}: IS {opt_result['best_score']:.2f} -> "
                f"OOS {eval_result['score']:.2f} "
                f"({fold_end_time - fold_start_time:.1f}s)"
            )

        wfo_end_time = time.time()
        total_time = wfo_end_time - wfo_start_time

        report = WFVReport(
            symbol, strategy_class.__name__, len(folds), fold_results
        )
        report.total_wfo_time_sec = total_time
        report.avg_fold_time_sec = total_time / len(folds) if folds else 0.0
        report.avg_trial_time_sec = total_time / total_trials if total_trials else 0.0

        report.compute()

        self._print_human_report(report)
        return report

    def _print_human_report(self, report: WFVReport) -> None:
        """Generates a Quant-style critical report."""

        def _bar(val, max_val=2.0, width=10):
            if val < 0:
                return "💀 " + " " * (width - 2)
            normalized = min(1.0, val / max_val)
            chars = int(normalized * width)
            return "█" * chars + "░" * (width - chars)

        def _col(text, width=10, align="<"):
            return f"{str(text):{align}{width}}"

        print(f"\n\n{'=' * 80}")
        print(f" WFV AUDIT REPORT: {report.strategy_name} @ {report.symbol}")
        print(f"{'=' * 80}")

        # 1. High Level Summary
        print(f"\n[EXECUTIVE SUMMARY]")
        print(f"  Verdict:        {report.verdict}")
        print(
            f"  Median OOS:     {report.median_oos_score:.4f} (Composite Score)"
        )
        print(
            f"  Perf Decay:     {report.median_degradation:+.1%} (IS -> OOS)"
        )
        print(
            f"  Skeptic Confidence (DSR): {report.avg_dsr:.0%} "
            f"(Prob. Skill > Noise)"
        )
        
        print(f"\n[COMPUTATIONAL PROFILE]")
        print(f"  Total WFO Runtime: {report.total_wfo_time_sec / 60:.1f} mins")
        print(f"  Average Fold Time: {report.avg_fold_time_sec:.1f} secs")
        print(f"  Average Trial Time: {report.avg_trial_time_sec:.3f} secs")

        if report.warnings:
            print(f"\n[RISK FLAGS]")
            for w in report.warnings:
                print(f"  ! {w}")

        # 2. Fold Detail
        print(f"\n[FOLD ANALYSIS]")
        header = (
            f"{_col('Fold', 4)} | "
            f"{_col('Period', 22)} | "
            f"{_col('IS Score', 8)} | "
            f"{_col('OOS Score', 9)} | "
            f"{_col('Decay', 7)} | "
            f"{_col('Visual', 12)} | "
            f"{_col('DD%', 6)}"
        )
        print(header)
        print("-" * len(header))

        for f in report.fold_results:
            if f.degradation < -0.5:
                decay_str = "CRASH"
            else:
                decay_str = f"{f.degradation:+.0%}"

            dd_val = f.oos_stats.get("max_drawdown", 0)
            dd_str = f"{dd_val:.1f}"

            print(
                f"{_col(str(f.fold_id), 4)} | "
                f"{_col(f'{f.test_start}..', 22)} | "
                f"{_col(f'{f.is_score:.2f}', 8)} | "
                f"{_col(f'{f.oos_score:.2f}', 9)} | "
                f"{_col(decay_str, 7)} | "
                f"{_col(_bar(f.oos_score, 1.5), 12)} | "
                f"{_col(dd_str, 6)}"
            )

        # 3. Params
        if report.candidate_params:
            print(f"\n[CANDIDATE PARAMETERS]")
            print(f"  {report.candidate_params}")
        else:
            print(
                "\n[NO CANDIDATE PARAMETERS] Strategy failed stability checks."
            )

        print(f"{'=' * 80}\n")
