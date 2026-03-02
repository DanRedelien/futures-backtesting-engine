"""
Research-Grade Fold Generator.

Implements Purged & Embargoed Cross-Validation (Walk-Forward).
Critical for preventing data leakage in time-series finance.
"""
from typing import Iterator, Tuple
import numpy as np
import pandas as pd

class PurgedFoldGenerator:
    """
    Generates Train/Test splits with Purging and Embargo.
    
    Diagram:
    [TRAIN ............] [PURGE] [EMBARGO] [TEST ...]
    """
    
    def __init__(
        self, 
        n_folds: int = 5,
        test_size: float = 0.2,     # Fraction of TOTAL data for tests? No, typically WFV implies rolling.
                                    # Actually, usually WFV is defined by 'test_size' relative to total, 
                                    # or specific window length. 
                                    # Let's use flexible definition: 
                                    # We iterate through time. test_size determines the step.
                                    
        purge_bars: int = 0,        # Gap between Train and Test to remove overlap
        embargo_bars: int = 0,      # Additional gap to prevent correlation leakage
        anchored: bool = False      # If True, Train grows. If False, Train rolls.
    ):
        """
        Args:
            n_folds: Number of folds (if test_size is None).
            test_size: Fraction of data for each test fold (e.g. 0.1).
            purge_bars: Samples to drop at train end (Label Overlap).
            embargo_bars: Samples to drop after purge (Correlation Decay).
            anchored: True for expanding window, False for rolling window.
        """
        self.n_folds = n_folds
        self.test_size = test_size
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.anchored = anchored

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields (train_indices, test_indices).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine sizes
        fold_size = int(n_samples * self.test_size)
        total_test_size = fold_size * self.n_folds
        
        # Minimum training size logic
        gap = self.purge_bars + self.embargo_bars
        initial_train_size = n_samples - total_test_size
        
        # If we don't have enough data for even 1 bar of training, raise error
        if initial_train_size <= gap:
            raise ValueError(
                f"Data too small! n_samples={n_samples}, "
                f"total_test_size={total_test_size}, gap={gap}. "
                f"Need at least {total_test_size + gap + 1} bars."
            )
            
        test_starts = range(initial_train_size, n_samples, fold_size)
        
        for i, test_start in enumerate(test_starts):
            if i >= self.n_folds:
                break  # We only yield up to n_folds
                
            test_end = min(test_start + fold_size, n_samples)
            if test_end <= test_start:
                break
                
            # Train End is BEFORE Purge+Embargo
            train_end_raw = test_start - self.purge_bars
            
            if train_end_raw <= 0:
                continue # Cannot train
            
            # Indices
            if self.anchored:
                # Expanding window (Start from 0)
                train_idx = indices[0:train_end_raw]
            else:
                # Rolling window (Fixed size = initial_train_size)
                # Note: train_end_raw pushes forward, so start moves forward too
                start_idx = max(0, train_end_raw - initial_train_size)
                train_idx = indices[start_idx:train_end_raw]
                
            # Embargo is usually right BEFORE the test set, or AFTER?
            # Standard: [Train] [Purge] [Test] [Embargo]
            # Since backtrader tests sequentially, embargo is skipped.
            # But here test_idx is strict.
            test_start_actual = test_start + self.embargo_bars
            if test_start_actual >= test_end:
                 continue # Embargo swallowed the whole test fold
                 
            test_idx = indices[test_start_actual:test_end]
            
            yield train_idx, test_idx
