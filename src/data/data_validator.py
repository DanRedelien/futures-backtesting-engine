"""
Data quality validation module.

Performs comprehensive checks on OHLCV data per VALIDATION_PROTOCOL:
    - Missing bar detection (gaps in time series)
    - OHLC consistency (High >= Low, etc.)
    - Volume anomaly detection
    - Data completeness metrics
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Container for validation results."""
    
    symbol: str
    timeframe: str
    total_bars: int
    missing_bars: int
    ohlc_violations: int
    volume_anomalies: int
    quality_score: float  # 0.0 to 1.0
    issues: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Data is valid if quality score >= 0.95."""
        return self.quality_score >= 0.95


class DataValidator:
    """
    Validates OHLCV data quality.
    
    Checks for common data issues that could affect backtest accuracy.
    Per VALIDATION_PROTOCOL guidelines.
    """
    
    # Gap thresholds for different timeframes (in minutes)
    GAP_THRESHOLDS = {
        "5m": 30,   # 6 bars = 30 min gap threshold
        "1h": 180,  # 3 bars = 3 hour gap threshold
    }
    
    def __init__(
        self,
        volume_zscore_threshold: float = 5.0,
    ):
        """
        Initialize validator.
        
        Args:
            volume_zscore_threshold: Z-score threshold for volume anomalies.
        """
        self.volume_zscore_threshold = volume_zscore_threshold
    
    def validate(
        self, 
        df: pd.DataFrame, 
        symbol: str = "UNKNOWN",
        timeframe: str = "5m",
    ) -> ValidationResult:
        """
        Run all validation checks on DataFrame.
        
        Args:
            df: OHLCV DataFrame with datetime index.
            symbol: Symbol name for reporting.
            timeframe: '5m' or '1h'.
            
        Returns:
            ValidationResult with all findings.
        """
        issues: List[str] = []
        
        if df.empty:
            return ValidationResult(
                symbol=symbol,
                timeframe=timeframe,
                total_bars=0,
                missing_bars=0,
                ohlc_violations=0,
                volume_anomalies=0,
                quality_score=0.0,
                issues=["Empty DataFrame"],
            )
        
        total_bars = len(df)
        max_gap = self.GAP_THRESHOLDS.get(timeframe, 30)
        
        # Check 1: Missing bars (gaps in time series)
        missing_bars, gap_issues = self._check_gaps(df, max_gap)
        issues.extend(gap_issues)
        
        # Check 2: OHLC consistency
        ohlc_violations, ohlc_issues = self._check_ohlc_consistency(df)
        issues.extend(ohlc_issues)
        
        # Check 3: Volume anomalies
        volume_anomalies, volume_issues = self._check_volume_anomalies(df)
        issues.extend(volume_issues)
        
        # Calculate quality score
        penalty = (missing_bars + ohlc_violations * 10 + volume_anomalies) / max(total_bars, 1)
        quality_score = max(0.0, min(1.0, 1.0 - penalty))
        
        return ValidationResult(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total_bars,
            missing_bars=missing_bars,
            ohlc_violations=ohlc_violations,
            volume_anomalies=volume_anomalies,
            quality_score=quality_score,
            issues=issues,
        )
    
    def _check_gaps(
        self, 
        df: pd.DataFrame, 
        max_gap_minutes: int
    ) -> Tuple[int, List[str]]:
        """
        Check for gaps in time series.
        
        Returns:
            Tuple of (gap_count, list of issue descriptions).
        """
        issues: List[str] = []
        gap_count = 0
        
        if len(df) < 2:
            return 0, []
        
        time_diffs = pd.Series(df.index).diff().dropna().reset_index(drop=True)
        threshold = timedelta(minutes=max_gap_minutes)
        large_gaps = time_diffs[time_diffs > threshold]
        
        gap_count = len(large_gaps)
        
        if gap_count > 0:
            issues.append(f"Found {gap_count} gaps > {max_gap_minutes} minutes")
            
            for idx in large_gaps.head(3).index:
                gap_start = df.index[idx]
                gap_end = df.index[idx + 1]
                issues.append(f"  Gap: {gap_start} to {gap_end}")
        
        return gap_count, issues
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Check OHLC bar consistency rules.
        
        Rules:
            - High >= Low
            - High >= Open, High >= Close
            - Low <= Open, Low <= Close
            
        Returns:
            Tuple of (violation_count, list of issue descriptions).
        """
        issues: List[str] = []
        violations = 0
        
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(df.columns.str.lower())):
            issues.append("Missing required OHLC columns")
            return 0, issues
        
        df_check = df.copy()
        df_check.columns = df_check.columns.str.lower()
        
        # Rule 1: High >= Low
        high_low_violations = (df_check["high"] < df_check["low"]).sum()
        if high_low_violations > 0:
            issues.append(f"High < Low violations: {high_low_violations}")
            violations += high_low_violations
        
        # Rule 2: High >= max(Open, Close)
        high_violations = (
            (df_check["high"] < df_check["open"]) |
            (df_check["high"] < df_check["close"])
        ).sum()
        if high_violations > 0:
            issues.append(f"High < Open/Close violations: {high_violations}")
            violations += high_violations
        
        # Rule 3: Low <= min(Open, Close)
        low_violations = (
            (df_check["low"] > df_check["open"]) |
            (df_check["low"] > df_check["close"])
        ).sum()
        if low_violations > 0:
            issues.append(f"Low > Open/Close violations: {low_violations}")
            violations += low_violations
        
        return violations, issues
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Check for volume anomalies using z-score.
        
        Returns:
            Tuple of (anomaly_count, list of issue descriptions).
        """
        issues: List[str] = []
        
        if "volume" not in df.columns.str.lower():
            return 0, []
        
        volume = df["volume"] if "volume" in df.columns else df["Volume"]
        
        mean_vol = volume.mean()
        std_vol = volume.std()
        
        if std_vol == 0:
            return 0, []
        
        z_scores = np.abs((volume - mean_vol) / std_vol)
        anomalies = (z_scores > self.volume_zscore_threshold).sum()
        
        if anomalies > 0:
            issues.append(
                f"Volume anomalies (z > {self.volume_zscore_threshold}): {anomalies}"
            )
        
        return anomalies, issues
    
    def generate_report(self, results: List[ValidationResult]) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            results: List of validation results to summarize.
            
        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            "",
        ]
        
        for r in results:
            status = "✓ PASS" if r.is_valid else "✗ FAIL"
            lines.append(f"{r.symbol} [{r.timeframe}]: {status} (Score: {r.quality_score:.2%})")
            lines.append(f"  Bars: {r.total_bars:,}")
            lines.append(f"  Missing: {r.missing_bars}, OHLC Violations: {r.ohlc_violations}")
            
            if r.issues:
                lines.append("  Issues:")
                for issue in r.issues[:5]:
                    lines.append(f"    - {issue}")
            
            lines.append("")
        
        valid_count = sum(1 for r in results if r.is_valid)
        lines.append("-" * 60)
        lines.append(f"SUMMARY: {valid_count}/{len(results)} datasets passed validation")
        
        return "\n".join(lines)
