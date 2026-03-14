"""
Framework-neutral result artifact loading for analytics UIs.

Methodology:
    Raw artifact loading stays separate from UI runtime caching so the same
    loader can be imported safely from Streamlit, FastAPI, tests, or future job
    workers. Request handlers should prefer `load_result_bundle_uncached()`
    (or `load_result_bundle(..., use_cache=False)`) to keep cache boundaries
    explicit. Long-lived UI shells may opt into the small in-process TTL cache.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd

from src.backtest_engine.analytics.artifact_contract import (
    ARTIFACT_SCHEMA_VERSION,
    RERUN_REQUIRED_FIELDS,
)
from src.backtest_engine.analytics.dashboard.core.paths import get_results_dir


BundleState = Literal["missing", "incomplete", "valid"]
REQUIRED_SINGLE_ARTIFACTS = ("history.parquet", "trades.parquet", "manifest.json")
REQUIRED_PORTFOLIO_ARTIFACTS = ("history.parquet", "trades.parquet", "manifest.json")


@lru_cache(maxsize=1)
def _result_bundle_cache_ttl_seconds() -> float:
    """Loads the interactive result-bundle cache TTL from shared settings."""
    try:
        from src.backtest_engine.settings import BacktestSettings

        return float(BacktestSettings().terminal_ui.terminal_result_bundle_cache_ttl_seconds)
    except Exception:
        return 15.0


@dataclass
class ArtifactMetadata:
    """Identity metadata used by caches, jobs, and UI loaders."""

    artifact_id: str
    schema_version: str
    engine_version: str
    artifact_created_at: str
    artifact_path: str


@dataclass
class ArtifactCompatibility:
    """Compatibility verdict for follow-up operations such as reruns."""

    is_rerunnable: bool
    reason: str = ""
    missing_fields: Tuple[str, ...] = ()


@dataclass
class ArtifactLoadStatus:
    """Integrity status for a result root before attempting to load it."""

    state: BundleState
    run_type: Optional[str]
    base_dir: Path
    target_dir: Path
    reason: str = ""
    missing_files: Tuple[str, ...] = ()


@dataclass
class ResultBundle:
    """Unified DTO containing loaded artifacts and safety metadata."""

    run_type: str
    history: pd.DataFrame
    trades: pd.DataFrame
    benchmark: Optional[pd.DataFrame] = None
    exposure: Optional[pd.DataFrame] = None
    strategy_pnl_daily: Optional[pd.DataFrame] = None
    instrument_closes: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None
    report: str = ""
    slots: Optional[Dict[str, str]] = None
    slot_weights: Optional[Dict[str, float]] = None
    artifact_metadata: Optional[ArtifactMetadata] = None
    compatibility: Optional[ArtifactCompatibility] = None
    artifact_state: BundleState = "valid"

    def __post_init__(self) -> None:
        self.metrics = dict(self.metrics or {})
        self.manifest = dict(self.manifest or {})
        self.slots = dict(self.slots or {})
        self.slot_weights = dict(self.slot_weights or {})
        if self.artifact_metadata is None:
            self.artifact_metadata = build_artifact_metadata(
                manifest=self.manifest,
                artifact_path=Path("."),
            )
        if self.compatibility is None:
            self.compatibility = assess_bundle_compatibility(
                run_type=self.run_type,
                manifest=self.manifest,
            )


@dataclass
class _ResultBundleCacheEntry:
    bundle: Optional[ResultBundle]
    expires_at: float


class ResultBundleCache:
    """Small explicit TTL cache for interactive dashboards."""

    def __init__(self, ttl_seconds: Optional[float] = None) -> None:
        self.ttl_seconds = (
            float(ttl_seconds)
            if ttl_seconds is not None
            else _result_bundle_cache_ttl_seconds()
        )
        self._entries: Dict[str, _ResultBundleCacheEntry] = {}
        self._lock = Lock()

    def get(self, cache_key: str) -> Optional[ResultBundle]:
        """Returns a cached bundle when the TTL window is still valid."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(cache_key, None)
                return None
            return entry.bundle

    def set(self, cache_key: str, bundle: Optional[ResultBundle]) -> None:
        """Stores a bundle result under an explicit cache key."""
        with self._lock:
            self._entries[cache_key] = _ResultBundleCacheEntry(
                bundle=bundle,
                expires_at=time.monotonic() + self.ttl_seconds,
            )

    def clear(self) -> None:
        """Drops all cached bundle entries."""
        with self._lock:
            self._entries.clear()


_RESULT_BUNDLE_CACHE = ResultBundleCache()


class ResultBundleService:
    """
    Framework-neutral service wrapper around result bundle inspection/loading.

    Methodology:
        The loader contract stays available as plain functions for
        backward compatibility, but exposes a small service object so future
        FastAPI routes and worker code can depend on one explicit interface.
    """

    def inspect_bundle(self, results_dir: Optional[str] = None) -> ArtifactLoadStatus:
        """Returns the integrity classification for a result root."""
        return inspect_result_bundle(results_dir=results_dir)

    def load_bundle(
        self,
        results_dir: Optional[str] = None,
        *,
        use_cache: bool = False,
    ) -> Optional[ResultBundle]:
        """Loads a result bundle through the shared artifact loader contract.

        Methodology:
            Defaults to `use_cache=False` so request handlers keep
            cache boundaries explicit. Long-lived UI shells that need the TTL
            cache must opt in by passing `use_cache=True`.
        """
        return load_result_bundle(results_dir=results_dir, use_cache=use_cache)

    def clear_cache(self) -> None:
        """Clears the shared in-process result bundle cache."""
        clear_result_bundle_cache()


result_bundle_service = ResultBundleService()


def _resolve_base_dir(results_dir: Optional[str]) -> Path:
    return Path(results_dir) if results_dir is not None else get_results_dir()


def _missing_files(target_dir: Path, required_files: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(name for name in required_files if not (target_dir / name).exists())


def build_artifact_metadata(manifest: Dict[str, Any], artifact_path: Path) -> ArtifactMetadata:
    """
    Normalizes artifact identity metadata from a manifest payload.

    Args:
        manifest: Parsed manifest.json payload.
        artifact_path: Loaded artifact directory.

    Returns:
        Bundle metadata with safe defaults for legacy artifacts.
    """
    fallback_id = f"legacy-{artifact_path.resolve().name}"
    return ArtifactMetadata(
        artifact_id=str(manifest.get("artifact_id") or manifest.get("run_id") or fallback_id),
        schema_version=str(manifest.get("schema_version") or ARTIFACT_SCHEMA_VERSION),
        engine_version=str(manifest.get("engine_version") or "unknown"),
        artifact_created_at=str(
            manifest.get("artifact_created_at") or manifest.get("generated_at") or ""
        ),
        artifact_path=str(manifest.get("artifact_path") or artifact_path.resolve()),
    )


def assess_bundle_compatibility(run_type: str, manifest: Dict[str, Any]) -> ArtifactCompatibility:
    """
    Determines whether a loaded bundle is safe to use for scenario reruns.

    Methodology:
        Old artifacts stay read-only unless they carry enough
        reproducibility metadata to reconstruct the baseline intentionally.
        This avoids silently falling back to example configs or other unsafe
        defaults during rerun workflows.

    Args:
        run_type: Loaded run type such as ``single`` or ``portfolio``.
        manifest: Parsed manifest payload.

    Returns:
        Compatibility verdict for rerun workflows.
    """
    if run_type != "portfolio":
        return ArtifactCompatibility(
            is_rerunnable=False,
            reason="Scenario reruns are only supported for portfolio artifacts.",
        )

    missing_fields = tuple(
        field_name
        for field_name in RERUN_REQUIRED_FIELDS
        if manifest.get(field_name) in (None, "", [])
    )
    if missing_fields:
        joined_fields = ", ".join(missing_fields)
        return ArtifactCompatibility(
            is_rerunnable=False,
            reason=(
                "Baseline artifact is view-only because rerun metadata is incomplete. "
                f"Missing: {joined_fields}."
            ),
            missing_fields=missing_fields,
        )

    source_config_path = Path(str(manifest["source_config_path"]))
    if not source_config_path.exists():
        return ArtifactCompatibility(
            is_rerunnable=False,
            reason=(
                "Baseline artifact is view-only because its `source_config_path` "
                "no longer exists on disk."
            ),
            missing_fields=("source_config_path",),
        )

    return ArtifactCompatibility(is_rerunnable=True)


def inspect_result_bundle(results_dir: Optional[str] = None) -> ArtifactLoadStatus:
    """
    Inspects a results root and classifies it as missing, incomplete, or valid.

    Methodology:
        The loader must fail closed when a bundle is partially written or
        ambiguous. A missing `.run_type` marker only infers `single` when the
        root cleanly contains the full single-run contract and no portfolio
        namespace is present.

    Args:
        results_dir: Optional override for tests or scenario namespaces.

    Returns:
        Structured integrity status for the requested result root.
    """
    base_dir = _resolve_base_dir(results_dir)
    portfolio_dir = base_dir / "portfolio"
    marker_path = base_dir / ".run_type"

    root_has_any = any((base_dir / name).exists() for name in REQUIRED_SINGLE_ARTIFACTS)
    portfolio_has_any = portfolio_dir.exists() and any(
        (portfolio_dir / name).exists() for name in REQUIRED_PORTFOLIO_ARTIFACTS
    )

    if marker_path.exists():
        run_type = marker_path.read_text(encoding="utf-8").strip()
        if run_type == "single":
            missing_files = _missing_files(base_dir, REQUIRED_SINGLE_ARTIFACTS)
            state: BundleState = "valid" if not missing_files else "incomplete"
            reason = "" if not missing_files else "Single-run artifact set is incomplete."
            return ArtifactLoadStatus(
                state=state,
                run_type="single",
                base_dir=base_dir,
                target_dir=base_dir,
                reason=reason,
                missing_files=missing_files,
            )

        if run_type == "portfolio":
            missing_files = _missing_files(portfolio_dir, REQUIRED_PORTFOLIO_ARTIFACTS)
            state = "valid" if not missing_files else "incomplete"
            reason = "" if not missing_files else "Portfolio artifact set is incomplete."
            return ArtifactLoadStatus(
                state=state,
                run_type="portfolio",
                base_dir=base_dir,
                target_dir=portfolio_dir,
                reason=reason,
                missing_files=missing_files,
            )

        return ArtifactLoadStatus(
            state="incomplete",
            run_type=None,
            base_dir=base_dir,
            target_dir=base_dir,
            reason=f"Unsupported `.run_type` value: {run_type!r}.",
        )

    if not root_has_any and not portfolio_has_any:
        return ArtifactLoadStatus(
            state="missing",
            run_type=None,
            base_dir=base_dir,
            target_dir=base_dir,
            reason="No result artifacts were found.",
        )

    if portfolio_has_any:
        return ArtifactLoadStatus(
            state="incomplete",
            run_type=None,
            base_dir=base_dir,
            target_dir=portfolio_dir,
            reason="Portfolio artifacts exist without a `.run_type` marker.",
        )

    missing_files = _missing_files(base_dir, REQUIRED_SINGLE_ARTIFACTS)
    if missing_files:
        return ArtifactLoadStatus(
            state="incomplete",
            run_type="single",
            base_dir=base_dir,
            target_dir=base_dir,
            reason="Single-run artifacts are partially written or missing files.",
            missing_files=missing_files,
        )

    return ArtifactLoadStatus(
        state="valid",
        run_type="single",
        base_dir=base_dir,
        target_dir=base_dir,
    )


def _load_optional_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    return frame


def load_result_bundle_uncached(results_dir: Optional[str] = None) -> Optional[ResultBundle]:
    """
    Loads a result bundle without using any process-local cache.

    Args:
        results_dir: Optional artifact root override.

    Returns:
        Loaded ResultBundle for valid artifact sets, else None.
    """
    status = inspect_result_bundle(results_dir=results_dir)
    if status.state != "valid" or status.run_type is None:
        return None

    manifest_path = status.target_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    history = pd.read_parquet(status.target_dir / "history.parquet")
    trades = pd.read_parquet(status.target_dir / "trades.parquet")

    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
    if "exit_time" in trades.columns:
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    if "entry_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])

    metrics_path = status.target_dir / "metrics.json"
    report_path = status.target_dir / "report.txt"

    bundle = ResultBundle(
        run_type=status.run_type,
        history=history,
        trades=trades,
        metrics=(
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        ),
        manifest=manifest,
        report=report_path.read_text(encoding="utf-8") if report_path.exists() else "",
        slots=manifest.get("slots", {}),
        slot_weights=manifest.get("slot_weights", {}),
        artifact_metadata=build_artifact_metadata(
            manifest=manifest,
            artifact_path=status.target_dir,
        ),
        compatibility=assess_bundle_compatibility(
            run_type=status.run_type,
            manifest=manifest,
        ),
        artifact_state=status.state,
    )

    bundle.exposure = _load_optional_frame(status.target_dir / "exposure.parquet")
    bundle.strategy_pnl_daily = _load_optional_frame(
        status.target_dir / "strategy_pnl_daily.parquet"
    )
    bundle.benchmark = _load_optional_frame(status.target_dir / "benchmark.parquet")
    bundle.instrument_closes = _load_optional_frame(
        status.target_dir / "instrument_closes.parquet"
    )
    return bundle


def load_result_bundle(
    results_dir: Optional[str] = None,
    *,
    use_cache: bool = True,
) -> Optional[ResultBundle]:
    """
    Loads a result bundle with an optional explicit TTL cache.

    Args:
        results_dir: Optional artifact root override.
        use_cache: Whether to use the process-local dashboard cache.

    Returns:
        Loaded ResultBundle for valid artifact sets, else None.
    """
    if not use_cache:
        return load_result_bundle_uncached(results_dir=results_dir)

    cache_key = str(_resolve_base_dir(results_dir).resolve())
    cached_bundle = _RESULT_BUNDLE_CACHE.get(cache_key)
    if cached_bundle is not None:
        return cached_bundle

    bundle = load_result_bundle_uncached(results_dir=results_dir)
    _RESULT_BUNDLE_CACHE.set(cache_key, bundle)
    return bundle


def clear_result_bundle_cache() -> None:
    """Clears the explicit result-bundle cache."""
    _RESULT_BUNDLE_CACHE.clear()


load_result_bundle.clear = clear_result_bundle_cache  # type: ignore[attr-defined]
