from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict
from uuid import uuid4


ARTIFACT_SCHEMA_VERSION = "1.1"
DEFAULT_ENGINE_VERSION = "workspace"
RERUN_REQUIRED_FIELDS = (
    "source_config_path",
    "run_seed",
    "config_hash",
    "data_version",
)


def build_artifact_identity(run_type: str, artifact_path: Path, project_root: Path) -> Dict[str, str]:
    """
    Builds stable identity metadata for a saved artifact bundle.

    Methodology:
        Artifact identity is separated from any specific UI runtime so that
        caches, async jobs, and FastAPI endpoints can all reference the same
        saved bundle without guessing from folder names alone.

    Args:
        run_type: Artifact namespace such as ``single`` or ``portfolio``.
        artifact_path: Directory that contains the persisted artifact files.
        project_root: Repository root used to resolve the engine version.

    Returns:
        JSON-safe metadata fields that become part of manifest.json.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    artifact_id = (
        f"{run_type}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{uuid4().hex[:8]}"
    )
    return {
        "artifact_id": artifact_id,
        "run_id": artifact_id,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "engine_version": resolve_engine_version(project_root),
        "artifact_created_at": created_at,
        "artifact_path": str(artifact_path.resolve()),
    }


@lru_cache(maxsize=16)
def resolve_engine_version(project_root: Path) -> str:
    """
    Resolves the current engine version from git when available.

    Methodology:
        A lightweight version marker is used before a packaged release process
        exists. The current git revision is specific enough for cache keys and
        async-job provenance, while gracefully degrading outside a git checkout.

    Args:
        project_root: Repository root used as the git working directory.

    Returns:
        Short git revision string, or ``workspace`` when unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return DEFAULT_ENGINE_VERSION

    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        return DEFAULT_ENGINE_VERSION
    return version
