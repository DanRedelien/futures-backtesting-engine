"""
Readiness and snapshot helpers for the terminal scenario queue service.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from .scenario_job_models import TerminalQueueConfig


def build_worker_start_command(
    config: TerminalQueueConfig,
    worker_manager: Optional[Any],
) -> str:
    """Returns the expected worker command for the configured queue."""
    if worker_manager is not None:
        return worker_manager.snapshot().command
    base_command = f'"{sys.executable}" -m rq worker'
    if config.redis_url:
        return f'{base_command} --url "{config.redis_url}" {config.queue_name}'
    return f"{base_command} {config.queue_name}"


def build_worker_snapshot(
    *,
    worker_manager: Optional[Any],
    worker_start_command: str,
) -> Dict[str, Any]:
    """Returns a JSON-safe worker snapshot."""
    if worker_manager is None:
        return {
            "state": "stopped",
            "is_running": False,
            "started_by_app": False,
            "pid": None,
            "started_at": "",
            "exit_code": None,
            "last_error": "",
            "log_path": "",
            "command": worker_start_command,
        }
    return worker_manager.snapshot().to_public_dict()


def build_redis_manager_snapshot(redis_manager: Optional[Any]) -> Dict[str, Any]:
    """Returns a JSON-safe managed-redis snapshot."""
    if redis_manager is None:
        return {
            "state": "stopped",
            "is_live": False,
            "started_by_app": False,
            "pid": None,
            "started_at": "",
            "exit_code": None,
            "last_error": "",
            "log_path": "",
            "host": "",
            "port": 0,
        }
    return redis_manager.snapshot().to_public_dict()


def build_missing_dependency_message(missing_packages: List[str]) -> str:
    """Builds a user-facing message for missing Python dependencies."""
    if not missing_packages:
        return ""
    if len(missing_packages) == 1:
        return (
            "Background worker is unavailable because Python package "
            f"{missing_packages[0]} is not installed in this environment."
        )
    packages = ", ".join(missing_packages)
    return (
        "Background worker is unavailable because these Python packages are "
        f"missing from this environment: {packages}."
    )


def build_readiness_summary(
    *,
    config: TerminalQueueConfig,
    redis_manager: Optional[Any],
    dependencies: Dict[str, Any],
    backend: Dict[str, Any],
    worker: Dict[str, Any],
    redis_mgr: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the user-facing readiness summary for Stress Testing."""
    missing_packages = list(dependencies.get("missing_packages", []))
    worker_state = str(worker.get("state", "stopped"))
    redis_state = str(redis_mgr.get("state", "stopped"))
    has_redis_manager = redis_manager is not None
    queueing_available = (
        bool(dependencies.get("rq_installed"))
        and bool(dependencies.get("redis_installed"))
        and bool(backend.get("redis_url_configured"))
        and bool(backend.get("redis_reachable"))
    )
    can_stop_redis = has_redis_manager and redis_state == "live"

    if missing_packages:
        return {
            "readiness_state": "missing_dependencies",
            "readiness_message": build_missing_dependency_message(missing_packages),
            "worker_status_label": "Install requirements first.",
            "can_start_worker": False,
            "ready_to_queue": False,
            "can_start_redis": False,
            "can_stop_redis": False,
        }
    if not bool(backend.get("redis_url_configured")):
        return {
            "readiness_state": "backend_not_configured",
            "readiness_message": "Redis URL is not configured. Set REDIS_URL in your environment or .env file.",
            "worker_status_label": "Redis not configured.",
            "can_start_worker": False,
            "ready_to_queue": False,
            "can_start_redis": False,
            "can_stop_redis": False,
        }
    if not bool(backend.get("redis_reachable")):
        if redis_state == "starting":
            return {
                "readiness_state": "redis_starting",
                "readiness_message": "Redis is starting. The panel will refresh automatically.",
                "worker_status_label": "Redis is starting.",
                "can_start_worker": False,
                "ready_to_queue": False,
                "can_start_redis": False,
                "can_stop_redis": False,
            }
        can_start_redis = has_redis_manager and redis_state not in {"starting", "live"}
        return {
            "readiness_state": "backend_unreachable",
            "readiness_message": f"Redis is not running at {config.redis_url}.",
            "worker_status_label": "Redis offline.",
            "can_start_worker": False,
            "ready_to_queue": False,
            "can_start_redis": can_start_redis,
            "can_stop_redis": False,
        }
    if worker_state == "crashed":
        return {
            "readiness_state": "worker_crashed",
            "readiness_message": (
                str(worker.get("last_error", "")).strip()
                or "Worker started, but exited immediately."
            ),
            "worker_status_label": "Managed worker crashed.",
            "can_start_worker": True,
            "ready_to_queue": False,
            "can_start_redis": False,
            "can_stop_redis": can_stop_redis,
        }
    if worker_state == "starting":
        return {
            "readiness_state": "worker_starting",
            "readiness_message": "Local worker is starting. The panel will refresh automatically.",
            "worker_status_label": "Managed worker is starting.",
            "can_start_worker": False,
            "ready_to_queue": False,
            "can_start_redis": False,
            "can_stop_redis": can_stop_redis,
        }
    if worker_state == "running":
        return {
            "readiness_state": "ready",
            "readiness_message": "Ready. Queue a stress test now.",
            "worker_status_label": "Managed worker is running.",
            "can_start_worker": False,
            "ready_to_queue": queueing_available,
            "can_start_redis": False,
            "can_stop_redis": can_stop_redis,
        }
    return {
        "readiness_state": "worker_stopped",
        "readiness_message": "Redis is live. Start the local worker to begin stress testing.",
        "worker_status_label": "Managed worker is stopped.",
        "can_start_worker": queueing_available,
        "ready_to_queue": False,
        "can_start_redis": False,
        "can_stop_redis": can_stop_redis,
    }
