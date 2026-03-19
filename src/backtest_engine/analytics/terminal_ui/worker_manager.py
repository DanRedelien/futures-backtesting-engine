from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Literal, Optional

from src.backtest_engine.analytics.dashboard.core.paths import get_results_dir
from src.backtest_engine.analytics.terminal_ui.jobs import TerminalQueueConfig


WorkerLifecycleState = Literal["stopped", "starting", "running", "crashed"]


def _utc_now_iso() -> str:
    """Returns the current UTC timestamp in ISO format for worker metadata."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ManagedWorkerSnapshot:
    """
    Captures the current state of the app-owned local worker process.

    Methodology:
        The UI needs one stable, JSON-safe snapshot that can describe the worker
        lifecycle without depending on subprocess objects or polling logic.
    """

    state: WorkerLifecycleState
    is_running: bool
    started_by_app: bool
    pid: Optional[int]
    started_at: str
    exit_code: Optional[int]
    last_error: str
    log_path: str
    command: str

    def to_public_dict(self) -> dict[str, object]:
        """Returns JSON-safe worker state for UI rendering and tests."""
        return asdict(self)


class LocalWorkerManager:
    """
    Manages one local RQ worker subprocess for the terminal dashboard session.

    Methodology:
        This manager owns only the local worker lifecycle. It does not replace
        queue semantics; it simply gives the Stress Testing UI a safe app-local
        way to start and observe a compatible worker.
    """

    def __init__(
        self,
        *,
        config: TerminalQueueConfig,
        results_dir: Optional[str],
        project_root: Path,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.results_root = Path(results_dir) if results_dir is not None else get_results_dir()
        self.log_path = self.results_root / "jobs" / "managed-worker.log"
        self._process: Optional[subprocess.Popen[str]] = None
        self._log_handle: Optional[IO[str]] = None
        self._started_at: str = ""
        self._exit_code: Optional[int] = None
        self._last_error: str = ""
        self._state_hint: WorkerLifecycleState = "stopped"
        self._lock = threading.Lock()

    def _resolve_rq_executable(self) -> str:
        """
        Resolves the rq CLI executable path for the active Python environment.

        Methodology:
            rq 1.x ships a console_scripts entry point ('rq') but has no __main__.py,
            so 'python -m rq' fails. We locate the installed script directly via
            shutil.which first, then fall back to the Scripts/bin directory adjacent
            to sys.executable so the correct virtualenv entry point is always used.
        """
        found = shutil.which("rq")
        if found:
            return found
        scripts_dir = Path(sys.executable).parent
        candidates = [
            scripts_dir / "rq.exe",
            scripts_dir / "rq",
            scripts_dir / "Scripts" / "rq.exe",
            scripts_dir / "Scripts" / "rq",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return "rq"

    def _build_command(self) -> list[str]:
        """
        Builds the exact worker command for the configured queue.

        On Windows, os.fork() is unavailable, so we use rq.SimpleWorker which
        executes jobs synchronously in-process instead of forking a child process.
        """
        command = [self._resolve_rq_executable(), "worker"]
        if self.config.redis_url:
            command.extend(["--url", self.config.redis_url])
        if sys.platform == "win32":
            command.extend([
                "--worker-class",
                "src.backtest_engine.analytics.terminal_ui.windows_worker.WindowsSimpleWorker",
            ])
        command.append(self.config.queue_name)
        return command

    def _command_display(self) -> str:
        """Returns the worker command formatted for user-facing diagnostics."""
        return subprocess.list2cmdline(self._build_command())

    def _write_log_banner(self, action: str) -> None:
        """Appends a small lifecycle banner to the managed worker log file."""
        if self._log_handle is None:
            return
        self._log_handle.write(f"\n[{_utc_now_iso()}] {action}\n")
        self._log_handle.flush()

    def _close_log_handle_locked(self) -> None:
        """Closes the current log file handle if one is open."""
        if self._log_handle is None:
            return
        try:
            self._log_handle.flush()
        except Exception:
            pass
        try:
            self._log_handle.close()
        except Exception:
            pass
        self._log_handle = None

    def _read_log_tail(self, max_chars: int = 400) -> str:
        """Reads the tail of the worker log for crash diagnostics."""
        if not self.log_path.exists():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        return text[-max_chars:].strip()

    def _snapshot_locked(self) -> ManagedWorkerSnapshot:
        """Builds the current worker snapshot while the state lock is held."""
        self._sync_process_state_locked()
        state = self._state_hint
        is_running = False
        pid: Optional[int] = None
        if self._process is not None and self._process.poll() is None:
            pid = int(self._process.pid) if self._process.pid is not None else None
            is_running = True
            if self._started_at:
                started_dt = datetime.fromisoformat(self._started_at)
                elapsed_seconds = (datetime.now(timezone.utc) - started_dt).total_seconds()
                if elapsed_seconds >= float(self.config.worker_start_grace_seconds):
                    state = "running"
                else:
                    state = "starting"
            else:
                state = "starting"
        elif state not in {"crashed"}:
            state = "stopped"
        return ManagedWorkerSnapshot(
            state=state,
            is_running=is_running,
            started_by_app=bool(self._started_at),
            pid=pid,
            started_at=self._started_at,
            exit_code=self._exit_code,
            last_error=self._last_error,
            log_path=str(self.log_path.resolve()),
            command=self._command_display(),
        )

    def _sync_process_state_locked(self) -> None:
        """Synchronizes persisted worker state from the current subprocess handle."""
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is None:
            return
        self._exit_code = int(exit_code)
        if exit_code == 0:
            self._state_hint = "stopped"
            if not self._last_error:
                self._last_error = ""
        else:
            self._state_hint = "crashed"
            if not self._last_error:
                tail = self._read_log_tail()
                tail_message = f" Log tail: {tail}" if tail else ""
                self._last_error = f"Worker exited with code {exit_code}.{tail_message}"
        self._process = None
        self._close_log_handle_locked()

    def snapshot(self) -> ManagedWorkerSnapshot:
        """Returns the current managed-worker snapshot for UI rendering."""
        with self._lock:
            return self._snapshot_locked()

    def start_worker(self) -> ManagedWorkerSnapshot:
        """
        Starts the local worker subprocess when it is not already running.

        Methodology:
            Duplicate launches are avoided by reusing the existing managed worker
            when it is already starting or running in this app session.
        """

        with self._lock:
            current = self._snapshot_locked()
            if current.state in {"starting", "running"}:
                return current
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._last_error = ""
            self._exit_code = None
            self._started_at = _utc_now_iso()
            self._state_hint = "starting"
            try:
                self._log_handle = self.log_path.open("a", encoding="utf-8")
                self._write_log_banner("Starting managed worker")
                cmd = self._build_command()
                env = None
                if sys.platform == "win32":
                    env = dict(os.environ)
                    project_root_str = str(self.project_root)
                    path_key = "PYTHONPATH"
                    env[path_key] = project_root_str if path_key not in env else f"{project_root_str}{os.pathsep}{env[path_key]}"
                self._process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            except Exception as exc:
                self._process = None
                self._state_hint = "crashed"
                self._last_error = str(exc)
                self._close_log_handle_locked()
            return self._snapshot_locked()

    def stop_worker(self) -> ManagedWorkerSnapshot:
        """Stops the managed worker subprocess if it is currently running."""
        with self._lock:
            self._sync_process_state_locked()
            if self._process is None or self._process.poll() is not None:
                self._state_hint = "stopped" if self._state_hint != "crashed" else self._state_hint
                return self._snapshot_locked()
            try:
                self._write_log_banner("Stopping managed worker")
                self._process.terminate()
                self._process.wait(timeout=float(self.config.worker_stop_timeout_seconds))
            except Exception:
                try:
                    self._process.kill()
                    self._process.wait(timeout=float(self.config.worker_stop_timeout_seconds))
                except Exception as exc:
                    self._state_hint = "crashed"
                    self._last_error = str(exc)
            finally:
                if self._process is not None:
                    self._exit_code = self._process.returncode
                self._process = None
                if self._state_hint != "crashed":
                    self._state_hint = "stopped"
                    self._last_error = ""
                self._close_log_handle_locked()
            return self._snapshot_locked()


RedisLifecycleState = Literal["stopped", "starting", "live", "error"]


@dataclass(frozen=True)
class ManagedRedisSnapshot:
    """
    Captures the current state of the app-owned local redis-server process.

    Methodology:
        Uses a TCP socket ping rather than a time-elapsed grace period so the
        UI shows a real liveness signal instead of a heuristic estimate.
    """

    state: RedisLifecycleState
    is_live: bool
    started_by_app: bool
    pid: Optional[int]
    started_at: str
    exit_code: Optional[int]
    last_error: str
    log_path: str
    host: str
    port: int

    def to_public_dict(self) -> dict[str, object]:
        """Returns JSON-safe redis state for UI rendering."""
        return asdict(self)


class LocalRedisManager:
    """
    Manages one local redis-server subprocess for the terminal dashboard session.

    Methodology:
        Mirrors LocalWorkerManager but targets redis-server instead of an RQ
        worker. Uses a TCP socket ping (no redis package required) to distinguish
        'starting' from 'live', giving accurate liveness feedback without
        depending on the Python redis client being installed.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        results_dir: Optional[str],
        project_root: Path,
    ) -> None:
        self._host = host
        self._port = port
        self.project_root = project_root
        self.results_root = Path(results_dir) if results_dir is not None else get_results_dir()
        self.log_path = self.results_root / "jobs" / "managed-redis.log"
        self._process: Optional[subprocess.Popen[str]] = None
        self._log_handle: Optional[IO[str]] = None
        self._started_at: str = ""
        self._exit_code: Optional[int] = None
        self._last_error: str = ""
        self._state_hint: RedisLifecycleState = "stopped"
        self._lock = threading.Lock()

    def _build_command(self) -> list[str]:
        """Builds the redis-server command for the configured host and port."""
        return ["redis-server", "--port", str(self._port), "--bind", self._host]

    def _command_display(self) -> str:
        """Returns the redis-server command formatted for user-facing diagnostics."""
        return subprocess.list2cmdline(self._build_command())

    def _ping(self) -> bool:
        """Checks whether redis-server is accepting TCP connections on the configured port."""
        import socket
        try:
            with socket.create_connection((self._host, self._port), timeout=0.5):
                return True
        except OSError:
            return False

    def _write_log_banner(self, action: str) -> None:
        """Appends a small lifecycle banner to the managed redis log file."""
        if self._log_handle is None:
            return
        self._log_handle.write(f"\n[{_utc_now_iso()}] {action}\n")
        self._log_handle.flush()

    def _close_log_handle_locked(self) -> None:
        """Closes the current log file handle if one is open."""
        if self._log_handle is None:
            return
        try:
            self._log_handle.flush()
        except Exception:
            pass
        try:
            self._log_handle.close()
        except Exception:
            pass
        self._log_handle = None

    def _read_log_tail(self, max_chars: int = 400) -> str:
        """Reads the tail of the redis log for crash diagnostics."""
        if not self.log_path.exists():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except Exception:
            return ""
        return text[-max_chars:].strip()

    def _sync_process_state_locked(self) -> None:
        """Synchronizes persisted state from the current subprocess handle."""
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is None:
            return
        self._exit_code = int(exit_code)
        if exit_code == 0:
            self._state_hint = "stopped"
        else:
            self._state_hint = "error"
            if not self._last_error:
                tail = self._read_log_tail()
                tail_msg = f" Log tail: {tail}" if tail else ""
                self._last_error = f"redis-server exited with code {exit_code}.{tail_msg}"
        self._process = None
        self._close_log_handle_locked()

    def _snapshot_locked(self) -> ManagedRedisSnapshot:
        """
        Builds the current redis snapshot while the state lock is held.

        Ping is checked regardless of whether the app owns the process so that
        Redis installed as an OS service (e.g. via winget on Windows) is detected
        as live without the app having launched it.
        """
        self._sync_process_state_locked()
        state = self._state_hint
        is_live = False
        pid: Optional[int] = None
        if self._process is not None and self._process.poll() is None:
            pid = int(self._process.pid) if self._process.pid is not None else None
            if self._ping():
                state = "live"
                is_live = True
            else:
                state = "starting"
        elif state not in {"error"}:
            if self._ping():
                state = "live"
                is_live = True
            else:
                state = "stopped"
        return ManagedRedisSnapshot(
            state=state,
            is_live=is_live,
            started_by_app=bool(self._started_at),
            pid=pid,
            started_at=self._started_at,
            exit_code=self._exit_code,
            last_error=self._last_error,
            log_path=str(self.log_path.resolve()),
            host=self._host,
            port=self._port,
        )

    def snapshot(self) -> ManagedRedisSnapshot:
        """Returns the current managed-redis snapshot for UI rendering."""
        with self._lock:
            return self._snapshot_locked()

    def start_redis(self) -> ManagedRedisSnapshot:
        """
        Starts the local redis-server subprocess when it is not already running.

        Methodology:
            Duplicate launches are avoided by reusing the existing managed
            redis-server when it is already starting or live in this app session.
        """
        with self._lock:
            current = self._snapshot_locked()
            if current.state in {"starting", "live"}:
                return current
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._last_error = ""
            self._exit_code = None
            self._started_at = _utc_now_iso()
            self._state_hint = "starting"
            try:
                self._log_handle = self.log_path.open("a", encoding="utf-8")
                self._write_log_banner("Starting managed redis-server")
                self._process = subprocess.Popen(
                    self._build_command(),
                    cwd=str(self.project_root),
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except FileNotFoundError:
                self._process = None
                self._state_hint = "error"
                self._last_error = (
                    "redis-server not found in PATH. "
                    "Install Redis for Windows: run 'winget install Redis.Redis' in a terminal, "
                    "then restart the dashboard."
                )
                self._close_log_handle_locked()
            except Exception as exc:
                self._process = None
                self._state_hint = "error"
                self._last_error = str(exc)
                self._close_log_handle_locked()
            return self._snapshot_locked()

    def stop_redis(self) -> ManagedRedisSnapshot:
        """Stops the managed redis-server subprocess if it is currently running."""
        with self._lock:
            self._sync_process_state_locked()
            if self._process is None or self._process.poll() is not None:
                if self._state_hint != "error":
                    self._state_hint = "stopped"
                return self._snapshot_locked()
            try:
                self._write_log_banner("Stopping managed redis-server")
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except Exception:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
                except Exception as exc:
                    self._state_hint = "error"
                    self._last_error = str(exc)
            finally:
                if self._process is not None:
                    self._exit_code = self._process.returncode
                self._process = None
                if self._state_hint != "error":
                    self._state_hint = "stopped"
                    self._last_error = ""
                self._close_log_handle_locked()
            return self._snapshot_locked()
