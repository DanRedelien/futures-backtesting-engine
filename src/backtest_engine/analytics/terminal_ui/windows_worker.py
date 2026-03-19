"""
Windows-compatible RQ worker that uses TimerDeathPenalty instead of UnixSignalDeathPenalty.

Methodology:
    UnixSignalDeathPenalty uses signal.SIGALRM, which does not exist on Windows.
    TimerDeathPenalty uses threading.Timer and skips setup when timeout <= 0.
    This module provides a drop-in worker for Windows that avoids SIGALRM.
"""
from __future__ import annotations

from rq import SimpleWorker
from rq.timeouts import TimerDeathPenalty


class WindowsSimpleWorker(SimpleWorker):
    """
    SimpleWorker that uses TimerDeathPenalty (Windows-safe) instead of UnixSignalDeathPenalty.

    Use this worker class on Windows via --worker-class when launching rq worker.
    """
    death_penalty_class = TimerDeathPenalty
