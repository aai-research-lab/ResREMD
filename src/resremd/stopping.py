"""Stopping cleanly when asked."""

from __future__ import annotations

import logging
import signal
import threading
from typing import Any

logger = logging.getLogger("resremd")


class StopRequests:
    """Turns SIGINT and SIGTERM into a request to stop at the next cycle.

    A batch scheduler sends SIGTERM before it kills a job. Stopping at a
    cycle boundary lets the run write a checkpoint and say it stopped,
    instead of losing everything since the last one.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "StopRequests":
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._previous[sig] = signal.signal(sig, self._note)
        return self

    def _note(self, number: int, _frame: Any) -> None:
        # A second interrupt from the keyboard means now. A second SIGTERM
        # usually means a wrapper and the scheduler both passed one on, and
        # is not a reason to lose the checkpoint.
        if self.requested and number == signal.SIGINT:
            raise KeyboardInterrupt
        if not self.requested:
            logger.warning("Signal %d: stopping at the next safe point, with "
                           "a checkpoint. Ctrl-C again stops at once.", number)
        self.requested = True

    def __exit__(self, *_exc: Any) -> bool:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False
