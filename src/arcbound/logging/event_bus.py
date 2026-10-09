"""In-memory event bus for live GUI notifications.

The decision engine emits short, human-readable events (e.g. "a fallback was
used and why", "epsilon exploration triggered") into this bus. The GUI polls
``/monitoring/events`` to drain new events and render them in the action
terminal with the appropriate color (red for errors, yellow for warnings).

This is deliberately separate from the loguru file logs: it is a low-latency,
bounded, in-memory channel for things the user needs to *see happen* in the
terminal, not a full audit trail.
"""

import threading
import time
from typing import Optional


class EventBus:
    """A bounded, thread-safe ring buffer of timestamped events.

    Each event is a dict: ``{"seq": int, "level": str, "message": str, "ts": float}``.
    ``seq`` is a monotonically increasing integer the GUI uses for incremental
    polling (``since``). ``level`` is one of ``"error"``, ``"warning"``, or
    ``"info"`` and maps to a terminal color.
    """

    def __init__(self, maxlen: int = 500):
        self._maxlen = maxlen
        self._events: list[dict] = []
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, level: str, message: str) -> int:
        """Append an event and return its seq number."""
        with self._lock:
            self._seq += 1
            self._events.append(
                {"seq": self._seq, "level": level, "message": message, "ts": time.time()}
            )
            if len(self._events) > self._maxlen:
                self._events = self._events[-self._maxlen:]
            return self._seq

    def since(self, seq: int = 0) -> list[dict]:
        """Return all events with seq > ``seq`` (oldest first)."""
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def clear(self) -> None:
        with self._lock:
            self._events = []
            self._seq = 0
