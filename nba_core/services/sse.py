"""Server-sent-event framing shared by the streaming workflows."""

from __future__ import annotations

import json
import queue
import threading
from typing import Iterator

__all__ = ["encode_event", "EventStream", "drain", "iter_queue"]


def encode_event(payload: dict) -> str:
    """Frame one payload as an SSE frame."""
    return f"data: {json.dumps(payload)}\n\n"


def drain(work_queue: "queue.Queue[dict]") -> Iterator[str]:
    """Yield every frame currently buffered in ``work_queue``."""
    while not work_queue.empty():
        try:
            payload = work_queue.get_nowait()
        except queue.Empty:
            break
        yield encode_event(payload)


def iter_queue(work_queue: "queue.Queue[dict]", completed: threading.Event,
               poll: float = 0.01) -> Iterator[str]:
    """Yield frames as producers push them until every producer has finished."""
    while not completed.is_set() or not work_queue.empty():
        try:
            payload = work_queue.get(timeout=poll)
        except queue.Empty:
            if completed.is_set() and work_queue.empty():
                break
            continue
        yield encode_event(payload)


class EventStream:
    """Fan-in coordinator for concurrent SSE producers."""

    def __init__(self, expected: int):
        self.expected = max(1, int(expected))
        self.queue: "queue.Queue[dict]" = queue.Queue()
        self.completed = threading.Event()
        self._lock = threading.Lock()
        self._finished = 0

    def emit(self, payload: dict) -> None:
        self.queue.put(payload)

    def finish(self) -> None:
        """Mark one producer as done; trips the barrier on the last one."""
        with self._lock:
            self._finished += 1
            if self._finished >= self.expected:
                self.completed.set()

    def frames(self) -> Iterator[str]:
        yield from iter_queue(self.queue, self.completed)
        yield from drain(self.queue)
