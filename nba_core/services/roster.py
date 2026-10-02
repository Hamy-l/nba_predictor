"""Roster catalogue service for the player-news workflow."""

from __future__ import annotations

import json
import threading
import time

from constants import NBA_ROSTER_FILE

__all__ = ["RosterService", "roster_service", "RosterUnavailable"]


class RosterUnavailable(RuntimeError):
    """Raised when the roster catalogue cannot be read."""


class RosterService:
    """Cached, thread-safe reader for the JSONL roster catalogue."""

    def __init__(self, path: str | None = None, ttl: float = 300.0):
        self.path = path or NBA_ROSTER_FILE
        self.ttl = float(ttl)
        self._lock = threading.RLock()
        self._cache: list[dict] | None = None
        self._stamp = 0.0

    def load(self, *, force: bool = False) -> list[dict]:
        """Return the roster entries, re-reading at most once per TTL."""
        with self._lock:
            fresh = (time.time() - self._stamp) < self.ttl
            if self._cache is not None and fresh and not force:
                return list(self._cache)

            try:
                entries = self._read()
            except OSError as exc:
                raise RosterUnavailable(str(exc)) from exc

            self._cache = entries
            self._stamp = time.time()
            return list(entries)

    def _read(self) -> list[dict]:
        entries: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entries.append(json.loads(line))
        return entries

    def team_names(self) -> list[str]:
        return [entry.get("team", "") for entry in self.load() if entry.get("team")]

    def invalidate(self) -> None:
        with self._lock:
            self._stamp = 0.0


roster_service = RosterService()
