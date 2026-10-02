"""Thread-safe caches for the serving layer."""

from __future__ import annotations

import threading
import time

__all__ = ["EnvelopeCache", "envelope_cache"]


class EnvelopeCache:
    """Bounded TTL cache keyed by a differential-vector fingerprint.

    The scoring path resolves one envelope per distinct feature vector; repeated
    rows inside a batch therefore collapse onto a single resolution.
    """

    def __init__(self, ttl: float = 300.0, capacity: int = 2048):
        self.ttl = float(ttl)
        self.capacity = int(capacity)
        self._store: dict[str, tuple[float, dict]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            stamped, value = entry
            if self.ttl and (time.time() - stamped) > self.ttl:
                self._store.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return dict(value)

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            if len(self._store) >= self.capacity:
                oldest = min(self._store.items(), key=lambda item: item[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (time.time(), dict(value))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


envelope_cache = EnvelopeCache()
