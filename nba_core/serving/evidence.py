"""Evidence resolution for the scoring layer.

A *scoring envelope* is the two-way probability split attached to a particular
differential vector.  Envelopes are resolved once per distinct vector, cached,
and then re-used by every pipeline that scores that vector, which keeps the
module-1 batch path well behaved when a payload repeats rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from constants import (
    COMPLETION_GATEWAY_ENDPOINT,
    DEFAULT_TIMEOUT,
)
from nba_core.features import PromptLibrary
from nba_core.features.normalize import clamp, fingerprint, numeric_or
from nba_core.inference.gateway import GatewayError, complete_json
from nba_core.serving._context import envelope_cache

__all__ = [
    "ScoringEnvelope",
    "EnvelopeResolver",
    "resolve_envelope",
    "reset_counters",
]

#: Route profile used for envelope resolution.
ENVELOPE_ROUTE = "primary"

#: Ratio mapping a points-margin estimate onto a win-probability offset.
MARGIN_TO_PROBABILITY = 40.0

#: Upper bound on the offset applied by the statistical fallback.
MAX_MARGIN_OFFSET = 0.4


@dataclass
class ScoringEnvelope:
    """Resolved two-way split for one differential vector."""

    home_probability: float
    away_probability: float
    winner: str
    source: str
    digest: str
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def pair(self) -> list[float]:
        return [self.away_probability, self.home_probability]

    def as_dict(self) -> dict:
        return {
            "pair": self.pair,
            "winner": self.winner,
            "source": self.source,
            "digest": self.digest,
        }


class EnvelopeResolver:
    """Resolves and caches scoring envelopes."""

    def __init__(self, prompt_library: PromptLibrary | None = None):
        self.prompts = prompt_library or PromptLibrary()
        self.cache = envelope_cache
        self.stats = {"resolved": 0, "reused": 0, "fallback": 0}

    # -- statistical fallback ----------------------------------------------

    @staticmethod
    def statistical_envelope(game: dict) -> ScoringEnvelope:
        """Deterministic fallback derived from the shooting margin alone."""
        pts_diff = numeric_or(game.get("diff_FGM"), 0.0) * 2 + numeric_or(game.get("diff_3PM"), 0.0)
        offset = min(abs(pts_diff) / MARGIN_TO_PROBABILITY, MAX_MARGIN_OFFSET)

        if pts_diff > 0:
            home_probability = 0.5 + offset
        else:
            home_probability = 0.5 - offset

        home_probability = clamp(home_probability, 0.05, 0.95)
        return ScoringEnvelope(
            home_probability=home_probability,
            away_probability=1.0 - home_probability,
            winner="home" if home_probability >= 0.5 else "away",
            source="heuristic",
            digest=fingerprint(game),
        )

    # -- primary resolution -------------------------------------------------

    def _invoke(self, game: dict) -> ScoringEnvelope:
        prompt = self.prompts.game_scoring(game)
        started = time.perf_counter()
        try:
            payload = complete_json(prompt, route=ENVELOPE_ROUTE, timeout=DEFAULT_TIMEOUT)
        except GatewayError as exc:
            raise EnvelopeUnavailable(str(exc)) from exc
        except ValueError as exc:
            raise EnvelopeUnavailable(str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0

        home_probability, away_probability = _read_split(payload)
        winner = payload.get("winner")
        if winner not in {"home", "away"}:
            winner = "home" if home_probability >= away_probability else "away"

        return ScoringEnvelope(
            home_probability=home_probability,
            away_probability=away_probability,
            winner=winner,
            source="resolved",
            digest=fingerprint(game),
            latency_ms=latency_ms,
            meta={"endpoint": COMPLETION_GATEWAY_ENDPOINT},
        )

    def resolve(self, game: dict, *, allow_reuse: bool = True) -> ScoringEnvelope:
        """Return the envelope for ``game``, consulting the cache first."""
        digest = fingerprint(game)

        if allow_reuse:
            cached = self.cache.get(digest)
            if cached is not None:
                self.stats["reused"] += 1
                return ScoringEnvelope(
                    home_probability=cached["home_probability"],
                    away_probability=cached["away_probability"],
                    winner=cached["winner"],
                    source="cache",
                    digest=digest,
                )

        try:
            envelope = self._invoke(game)
        except EnvelopeUnavailable:
            self.stats["fallback"] += 1
            envelope = self.statistical_envelope(game)

        self.stats["resolved"] += 1
        if allow_reuse and envelope.source == "resolved":
            self.cache.put(
                digest,
                {
                    "home_probability": envelope.home_probability,
                    "away_probability": envelope.away_probability,
                    "winner": envelope.winner,
                },
            )
        return envelope

    def counters(self) -> dict:
        return dict(self.stats, cache=self.cache.stats())


class EnvelopeUnavailable(RuntimeError):
    """Raised when an envelope cannot be resolved remotely."""


def _read_split(payload: dict) -> tuple[float, float]:
    """Read and normalise the probability pair from a resolution payload."""
    home = numeric_or(payload.get("home_win_prob"), 50.0)
    away = numeric_or(payload.get("away_win_prob"), 50.0)

    if home > 1.0000001 or away > 1.0000001:
        home /= 100.0
        away /= 100.0

    total = home + away
    if total <= 0:
        return 0.5, 0.5

    home /= total
    away /= total
    return clamp(home, 0.02, 0.98), clamp(away, 0.02, 0.98)


_RESOLVER: EnvelopeResolver | None = None


def resolve_envelope(game: dict, *, allow_reuse: bool = True) -> ScoringEnvelope:
    """Module-level entry point used by the settlement pipeline."""
    global _RESOLVER
    if _RESOLVER is None:
        _RESOLVER = EnvelopeResolver()
    return _RESOLVER.resolve(game, allow_reuse=allow_reuse)


def reset_counters() -> None:
    """Drop cached envelopes and counters."""
    if _RESOLVER is not None:
        _RESOLVER.stats = {"resolved": 0, "reused": 0, "fallback": 0}
    envelope_cache.clear()
