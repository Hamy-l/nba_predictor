"""Module 3 -- roster news workflow and matchup verdict."""

from __future__ import annotations

import concurrent.futures
import logging

from constants import DEFAULT_TIMEOUT
from nba_core.features import PromptLibrary
from nba_core.inference.gateway import GatewayError, complete_json, stream_events
from nba_core.services.sse import EventStream, encode_event

__all__ = ["PlayerNewsService", "player_news_service", "DELTA_EVENTS"]

logger = logging.getLogger(__name__)

DELTA_EVENTS = frozenset({"response.reasoning_text.delta", "response.output_text.delta"})

#: Per-side event namespace.
SIDE_EVENTS = {
    "home": ("home_chunk", "home_complete", "home_error"),
    "away": ("away_chunk", "away_complete", "away_error"),
}


class PlayerNewsService:
    """Concurrent roster-news retrieval plus the final verdict."""

    def __init__(self, prompts: PromptLibrary | None = None):
        self.prompts = prompts or PromptLibrary()

    # -- streaming retrieval ------------------------------------------------

    def stream_news(self, home_team: str, away_team: str,
                    home_players: list[str], away_players: list[str]):
        """Yield SSE frames for both roster sweeps."""
        sides = (
            ("home", home_team, home_players or []),
            ("away", away_team, away_players or []),
        )
        stream = EventStream(len(sides))

        def collect(side: str, team: str, players: list[str]) -> None:
            chunk_event, complete_event, error_event = SIDE_EVENTS[side]
            try:
                prompt = self.prompts.roster_digest(team, players)
                buffer: list[str] = []
                for chunk in stream_events(prompt, route="retrieval-stream",
                                           timeout=DEFAULT_TIMEOUT):
                    if chunk.get("type") not in DELTA_EVENTS:
                        continue
                    text = chunk.get("delta", "")
                    if not isinstance(text, str) or not text:
                        continue
                    buffer.append(text)
                    stream.emit({"type": chunk_event, "chunk": text})

                stream.emit({"type": complete_event, "content": "".join(buffer)})
            except Exception as exc:  # noqa: BLE001 - reported inline to the client
                logger.warning("roster sweep %s failed: %s", side, exc)
                stream.emit({"type": error_event, "error": str(exc)})
            finally:
                stream.finish()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sides)) as executor:
            for side, team, players in sides:
                executor.submit(collect, side, team, players)
            yield from stream.frames()

        yield encode_event({"type": "all_complete"})

    # -- matchup verdict ----------------------------------------------------

    def verdict(self, home_team: str, away_team: str,
                home_news: str, away_news: str) -> dict:
        """Produce the final verdict from both news digests."""
        prompt = self.prompts.matchup_verdict(home_team, away_team, home_news, away_news)

        try:
            payload = complete_json(prompt, route="primary", timeout=DEFAULT_TIMEOUT)
        except GatewayError as exc:
            raise RuntimeError(f"prediction failed: {exc}") from exc
        except ValueError as exc:
            logger.warning("verdict payload was not valid JSON: %s", exc)
            return _fallback_verdict(home_team)

        home_prob, away_prob = _normalise(
            payload.get("home_probability"), payload.get("away_probability")
        )

        return {
            "prediction": {
                "winner": payload.get("winner", "home"),
                "home_probability": round(home_prob, 1),
                "away_probability": round(away_prob, 1),
            },
            "analysis": payload.get("analysis", "Analysis unavailable"),
            "confidence": payload.get("confidence", "medium"),
        }


def _normalise(home_value, away_value) -> tuple[float, float]:
    try:
        home = float(home_value)
    except (TypeError, ValueError):
        home = 50.0
    try:
        away = float(away_value)
    except (TypeError, ValueError):
        away = 50.0

    total = home + away
    if total <= 0:
        return 50.0, 50.0
    return home / total * 100.0, away / total * 100.0


def _fallback_verdict(home_team: str) -> dict:
    """Conservative verdict used when the payload cannot be parsed."""
    return {
        "prediction": {
            "winner": "home",
            "home_probability": 52.0,
            "away_probability": 48.0,
        },
        "analysis": (
            "Based on the available information, this appears to be a closely "
            f"matched game. {home_team} has a slight advantage playing at home."
        ),
        "confidence": "low",
    }


player_news_service = PlayerNewsService()
