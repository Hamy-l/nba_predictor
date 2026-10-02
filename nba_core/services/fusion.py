"""Module 2 -- multi-source fusion workflow.

Stages: structured feature assembly, concurrent evidence retrieval, evidence
condensation and the fused verdict.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass

from constants import (
    DEFAULT_TIMEOUT,
    SEARCH_FANOUT,
)
from nba_core.features import PromptLibrary
from nba_core.inference.gateway import GatewayError, complete_json, stream_events
from nba_core.services.datasets import get_nba_processor
from nba_core.services.sse import EventStream, encode_event

__all__ = ["FusionService", "fusion_service", "SearchLane"]

logger = logging.getLogger(__name__)

#: Event types carrying text deltas inside a retrieval stream.
DELTA_EVENTS = frozenset({"response.reasoning_text.delta", "response.output_text.delta"})

#: Label template for each retrieval lane.
LANE_LABELS = {
    "home_injury": "{team} Injury Reports",
    "away_injury": "{team} Injury Reports",
    "home_news": "{team} Recent News",
    "away_news": "{team} Recent News",
    "home_tactical": "{team} Tactical Analysis",
    "away_tactical": "{team} Tactical Analysis",
}

DISPLAY_FIELDS = (
    ("Minutes", "diff_MIN", 1),
    ("FG Made", "diff_FGM", 1),
    ("FG Attempts", "diff_FGA", 1),
    ("FG %", "diff_FG%", 3),
    ("3P Made", "diff_3PM", 1),
    ("3P Attempts", "diff_3PA", 1),
    ("3P %", "diff_3P%", 3),
    ("FT Made", "diff_FTM", 1),
    ("FT Attempts", "diff_FTA", 1),
    ("FT %", "diff_FT%", 3),
    ("Off Rebounds", "diff_OREB", 1),
    ("Def Rebounds", "diff_DREB", 1),
    ("Total Rebounds", "diff_REB", 1),
    ("Assists", "diff_AST", 1),
    ("Steals", "diff_STL", 1),
    ("Blocks", "diff_BLK", 1),
    ("Turnovers", "diff_TOV", 1),
    ("Personal Fouls", "diff_PF", 1),
)


@dataclass(frozen=True)
class SearchLane:
    """One retrieval lane in the concurrent evidence sweep."""

    category: str
    team: str
    opponent: str
    season: str
    game_date: str
    title: str
    prompt: str


class FusionService:
    """Workflow facade for the module-2 screens."""

    def __init__(self, prompts: PromptLibrary | None = None, fanout: int = SEARCH_FANOUT):
        self.prompts = prompts or PromptLibrary()
        self.fanout = max(1, int(fanout))

    # -- reference data -----------------------------------------------------

    def seasons(self) -> list[str]:
        return get_nba_processor().get_available_seasons()

    def teams(self, season: str) -> list[str]:
        return get_nba_processor().get_teams_by_season(season)

    def game_dates(self, season: str, home_team: str, away_team: str) -> list[str]:
        return get_nba_processor().get_game_dates(season, home_team, away_team)

    # -- structured assembly ------------------------------------------------

    def structured_view(self, season: str, home_team: str, away_team: str,
                        game_date: str) -> dict:
        """Assemble the structured half of the fusion input."""
        processor = get_nba_processor()
        prepared = processor.prepare_prediction_data(season, home_team, away_team, game_date)

        if prepared["home_recent_stats"] is None or prepared["away_recent_stats"] is None:
            raise LookupError("insufficient historical data for prediction")

        return {
            "home_stats": processor.format_stats_for_display(prepared["home_recent_stats"]),
            "away_stats": processor.format_stats_for_display(prepared["away_recent_stats"]),
            "game_data": self._display_differentials(prepared["game_data"]),
        }

    @staticmethod
    def _display_differentials(raw: dict | None) -> dict | None:
        """Format the in-game differential row for display."""
        if raw is None:
            return None
        formatted = {}
        for label, column, precision in DISPLAY_FIELDS:
            value = raw.get(column, 0)
            try:
                formatted[label] = f"{float(value):.{precision}f}"
            except (TypeError, ValueError):
                formatted[label] = "0" if precision == 1 else "0.000"
        return formatted

    # -- retrieval planning -------------------------------------------------

    def plan_lanes(self, season: str, home_team: str, away_team: str,
                   game_date: str) -> list[SearchLane]:
        """Build the six-lane evidence sweep plan."""
        lane_specs = (
            ("home_injury", home_team, away_team),
            ("away_injury", away_team, home_team),
            ("home_news", home_team, away_team),
            ("away_news", away_team, home_team),
            ("home_tactical", home_team, away_team),
            ("away_tactical", away_team, home_team),
        )

        lanes = []
        for category, team, opponent in lane_specs:
            lanes.append(
                SearchLane(
                    category=category,
                    team=team,
                    opponent=opponent,
                    season=season,
                    game_date=game_date,
                    title=LANE_LABELS[category].format(team=team),
                    prompt=self.prompts.evidence_query(
                        category, team, opponent, season, game_date
                    ),
                )
            )
        return lanes

    # -- streaming retrieval ------------------------------------------------

    def stream_evidence(self, season: str, home_team: str, away_team: str,
                        game_date: str):
        """Yield SSE frames for the concurrent evidence sweep."""
        lanes = self.plan_lanes(season, home_team, away_team, game_date)
        stream = EventStream(len(lanes))

        for lane in lanes:
            stream.emit(
                {"type": "task_start", "category": lane.category, "title": lane.title}
            )

        def collect(lane: SearchLane) -> None:
            try:
                buffer: list[str] = []
                for chunk in stream_events(lane.prompt, route="retrieval-stream",
                                           timeout=DEFAULT_TIMEOUT):
                    if chunk.get("type") not in DELTA_EVENTS:
                        continue
                    text = chunk.get("delta", "")
                    if not isinstance(text, str) or not text:
                        continue
                    buffer.append(text)
                    stream.emit(
                        {"type": "task_stream", "category": lane.category, "chunk": text}
                    )

                stream.emit(
                    {
                        "type": "task_complete",
                        "category": lane.category,
                        "title": lane.title,
                        "content": "".join(buffer),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - a lane failure is reported inline
                logger.warning("evidence lane %s failed: %s", lane.category, exc)
                stream.emit(
                    {
                        "type": "task_error",
                        "category": lane.category,
                        "title": lane.title,
                        "error": str(exc),
                    }
                )
            finally:
                stream.finish()

        workers = max(1, min(self.fanout, len(lanes)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for lane in lanes:
                executor.submit(collect, lane)
            yield from stream.frames()

        yield encode_event({"type": "all_complete", "total_tasks": len(lanes)})

    # -- fused verdict ------------------------------------------------------

    def fuse(self, season: str, home_team: str, away_team: str, game_date: str,
             raw_unstructured: str, structured_data: dict) -> dict:
        """Condense the evidence and produce the fused prediction."""
        structured_text = self._render_structured(structured_data, home_team, away_team)
        prompt = self.prompts.fusion_summary(
            season, home_team, away_team, game_date, structured_text, raw_unstructured
        )

        try:
            payload = complete_json(prompt, route="retrieval", timeout=DEFAULT_TIMEOUT)
        except GatewayError as exc:
            raise RuntimeError(f"prediction failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"failed to parse model response: {exc}") from exc

        prediction = payload.get("prediction", {}) or {}
        home_prob, away_prob = _normalise(prediction)

        return {
            "prediction": {
                "winner": prediction.get("winner", "home"),
                "home_team": home_team,
                "away_team": away_team,
                "home_win_probability": round(home_prob, 1),
                "away_win_probability": round(away_prob, 1),
            },
            "top_factors": payload.get("top_factors", []),
            "unstructured_summary": payload.get("unstructured_summary", {}),
            "actual_result": self._actual_result(season, home_team, away_team, game_date),
        }

    @staticmethod
    def _render_structured(structured_data: dict, home_team: str, away_team: str) -> str:
        home_stats = structured_data.get("home_stats", {}) or {}
        away_stats = structured_data.get("away_stats", {}) or {}

        text = f"\n### {home_team} Recent Performance (Last 10 Games Average):\n"
        for key, value in home_stats.items():
            text += f"- {key}: {value}\n"

        text += f"\n### {away_team} Recent Performance (Last 10 Games Average):\n"
        for key, value in away_stats.items():
            text += f"- {key}: {value}\n"
        return text

    @staticmethod
    def _actual_result(season: str, home_team: str, away_team: str,
                       game_date: str) -> dict | None:
        """Ground-truth winner when the historical row is present."""
        processor = get_nba_processor()
        prepared = processor.prepare_prediction_data(season, home_team, away_team, game_date)
        game_data = prepared.get("game_data")
        if not game_data:
            return None

        home_won = game_data.get("home_win")
        winner = "home" if home_won == 1 else "away"
        return {
            "winner": winner,
            "winner_name": home_team if winner == "home" else away_team,
        }


def _normalise(prediction: dict) -> tuple[float, float]:
    """Scale the reported probability pair to percentages."""
    try:
        home = float(prediction.get("home_win_probability", 50))
    except (TypeError, ValueError):
        home = 50.0
    try:
        away = float(prediction.get("away_win_probability", 50))
    except (TypeError, ValueError):
        away = 50.0

    total = home + away
    if total <= 0:
        return 50.0, 50.0
    return home / total * 100.0, away / total * 100.0


fusion_service = FusionService()
