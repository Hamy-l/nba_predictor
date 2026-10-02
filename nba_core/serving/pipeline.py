"""Batch orchestration for the scoring layer.

The pipeline owns fan-out and fan-in: it schedules every (game, pipeline) pair
onto a bounded thread pool, collects structured records, and regroups them for
presentation.  Web handlers never touch threads directly.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from constants import MAX_WORKERS
from nba_core.serving.evidence import resolve_envelope
from nba_core.serving.registry import registry
from nba_core.serving.scoring import ScoreRecord, ScoreRequest, scorer_for

__all__ = ["ServingPipeline", "PipelineReport", "pipeline", "GPU_FREE_WORKERS"]

logger = logging.getLogger(__name__)

#: Upper bound on the worker pool regardless of the configured budget.
GPU_FREE_WORKERS = 10


@dataclass
class PipelineReport:
    """Aggregated outcome of a batch run."""

    records: list[ScoreRecord]
    games: list[dict]
    models: list[str]
    duration_ms: float
    failures: int = 0

    @property
    def total_games(self) -> int:
        return len(self.games)

    @property
    def total_models(self) -> int:
        return len(self.models)

    def by_game(self) -> list[dict]:
        """Regroup records into the per-game presentation shape."""
        grouped: list[dict] = []

        for game in self.games:
            row_index = game.get("index", 0)
            entry = {
                "index": row_index,
                "home_team": game.get("h_team_name", "Home Team"),
                "away_team": game.get("o_team_name", "Away Team"),
                "predictions": [],
            }

            for record in self.records:
                if record.row_index != row_index:
                    continue
                descriptor = registry.get(record.model_key)
                entry["predictions"].append(
                    {
                        "model_name": descriptor.name,
                        "model_key": record.model_key,
                        "winner": record.winner,
                        "home_win_prob": record.home_win_prob,
                        "away_win_prob": record.away_win_prob,
                    }
                )

            entry["predictions"].sort(key=lambda item: item["model_name"])
            grouped.append(entry)

        return grouped

    def matrix(self) -> dict:
        """Regroup records into the model-major shape used by comparisons."""
        result: dict[str, dict] = {}
        for model_key in self.models:
            descriptor = registry.get(model_key)
            rows = [record for record in self.records if record.model_key == model_key]
            rows.sort(key=lambda record: record.row_index)
            result[model_key] = {
                "model_name": descriptor.name,
                "predictions": [
                    {
                        "prediction": record.prediction,
                        "home_win_prob": record.probabilities[1],
                        "away_win_prob": record.probabilities[0],
                    }
                    for record in rows
                ],
            }
        return result


class ServingPipeline:
    """Fan-out executor over the registered scoring pipelines."""

    def __init__(self, max_workers: int | None = None):
        budget = max_workers or MAX_WORKERS
        self.max_workers = max(1, min(GPU_FREE_WORKERS, int(budget)))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _normalise(games: Sequence[dict]) -> list[dict]:
        prepared = []
        for position, game in enumerate(games):
            row = dict(game)
            row.setdefault("index", position)
            prepared.append(row)
        return prepared

    def _plan(self, games: Sequence[dict], models: Iterable[str]) -> list[tuple[dict, str]]:
        plan = []
        for game in games:
            for model_key in models:
                if model_key in registry:
                    plan.append((game, model_key))
        return plan

    def _run_one(self, game: dict, model_key: str) -> ScoreRecord | None:
        try:
            scorer = scorer_for(model_key)
            return scorer(game, int(game.get("index", 0)))
        except Exception:  # noqa: BLE001 - a single failure must not abort the batch
            logger.exception("scoring failed for row=%s model=%s", game.get("index"), model_key)
            return None

    # -- public surface -----------------------------------------------------

    def execute(self, games: Sequence[dict], models: Iterable[str]) -> PipelineReport:
        """Score every (game, model) pair and return an aggregated report."""
        prepared = self._normalise(games)
        model_list = [key for key in models]
        plan = self._plan(prepared, model_list)

        started = time.perf_counter()
        records: list[ScoreRecord] = []
        failures = 0

        if plan:
            workers = max(1, min(self.max_workers, len(plan)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._run_one, game, model_key)
                    for game, model_key in plan
                ]
                for future in concurrent.futures.as_completed(futures):
                    outcome = future.result()
                    if outcome is None:
                        failures += 1
                    else:
                        records.append(outcome)

        duration_ms = (time.perf_counter() - started) * 1000.0
        return PipelineReport(
            records=records,
            games=prepared,
            models=model_list,
            duration_ms=duration_ms,
            failures=failures,
        )

    def compare(self, games: Sequence[dict], *, dwell: float = 0.05) -> PipelineReport:
        """Score every game with every registered pipeline, serially paced."""
        prepared = self._normalise(games)
        model_list = registry.keys()

        started = time.perf_counter()
        records: list[ScoreRecord] = []
        failures = 0

        for model_key in model_list:
            time.sleep(dwell)
            for game in prepared:
                outcome = self._run_one(game, model_key)
                if outcome is None:
                    failures += 1
                else:
                    records.append(outcome)

        duration_ms = (time.perf_counter() - started) * 1000.0
        return PipelineReport(
            records=records,
            games=prepared,
            models=model_list,
            duration_ms=duration_ms,
            failures=failures,
        )

    def warm(self, games: Sequence[dict]) -> int:
        """Pre-resolve envelopes for a payload, collapsing duplicate vectors."""
        warmed = 0
        for game in games:
            envelope = resolve_envelope(game)
            if envelope.source in {"resolved", "cache"}:
                warmed += 1
        return warmed


pipeline = ServingPipeline()
