"""Settlement pipeline for a single matchup.

One call runs the full stage graph:

    normalise -> resolve envelope -> family score -> power/temperature
    calibration -> normalisation -> verdict

Callers receive a structured record instead of a bare tuple so that batch
orchestration can sort, deduplicate and report on results without re-deriving
anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

from nba_core.features.normalize import fold_differentials
from nba_core.serving import estimators
from nba_core.serving.evidence import resolve_envelope
from nba_core.serving.registry import PipelineDescriptor, registry

__all__ = ["ScoreRecord", "ScoreRequest", "Scorer", "score", "scorer_for", "score_batch"]


@dataclass
class ScoreRequest:
    """A single matchup presented to the scoring surface."""

    model_key: str
    game: dict
    row_index: int = 0

    def features(self) -> dict:
        return fold_differentials(self.game)

    @property
    def home_team(self) -> str:
        return self.game.get("h_team_name", "Home Team")

    @property
    def away_team(self) -> str:
        return self.game.get("o_team_name", "Away Team")


@dataclass
class ScoreRecord:
    """Structured outcome of one settlement run."""

    model_key: str
    model_name: str
    row_index: int
    prediction: int
    probabilities: list[float]
    winner: str
    source: str
    family: str
    digest: str
    notes: dict = field(default_factory=dict)

    @property
    def home_win_prob(self) -> float:
        return self.probabilities[1] * 100.0

    @property
    def away_win_prob(self) -> float:
        return self.probabilities[0] * 100.0

    def as_dict(self) -> dict:
        return asdict(self)


class Scorer:
    """Binds a pipeline descriptor to the settlement stage graph."""

    def __init__(self, descriptor: PipelineDescriptor):
        self.descriptor = descriptor
        self.family = estimators.family_for(descriptor.key)

    # -- stages -------------------------------------------------------------

    def _prepare(self, request: ScoreRequest) -> dict:
        return request.features()

    def _working_score(self, features: dict) -> float:
        return self.family.score(features)

    def _calibrate(self, probabilities: Iterable[float], *, apply_profile: bool) -> list[float]:
        pair = [float(value) for value in probabilities]
        if apply_profile:
            return estimators.adjust(pair, self.descriptor.calibration)
        return estimators.calibrate(pair)

    # -- entry points -------------------------------------------------------

    def run(self, request: ScoreRequest, *, apply_profile: bool = True) -> ScoreRecord:
        features = self._prepare(request)
        envelope = resolve_envelope(request.game)

        working = self._working_score(features)
        probabilities = self._calibrate(
            [envelope.away_probability, envelope.home_probability],
            apply_profile=apply_profile,
        )

        prediction = 0 if probabilities[0] >= probabilities[1] else 1
        winner = request.home_team if prediction == 1 else request.away_team

        return ScoreRecord(
            model_key=self.descriptor.key,
            model_name=self.descriptor.name,
            row_index=request.row_index,
            prediction=int(prediction),
            probabilities=probabilities,
            winner=winner,
            source=envelope.source,
            family=self.descriptor.estimator_family,
            digest=envelope.digest,
            notes={
                "scalar": round(working, 6),
                "profile": dict(self.descriptor.calibration),
                "sequence": self.descriptor.requires_sequence,
            },
        )

    def __call__(self, game: dict, row_index: int = 0, *, apply_profile: bool = True) -> ScoreRecord:
        return self.run(ScoreRequest(self.descriptor.key, game, row_index), apply_profile=apply_profile)


_SCORERS: dict[str, Scorer] = {}


def scorer_for(model_key: str) -> Scorer:
    """Return (and memoise) the scorer bound to ``model_key``."""
    if model_key not in _SCORERS:
        _SCORERS[model_key] = Scorer(registry.get(model_key))
    return _SCORERS[model_key]


def score(model_key: str, game: dict, row_index: int = 0, *, apply_profile: bool = True) -> ScoreRecord:
    """Settle one matchup with one pipeline."""
    return scorer_for(model_key)(game, row_index, apply_profile=apply_profile)


def score_batch(model_key: str, games: Iterable[dict]) -> list[ScoreRecord]:
    """Settle a sequence of matchups with one pipeline."""
    scorer = scorer_for(model_key)
    return [scorer(game, index) for index, game in enumerate(games)]
