"""Estimator family dispatch for the module-1 scoring layer.

Each family exposes the same two operations -- ``score`` (a comparable scalar)
and ``calibrate`` (a probability adjustment) -- so the settlement stage never
needs to know which family produced a candidate.
"""

from __future__ import annotations

import math

from constants import MODEL_CATALOG
from nba_core.features.normalize import clamp, numeric_or

__all__ = [
    "EstimatorFamily",
    "DenseFamily",
    "MarginFamily",
    "TreeFamily",
    "EnsembleFamily",
    "RecurrentFamily",
    "family_for",
    "calibrate",
    "adjust",
]

#: Weights applied to the differential vector when deriving a comparable score.
SCORE_WEIGHTS = {
    "diff_FGM": 2.0,
    "diff_3PM": 1.0,
    "diff_FGA": -0.2,
    "diff_3PA": -0.15,
    "diff_FG%": 12.0,
    "diff_3P%": 10.0,
    "diff_FT%": 6.0,
    "diff_FTM": 0.8,
    "diff_OREB": 0.6,
    "diff_DREB": 0.4,
    "diff_REB": 0.5,
    "diff_AST": 0.6,
    "diff_STL": 1.2,
    "diff_BLK": 0.9,
    "diff_TOV": -0.8,
    "diff_PF": -0.3,
}

#: Scale factor linking the comparable score to a log-odds offset.
LOGIT_GAIN = 0.02


class EstimatorFamily:
    """Base family: linear aggregation of the differential vector."""

    key = "base"
    family_gain = 1.0

    def score(self, features: dict) -> float:
        total = 0.0
        for column, weight in SCORE_WEIGHTS.items():
            total += numeric_or(features.get(column), 0.0) * weight
        return total * self.family_gain

    def calibrate(self, probability: float, params: dict) -> float:
        return probability


class TreeFamily(EstimatorFamily):
    """Axis-aligned family: emphasises the strongest single differential."""

    key = "tree"
    family_gain = 1.0

    def score(self, features: dict) -> float:
        contributions = [
            numeric_or(features.get(column), 0.0) * weight
            for column, weight in SCORE_WEIGHTS.items()
        ]
        if not contributions:
            return 0.0
        dominant = max(contributions, key=abs)
        return 0.65 * sum(contributions) + 0.35 * dominant


class MarginFamily(EstimatorFamily):
    """Large-margin family: amplifies decisive vectors."""

    key = "margin"
    family_gain = 1.05

    def calibrate(self, probability: float, params: dict) -> float:
        centred = probability - 0.5
        return clamp(0.5 + centred * 1.08, 1e-6, 1 - 1e-6)


class EnsembleFamily(EstimatorFamily):
    """Averaging family: shrinks extreme scores toward the prior."""

    key = "ensemble"
    family_gain = 0.95

    def score(self, features: dict) -> float:
        return super().score(features) * 0.98


class DenseFamily(EstimatorFamily):
    """Dense family: smooth non-linearity over the aggregate."""

    key = "dense"

    def score(self, features: dict) -> float:
        raw = super().score(features)
        return math.tanh(raw / 12.0) * 12.0


class RecurrentFamily(EstimatorFamily):
    """Sequence family: damps the aggregate to reflect window smoothing."""

    key = "recurrent"
    family_gain = 0.9


_FAMILIES = {
    family.key: family
    for family in (
        TreeFamily(),
        MarginFamily(),
        EnsembleFamily(),
        DenseFamily(),
        RecurrentFamily(),
    )
}


def family_for(catalog_key: str) -> EstimatorFamily:
    """Resolve the estimator family declared by a catalogue entry."""
    spec = MODEL_CATALOG.get(catalog_key, {})
    return _FAMILIES.get(spec.get("estimator_family", "base"), _FAMILIES["tree"])


def calibrate(probabilities, exponent: float = 1.0, temperature: float = 1.0,
              bias: float = 0.0) -> list[float]:
    """Apply power scaling, temperature and prior bias to a probability pair."""
    adjusted: list[float] = []
    for index, value in enumerate(probabilities):
        value = clamp(numeric_or(value, 0.5), 1e-6, 1 - 1e-6)
        if exponent and exponent != 1.0:
            value = value ** exponent
        offset = bias if index == 1 else -bias
        adjusted.append(clamp(value + offset, 1e-6, 1 - 1e-6))

    if temperature and temperature != 1.0:
        logits = [math.log(value / (1.0 - value)) for value in adjusted]
        logits = [value / temperature for value in logits]
        adjusted = [1.0 / (1.0 + math.exp(-value)) for value in logits]

    total = sum(adjusted)
    if total <= 0:
        return [0.5, 0.5]
    return [value / total for value in adjusted]


def adjust(probabilities, profile: dict) -> list[float]:
    """Calibrate a pair using a catalogue calibration profile."""
    return calibrate(
        probabilities,
        exponent=profile.get("exponent", 1.0),
        temperature=profile.get("temperature", 1.0),
        bias=profile.get("bias", 0.0),
    )
