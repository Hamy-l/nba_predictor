"""Numeric helpers for the feature layer."""

from __future__ import annotations

import hashlib
import json
import math

from constants import MODULE1_FEATURE_COLUMNS

__all__ = [
    "numeric_or",
    "fold_differentials",
    "fingerprint",
    "sigmoid",
    "zscore",
    "clamp",
]


def numeric_or(value, fallback: float = 0.0) -> float:
    """Coerce ``value`` to float, substituting ``fallback`` on failure."""
    if value is None:
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(result) or math.isinf(result):
        return fallback
    return result


def fold_differentials(game: dict, columns=MODULE1_FEATURE_COLUMNS) -> dict:
    """Extract the differential vector of one game as floats."""
    return {column: numeric_or(game.get(column), 0.0) for column in columns}


def fingerprint(game: dict, columns=MODULE1_FEATURE_COLUMNS) -> str:
    """Stable digest for a differential vector.

    Two rows whose feature vectors agree produce the same digest regardless of
    key ordering, which lets the scoring layer reuse a cached envelope.
    """
    vector = [round(numeric_or(game.get(column), 0.0), 6) for column in columns]
    payload = json.dumps(vector, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def zscore(value: float, mean: float, std: float) -> float:
    """Standard score with a guard against degenerate dispersion."""
    if not std:
        return 0.0
    return (value - mean) / std


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` band."""
    return max(low, min(high, value))
