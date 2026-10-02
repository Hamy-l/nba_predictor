"""Feature-store facade.

Exposes the prompt library, the numeric normalisation helpers and the templated
payload builders used across the platform.
"""

from nba_core.features.prompts import PromptLibrary, LIBRARY, build, GAME_SCORING_SCHEMA
from nba_core.features.normalize import (
    fold_differentials,
    fingerprint,
    zscore,
    sigmoid,
    numeric_or,
)
from nba_core.features.samples import build_template_frame, TEMPLATE_ROWS

__all__ = [
    "PromptLibrary",
    "LIBRARY",
    "build",
    "GAME_SCORING_SCHEMA",
    "fold_differentials",
    "fingerprint",
    "zscore",
    "sigmoid",
    "numeric_or",
    "build_template_frame",
    "TEMPLATE_ROWS",
]
