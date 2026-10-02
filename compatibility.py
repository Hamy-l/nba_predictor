"""Compatibility surface.

The public functions that earlier revisions of this project exposed are kept
here so that notebooks, scripts and integrations written against them continue to
work.  Each one is a thin adapter over the layered implementation.
"""

from __future__ import annotations

import logging

from constants import MODULE1_FEATURE_COLUMNS
from nba_core.serving.registry import registry
from nba_core.serving.scoring import score

__all__ = [
    "MODEL_CONFIG",
    "allowed_file",
    "validate_data",
    "call_model_api",
    "score_model_prediction",
    "simulate_model_prediction",
]

logger = logging.getLogger(__name__)

#: Catalogue in the shape the original entry point exposed.
MODEL_CONFIG = registry.public_catalog()

#: Identifier columns required by the payload validator.
REQUIRED_COLUMNS = ("h_team_name", "o_team_name")


def allowed_file(filename: str, allowed=("csv", "xlsx", "xls")) -> bool:
    """True when ``filename`` carries an accepted payload extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in set(allowed)


def validate_data(frame) -> tuple[bool, str | None]:
    """Validate an uploaded payload frame."""
    columns = set(getattr(frame, "columns", []))

    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        return False, f"Missing required column: {missing[0]}"

    if not any(column in columns for column in MODULE1_FEATURE_COLUMNS):
        return False, (
            "Missing feature columns, need at least one of: "
            + ", ".join(MODULE1_FEATURE_COLUMNS)
        )

    return True, None


def call_model_api(prompt: str, timeout: int | None = None) -> str | None:
    """Send a prompt through the primary route and return the text body."""
    from nba_core.inference.gateway import GatewayError, complete_text

    try:
        return complete_text(prompt, route="primary", timeout=timeout)
    except GatewayError as exc:
        logger.warning("routed call failed: %s", exc)
        return None


def score_model_prediction(model_name: str, data_row: dict) -> tuple[int, list[float]]:
    """Score one row with ``model_name`` and return ``(prediction, pair)``."""
    record = score(model_name, data_row, row_index=int(data_row.get("index", 0)))
    return record.prediction, list(record.probabilities)


def simulate_model_prediction(model_name: str, data_row: dict) -> tuple[int, list[float]]:
    """Deprecated alias for :func:`score_model_prediction`."""
    return score_model_prediction(model_name, data_row)
