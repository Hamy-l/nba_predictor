"""Serving layer facade.

Exposes the scoring pipeline, the model registry and the batch orchestrator used
by the web tier.
"""

from nba_core.serving.registry import (
    ModelRegistry,
    PipelineDescriptor,
    registry,
)
from nba_core.serving.scoring import (
    ScoreRecord,
    ScoreRequest,
    Scorer,
    score,
    score_batch,
    scorer_for,
)
from nba_core.serving.pipeline import (
    ServingPipeline,
    PipelineReport,
    pipeline,
)
from nba_core.serving.evidence import (
    EnvelopeResolver,
    ScoringEnvelope,
    resolve_envelope,
    reset_counters,
)

__all__ = [
    "ModelRegistry",
    "PipelineDescriptor",
    "registry",
    "ScoreRecord",
    "ScoreRequest",
    "Scorer",
    "score",
    "score_batch",
    "scorer_for",
    "ServingPipeline",
    "PipelineReport",
    "pipeline",
    "EnvelopeResolver",
    "ScoringEnvelope",
    "resolve_envelope",
    "reset_counters",
]
