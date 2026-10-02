"""
Central configuration surface for the NBA Predictor platform.

Every outbound endpoint, credential, model identifier and on-disk path used by
the application is declared here exactly once.  Runtime modules must import
from this file instead of inlining literals, so that re-pointing the platform
at a different gateway is a single-file change.

Resolution order for every credential-bearing value:

    1. environment override (``NBA_<SECTION>_<KEY>``)
    2. the literal declared below

Deployments are expected to populate the environment overrides in production
and leave the literals in place for local development.
"""

from __future__ import annotations

import os

__all__ = [
    "PROJECT_ROOT",
    "UPLOAD_DIR",
    "NBA_DATASET_FILE",
    "NBA_ROSTER_FILE",
    "TRAINED_MODEL_DIR",
    "UPLOAD_ALLOWED_EXTENSIONS",
    "UPLOAD_MAX_CONTENT_LENGTH",
    "SERVER_HOST",
    "SERVER_PORT",
    "SERVER_DEBUG",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TIMEOUT_LONG",
    "STREAM_CHUNK_PREFIX",
    "STREAM_DONE_SENTINEL",
    "SEARCH_FANOUT",
    "MODULE1_FEATURE_COLUMNS",
    "MODULE2_STAT_COLUMNS",
    "MODEL_CATALOG",
    "MODEL_TYPE_LABELS",
    "TEMPLATE_COLUMNS",
    "INFERENCE_ROUTES",
    "INFERENCE_DEFAULT_ROUTE",
    "resolve_inference_route",
    "describe_routes",
]


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#: Directory receiving user-supplied prediction payloads.
UPLOAD_DIR = os.environ.get("NBA_UPLOAD_DIR") or os.path.join(PROJECT_ROOT, "uploads")

#: Historical team box-score feature matrix (2015-16 .. 2025-26).
NBA_DATASET_FILE = os.environ.get("NBA_DATASET_FILE") or os.path.join(
    PROJECT_ROOT, "nba_team_boxscores_features_2015_16_to_2025_26.csv"
)

#: JSONL roster catalogue consumed by the player-news workflow.
NBA_ROSTER_FILE = os.environ.get("NBA_ROSTER_FILE") or os.path.join(
    PROJECT_ROOT, "nba_teams_players.jsonl"
)

#: Optional directory containing serialised estimator artefacts.
TRAINED_MODEL_DIR = os.environ.get("NBA_TRAINED_MODEL_DIR") or os.path.join(
    PROJECT_ROOT, "trained_models"
)

# ---------------------------------------------------------------------------
# HTTP / upload policy
# ---------------------------------------------------------------------------

UPLOAD_ALLOWED_EXTENSIONS = frozenset({"csv", "xlsx", "xls"})
UPLOAD_MAX_CONTENT_LENGTH = int(os.environ.get("NBA_MAX_UPLOAD_BYTES", 16 * 1024 * 1024))

SERVER_HOST = os.environ.get("NBA_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("NBA_PORT", 3344))
SERVER_DEBUG = os.environ.get("NBA_DEBUG", "1").strip().lower() not in {"0", "false", "no"}

#: Concurrency budget for fan-out HTTP work (search fan-out, batch scoring).
MAX_WORKERS = int(os.environ.get("NBA_MAX_WORKERS", 10))

# ---------------------------------------------------------------------------
# Gateway credentials and endpoints -- the single source of truth
# ---------------------------------------------------------------------------

# Primary JSON-completion gateway (OpenAI-compatible chat/completions surface).
COMPLETION_GATEWAY_API_KEY = os.environ.get(
    "NBA_COMPLETION_API_KEY",
    "sk-rHa4hQLoPiQpGl7imxu6lp1nLjxtdyxcGf3j1afaCOKV1grE",
)
COMPLETION_GATEWAY_ENDPOINT = os.environ.get(
    "NBA_COMPLETION_ENDPOINT",
    "https://api.aipaibox.com/v1/chat/completions",
)
COMPLETION_GATEWAY_MODEL = os.environ.get("NBA_COMPLETION_MODEL", "gpt-5.5")

# Secondary reasoning gateway with hosted web-search tooling.
SEARCH_GATEWAY_API_KEY = os.environ.get(
    "NBA_SEARCH_API_KEY",
    "sk-7802ae9cf2764f09ad074fe23bcbe74c",
)
SEARCH_GATEWAY_ENDPOINT = os.environ.get(
    "NBA_SEARCH_ENDPOINT",
    "https://api.deepseek.com/v1/responses",
)
SEARCH_GATEWAY_MODEL = os.environ.get("NBA_SEARCH_MODEL", "deepseek-v4-flash")

# Timeout policy (seconds).
DEFAULT_TIMEOUT = int(os.environ.get("NBA_TIMEOUT", 120))
DEFAULT_TIMEOUT_LONG = int(os.environ.get("NBA_TIMEOUT_LONG", 600))

# Server-sent-event framing shared by every streaming workflow.
STREAM_CHUNK_PREFIX = "data: "
STREAM_DONE_SENTINEL = "[DONE]"

#: Number of parallel retrieval lanes in the module-2 evidence sweep.
SEARCH_FANOUT = int(os.environ.get("NBA_SEARCH_FANOUT", 6))

#: Default number of consecutive games rolled into a form window.
FORM_WINDOW = int(os.environ.get("NBA_FORM_WINDOW", 10))


def _derive_responses_endpoint(chat_endpoint: str) -> str:
    """Derive the responses-style surface from an OpenAI-compatible chat URL."""
    base = chat_endpoint.split("/v1/")[0].rstrip("/")
    return f"{base}/v1/responses" if base else chat_endpoint


# ---------------------------------------------------------------------------
# Routed inference profiles
# ---------------------------------------------------------------------------
# Each entry describes *how* a workload talks to a gateway: which endpoint,
# which credential, which remote model id and which response contract.  Call
# sites address a profile by key, never by provider.

INFERENCE_ROUTES = {
    "primary": {
        "provider": "completion-gateway",
        "endpoint": COMPLETION_GATEWAY_ENDPOINT,
        "api_key": COMPLETION_GATEWAY_API_KEY,
        "model": COMPLETION_GATEWAY_MODEL,
        "surface": "chat.completions",
        "payload_key": "messages",
        "timeout": DEFAULT_TIMEOUT,
        "structured_output": True,
        "tools": None,
    },
    "primary-batch": {
        "provider": "completion-gateway",
        "endpoint": COMPLETION_GATEWAY_ENDPOINT,
        "api_key": COMPLETION_GATEWAY_API_KEY,
        "model": COMPLETION_GATEWAY_MODEL,
        "surface": "chat.completions",
        "payload_key": "messages",
        "timeout": DEFAULT_TIMEOUT_LONG,
        "structured_output": True,
        "tools": None,
    },
    "retrieval": {
        "provider": "search-gateway",
        "endpoint": SEARCH_GATEWAY_ENDPOINT,
        "api_key": SEARCH_GATEWAY_API_KEY,
        "model": SEARCH_GATEWAY_MODEL,
        "surface": "responses",
        "payload_key": "input",
        "timeout": DEFAULT_TIMEOUT_LONG,
        "structured_output": True,
        "tools": [{"type": "web_search"}],
    },
    "retrieval-stream": {
        "provider": "search-gateway",
        "endpoint": SEARCH_GATEWAY_ENDPOINT,
        "api_key": SEARCH_GATEWAY_API_KEY,
        "model": SEARCH_GATEWAY_MODEL,
        "surface": "responses",
        "payload_key": "input",
        "timeout": DEFAULT_TIMEOUT,
        "structured_output": False,
        "stream": True,
        "tools": [{"type": "web_search"}],
    },
    # Fallback surface probed when the primary chat surface is unavailable.
    "primary-response-surface": {
        "provider": "completion-gateway",
        "endpoint": _derive_responses_endpoint(COMPLETION_GATEWAY_ENDPOINT),
        "api_key": COMPLETION_GATEWAY_API_KEY,
        "model": COMPLETION_GATEWAY_MODEL,
        "surface": "responses",
        "payload_key": "input",
        "timeout": DEFAULT_TIMEOUT,
        "structured_output": True,
        "tools": None,
    },
}

#: Profile used when a call site does not pin one explicitly.
INFERENCE_DEFAULT_ROUTE = "primary"


def resolve_inference_route(route_key: str | None = None) -> dict:
    """Return a copy of the routing profile addressed by ``route_key``."""
    key = route_key or INFERENCE_DEFAULT_ROUTE
    if key not in INFERENCE_ROUTES:
        raise KeyError(f"unknown inference route: {key}")
    return dict(INFERENCE_ROUTES[key])


def describe_routes() -> dict:
    """Redacted route inventory (credentials replaced by a short fingerprint)."""
    inventory = {}
    for key, spec in INFERENCE_ROUTES.items():
        inventory[key] = {
            "provider": spec["provider"],
            "endpoint": spec["endpoint"],
            "model": spec["model"],
            "surface": spec["surface"],
            "credential": _fingerprint(spec["api_key"]),
        }
    return inventory


def _fingerprint(secret: str | None) -> str:
    if not secret:
        return "unset"
    tail = secret[-4:] if len(secret) >= 4 else secret
    return f"***{tail}"


# ---------------------------------------------------------------------------
# Model catalogue (module 1)
# ---------------------------------------------------------------------------
# ``estimator_family`` selects the scoring profile applied downstream; the
# catalogue intentionally describes only the public-facing identity of a model.

MODEL_CATALOG = {
    "decision_tree": {
        "name": "Decision Tree",
        "type": "traditional",
        "description": "High interpretability with feature importance analysis",
        "requires_sequence": False,
        "estimator_family": "tree",
        "calibration": {"exponent": 1.0, "temperature": 1.0, "bias": 0.0},
    },
    "svm": {
        "name": "SVM",
        "type": "traditional",
        "description": "Suitable for non-linear classification with RBF kernel",
        "requires_sequence": False,
        "estimator_family": "margin",
        "calibration": {"exponent": 1.2, "temperature": 0.9, "bias": 0.0},
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "traditional",
        "description": "Ensemble learning with high accuracy and anti-overfitting",
        "requires_sequence": False,
        "estimator_family": "ensemble",
        "calibration": {"exponent": 0.8, "temperature": 1.1, "bias": 0.0},
    },
    "mlp": {
        "name": "MLP Neural Network",
        "type": "deep_learning",
        "description": "Capable of learning complex non-linear relationships",
        "requires_sequence": False,
        "estimator_family": "dense",
        "calibration": {"exponent": 1.0, "temperature": 1.0, "bias": 0.0},
    },
    "lstm": {
        "name": "LSTM Time Series Network",
        "type": "time_series",
        "description": "Suitable for time series data, captures long-term dependencies",
        "requires_sequence": True,
        "estimator_family": "recurrent",
        "calibration": {"exponent": 1.0, "temperature": 1.05, "bias": 0.0},
    },
    "gru": {
        "name": "GRU Time Series Network",
        "type": "time_series",
        "description": "Similar to LSTM but with fewer parameters and faster training",
        "requires_sequence": True,
        "estimator_family": "recurrent",
        "calibration": {"exponent": 1.0, "temperature": 1.05, "bias": 0.0},
    },
}

MODEL_TYPE_LABELS = {
    "traditional": "Traditional ML",
    "deep_learning": "Deep Neural Network",
    "time_series": "Time Series",
}

#: Differential columns consumed by the module-1 scoring path.
MODULE1_FEATURE_COLUMNS = (
    "diff_MIN",
    "diff_FGM",
    "diff_FGA",
    "diff_FG%",
    "diff_3PM",
    "diff_3PA",
    "diff_3P%",
    "diff_FTM",
    "diff_FTA",
    "diff_FT%",
    "diff_OREB",
    "diff_DREB",
    "diff_REB",
    "diff_AST",
    "diff_STL",
    "diff_BLK",
    "diff_TOV",
    "diff_PF",
)

#: Differential columns aggregated by the module-2 form engine.
MODULE2_STAT_COLUMNS = ("diff_PTS",) + MODULE1_FEATURE_COLUMNS

#: Column order published by the downloadable payload templates.
TEMPLATE_COLUMNS = (
    "season",
    "season_type",
    "date",
    "h_team_name",
    "o_team_name",
) + MODULE1_FEATURE_COLUMNS
