"""Module 2 routes: multi-source fusion."""

from __future__ import annotations

import logging

from flask import Blueprint, Response, jsonify, render_template, request

from nba_core.services.datasets import DataUnavailable
from nba_core.services.fusion import fusion_service
from nba_core.services.sse import encode_event

__all__ = ["module2_bp"]

logger = logging.getLogger(__name__)

module2_bp = Blueprint("module2", __name__)

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def _error(message: str, status: int = 200):
    return jsonify({"success": False, "message": message}), status


def _stream_response(generator) -> Response:
    response = Response(generator, mimetype="text/event-stream")
    for header, value in STREAM_HEADERS.items():
        response.headers[header] = value
    return response


@module2_bp.route("/module2")
def page():
    """Fusion workbench screen."""
    return render_template("module2.html")


@module2_bp.route("/api/module2/seasons", methods=["GET"])
def seasons():
    try:
        return jsonify({"success": True, "seasons": fusion_service.seasons()})
    except DataUnavailable as exc:
        return _error(str(exc))


@module2_bp.route("/api/module2/teams", methods=["GET"])
def teams():
    season = request.args.get("season")
    if not season:
        return _error("Season parameter required")
    try:
        return jsonify({"success": True, "teams": fusion_service.teams(season)})
    except DataUnavailable as exc:
        return _error(str(exc))


@module2_bp.route("/api/module2/game_dates", methods=["GET"])
def game_dates():
    season = request.args.get("season")
    home_team = request.args.get("home_team")
    away_team = request.args.get("away_team")

    if not all([season, home_team, away_team]):
        return _error("Missing required parameters")

    try:
        dates = fusion_service.game_dates(season, home_team, away_team)
        return jsonify({"success": True, "dates": dates})
    except DataUnavailable as exc:
        return _error(str(exc))


@module2_bp.route("/api/module2/structured_data", methods=["POST"])
def structured_data():
    payload = request.get_json(silent=True) or {}
    season = payload.get("season")
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")
    game_date = payload.get("game_date")

    if not all([season, home_team, away_team, game_date]):
        return _error("Missing required parameters")

    try:
        view = fusion_service.structured_view(season, home_team, away_team, game_date)
    except DataUnavailable as exc:
        return _error(str(exc))
    except LookupError as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("structured data assembly failed")
        return _error(f"Failed to get structured data: {exc}")

    return jsonify({"success": True, "structured_data": view})


@module2_bp.route("/api/module2/search_unstructured", methods=["POST"])
def search_unstructured():
    payload = request.get_json(silent=True) or {}
    season = payload.get("season")
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")
    game_date = payload.get("game_date")

    def generate():
        try:
            yield from fusion_service.stream_evidence(
                season, home_team, away_team, game_date
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("evidence sweep failed")
            yield encode_event({"type": "error", "message": str(exc)})

    return _stream_response(generate())


@module2_bp.route("/api/module2/summarize_and_predict", methods=["POST"])
def summarize_and_predict():
    payload = request.get_json(silent=True) or {}
    season = payload.get("season")
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")
    game_date = payload.get("game_date")
    raw_unstructured = payload.get("raw_unstructured", "")
    structured_data = payload.get("structured_data", {}) or {}

    if not all([season, home_team, away_team, game_date]):
        return _error("Missing required parameters")

    try:
        fused = fusion_service.fuse(
            season, home_team, away_team, game_date, raw_unstructured, structured_data
        )
    except DataUnavailable as exc:
        return _error(str(exc))
    except RuntimeError as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("fusion verdict failed")
        return _error(f"Prediction failed: {exc}")

    return jsonify({"success": True, **fused})
