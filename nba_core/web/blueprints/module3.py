"""Module 3 routes: roster news and matchup verdict."""

from __future__ import annotations

import logging

from flask import Blueprint, Response, jsonify, render_template, request

from nba_core.services.players import player_news_service
from nba_core.services.roster import RosterUnavailable, roster_service
from nba_core.services.sse import encode_event

__all__ = ["module3_bp"]

logger = logging.getLogger(__name__)

module3_bp = Blueprint("module3", __name__)

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


@module3_bp.route("/module3")
def page():
    """Player-news workbench screen."""
    return render_template("module3.html")


@module3_bp.route("/api/module3/teams", methods=["GET"])
def teams():
    try:
        return jsonify({"success": True, "teams": roster_service.load()})
    except RosterUnavailable as exc:
        return _error(f"Failed to load teams: {exc}")


@module3_bp.route("/api/module3/search_news", methods=["POST"])
def search_news():
    payload = request.get_json(silent=True) or {}
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")
    home_players = payload.get("home_players", []) or []
    away_players = payload.get("away_players", []) or []

    def generate():
        try:
            yield from player_news_service.stream_news(
                home_team, away_team, home_players, away_players
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("roster sweep failed")
            yield encode_event({"type": "error", "message": str(exc)})

    return _stream_response(generate())


@module3_bp.route("/api/module3/final_prediction", methods=["POST"])
def final_prediction():
    payload = request.get_json(silent=True) or {}
    home_team = payload.get("home_team")
    away_team = payload.get("away_team")
    home_news = payload.get("home_news", "")
    away_news = payload.get("away_news", "")

    if not all([home_team, away_team]):
        return _error("Missing required parameters")

    try:
        verdict = player_news_service.verdict(home_team, away_team, home_news, away_news)
    except RuntimeError as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("verdict failed")
        return _error(f"Prediction failed: {exc}")

    return jsonify({"success": True, **verdict})
