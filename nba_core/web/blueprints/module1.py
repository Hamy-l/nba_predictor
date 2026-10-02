"""Module 1 routes: custom payload scoring."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, send_file

from nba_core.services.templates import TemplateUnavailable, template_service
from nba_core.services.uploads import UploadRejected, upload_store
from nba_core.serving.pipeline import pipeline

__all__ = ["module1_bp"]

logger = logging.getLogger(__name__)

module1_bp = Blueprint("module1", __name__)


def _error(message: str, status: int = 200):
    return jsonify({"success": False, "message": message}), status


@module1_bp.route("/api/download_template/<template_type>", methods=["GET"])
def download_template(template_type: str):
    """Publish and stream a payload template."""
    try:
        filename, path = template_service.publish(template_type)
    except TemplateUnavailable:
        return _error(f"unknown template type: {template_type}", 404)
    return send_file(path, as_attachment=True, download_name=filename)


@module1_bp.route("/api/upload", methods=["POST"])
def upload():
    """Accept, validate and preview a payload."""
    if "file" not in request.files:
        return _error("No file uploaded")

    staged = None
    try:
        staged = upload_store.persist(request.files["file"])
        frame = upload_store.load_frame(staged)

        valid, reason = upload_store.validate(frame)
        if not valid:
            upload_store.discard(staged)
            return _error(f"Data format error: {reason}")

        return jsonify(
            {
                "success": True,
                "message": "File uploaded successfully",
                "filename": staged,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "preview": upload_store.preview(frame),
            }
        )
    except UploadRejected as exc:
        if staged:
            upload_store.discard(staged)
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced to the client verbatim
        logger.exception("upload failed")
        if staged:
            upload_store.discard(staged)
        return _error(f"File processing failed: {exc}")


@module1_bp.route("/api/predict", methods=["POST"])
def predict():
    """Score a stored payload with the selected pipelines."""
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename")
    selected_models = payload.get("models") or []

    if not filename or not selected_models:
        return _error("Missing required parameters")

    try:
        frame = upload_store.load_frame(filename)
    except UploadRejected as exc:
        return _error(str(exc))

    try:
        games = upload_store.to_records(frame)
        report = pipeline.execute(games, selected_models)
        return jsonify(
            {
                "success": True,
                "results": report.by_game(),
                "total_games": report.total_games,
                "total_models": len(selected_models),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        return _error(f"Prediction failed: {exc}")


@module1_bp.route("/api/predict_comparison", methods=["POST"])
def predict_comparison():
    """Score a stored payload with every registered pipeline plus an ensemble."""
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename")

    if not filename:
        return _error("Missing filename")

    try:
        frame = upload_store.load_frame(filename)
    except UploadRejected as exc:
        return _error(str(exc))

    try:
        games = upload_store.to_records(frame)
        report = pipeline.compare(games)
        individual = report.matrix()
        ensemble = _ensemble(individual, games)

        return jsonify(
            {
                "success": True,
                "individual_results": individual,
                "ensemble_predictions": ensemble,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("comparison failed")
        return _error(f"Comparison prediction failed: {exc}")


def _ensemble(individual: dict, games: list[dict]) -> list[dict]:
    """Majority vote over pipelines with an averaged probability pair."""
    ensemble: list[dict] = []

    for row, game in enumerate(games):
        votes: list[int] = []
        home_probs: list[float] = []
        away_probs: list[float] = []

        for result in individual.values():
            predictions = result["predictions"]
            if row >= len(predictions):
                continue
            entry = predictions[row]
            votes.append(entry["prediction"])
            home_probs.append(entry["home_win_prob"])
            away_probs.append(entry["away_win_prob"])

        if not votes:
            continue

        home_team = game.get("h_team_name", "Home Team")
        away_team = game.get("o_team_name", "Away Team")
        winner_home = sum(votes) > len(votes) / 2

        ensemble.append(
            {
                "index": row,
                "home_team": home_team,
                "away_team": away_team,
                "prediction": 1 if winner_home else 0,
                "winner": home_team if winner_home else away_team,
                "home_win_prob": float(sum(home_probs) / len(home_probs) * 100.0),
                "away_win_prob": float(sum(away_probs) / len(away_probs) * 100.0),
            }
        )

    return ensemble
