"""Application factory and shared HTTP concerns."""

from __future__ import annotations

import logging
import os
import time

from flask import Flask, jsonify, request

from constants import (
    PROJECT_ROOT,
    SERVER_DEBUG,
    UPLOAD_DIR,
    UPLOAD_MAX_CONTENT_LENGTH,
    UPLOAD_ALLOWED_EXTENSIONS,
)
from nba_core import __version__

__all__ = ["create_app", "configure_logging"]

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Idempotent logging setup for both dev and WSGI entry points."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.DEBUG if SERVER_DEBUG else logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        )

    # Third-party transports are chatty at DEBUG; keep their signal at WARNING.
    for noisy in ("urllib3", "requests", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_SUFFIXES = (".json",)


def _wants_json() -> bool:
    if request.path.startswith("/api/"):
        return True
    return request.path.endswith(_SUFFIXES)


def create_app(config: dict | None = None) -> Flask:
    """Build and configure the Flask application."""
    configure_logging()

    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, "templates"),
    )

    app.config.update(
        UPLOAD_FOLDER=UPLOAD_DIR,
        MAX_CONTENT_LENGTH=UPLOAD_MAX_CONTENT_LENGTH,
        ALLOWED_EXTENSIONS=set(UPLOAD_ALLOWED_EXTENSIONS),
        JSON_SORT_KEYS=False,
        APP_VERSION=__version__,
    )
    if config:
        app.config.update(config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    _register_blueprints(app)
    _register_observers(app)
    _register_error_handlers(app)

    logger.info("nba-predictor %s ready (uploads=%s)", __version__, app.config["UPLOAD_FOLDER"])
    return app


def _register_blueprints(app: Flask) -> None:
    from nba_core.web.blueprints import ALL_BLUEPRINTS

    for blueprint in ALL_BLUEPRINTS:
        app.register_blueprint(blueprint)


def _register_observers(app: Flask) -> None:
    @app.before_request
    def _stamp_start() -> None:
        request.environ["nba.started_at"] = time.perf_counter()

    @app.after_request
    def _tag_response(response):
        started = request.environ.get("nba.started_at")
        if started is not None:
            response.headers["X-Elapsed-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        response.headers["X-Service-Version"] = __version__
        return response


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def _not_found(error):
        if _wants_json():
            return jsonify({"success": False, "message": "resource not found"}), 404
        return error, 404

    @app.errorhandler(413)
    def _too_large(error):
        return jsonify({"success": False, "message": "uploaded payload exceeds the size limit"}), 413

    @app.errorhandler(500)
    def _server_error(error):
        logger.exception("unhandled application error")
        if _wants_json():
            return jsonify({"success": False, "message": "internal server error"}), 500
        return error, 500

    @app.errorhandler(Exception)
    def _unexpected(error):
        if isinstance(error, SystemExit):
            raise error
        logger.exception("unhandled exception: %s", error)
        if _wants_json():
            return jsonify({"success": False, "message": f"unexpected error: {error}"}), 500
        raise error
