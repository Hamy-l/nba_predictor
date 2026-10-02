"""Landing page and model catalogue."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from nba_core.serving.registry import registry

__all__ = ["platform_bp"]

platform_bp = Blueprint("platform", __name__)


@platform_bp.route("/")
def index():
    """Landing page for the custom-data prediction tool."""
    return render_template("index.html", models=registry.public_catalog())


@platform_bp.route("/api/models", methods=["GET"])
def models():
    """Public catalogue of the selectable scoring pipelines."""
    return jsonify({"success": True, "models": registry.public_catalog()})
