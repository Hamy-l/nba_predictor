"""Diagnostics routes (additive; no existing endpoint changes)."""

from __future__ import annotations

from flask import Blueprint, jsonify

from nba_core import __version__
from nba_core.inference.gateway import route_inventory
from nba_core.services.datasets import dataset_status
from nba_core.serving.evidence import EnvelopeResolver
from nba_core.serving.registry import registry

__all__ = ["system_bp"]

system_bp = Blueprint("system", __name__)

#: Shared resolver used purely for counter reporting.
_resolver = EnvelopeResolver()


@system_bp.route("/api/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "version": __version__,
            "pipelines": len(registry.keys()),
        }
    )


@system_bp.route("/api/diagnostics", methods=["GET"])
def diagnostics():
    """Read-only view of the runtime wiring."""
    return jsonify(
        {
            "success": True,
            "version": __version__,
            "dataset": dataset_status(),
            "routes": route_inventory(),
            "envelope_counters": _resolver.counters(),
            "pipelines": {
                key: {"name": descriptor.name, "family": descriptor.estimator_family}
                for key, descriptor in registry.items()
            },
        }
    )
