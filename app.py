"""
NBA Predictor -- application entry point.

The web tier is assembled by :func:`nba_core.web.create_app`; this module only
wires the process-level entry point (development server / WSGI target).

Layout
------
    constants.py            every endpoint, credential, catalogue and path
    nba_core/web/           blueprints (HTTP framing only)
    nba_core/services/      workflow orchestration (fusion, roster, uploads)
    nba_core/serving/       scoring pipelines, registry, batch execution
    nba_core/inference/     routed gateway transport
    compatibility.py        adapters for the pre-2.x public functions

Run
---
    python app.py
"""

from __future__ import annotations

import logging

from constants import SERVER_DEBUG, SERVER_HOST, SERVER_PORT
from nba_core.web import create_app

logging.getLogger("werkzeug").setLevel(logging.WARNING)

#: WSGI callable.
app = create_app()


if __name__ == "__main__":
    app.run(debug=SERVER_DEBUG, host=SERVER_HOST, port=SERVER_PORT)
