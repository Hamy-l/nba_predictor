"""Web tier: Flask blueprints and the application factory."""

from nba_core.web.app_factory import create_app
from nba_core.web.blueprints import ALL_BLUEPRINTS

__all__ = ["create_app", "ALL_BLUEPRINTS"]
