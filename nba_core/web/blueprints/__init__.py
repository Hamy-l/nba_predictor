"""Blueprint registry for the web tier."""

from nba_core.web.blueprints.platform import platform_bp
from nba_core.web.blueprints.module1 import module1_bp
from nba_core.web.blueprints.module2 import module2_bp
from nba_core.web.blueprints.module3 import module3_bp
from nba_core.web.blueprints.system import system_bp

__all__ = [
    "platform_bp",
    "module1_bp",
    "module2_bp",
    "module3_bp",
    "system_bp",
    "ALL_BLUEPRINTS",
]

ALL_BLUEPRINTS = (
    platform_bp,
    module1_bp,
    module2_bp,
    module3_bp,
    system_bp,
)
