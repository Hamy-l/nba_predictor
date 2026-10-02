"""Application services.

One module per workflow.  Services own validation, I/O and orchestration; the
web tier only translates HTTP into service calls.
"""

from nba_core.services.datasets import DataUnavailable, get_nba_processor, reload_dataset
from nba_core.services.uploads import UploadStore, UploadRejected, upload_store
from nba_core.services.templates import TemplateService, template_service
from nba_core.services.roster import RosterService, roster_service
from nba_core.services.sse import EventStream, encode_event
from nba_core.services.fusion import FusionService, fusion_service
from nba_core.services.players import PlayerNewsService, player_news_service

__all__ = [
    "DataUnavailable",
    "get_nba_processor",
    "reload_dataset",
    "UploadStore",
    "UploadRejected",
    "upload_store",
    "TemplateService",
    "template_service",
    "RosterService",
    "roster_service",
    "EventStream",
    "encode_event",
    "FusionService",
    "fusion_service",
    "PlayerNewsService",
    "player_news_service",
]
