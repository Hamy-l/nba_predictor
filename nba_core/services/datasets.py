"""Dataset access for the historical NBA feature matrix."""

from __future__ import annotations

import logging
import threading

from constants import NBA_DATASET_FILE
from process_data import NBADataProcessor

__all__ = ["DataUnavailable", "get_nba_processor", "reload_dataset", "dataset_status"]

logger = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """Raised when a workflow needs the historical dataset but it is not loaded."""


_LOCK = threading.RLock()
_PROCESSOR: NBADataProcessor | None = None
_LOAD_ERROR: str | None = None


def _load(path: str) -> NBADataProcessor | None:
    global _PROCESSOR, _LOAD_ERROR
    try:
        _PROCESSOR = NBADataProcessor(path)
        _LOAD_ERROR = None
        logger.info("historical dataset loaded: %s", path)
    except Exception as exc:  # noqa: BLE001 - startup must survive a missing dataset
        _PROCESSOR = None
        _LOAD_ERROR = str(exc)
        logger.warning("historical dataset unavailable: %s", exc)
    return _PROCESSOR


def get_nba_processor() -> NBADataProcessor:
    """Return the shared dataset processor, loading it on first use."""
    with _LOCK:
        if _PROCESSOR is None:
            _load(NBA_DATASET_FILE)
    if _PROCESSOR is None:
        raise DataUnavailable(_LOAD_ERROR or "historical dataset is not available")
    return _PROCESSOR


def reload_dataset(path: str | None = None) -> NBADataProcessor | None:
    """Force a reload from ``path`` (defaults to the configured dataset)."""
    with _LOCK:
        return _load(path or NBA_DATASET_FILE)


def dataset_status() -> dict:
    """Diagnostic view of the dataset binding."""
    with _LOCK:
        return {
            "path": NBA_DATASET_FILE,
            "loaded": _PROCESSOR is not None,
            "error": _LOAD_ERROR,
        }


# Warm the dataset at import time, matching the original startup behaviour.
_load(NBA_DATASET_FILE)
