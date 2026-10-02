"""Payload template publication."""

from __future__ import annotations

import os

from constants import UPLOAD_DIR
from nba_core.features.samples import build_template_frame, template_filename

__all__ = ["TemplateService", "template_service", "TemplateUnavailable"]


class TemplateUnavailable(KeyError):
    """Raised when an unknown template kind is requested."""


class TemplateService:
    """Materialises downloadable payload templates on demand."""

    KINDS = ("single", "time_series")

    def __init__(self, root: str | None = None):
        self.root = root or UPLOAD_DIR
        os.makedirs(self.root, exist_ok=True)

    def publish(self, kind: str) -> tuple[str, str]:
        """Write a template and return ``(filename, absolute_path)``."""
        if kind not in self.KINDS:
            raise TemplateUnavailable(kind)

        filename = template_filename(kind)
        path = os.path.join(self.root, filename)
        frame = build_template_frame(kind)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return filename, path

    def catalogue(self) -> dict:
        return {
            "single": {
                "filename": template_filename("single"),
                "description": "Cross-sectional matchup rows for the module-1 tool",
            },
            "time_series": {
                "filename": template_filename("time_series"),
                "description": "Ordered form window for the sequence pipelines",
            },
        }


template_service = TemplateService()
