"""Upload store: persistence, validation and preview for user payloads."""

from __future__ import annotations

import os
import threading
from datetime import datetime

import pandas as pd
from werkzeug.utils import secure_filename

from constants import (
    MODULE1_FEATURE_COLUMNS,
    UPLOAD_ALLOWED_EXTENSIONS,
    UPLOAD_DIR,
)

__all__ = ["UploadStore", "UploadRejected", "upload_store", "REQUIRED_COLUMNS"]

#: Columns every payload must carry.
REQUIRED_COLUMNS = ("h_team_name", "o_team_name")

_SPREADSHEET_SUFFIXES = (".xlsx", ".xls")


class UploadRejected(ValueError):
    """Raised when a payload fails validation."""


class UploadStore:
    """Filesystem-backed store for prediction payloads."""

    def __init__(self, root: str | None = None,
                 allowed: frozenset[str] | None = None):
        self.root = root or UPLOAD_DIR
        self.allowed = allowed or UPLOAD_ALLOWED_EXTENSIONS
        self._lock = threading.RLock()
        os.makedirs(self.root, exist_ok=True)

    # -- naming -------------------------------------------------------------

    def is_allowed(self, filename: str) -> bool:
        if "." not in filename:
            return False
        return filename.rsplit(".", 1)[1].lower() in self.allowed

    def stage_name(self, filename: str) -> str:
        """Timestamped, sanitised name used on disk."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{secure_filename(filename)}"

    def path_for(self, filename: str) -> str:
        """Absolute path for a stored payload, guarding against traversal."""
        candidate = os.path.basename(filename or "")
        if not candidate:
            raise UploadRejected("empty filename")
        return os.path.join(self.root, candidate)

    def exists(self, filename: str) -> bool:
        try:
            return os.path.exists(self.path_for(filename))
        except UploadRejected:
            return False

    # -- persistence --------------------------------------------------------

    def persist(self, storage) -> str:
        """Store an uploaded file object and return its staged name."""
        if storage is None or not getattr(storage, "filename", ""):
            raise UploadRejected("no file uploaded")
        if not self.is_allowed(storage.filename):
            raise UploadRejected(
                "unsupported file format, please upload a CSV or Excel file"
            )

        staged = self.stage_name(storage.filename)
        target = self.path_for(staged)
        with self._lock:
            storage.save(target)
        return staged

    def load_frame(self, filename: str) -> pd.DataFrame:
        """Read a stored payload into a DataFrame."""
        path = self.path_for(filename)
        if not os.path.exists(path):
            raise UploadRejected("file not found")

        with self._lock:
            if path.lower().endswith(_SPREADSHEET_SUFFIXES):
                frame = pd.read_excel(path)
            else:
                frame = pd.read_csv(path)

        return frame.fillna(0)

    def discard(self, filename: str) -> None:
        """Remove a stored payload, ignoring a missing file."""
        try:
            path = self.path_for(filename)
        except UploadRejected:
            return
        with self._lock:
            if os.path.exists(path):
                os.remove(path)

    # -- validation ---------------------------------------------------------

    @staticmethod
    def validate(frame: pd.DataFrame) -> tuple[bool, str | None]:
        """Check that a frame carries the identifiers and at least one feature."""
        columns = set(frame.columns)

        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            return False, f"Missing required column: {missing[0]}"

        if not any(column in columns for column in MODULE1_FEATURE_COLUMNS):
            return False, (
                "Missing feature columns, need at least one of: "
                + ", ".join(MODULE1_FEATURE_COLUMNS)
            )

        return True, None

    @classmethod
    def preview(cls, frame: pd.DataFrame, rows: int = 5) -> list[dict]:
        """Serialisable head of a frame."""
        return frame.head(rows).to_dict("records")

    @classmethod
    def to_records(cls, frame: pd.DataFrame) -> list[dict]:
        """Row-major records with a stable index column."""
        records: list[dict] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            record = row.to_dict()
            record["index"] = int(position)
            records.append(record)
        return records


upload_store = UploadStore()
