"""Transport helpers shared by the outbound gateway layer."""

from __future__ import annotations

import json
import re

from constants import STREAM_CHUNK_PREFIX, STREAM_DONE_SENTINEL

__all__ = [
    "redact",
    "ensure_message_list",
    "extract_json_object",
    "normalise_pair",
    "parse_sse_data",
]


def redact(secret: str | None) -> str:
    """Return a log-safe fingerprint of a credential."""
    if not secret:
        return "unset"
    tail = secret[-4:] if len(secret) >= 4 else ""
    return f"***{tail}"


def ensure_message_list(prompt: str | list) -> list:
    """Coerce a prompt into the chat message array shape."""
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": prompt}]


def extract_json_object(raw: str) -> dict:
    """Best-effort extraction of a JSON object from a model response body.

    Handles fenced code blocks, a leading ``json`` tag and prose surrounding a
    single object literal.  Raises ``ValueError`` when nothing parseable is
    found.
    """
    if raw is None:
        raise ValueError("empty completion body")

    text = raw.strip()

    fence = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    if text[:4].lower() == "json":
        text = text[4:].strip()

    if not text:
        raise ValueError("empty completion body after cleanup")

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return json.loads(text)


def normalise_pair(
    first: float | int | str | None,
    second: float | int | str | None,
    scale: float = 100.0,
    fallback: float = 50.0,
) -> tuple[float, float]:
    """Scale a two-way probability pair so it sums to ``scale``."""
    try:
        a = float(first)
    except (TypeError, ValueError):
        a = fallback
    try:
        b = float(second)
    except (TypeError, ValueError):
        b = fallback

    total = a + b
    if total <= 0:
        return fallback, fallback
    return a / total * scale, b / total * scale


def parse_sse_data(line: bytes | str):
    """Decode one ``data:`` line from a server-sent-event stream.

    Returns ``None`` for framing noise and for the terminal sentinel.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if not line or not line.startswith(STREAM_CHUNK_PREFIX):
        return None

    payload = line[len(STREAM_CHUNK_PREFIX) :].strip()
    if not payload or payload == STREAM_DONE_SENTINEL:
        return None

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
