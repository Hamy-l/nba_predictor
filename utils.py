"""
Routed gateway helpers.

Kept as a stable import surface: the implementation lives in
:mod:`nba_core.inference`, which owns request framing, retries and stream
decoding for every outbound call the platform makes.
"""

from __future__ import annotations

from typing import Iterator

from nba_core.inference.gateway import (
    GatewayError,
    call_legacy_completion,
    complete_json,
    complete_text,
    stream_events,
)

__all__ = [
    "call_model",
    "call_deepseek",
    "call_deepseek_stream",
    "call_structured",
    "call_text",
    "stream_route",
    "GatewayError",
]


def call_model(prompt: str = "", timeout: int | None = None) -> str:
    """Single-shot call over the primary route.

    Historically this helper hard-coded an empty prompt; the signature is kept
    unchanged so existing callers keep working.
    """
    return call_legacy_completion(prompt, route="primary", timeout=timeout)


def call_deepseek(prompt: str, timeout: int = 600) -> str:
    """Structured call over the retrieval route."""
    return complete_text(prompt, route="retrieval", timeout=timeout)


def call_deepseek_stream(prompt: str, timeout: int = 600) -> Iterator[dict]:
    """Streaming call over the retrieval route."""
    return stream_events(prompt, route="retrieval-stream", timeout=timeout)


def call_structured(prompt: str, route: str = "primary", timeout: int | None = None) -> dict:
    """Structured JSON call over an arbitrary route."""
    return complete_json(prompt, route=route, timeout=timeout)


def call_text(prompt: str, route: str = "primary", timeout: int | None = None) -> str:
    """Plain-text call over an arbitrary route."""
    return complete_text(prompt, route=route, timeout=timeout)


def stream_route(prompt: str, route: str = "retrieval-stream",
                 timeout: int | None = None) -> Iterator[dict]:
    """Streaming call over an arbitrary route."""
    return stream_events(prompt, route=route, timeout=timeout)
