"""Outbound gateway client.

Every workload in the platform funnels through :class:`GatewayClient`, which
resolves a routing profile from :mod:`constants`, frames the request envelope
for the profile's surface, applies the retry policy and yields decoded events.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import requests

from constants import (
    INFERENCE_DEFAULT_ROUTE,
    INFERENCE_ROUTES,
    describe_routes,
    resolve_inference_route,
)
from nba_core.inference._transport import (
    ensure_message_list,
    extract_json_object,
    parse_sse_data,
    redact,
)

__all__ = [
    "GatewayError",
    "GatewayClient",
    "get_client",
    "complete_text",
    "complete_json",
    "stream_events",
    "call_legacy_completion",
    "route_inventory",
    "credential_fingerprints",
]

#: HTTP statuses worth another attempt.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    """Raised when a routed inference call cannot be satisfied."""

    def __init__(self, message: str, *, route: str = "", status: int | None = None):
        super().__init__(message)
        self.route = route
        self.status = status


def _log_attempt(route_name: str, attempt: int, attempts: int, exc: Exception) -> None:
    """Report a failed attempt, downgrading transient statuses to debug."""
    transient = isinstance(exc, GatewayError) and exc.status in RETRYABLE_STATUS
    level = logging.DEBUG if transient else logging.WARNING
    logger.log(level, "route %s attempt %s/%s failed: %s", route_name, attempt, attempts, exc)


def _build_envelope(spec: dict, prompt: str, *, stream: bool, structured: bool) -> dict:
    """Frame a request body for the surface described by ``spec``."""
    payload: dict[str, Any] = {
        "model": spec["model"],
        spec.get("payload_key", "messages"): ensure_message_list(prompt),
    }

    if spec.get("surface") == "chat.completions":
        payload["stream"] = bool(stream)
        if structured and spec.get("structured_output"):
            payload["response_format"] = {"type": "json_object"}
    else:
        if structured and spec.get("structured_output"):
            payload["response_format"] = {"type": "json_object"}
        if stream:
            payload["stream"] = True

    tools = spec.get("tools")
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    return payload


def _build_headers(spec: dict) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {spec['api_key']}",
    }


def _extract_chat_text(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GatewayError(f"unexpected completion envelope: {exc}") from exc


def _extract_response_text(body: dict) -> str:
    """Flatten the ``output`` array of a responses-style envelope."""
    fragments: list[str] = []
    for item in body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and content.get("text"):
                fragments.append(content["text"])
    return "".join(fragments)


class GatewayClient:
    """Retrying HTTP client for the routed gateway profiles."""

    def __init__(self, max_attempts: int = 3, backoff: float = 0.6):
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = float(backoff)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "nba-predictor/2.x"})

    # -- primitives ---------------------------------------------------------

    def _post(self, spec: dict, envelope: dict, *, stream: bool) -> requests.Response:
        route_name = spec.get("route_key", INFERENCE_DEFAULT_ROUTE)
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._session.post(
                    spec["endpoint"],
                    headers=_build_headers(spec),
                    json=envelope,
                    timeout=spec.get("timeout", 120),
                    stream=stream,
                )
            except requests.RequestException as exc:
                last_error = GatewayError(f"transport failure: {exc}", route=route_name)
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS:
                    raise GatewayError(
                        f"gateway rejected request ({response.status_code})",
                        route=route_name,
                        status=response.status_code,
                    )
                last_error = GatewayError(
                    f"gateway returned {response.status_code}",
                    route=route_name,
                    status=response.status_code,
                )
                response.close()

            if attempt < self.max_attempts:
                _log_attempt(route_name, attempt, self.max_attempts, last_error)
                time.sleep(self.backoff * attempt)

        if last_error is not None:
            _log_attempt(route_name, self.max_attempts, self.max_attempts, last_error)
        raise last_error or GatewayError("gateway call failed", route=route_name)

    # -- public surface -----------------------------------------------------

    def complete(self, prompt: str, *, route: str | None = None,
                 timeout: int | None = None, structured: bool = False) -> str:
        """Return the text body produced for ``prompt``."""
        spec = resolve_inference_route(route)
        spec["route_key"] = route or INFERENCE_DEFAULT_ROUTE
        if timeout:
            spec["timeout"] = timeout

        envelope = _build_envelope(spec, prompt, stream=False, structured=structured)
        response = self._post(spec, envelope, stream=False)
        body = response.json()

        if spec.get("surface") == "chat.completions":
            return _extract_chat_text(body)
        return _extract_response_text(body)

    def complete_structured(self, prompt: str, *, route: str | None = None,
                            timeout: int | None = None) -> dict:
        """Return a parsed JSON object produced for ``prompt``."""
        raw = self.complete(prompt, route=route, timeout=timeout, structured=True)
        return extract_json_object(raw)

    def stream(self, prompt: str, *, route: str | None = None,
               timeout: int | None = None) -> Iterator[dict]:
        """Yield decoded events from a streaming gateway call."""
        spec = resolve_inference_route(route)
        spec["route_key"] = route or INFERENCE_DEFAULT_ROUTE
        if timeout:
            spec["timeout"] = timeout

        envelope = _build_envelope(spec, prompt, stream=True, structured=False)
        response = self._post(spec, envelope, stream=True)

        for line in response.iter_lines():
            event = parse_sse_data(line)
            if event is None:
                continue
            if isinstance(event, dict) and event.get("type") == "error":
                raise GatewayError(
                    f"gateway stream error: {event.get('message', 'unknown')}",
                    route=spec["route_key"],
                )
            yield event

        response.close()


_CLIENT: GatewayClient | None = None


def get_client() -> GatewayClient:
    """Process-wide gateway client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = GatewayClient()
    return _CLIENT


def complete_text(prompt: str, *, route: str | None = None,
                  timeout: int | None = None) -> str:
    return get_client().complete(prompt, route=route, timeout=timeout)


def complete_json(prompt: str, *, route: str | None = None,
                  timeout: int | None = None) -> dict:
    return get_client().complete_structured(prompt, route=route, timeout=timeout)


def stream_events(prompt: str, *, route: str | None = None,
                  timeout: int | None = None) -> Iterator[dict]:
    return get_client().stream(prompt, route=route, timeout=timeout)


def call_legacy_completion(prompt: str, *, route: str | None = None,
                           timeout: int | None = None) -> str:
    """Compatibility entry point for the original single-shot helper."""
    return get_client().complete(prompt, route=route or "primary", timeout=timeout)


def route_inventory() -> dict:
    """Redacted view of the configured routes (safe for diagnostics)."""
    return describe_routes()


def credential_fingerprints() -> dict:
    """Provider + credential fingerprint per configured route."""
    return {
        key: {
            "provider": spec["provider"],
            "endpoint": spec["endpoint"],
            "credential": redact(spec["api_key"]),
        }
        for key, spec in INFERENCE_ROUTES.items()
    }
