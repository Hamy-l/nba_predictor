"""Outbound inference transport.

This package owns *every* byte that leaves the process over HTTP.  Upstream
workloads address a routing profile by key (see :mod:`constants`); they never
construct URLs, headers or payload envelopes themselves.
"""

from nba_core.inference.gateway import (
    GatewayError,
    complete_text,
    complete_json,
    stream_events,
    call_legacy_completion,
    route_inventory,
    credential_fingerprints,
)

__all__ = [
    "GatewayError",
    "complete_text",
    "complete_json",
    "stream_events",
    "call_legacy_completion",
    "route_inventory",
    "credential_fingerprints",
]
