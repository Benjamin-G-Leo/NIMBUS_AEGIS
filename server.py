"""ReliefMatch FastMCP inspector server.

Exposes the ReliefMatch pipeline as standard FastMCP tools so an internal
inspector (any MCP client) can call them directly over stdio, HTTP or SSE:

    * ``evaluate_offer``       -- triage a free-text offer (extraction + metrics + drafted email)
    * ``detect_clashes``       -- list over-pledged (community, item) cells with a recommended fix
    * ``apply_reallocation``   -- shift/split a pledge and report before/after metrics

The tools delegate to the exact same pure domain functions that the FastAPI layer
in ``app.py`` uses (``app.run_triage`` / ``app.run_reallocation``) and to the
deterministic :mod:`core` module. This guarantees the inspector observes the same
behavior as the HTTP API. Because the pipeline functions are synchronous and
self-contained, importing this module never starts a web server or causes request
loops.

Run locally (stdio, the MCP default):
    python server.py

Run over streamable HTTP for a remote inspector:
    MCP_TRANSPORT=streamable-http MCP_PORT=8001 python server.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

import app
import core

mcp = FastMCP("ReliefMatch Inspector")


@mcp.tool
def evaluate_offer(text: str, team: Optional[str] = None) -> Dict[str, Any]:
    """Triage a free-text offer.

    Extracts offered quantities using OpenRouter's Qwen model (temperature=0.0,
    JSON-only) with an offline regex fallback if the LLM is unavailable, computes
    the per-item shortfall via ``core.calculate_gap``, detects the current
    distribution clashes, and returns the algorithmically drafted response email.

    Args:
        text: The raw offer text block.
        team: Optional team name used in the email salutation.

    Returns:
        A dict with ``extraction``, ``source`` ("qwen" or "regex-fallback"),
        ``fallback_reason``, ``metrics`` (per-item gaps + clashes) and
        ``drafted_email``.
    """
    return app.run_triage(text, team=team)


@mcp.tool
def detect_clashes() -> List[Dict[str, Any]]:
    """Return every over-pledged (community, item) cell and a recommended fix.

    Each entry includes ``item``, ``community``, ``need``, ``pledged``,
    ``overage``, ``severity`` (HIGH/MEDIUM) and a ``recommendation`` payload
    (``{"pledge_id", "to_community", "qty"}``) or ``None`` when no safe target
    exists.
    """
    return core.detect_clashes()


@mcp.tool
def apply_reallocation(pledge_id: str, to_community: str, qty: int) -> Dict[str, Any]:
    """Shift or split an existing pledge to a new community.

    Delegates to the same reallocation pipeline the HTTP API uses and returns the
    applied change plus before/after gap tables, remaining clashes, and a
    human-readable summary.

    Args:
        pledge_id: The id of the pledge to relocate.
        to_community: Target community name.
        qty: Positive integer quantity to move.

    Returns:
        A dict with ``applied``, ``state_after``, ``before`` and ``after``. If the
        request is invalid, returns ``{"error": <message>}`` instead of raising.
    """
    try:
        return app.run_reallocation(pledge_id, to_community, qty)
    except ValueError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport in ("http", "sse", "streamable-http"):
        mcp.run(
            transport=transport,
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8001")),
        )
    else:
        mcp.run()
