"""ReliefMatch orchestration engine (FastAPI).

This module is the HTTP + AI orchestration layer for ReliefMatch:

    * Serves the hand-coded frontend in ``static/`` (mounted at ``/static``) and
      the console UI at ``/`` via :class:`FileResponse`.
    * Exposes the pipeline API: ``/api/grid_state`` (GET), ``/api/triage`` (POST)
      and ``/api/reallocate`` (POST).
    * Extracts offered quantities from free-text offers using OpenRouter's Qwen
      model through the OpenAI client (``temperature=0.0``). The LLM is strictly
      constrained to return raw JSON of ``{"item_name": quantity}`` and NEVER
      computes any metric.
    * All metric math (gaps, clashes) and state mutation are delegated to the
      deterministic :mod:`core` module; the LLM's numbers are fed into
      ``core.calculate_gap`` and the resulting metrics are used to algorithmically
      assemble the drafted response email.
    * If the OpenRouter connection is missing/drops/fails (or the LLM output is
      unparseable), an inline try/except degrades to a deterministic regex parser
      and drafts a template fallback email, so triage never hard-fails.

The domain logic lives in the pure functions :func:`run_triage`,
:func:`run_reallocation` and :func:`build_grid_state`; the FastAPI routes are thin
wrappers so the same logic can be reused verbatim by the FastMCP inspector in
``server.py`` without request loops.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from core import (
    apply_reallocation,
    calculate_gap,
    detect_clashes,
    load_state,
    singular,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("reliefmatch")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# --------------------------------------------------------------------------- #
# OpenRouter / Qwen configuration
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Any OpenRouter Qwen slug works; override via OPENROUTER_MODEL to your tier.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-3-30b-a3b")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "30"))
_APP_TITLE = "ReliefMatch Triage"


class OpenRouterNotConfigured(RuntimeError):
    """Raised when the OpenRouter API key is missing or empty."""


_client: Optional[OpenAI] = None


def _get_openrouter_client() -> OpenAI:
    """Lazily build and cache the OpenAI-compatible OpenRouter client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise OpenRouterNotConfigured(
                "OPENROUTER_API_KEY is not set; offline regex fallback is active."
            )
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=OPENROUTER_TIMEOUT,
            default_headers={
                "HTTP-Referer": "https://reliefmatch.example",
                "X-Title": _APP_TITLE,
            },
        )
    return _client


# --------------------------------------------------------------------------- #
# LLM extraction (Qwen via OpenRouter) — quantities ONLY, never metrics
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM_PROMPT = (
    "You are the extraction module of the ReliefMatch triage system.\n"
    "Your ONLY task is to read the provided offer text and extract the quantity "
    "of each distinct relief item being offered.\n\n"
    "STRICT OUTPUT RULES:\n"
    "- Return a SINGLE raw JSON object and nothing else: no prose, no markdown, "
    "no code fences, no explanations, no trailing commas.\n"
    "- The object maps item names to non-negative integers only.\n"
    "- Keys: the item name, lower-cased, with single spaces "
    "(e.g. \"notebook\", \"hygiene kit\").\n"
    "- Values: an integer quantity. Never a decimal, string, range, or unit word.\n"
    "- You must NOT compute or infer ANY metric. Do not output need, shortfall, "
    "gap, total, percentage, stock, pledged, or any comparison. "
    "You report ONLY the offered quantities.\n"
    "- If an item is mentioned without an explicit number, omit it.\n"
    "- If no quantities can be found, return exactly: {}\n"
)


def _call_qwen(text: str) -> str:
    """Call OpenRouter's Qwen model and return the raw completion text."""
    client = _get_openrouter_client()
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Extract the offered relief-item quantities from the "
                "text below:\n\n" + text,
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter returned an empty completion.")
    return content


def _parse_llm_json(raw: str) -> Dict[str, int]:
    """Parse a strict ``{"item_name": int}`` object from the LLM output.

    Defensive: tolerates accidental code fences, coerces values to int, and
    normalizes keys via :func:`core.singular`. Raises on anything non-numeric so
    the caller can fall back to the offline parser.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM JSON is not an object.")
    result: Dict[str, int] = {}
    for key, value in data.items():
        qty = int(value)
        if qty < 0:
            raise ValueError(f"Negative quantity for item {key!r}.")
        item = singular(str(key))
        result[item] = result.get(item, 0) + qty
    return result


# --------------------------------------------------------------------------- #
# Offline fallback — deterministic regex quantity extraction
# --------------------------------------------------------------------------- #
# Matches "<integer> <item>" for the canonical relief items the system tracks.
# The quantity accepts a plain integer or a comma-grouped thousands form
# (e.g. "1,000"). Extend the item alternation below for new standard items.
_QUANTITY_RE = re.compile(
    r"(?P<qty>\d{1,3}(?:,\d{3})+|\d+)\s+"
    r"(?P<item>(?:hygiene\s+kits?|pencil\s+boxes?|notebooks?|jackets?|blankets?))\b",
    re.IGNORECASE,
)


def _regex_extract(text: str) -> Dict[str, int]:
    """Extract standard quantities from free text with a deterministic regex."""
    found: Dict[str, int] = {}
    for match in _QUANTITY_RE.finditer(text):
        qty = int(match.group("qty").replace(",", ""))
        item = singular(match.group("item").lower())
        found[item] = found.get(item, 0) + qty
    return found


def extract_quantities(text: str) -> Tuple[Dict[str, int], str, str]:
    """Extract offered quantities, preferring Qwen and degrading to regex.

    Returns ``(quantities, source, reason)`` where ``source`` is ``"qwen"`` on a
    successful LLM call, or ``"regex-fallback"`` when the OpenRouter connection,
    the LLM output, or JSON parsing fails and the inline offline parser is used.
    Any LLM-side failure is swallowed here (and logged) so triage never crashes.
    """
    try:
        raw = _call_qwen(text)
        return _parse_llm_json(raw), "qwen", ""
    except OpenRouterNotConfigured as exc:
        logger.warning("OpenRouter not configured -> regex fallback: %s", exc)
        return _regex_extract(text), "regex-fallback", str(exc)
    except Exception as exc:  # noqa: BLE001 - intentional broad offline insulation
        logger.warning("OpenRouter/LLM failure -> regex fallback: %s", exc)
        return _regex_extract(text), "regex-fallback", f"LLM unavailable: {exc}"


# --------------------------------------------------------------------------- #
# Deterministic metric helpers (the math lives in core, not the LLM)
# --------------------------------------------------------------------------- #
def item_breakdown(item: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Return the need/pledged/stock/gap breakdown for a single item.

    The gap is the authoritative value produced by :func:`core.calculate_gap`;
    the other three are read directly from state for display only.
    """
    if state is None:
        state = load_state()
    key = singular(item)
    need = sum(needs.get(key, 0) for needs in state.get("community_needs", {}).values())
    pledged = sum(
        p.get("qty", 0)
        for p in state.get("active_pledges", [])
        if singular(p.get("item", "")) == key
    )
    stock = state.get("warehouse_stock", {}).get(key, 0)
    return {
        "need": need,
        "pledged": pledged,
        "stock": stock,
        "gap": calculate_gap(item, state),
    }


def _all_items(state: Dict[str, Any]) -> List[str]:
    """Every item referenced anywhere in the state, sorted."""
    items = set(state.get("warehouse_stock", {}).keys())
    for needs in state.get("community_needs", {}).values():
        items.update(needs.keys())
    for p in state.get("active_pledges", []):
        items.add(singular(p.get("item", "")))
    return sorted(items)


def build_grid_state() -> Dict[str, Any]:
    """The full snapshot the frontend renders (stock, needs, pledges, gaps, clashes)."""
    state = load_state()
    breakdowns = {item: item_breakdown(item, state) for item in _all_items(state)}
    return {
        "warehouse_stock": state.get("warehouse_stock", {}),
        "community_needs": state.get("community_needs", {}),
        "active_pledges": state.get("active_pledges", []),
        "gaps": {item: breakdowns[item]["gap"] for item in breakdowns},
        "item_breakdown": breakdowns,
        "clashes": detect_clashes(state),
    }


# --------------------------------------------------------------------------- #
# Drafted response email (assembled algorithmically from the metrics)
# --------------------------------------------------------------------------- #
def build_triage_email(
    quantities: Dict[str, int],
    source: str,
    reason: str,
    team: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> str:
    """Assemble the response email deterministically from extraction + metrics."""
    if state is None:
        state = load_state()
    addressee = team.strip() if (team and team.strip()) else "team"

    lines: List[str] = []
    lines.append("Subject: ReliefMatch Triage — Offer Review")
    lines.append("")
    lines.append(f"Hello {addressee},")
    lines.append("")
    lines.append("We've reviewed your latest offer against live distribution requirements.")
    if source != "qwen":
        lines.append("")
        lines.append(
            f"Note: automated extraction ran on the offline fallback "
            f"({reason.rstrip('.').strip()}). Please verify the quantities below."
        )
    lines.append("")

    if not quantities:
        lines.append("We could not confirm any specific item quantities in your message.")
        lines.append('Please reply with explicit counts, e.g. "200 notebooks, 50 jackets".')
        lines.append("")
        lines.append("— ReliefMatch Triage")
        return "\n".join(lines)

    lines.append("Offered quantities and current shortfall for each item you mentioned:")
    lines.append("")
    for item, qty in quantities.items():
        b = item_breakdown(item, state)
        lines.append(
            f"  - {item}: offered {qty} · current shortfall {b['gap']} "
            f"(need {b['need']}, pledged {b['pledged']}, on hand {b['stock']})"
        )
    lines.append("")

    clashes = detect_clashes(state)
    if clashes:
        lines.append("Distribution warnings detected on the current grid:")
        lines.append("")
        for c in clashes:
            rec = c.get("recommendation")
            rec_text = (
                f"Recommendation: move {rec['qty']} {c['item']} from {c['community']} "
                f"to {rec['to_community']}."
                if rec
                else "No safe reallocation is available for this item."
            )
            lines.append(
                f"  - {c['item']} @ {c['community']} is over-pledged by {c['overage']} "
                f"[{c['severity']}]. {rec_text}"
            )
        lines.append("")
    else:
        lines.append("No distribution clashes are detected; all active pledges are within need.")
        lines.append("")

    lines.append("Please confirm these quantities so we can update the allocation grid.")
    lines.append("")
    lines.append("— ReliefMatch Triage")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Domain pipeline (shared by FastAPI and the FastMCP inspector)
# --------------------------------------------------------------------------- #
def run_triage(text: str, team: Optional[str] = None) -> Dict[str, Any]:
    """Extract quantities, compute metrics, and draft the response email."""
    state = load_state()
    quantities, source, reason = extract_quantities(text)
    gaps_per_item = {item: item_breakdown(item, state) for item in quantities}
    return {
        "extraction": quantities,
        "source": source,
        "fallback_reason": reason,
        "metrics": {
            "gaps": {item: b["gap"] for item, b in gaps_per_item.items()},
            "gaps_per_item": gaps_per_item,
            "clashes": detect_clashes(state),
        },
        "drafted_email": build_triage_email(
            quantities, source, reason, team=team, state=state
        ),
    }


def _summarize_reallocation(
    pledge_id: str,
    to_community: str,
    qty: int,
    source_pledge: Dict[str, Any],
    new_state: Dict[str, Any],
) -> str:
    """Human-readable one-liner describing what the reallocation did."""
    item = source_pledge.get("item")
    if qty == source_pledge.get("qty"):
        return f"Shifted {qty} x {item} from pledge {pledge_id!r} to {to_community!r}."
    remaining = next(
        (
            p.get("qty")
            for p in new_state.get("active_pledges", [])
            if p.get("pledge_id") == pledge_id
        ),
        0,
    )
    return (
        f"Split pledge {pledge_id!r}: moved {qty} x {item} to {to_community!r}, "
        f"leaving {remaining} x {item} at the original destination."
    )


def run_reallocation(pledge_id: str, to_community: str, qty: int) -> Dict[str, Any]:
    """Apply a reallocation via core and return before/after metrics + a summary.

    Raises :class:`ValueError` for any invalid input so the HTTP layer can map it
    to a 400 response.
    """
    before = load_state()
    before_gaps = {i: item_breakdown(i, before)["gap"] for i in _all_items(before)}
    before_clashes = detect_clashes(before)

    source_pledge = next(
        (p for p in before.get("active_pledges", []) if p.get("pledge_id") == pledge_id),
        None,
    )
    if source_pledge is None:
        raise ValueError(f"pledge_id {pledge_id!r} not found in active_pledges.")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ValueError("qty must be a positive integer.")
    if qty > source_pledge.get("qty", 0):
        raise ValueError(
            f"cannot move {qty} units; pledge {pledge_id!r} only holds "
            f"{source_pledge.get('qty', 0)}."
        )
    target = str(to_community).strip()
    if not target:
        raise ValueError("to_community must be a non-empty string.")

    if target == source_pledge.get("to_community"):
        # Re-targeting a pledge to the community it already targets is a no-op.
        # Report it honestly instead of implying a move (core also guards this).
        return {
            "applied": {"pledge_id": pledge_id, "to_community": to_community, "qty": qty},
            "no_op": True,
            "state_after": {
                "warehouse_stock": before.get("warehouse_stock", {}),
                "community_needs": before.get("community_needs", {}),
                "active_pledges": before.get("active_pledges", []),
            },
            "before": {"gaps": before_gaps, "clashes": before_clashes},
            "after": {
                "gaps": before_gaps,
                "clashes": before_clashes,
                "summary": f"No change: pledge {pledge_id!r} already targets {target!r}.",
            },
        }

    new_state = apply_reallocation(pledge_id, to_community, qty)

    after_gaps = {i: item_breakdown(i, new_state)["gap"] for i in _all_items(new_state)}
    after_clashes = detect_clashes(new_state)
    return {
        "applied": {"pledge_id": pledge_id, "to_community": to_community, "qty": qty},
        "state_after": {
            "warehouse_stock": new_state.get("warehouse_stock", {}),
            "community_needs": new_state.get("community_needs", {}),
            "active_pledges": new_state.get("active_pledges", []),
        },
        "before": {"gaps": before_gaps, "clashes": before_clashes},
        "after": {
            "gaps": after_gaps,
            "clashes": after_clashes,
            "summary": _summarize_reallocation(
                pledge_id, to_community, qty, source_pledge, new_state
            ),
        },
    }


# --------------------------------------------------------------------------- #
# FastAPI request models
# --------------------------------------------------------------------------- #
class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw offer text block to triage.")
    team: Optional[str] = Field(
        default=None, description="Optional team name for the email salutation."
    )


class ReallocationRequest(BaseModel):
    pledge_id: str = Field(..., min_length=1, description="The pledge to relocate.")
    to_community: str = Field(..., min_length=1, description="Target community.")
    qty: int = Field(..., gt=0, description="Positive integer quantity to move.")


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
app = FastAPI(title="ReliefMatch Orchestration Engine", version="1.0.0")


@app.get("/", tags=["ui"])
def root() -> FileResponse:
    """Serve the hand-coded console UI directly via FileResponse."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=503, detail="static/index.html is missing.")
    return FileResponse(INDEX_HTML)


@app.get("/api/grid_state", tags=["api"])
def api_grid_state() -> Dict[str, Any]:
    """Return the full allocation grid snapshot (stock, needs, pledges, gaps, clashes)."""
    return build_grid_state()


@app.post("/api/triage", tags=["api"])
def api_triage(payload: TriageRequest) -> Dict[str, Any]:
    """Triage an offer: extract quantities, compute metrics, draft the response email."""
    return run_triage(payload.text, team=payload.team)


@app.post("/api/reallocate", tags=["api"])
def api_reallocate(payload: ReallocationRequest) -> Dict[str, Any]:
    """Apply a reallocation and return before/after metrics plus a summary."""
    try:
        return run_reallocation(payload.pledge_id, payload.to_community, payload.qty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Mount static assets last so the explicit routes above always win.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
