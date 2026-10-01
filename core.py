"""ReliefMatch deterministic core.

This module implements the pure, deterministic business logic for the
ReliefMatch relief-distribution system:

    * load_state / save_state   -- persistence for core/state.json
    * singular                  -- item-name text normalizer (plural -> canonical)
    * calculate_gap             -- per-item shortfall equation
    * detect_clashes            -- find over-pledged (community, item) cells
    * apply_reallocation        -- safely shift or split a pledge into state

Design intent: every read-only logic function accepts an optional ``state``
dict (defaulting to a freshly loaded copy) and returns new data without
touching the on-disk seed. Only :func:`apply_reallocation` writes to disk, and
it always writes through :func:`save_state`, so the seed file is never
corrupted in place.
"""

from __future__ import annotations

import json
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_STATE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(_STATE_DIR, "state.json")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_state(path: str = STATE_PATH) -> dict:
    """Load and return the JSON state as a dict.

    Args:
        path: Optional override for the state file location.

    Returns:
        The parsed state dict.

    Raises:
        FileNotFoundError: If the state file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict, path: str = STATE_PATH) -> None:
    """Persist ``state`` to ``path`` as pretty-printed JSON.

    A trailing newline is written for friendlier diffs.

    Args:
        state: The state dict to write.
        path: Optional override for the state file location.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# Item-name normalization
# --------------------------------------------------------------------------- #
# Whole-phrase canonical forms. Multi-word items are matched first so that
# "hygiene kits" resolves to "hygiene kit" instead of a bare word-level hit.
_PHRASE_MAP = {
    "hygiene kits": "hygiene kit",
    "hygiene kit": "hygiene kit",
    "pencil boxes": "pencil box",
    "pencil box": "pencil box",
    "jackets": "jacket",
    "jacket": "jacket",
    "blankets": "blanket",
    "blanket": "blanket",
    "notebooks": "notebook",
    "notebook": "notebook",
}

# Word-level plural -> singular fallback, applied to the final token only.
_WORD_MAP = {
    "jackets": "jacket",
    "jacket": "jacket",
    "blankets": "blanket",
    "blanket": "blanket",
    "notebooks": "notebook",
    "notebook": "notebook",
    "boxes": "box",
    "box": "box",
    "kits": "kit",
    "kit": "kit",
}


def singular(item_name: str) -> str:
    """Normalize an item name to its canonical singular form.

    Whitespace is collapsed and the value is lower-cased. Whole-phrase forms
    are matched first (so multi-word items are handled atomically), then a
    word-level transform is applied to the final token, with a generic
    trailing-'s' strip as the last resort.

    Examples:
        >>> singular("boxes")
        'box'
        >>> singular("kits")
        'kit'
        >>> singular("jackets")
        'jacket'
        >>> singular("hygiene kits")
        'hygiene kit'
        >>> singular("pencil boxes")
        'pencil box'

    Args:
        item_name: The raw item label.

    Returns:
        The canonical singular item key.

    Raises:
        ValueError: If ``item_name`` is not a string or is empty.
    """
    if not isinstance(item_name, str):
        raise ValueError("item_name must be a string")
    text = " ".join(item_name.split()).lower()
    if not text:
        raise ValueError("item_name must be a non-empty string")

    if text in _PHRASE_MAP:
        return _PHRASE_MAP[text]

    tokens = text.split(" ")
    last = tokens[-1]
    if last in _WORD_MAP:
        tokens[-1] = _WORD_MAP[last]
    elif last.endswith("s") and not last.endswith("ss"):
        tokens[-1] = last[:-1]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# Shortfall math
# --------------------------------------------------------------------------- #
def calculate_gap(item: str, state: dict = None) -> int:
    """Compute the shortfall for a single item.

    Equation (strict, per spec)::

        Gap = max(Need - Pledges - Stock, 0)

    where:
        * ``Need``    = total community need across *all* communities for
          ``item``.
        * ``Pledges`` = total active pledge quantity for ``item`` across all
          communities.
        * ``Stock``   = warehouse stock on hand for ``item``.

    Args:
        item: Item label (normalized internally via :func:`singular`).
        state: Optional state dict; loaded from disk when ``None``.

    Returns:
        The non-negative gap integer.
    """
    if state is None:
        state = load_state()
    key = singular(item)

    need = sum(
        needs.get(key, 0)
        for needs in state.get("community_needs", {}).values()
    )
    pledges = sum(
        pledge.get("qty", 0)
        for pledge in state.get("active_pledges", [])
        if singular(pledge.get("item", "")) == key
    )
    stock = state.get("warehouse_stock", {}).get(key, 0)

    return max(need - pledges - stock, 0)


# --------------------------------------------------------------------------- #
# Clash detection
# --------------------------------------------------------------------------- #
def _pledge_sum_by(state: dict, community: str, item_key: str) -> int:
    """Total active pledge quantity targeting ``community`` for ``item_key``."""
    return sum(
        pledge.get("qty", 0)
        for pledge in state.get("active_pledges", [])
        if pledge.get("to_community") == community
        and singular(pledge.get("item", "")) == item_key
    )


def _recommend_relocation(
    state: dict, over_pledging_community: str, item_key: str, overage: int
) -> dict:
    """Build a safe recommendation payload for an over-pledged cell.

    The payload has the shape ``{"pledge_id", "to_community", "qty"}`` and is
    directly consumable by :func:`apply_reallocation`.

    Selection is fully deterministic:
        * Alternative community = largest remaining capacity for the item
          (tie-break on community name).
        * Source pledge         = largest pledge in the over-pledged group
          (tie-break on ``pledge_id``).
        * Qty                   = min(overage, source qty, alt capacity).

    Returns ``None`` when there is no viable alternative community.
    """
    communities = state.get("community_needs", {})

    candidates = []
    for other, needs in communities.items():
        if other == over_pledging_community:
            continue
        their_need = needs.get(item_key, 0)
        their_pledged = _pledge_sum_by(state, other, item_key)
        capacity = their_need - their_pledged
        if capacity > 0:
            candidates.append((other, capacity))

    if not candidates:
        return None

    alt_community, alt_capacity = sorted(candidates, key=lambda t: (-t[1], t[0]))[0]

    source_pledges = [
        pledge
        for pledge in state.get("active_pledges", [])
        if pledge.get("to_community") == over_pledging_community
        and singular(pledge.get("item", "")) == item_key
    ]
    if not source_pledges:
        return None

    source = sorted(
        source_pledges, key=lambda p: (-p.get("qty", 0), p.get("pledge_id", ""))
    )[0]

    qty = min(overage, source.get("qty", 0), alt_capacity)
    if qty <= 0:
        return None

    return {
        "pledge_id": source.get("pledge_id"),
        "to_community": alt_community,
        "qty": qty,
    }


def detect_clashes(state: dict = None) -> list:
    """Detect over-pledged ``(community, item)`` cells and suggest a fix.

    A clash exists when the sum of active pledges for a resource in a community
    exceeds that community's need for it. Severity is ``"HIGH"`` when the
    overage is greater than 30, otherwise ``"MEDIUM"``. Each clash carries a
    concrete recommendation payload (``{"pledge_id", "to_community", "qty"}``)
    that would relieve the excess by redirecting quantity to an alternative
    community that still needs the same resource.

    Args:
        state: Optional state dict; loaded from disk when ``None``.

    Returns:
        A list of clash dicts (possibly empty), each with keys ``item``,
        ``community``, ``need``, ``pledged``, ``overage``, ``severity`` and
        ``recommendation``.
    """
    if state is None:
        state = load_state()

    communities = state.get("community_needs", {})
    clashes = []

    for community, needs in communities.items():
        for item, need in needs.items():
            item_key = singular(item)
            pledged = _pledge_sum_by(state, community, item_key)
            if pledged <= need:
                continue

            overage = pledged - need
            severity = "HIGH" if overage > 30 else "MEDIUM"
            recommendation = _recommend_relocation(
                state, community, item_key, overage
            )
            clashes.append(
                {
                    "item": item_key,
                    "community": community,
                    "need": need,
                    "pledged": pledged,
                    "overage": overage,
                    "severity": severity,
                    "recommendation": recommendation,
                }
            )
    return clashes


# --------------------------------------------------------------------------- #
# Reallocation
# --------------------------------------------------------------------------- #
def _unique_pledge_id(pledges: list, base: str) -> str:
    """Return a collision-free pledge id derived from ``base``."""
    taken = {pledge.get("pledge_id") for pledge in pledges}
    candidate = base
    counter = 2
    while candidate in taken:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def apply_reallocation(
    pledge_id: str, to_community: str, qty: int, path: str = STATE_PATH
) -> dict:
    """Safely shift or split an existing pledge to a new community.

    * If ``qty == pledge.qty`` the pledge is **shifted** in place (its
      ``to_community`` is updated).
    * Otherwise the pledge is **split**: the original is reduced by ``qty``
      and a new, uniquely-id'd pledge carrying ``qty`` is appended targeting
      ``to_community``.

    Records are never corrupted: the original pledge is preserved (reduced or
    re-pointed) and any split portion receives a fresh, collision-free id. The
    updated state is written back to ``path`` and returned.

    Args:
        pledge_id: The id of the pledge to relocate.
        to_community: Target community name.
        qty: Positive integer quantity to move.
        path: State file to read/write (defaults to the seed).

    Returns:
        The updated state dict.

    Raises:
        ValueError: For an unknown pledge id, a non-positive or excessive qty,
            or an empty target community.
    """
    state = load_state(path)
    pledges = state.get("active_pledges", [])

    source = next(
        (pledge for pledge in pledges if pledge.get("pledge_id") == pledge_id), None
    )
    if source is None:
        raise ValueError(f"pledge_id {pledge_id!r} not found in active_pledges")

    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ValueError("qty must be a positive integer")

    source_qty = source.get("qty", 0)
    if qty > source_qty:
        raise ValueError(
            f"cannot move {qty} units; pledge {pledge_id!r} only holds "
            f"{source_qty}"
        )

    target = str(to_community).strip()
    if not target:
        raise ValueError("to_community must be a non-empty string")

    if target == source.get("to_community"):
        # Re-targeting to the same community is a safe no-op.
        return state

    if qty == source_qty:
        # Full shift: re-point the existing pledge.
        source["to_community"] = target
    else:
        # Split: carve ``qty`` off into a new pledge, keep the remainder.
        source["qty"] = source_qty - qty
        new_id = _unique_pledge_id(pledges, f"{pledge_id}_relocated")
        state["active_pledges"].append(
            {
                "pledge_id": new_id,
                "team": source.get("team", ""),
                "item": source.get("item"),
                "qty": qty,
                "to_community": target,
            }
        )

    save_state(state, path)
    return state


# --------------------------------------------------------------------------- #
# Self-validation (runs only when executed directly, never on import)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import shutil
    import tempfile

    state = load_state()

    # 1) Normalization spot-checks (the exact examples from the spec).
    assert singular("boxes") == "box"
    assert singular("kits") == "kit"
    assert singular("jackets") == "jacket"
    assert singular("Hygiene Kits") == "hygiene kit"
    assert singular("Pencil Boxes") == "pencil box"

    # 2) Gap report for every item that appears anywhere in the state.
    all_items = set(state.get("warehouse_stock", {}).keys())
    for needs in state.get("community_needs", {}).values():
        all_items.update(needs.keys())
    for pledge in state.get("active_pledges", []):
        all_items.add(singular(pledge.get("item", "")))

    print("=== Gap report  (Gap = max(Need - Pledges - Stock, 0)) ===")
    for item in sorted(all_items):
        print(f"  {item:<14} -> {calculate_gap(item, state)}")

    # 3) Clash detection.
    print("\n=== Clash detection ===")
    clashes = detect_clashes(state)
    if not clashes:
        print("  no clashes detected")
    for clash in clashes:
        print(
            f"  [{clash['severity']}] {clash['item']} @ {clash['community']}: "
            f"pledged={clash['pledged']} need={clash['need']} "
            f"overage={clash['overage']}"
        )
        rec = clash.get("recommendation")
        if rec:
            print(
                f"      -> move pledge {rec['pledge_id']} "
                f"({rec['qty']} x {clash['item']}) to {rec['to_community']}"
            )

    # 4) Simulate a reallocation on a THROWAWAY copy so the seed stays intact.
    actionable = next((c for c in clashes if c.get("recommendation")), None)
    if actionable:
        rec = actionable["recommendation"]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = os.path.join(tmp, "state.json")
            shutil.copyfile(STATE_PATH, tmp_path)
            new_state = apply_reallocation(
                rec["pledge_id"], rec["to_community"], rec["qty"], path=tmp_path
            )

        print("\n=== Reallocation simulation (on a temp copy) ===")
        print(f"  applied: {rec}")

        # Invariant A: total pledged units of the item are conserved.
        total_before = sum(
            p["qty"]
            for p in state["active_pledges"]
            if singular(p["item"]) == actionable["item"]
        )
        total_after = sum(
            p["qty"]
            for p in new_state["active_pledges"]
            if singular(p["item"]) == actionable["item"]
        )
        assert total_before == total_after, "pledge quantity not conserved!"
        print(f"  conserved {total_before} total units of {actionable['item']!r}")

        # Invariant B: the source cell's overage is relieved.
        clashes_after = detect_clashes(new_state)
        remaining = next(
            (
                c["overage"]
                for c in clashes_after
                if c["community"] == actionable["community"]
                and c["item"] == actionable["item"]
            ),
            0,
        )
        print(
            f"  remaining overage @ {actionable['community']}/"
            f"{actionable['item']!r}: {remaining} (was {actionable['overage']})"
        )
        assert remaining < actionable["overage"], "clash not relieved"

    # 5) Confirm the seed file was never mutated.
    reloaded = load_state()
    assert reloaded == state, "seed state.json was modified by validation!"

    print("\nVALIDATION PASSED: core.py executed with zero crashes.")




