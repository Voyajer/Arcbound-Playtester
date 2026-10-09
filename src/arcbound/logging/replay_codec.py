"""Lossless replay file codec.

Replay files are stored as gzip-compressed JSON (``.json.gz``) with a
card-table deduplication scheme that shrinks them dramatically without losing
any information.

Why the file is so big without this
-----------------------------------
Every move stores a full board state, and every card in every zone is
serialized in full (name, text, mana cost, types, keywords, ...). The same card
appears in dozens of consecutive moves with only a few fields changed (tapped,
damage, zone, ...). A single AI-vs-AI match can therefore be hundreds of MB.

How the codec works (lossless)
------------------------------
* A top-level ``cards`` table holds one entry per *distinct stable card
  content*. "Stable" = every field except the volatile per-instance state
  (``id``, ``zone``, ``owner``, ``controller``, ``power``, ``toughness``,
  ``damage``, ``counters``, ``loyalty``, ``tapped``, ``attacking``,
  ``blocking``, ``blocked``, ``flipped``, ``face_down``, ``revealed_to``,
  ``count``). Two references with identical stable content share one table
  entry.
* Each card occurrence in a board state is replaced by a small reference
  object: ``{"c": <table index>, ...volatile fields...}``.
* The reader merges the table entry with the volatile fields to reconstruct
  the exact original card dict.

The table is keyed on the *full stable content* (not on the card name or id)
so it stays lossless even when Forge reuses instance ids across different
cards or emits the same card with different cosmetic text highlighting.

Files are written as ``<name>.json.gz`` (gzip level 9). A 287 MB match file
becomes ~1 MB. Legacy ``.json`` files (no codec marker) are still readable.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Marker written into the encoded document so the reader knows the card table
# is present and must be expanded.
CODEC_VERSION = 1

# Per-instance card state that changes between moves. Everything else on a card
# is treated as stable identity and deduplicated into the card table.
VOLATILE = frozenset({
    "id", "zone", "owner", "controller",
    "power", "toughness", "damage", "counters", "loyalty",
    "tapped", "attacking", "blocking", "blocked", "flipped", "face_down",
    "revealed_to", "count",
})

# Player zones that hold card lists, plus the board-level stack.
CARD_ZONES = (
    "battlefield", "hand", "graveyard", "exile", "command_zone",
    "commanders", "decklist",
)


def _iter_card_slots(board_state: Dict[str, Any]) -> Iterator[Tuple[List, int]]:
    """Yield ``(container_list, index)`` for every card slot in a board state."""
    for pl in board_state.get("players", []) or []:
        if not isinstance(pl, dict):
            continue
        for zone in CARD_ZONES:
            lst = pl.get(zone)
            if isinstance(lst, list):
                for i, _c in enumerate(lst):
                    yield lst, i
    stack = board_state.get("stack")
    if isinstance(stack, list):
        for i, _c in enumerate(stack):
            yield stack, i


def _iter_board_states(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every board-state dict in a replay document (both formats)."""
    for game in data.get("games", []) or []:
        if not isinstance(game, dict):
            continue
        for move in game.get("moves", []) or []:
            bs = move.get("board_state")
            if isinstance(bs, dict):
                yield bs
    for dec in data.get("decisions", []) or []:
        bs = dec.get("board_state")
        if isinstance(bs, dict):
            yield bs


def _encode_card_list(lst: List, table: List[Dict[str, Any]], index: Dict[str, int]) -> List:
    """Return a new list with each card replaced by a deduplicated reference.

    The input list is not mutated. Card *content* (strings, etc.) is shared by
    reference with the table, so this is cheap; only the list and the small
    reference dicts are newly allocated.
    """
    out: List = []
    for c in lst:
        if not (isinstance(c, dict) and "name" in c):
            out.append(c)
            continue
        stable = {k: v for k, v in c.items() if k not in VOLATILE}
        key = json.dumps(stable, sort_keys=True, ensure_ascii=False)
        idx = index.get(key)
        if idx is None:
            idx = len(table)
            table.append(stable)
            index[key] = idx
        ref: Dict[str, Any] = {"c": idx}
        for k in VOLATILE:
            if k in c:
                ref[k] = c[k]
        out.append(ref)
    return out


def _copy_board_state(bs: Dict[str, Any], table: List[Dict[str, Any]], index: Dict[str, int]) -> Dict[str, Any]:
    """Return a copy of a board state with its cards deduplicated into refs.

    Only the container dicts/lists are copied; card content is shared by
    reference, so the input board state is left untouched.
    """
    bs_c = dict(bs)
    players_c: List = []
    for pl in bs.get("players", []) or []:
        if not isinstance(pl, dict):
            players_c.append(pl)
            continue
        pl_c = dict(pl)
        for zone in CARD_ZONES:
            lst = pl.get(zone)
            if isinstance(lst, list):
                pl_c[zone] = _encode_card_list(lst, table, index)
        players_c.append(pl_c)
    bs_c["players"] = players_c
    stack = bs.get("stack")
    if isinstance(stack, list):
        bs_c["stack"] = _encode_card_list(stack, table, index)
    return bs_c


def encode(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with card references deduplicated into a table.

    The input is not mutated. The result carries a top-level ``codec`` marker
    and a ``cards`` table; every card in every board state is replaced by a
    small reference object.
    """
    if "codec" in data:
        # Already encoded — return a shallow copy to avoid surprising callers.
        return dict(data)

    table: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}

    out: Dict[str, Any] = {"codec": CODEC_VERSION, "cards": table}
    for k, v in data.items():
        out[k] = v

    # Rebuild the "games" list with deduplicated board states (only if present,
    # so the output structure matches the input exactly).
    if "games" in data:
        games_out: List = []
        for game in data["games"] or []:
            moves_out: List = []
            for move in game.get("moves", []) or []:
                move_c = dict(move)
                bs = move.get("board_state")
                if isinstance(bs, dict):
                    move_c["board_state"] = _copy_board_state(bs, table, index)
                moves_out.append(move_c)
            game_c = dict(game)
            game_c["moves"] = moves_out
            games_out.append(game_c)
        out["games"] = games_out

    # Same for the legacy per-game "decisions" list.
    if "decisions" in data:
        decisions_out: List = []
        for dec in data["decisions"] or []:
            dec_c = dict(dec)
            bs = dec.get("board_state")
            if isinstance(bs, dict):
                dec_c["board_state"] = _copy_board_state(bs, table, index)
            decisions_out.append(dec_c)
        out["decisions"] = decisions_out

    return out


def decode(data: Dict[str, Any]) -> Dict[str, Any]:
    """Expand card references back into full card dicts (inverse of encode).

    Documents without the ``codec`` marker (legacy files) are returned as-is.
    """
    if "codec" not in data:
        return data

    table = data.get("cards", [])
    for bs in _iter_board_states(data):
        for lst, i in _iter_card_slots(bs):
            ref = lst[i]
            if not isinstance(ref, dict) or "c" not in ref:
                continue
            card = dict(table[ref["c"]])
            for k, v in ref.items():
                if k != "c":
                    card[k] = v
            lst[i] = card

    data.pop("codec", None)
    data.pop("cards", None)
    return data


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_replay(path: Path, data: Dict[str, Any], summary: Optional[str] = None) -> Path:
    """Encode ``data`` and write it to ``path`` as gzip-compressed JSON.

    ``path`` should end in ``.json.gz``; if it ends in ``.json`` (or has no
    extension) the ``.gz`` suffix is appended so the on-disk name is
    self-describing. The human-readable ``summary`` (when provided) is embedded
    as a top-level ``summary`` field so it survives compression.

    Returns the path actually written.
    """
    if summary is not None:
        data = dict(data)
        data["summary"] = summary

    encoded = encode(data)
    payload = json.dumps(encoded, separators=(",", ":"), ensure_ascii=False)

    target = Path(path)
    if not target.name.endswith(".gz"):
        target = target.with_name(target.name + ".gz")
    target.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(target, "wb") as f:
        f.write(payload.encode("utf-8"))
    return target


def read_replay(path: Path) -> Dict[str, Any]:
    """Read a replay file (new ``.json.gz`` or legacy ``.json``) and decode it.

    Returns the fully-expanded document (card references resolved). The
    embedded ``summary`` field, if present, is included in the returned dict.
    """
    path = Path(path)
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        data = json.loads(raw.decode("utf-8"))
    else:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Legacy files may have a trailing human-readable summary block after
        # the JSON document. Parse the first JSON value and ignore the rest.
        stripped = content.lstrip()
        try:
            data, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            for marker in ("\n\n=== GAME SUMMARY ===", "\n\n=== MATCH SUMMARY ==="):
                idx = content.find(marker)
                if idx != -1:
                    content = content[:idx]
                    break
            data = json.loads(content)
    return decode(data)


def list_replays(replays_dir: Path) -> List[Path]:
    """List all replay files (new ``.json.gz`` and legacy ``.json``), sorted.

    Recurses into subdirectories so a folder selected in the GUI (or passed to
    the trainer) picks up replays stored in nested folders (e.g. per-match or
    per-opponent subfolders) as well as the top level.
    """
    replays_dir = Path(replays_dir)
    if not replays_dir.exists():
        return []
    files = set(replays_dir.rglob("*.json.gz")) | set(replays_dir.rglob("*.json"))
    return sorted(files)
