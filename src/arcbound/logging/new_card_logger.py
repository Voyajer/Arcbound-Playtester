"""New-card detection — logs each card the AI encounters for the first time.

When the external AI sees a card it has never seen before (in any zone, the
stack, or the focal player's decklist), this module prints the card's name and
every piece of information the AI can see about it to the server log (the
Python terminal). Each card is logged exactly once per server session, so the
output is not repeated across games or decisions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from loguru import logger

# Card names already reported. Module-level so it persists for the life of the
# server process (a card is "new" the first time the AI ever sees it).
_seen: set[str] = set()


def reset_seen() -> None:
    """Clear the set of seen cards (e.g. for tests or a fresh session)."""
    _seen.clear()


def _iter_cards(board_state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield every card dict in a board state (all zones + stack + decklists)."""
    for player in board_state.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        for zone in (
            "hand",
            "battlefield",
            "graveyard",
            "exile",
            "command_zone",
            "decklist",
        ):
            for card in player.get(zone, []) or []:
                if isinstance(card, dict):
                    yield card
    for card in board_state.get("stack", []) or []:
        if isinstance(card, dict):
            yield card


def _format_card(card: Dict[str, Any]) -> str:
    """Render the full card info the AI can see, as a multi-line string."""
    lines: List[str] = []
    name = card.get("name", "?")
    lines.append(f"NEW CARD: {name}")

    mana = card.get("mana_cost")
    cmc = card.get("converted_mana_cost")
    if mana or cmc:
        lines.append(f"  Mana cost: {mana or '?'} (CMC {cmc if cmc is not None else '?'})")

    colors = card.get("colors") or []
    if colors:
        lines.append(f"  Colors: {', '.join(colors)}")

    identity = card.get("color_identity") or []
    if identity:
        lines.append(f"  Color identity: {', '.join(identity)}")

    types = card.get("types") or []
    supertypes = card.get("supertypes") or []
    subtypes = card.get("subtypes") or []
    type_parts = [t for t in (supertypes + types + subtypes) if t]
    if type_parts:
        lines.append(f"  Types: {', '.join(type_parts)}")

    power = card.get("power")
    toughness = card.get("toughness")
    if power is not None or toughness is not None:
        lines.append(f"  P/T: {power if power is not None else '?'} / {toughness if toughness is not None else '?'}")

    loyalty = card.get("loyalty")
    if loyalty:
        lines.append(f"  Loyalty: {loyalty}")

    keywords = card.get("keywords") or []
    if keywords:
        lines.append(f"  Keywords: {', '.join(keywords)}")

    text = card.get("text")
    if text:
        lines.append(f"  Text: {text}")

    return "\n".join(lines)


def report_new_cards(board_state: Dict[str, Any]) -> List[str]:
    """Log any cards in ``board_state`` not previously seen.

    Returns the list of newly-seen card names (empty if nothing is new).
    """
    new_cards: List[str] = []
    for card in _iter_cards(board_state):
        name = card.get("name")
        if not name or name in _seen:
            continue
        _seen.add(name)
        new_cards.append(name)
        logger.info(_format_card(card))
    return new_cards
