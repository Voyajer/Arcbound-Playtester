"""Card vocabulary for identity embeddings.

Builds a deterministic card-name -> index mapping from replay files so the
model can learn a distinct embedding per card. Index 0 is reserved for
``<UNK>`` (cards not in the vocabulary, e.g. cards that appear only after the
vocabulary was built).

The vocabulary is built from replays (the "replay vocab" style): every card
name found in any board state or decklist gets an entry. Indices are assigned
in sorted order so the mapping is stable across training runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from arcbound.logging.replay_codec import list_replays, read_replay

UNK_INDEX = 0
UNK_NAME = "<UNK>"


class CardVocabulary:
    """Maps card names to integer indices for the card identity embedding."""

    def __init__(self, names: Optional[List[str]] = None):
        # names[0] must be the UNK sentinel; the rest are card names.
        self.names: List[str] = [UNK_NAME] + list(names or [])
        self._index: Dict[str, int] = {name: i for i, name in enumerate(self.names)}

    # ------------------------------------------------------------------
    # Lookup / mutation
    # ------------------------------------------------------------------
    def index_of(self, name: Optional[str]) -> int:
        """Return the index for a card name, or UNK_INDEX if unknown."""
        if not name:
            return UNK_INDEX
        return self._index.get(name, UNK_INDEX)

    def add(self, name: Optional[str]) -> None:
        """Add a card name if not already present (appends, stable indices)."""
        if not name or name in self._index:
            return
        self._index[name] = len(self.names)
        self.names.append(name)

    def add_many(self, names: Iterable[str]) -> None:
        for name in names:
            self.add(name)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"unk": UNK_INDEX, "names": self.names}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CardVocabulary":
        with open(path) as f:
            data = json.load(f)
        names = data.get("names", [])
        # The saved list starts with the UNK sentinel; strip it (the
        # constructor re-prepends it).
        if names and names[0] == UNK_NAME:
            names = names[1:]
        return cls(names)

    # ------------------------------------------------------------------
    # Building from replays
    # ------------------------------------------------------------------
    @classmethod
    def build_from_replays(cls, replay_dir: Path) -> "CardVocabulary":
        """Scan all replay JSON files and collect every card name.

        Collects names from board states (all zones + stack) and from
        decklists. Indices are assigned in sorted order for determinism.
        """
        names: Set[str] = set()
        for path in list_replays(replay_dir):
            try:
                data = read_replay(path)
            except Exception:
                continue
            _collect_card_names(data, names)
        vocab = cls(sorted(names))
        return vocab


def _collect_card_names(data: dict, names: Set[str]) -> None:
    """Recursively collect card names from a replay dict."""
    # Decklists: {player: [card names]}
    decklists = data.get("decklists")
    if isinstance(decklists, dict):
        for cards in decklists.values():
            if isinstance(cards, list):
                for c in cards:
                    if isinstance(c, str) and c:
                        names.add(c)
    for game in data.get("games", []) or []:
        if not isinstance(game, dict):
            continue
        _collect_card_names(game, names)
        for move in game.get("moves", []) or []:
            _collect_board_state_names(move.get("board_state"), names)
    # Legacy per-game format
    for dec in data.get("decisions", []) or []:
        _collect_board_state_names(dec.get("board_state"), names)


def _collect_board_state_names(board_state: Optional[dict], names: Set[str]) -> None:
    if not isinstance(board_state, dict):
        return
    for player in board_state.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        for zone in ("hand", "battlefield", "graveyard", "exile", "command_zone"):
            for card in player.get(zone, []) or []:
                _add_card_name(card, names)
    for card in board_state.get("stack", []) or []:
        _add_card_name(card, names)


def _add_card_name(card: object, names: Set[str]) -> None:
    if isinstance(card, dict):
        name = card.get("name")
        if isinstance(name, str) and name:
            names.add(name)


# ---------------------------------------------------------------------------
# Keyword vocabulary
# ---------------------------------------------------------------------------

KW_UNK_NAME = "<KW_UNK>"


class KeywordVocabulary:
    """Maps MTG keyword strings to integer indices for the keyword embedding.

    Mirrors :class:`CardVocabulary`: index 0 is reserved for ``<KW_UNK>``
    (keywords not in the vocabulary). Keywords are the small closed set of
    ability keywords (Flying, Trample, Haste, ...); each card token carries the
    index of its *first* keyword (0 when the card has none), which the model
    looks up in a learned embedding.
    """

    def __init__(self, names: Optional[List[str]] = None):
        self.names: List[str] = [KW_UNK_NAME] + list(names or [])
        self._index: Dict[str, int] = {name: i for i, name in enumerate(self.names)}

    def index_of(self, name: Optional[str]) -> int:
        if not name:
            return UNK_INDEX
        return self._index.get(name, UNK_INDEX)

    def add(self, name: Optional[str]) -> None:
        if not name or name in self._index:
            return
        self._index[name] = len(self.names)
        self.names.append(name)

    def add_many(self, names: Iterable[str]) -> None:
        for name in names:
            self.add(name)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"unk": UNK_INDEX, "names": self.names}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "KeywordVocabulary":
        with open(path) as f:
            data = json.load(f)
        names = data.get("names", [])
        if names and names[0] == KW_UNK_NAME:
            names = names[1:]
        return cls(names)

    @classmethod
    def build_from_replays(cls, replay_dir: Path) -> "KeywordVocabulary":
        """Scan all replay JSON files and collect every keyword string."""
        names: Set[str] = set()
        for path in list_replays(replay_dir):
            try:
                data = read_replay(path)
            except Exception:
                continue
            _collect_keywords(data, names)
        return cls(sorted(names))


def _collect_keywords(data: dict, names: Set[str]) -> None:
    """Recursively collect keyword strings from a replay dict."""
    for game in data.get("games", []) or []:
        if not isinstance(game, dict):
            continue
        _collect_keywords(game, names)
        for move in game.get("moves", []) or []:
            _collect_board_state_keywords(move.get("board_state"), names)
    for dec in data.get("decisions", []) or []:
        _collect_board_state_keywords(dec.get("board_state"), names)


def _collect_board_state_keywords(board_state: Optional[dict], names: Set[str]) -> None:
    if not isinstance(board_state, dict):
        return
    for player in board_state.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        for zone in ("hand", "battlefield", "graveyard", "exile", "command_zone", "decklist"):
            for card in player.get(zone, []) or []:
                _add_card_keywords(card, names)
        for card in player.get("commanders", []) or []:
            _add_card_keywords(card, names)
    for card in board_state.get("stack", []) or []:
        _add_card_keywords(card, names)


def _add_card_keywords(card: object, names: Set[str]) -> None:
    if isinstance(card, dict):
        for kw in card.get("keywords", []) or []:
            if isinstance(kw, str) and kw:
                names.add(kw)
