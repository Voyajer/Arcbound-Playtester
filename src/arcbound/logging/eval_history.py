"""Unified per-action evaluation history for the live game.

Tracks the model's value estimate after *every* player's action (the external
AI's decisions *and* the opponent's event-based moves). For each action, the
value is computed from *each* player's own perspective and stored, so every
player gets its own self-assessment series.

A value from a given player's perspective is that player's own self-assessment,
computed from its own hidden information (its hand, library, decklist) with the
opponent's hidden zones masked out. No zero-sum negation is applied, because MTG
has hidden information — the opponent's self-assessment is a different
information set and cannot be re-expressed as "the negative" of this player's
view. The opponent's self-assessment is shown in the opponent's own series
instead.

This is what lets the GUI show, for example, "how the external AI thinks the
human is doing" (the human's self-assessment, computed with the human's hand
visible and the AI's hand masked) alongside the AI's own self-assessment.
"""

import threading
from typing import Any, Dict, List, Optional


class EvalHistory:
    """In-memory store of per-action evaluation points for the current game."""

    def __init__(self):
        self._lock = threading.RLock()
        self._game_id: Optional[str] = None
        # Raw points for the current game. Each point's ``values`` maps a player
        # name to the model's value estimate from that player's own perspective
        # (that player's self-assessment, computed with its own hidden info).
        self._points: List[Dict[str, Any]] = []
        self._seq: int = 0
        # All player names seen this game, in first-seen order (AI + human + bot).
        self._players: List[str] = []
        # Ordered list of the external AI player names (may be more than one in
        # AI-vs-AI). Populated from /match/start's player_types and refreshed
        # from each decision's focal player.
        self._ai_players: List[str] = []
        # The most recent external AI to act (for the GUI's auto-switch).
        self._active_ai: Optional[str] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_player(self, name: Optional[str]):
        """Register a player name (any role) so it gets a series. Idempotent."""
        if name:
            with self._lock:
                if name not in self._players:
                    self._players.append(name)

    def set_ai_player(self, name: Optional[str]):
        """Register an external AI player name (for per-player series).

        Idempotent: calling it repeatedly with the same name is a no-op. In
        AI-vs-AI both AIs are registered, so each gets its own series. Also
        registers the name as a general player so it appears in the series map.
        """
        if name:
            with self._lock:
                if name not in self._ai_players:
                    self._ai_players.append(name)
                if name not in self._players:
                    self._players.append(name)

    @property
    def players(self) -> List[str]:
        """All player names seen this game, in first-seen order."""
        with self._lock:
            return list(self._players)

    @property
    def ai_players(self) -> List[str]:
        with self._lock:
            return list(self._ai_players)

    @property
    def ai_player(self) -> Optional[str]:
        """The first registered AI player (kept for backward compatibility)."""
        with self._lock:
            return self._ai_players[0] if self._ai_players else None

    @property
    def active_ai(self) -> Optional[str]:
        """The most recent external AI to act, or None."""
        with self._lock:
            return self._active_ai

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def add(
        self,
        game_id: Optional[str],
        turn: Optional[int],
        values: Optional[Dict[str, float]],
        actor: Optional[str],
        kind: str,
    ):
        """Append an evaluation point for ``game_id``.

        ``values`` maps a player name to the model's value estimate from that
        player's own perspective (that player's self-assessment). ``None``
        values are dropped; a point with no remaining values is ignored. When a
        new ``game_id`` is seen, the history is reset so each game has its own
        clean series.
        """
        clean = {p: v for p, v in (values or {}).items() if v is not None}
        if not clean:
            return
        with self._lock:
            # Register every player that has a value so it gets a series.
            for p in clean:
                if p not in self._players:
                    self._players.append(p)
            if self._game_id != game_id:
                self._game_id = game_id
                self._points = []
                self._seq = 0
            self._seq += 1
            self._points.append(
                {
                    "seq": self._seq,
                    "turn": turn,
                    "values": clean,
                    "player": actor,
                    "kind": kind,
                }
            )
            if actor in self._ai_players:
                self._active_ai = actor

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def _series_for(self, player: str) -> List[Dict[str, Any]]:
        """Return ``player``'s own self-assessment points for the current game.

        Only the points where ``player`` has a value are included, each kept
        as-is (from ``player``'s own perspective). No negation is applied: with
        hidden information, the opponent's self-assessment is a different
        information set and cannot be re-expressed in this player's frame.
        """
        return [
            {
                "seq": pt["seq"],
                "turn": pt["turn"],
                "value": pt["values"][player],
                "player": pt["player"],
                "kind": pt["kind"],
            }
            for pt in self._points
            if player in pt["values"]
        ]

    def get_all(self) -> Dict[str, Any]:
        """Return per-player series for the current game.

        Shape::

            {
              "players": ["ai-0", "human-0"],   # all players seen, in order
              "ai_players": ["ai-0"],           # the external AIs
              "active": "ai-0",                 # most recent AI to act (or None)
              "series": {
                "ai-0": [ {seq, turn, value, player, kind}, ... ],  # ai-0's self-assessment
                "human-0": [ ... ],                              # human's self-assessment
              },
            }

        Each series contains that player's own self-assessment (its value from
        its own perspective, with its own hidden info), at every action where it
        was computed (no negation — see module docstring).
        """
        with self._lock:
            return {
                "players": list(self._players),
                "ai_players": list(self._ai_players),
                "active": self._active_ai,
                "series": {p: self._series_for(p) for p in self._players},
            }

    def get(self, game_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the current game's evaluation points (a copy).

        Backward-compatible single-series accessor: returns the first registered
        AI player's own self-assessment series, or the first player's series
        when no AI player is known. If ``game_id`` is provided and differs from
        the current game, an empty list is returned.
        """
        with self._lock:
            if game_id is not None and self._game_id != game_id:
                return []
            if self._ai_players:
                return self._series_for(self._ai_players[0])
            if self._players:
                return self._series_for(self._players[0])
            return []

    @property
    def game_id(self) -> Optional[str]:
        with self._lock:
            return self._game_id

    def clear(self):
        with self._lock:
            self._game_id = None
            self._points = []
            self._seq = 0
            self._active_ai = None
