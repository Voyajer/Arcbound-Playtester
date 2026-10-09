"""Replay file reader for loading and parsing game replays."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from arcbound.logging.replay_codec import read_replay, list_replays


class ReplayReader:
    """Read and parse replay files."""

    @staticmethod
    def read(replay_path: Path) -> Dict[str, Any]:
        """Read a replay file and return its contents as a dictionary.

        Handles both the new lossless format (``.json.gz`` with a card table,
        see :mod:`arcbound.logging.replay_codec`) and legacy ``.json`` files
        (which may carry a trailing human-readable summary block). Card
        references are expanded back into full card dicts, so callers always
        receive the original, uncompressed document.
        """
        return read_replay(replay_path)

    @staticmethod
    def list_replays(replays_dir: Path) -> List[Path]:
        """List all replay files (``.json.gz`` and legacy ``.json``), sorted."""
        return list_replays(replays_dir)

    @staticmethod
    def get_summary(replay_path: Path) -> Dict[str, Any]:
        """Extract just the metadata summary from a replay file."""
        data = ReplayReader.read(replay_path)
        return data.get("metadata", {})

    @staticmethod
    def get_decisions(replay_path: Path) -> List[Dict[str, Any]]:
        """Extract decisions from a replay file.

        Supports both formats:
        - Legacy per-game format: a top-level ``decisions`` list.
        - Match format: decision records stored inside ``games[].moves[]``
          and tagged with ``kind="decision"``.
        """
        data = ReplayReader.read(replay_path)
        decisions = data.get("decisions")
        if decisions is not None:
            return decisions
        # Match format: collect decision records from each game's moves.
        collected: List[Dict[str, Any]] = []
        for game in data.get("games", []):
            for move in game.get("moves", []):
                if move.get("kind") == "decision":
                    collected.append(move)
        return collected

    @staticmethod
    def get_moves(replay_path: Path) -> List[Dict[str, Any]]:
        """Extract ALL moves from a replay file (event-based + decision records).

        Unlike :meth:`get_decisions`, this returns every recorded move — including
        the human player's event-based moves (spells, lands, attackers, blockers,
        mulligans, zone changes) that are not decision records. This is what the
        analysis tool uses to rate every player's play, not just the AI's.

        Supports both formats:
        - Legacy per-game format: a top-level ``decisions`` list.
        - Match format: every record in ``games[].moves[]`` (any ``kind``).
        """
        data = ReplayReader.read(replay_path)
        decisions = data.get("decisions")
        if decisions is not None:
            return decisions
        collected: List[Dict[str, Any]] = []
        for game in data.get("games", []):
            for move in game.get("moves", []):
                collected.append(move)
        return collected

    @staticmethod
    def get_decklists(replay_path: Path) -> Dict[str, List[str]]:
        """Extract decklists from a replay file."""
        data = ReplayReader.read(replay_path)
        return data.get("decklists", {})

    @staticmethod
    def get_player_types(replay_path: Path) -> Dict[str, str]:
        """Extract the player name -> role map from a replay file.

        Roles are "human", "bot", "ai", or "unknown". Returns an empty dict for
        legacy replays recorded before player types were captured.
        """
        data = ReplayReader.read(replay_path)
        match = data.get("match") or {}
        pt = match.get("player_types")
        return pt if isinstance(pt, dict) else {}

    @staticmethod
    def get_composition(replay_path: Path) -> str:
        """Extract the match composition label (e.g. "human_vs_ai").

        Falls back to deriving it from ``player_types`` when the stored label is
        missing (legacy replays), and to "unknown" when neither is present.
        """
        data = ReplayReader.read(replay_path)
        match = data.get("match") or {}
        comp = match.get("composition")
        if comp:
            return str(comp)
        pt = match.get("player_types")
        if isinstance(pt, dict) and pt:
            roles = [str(v).strip().lower() or "unknown" for v in pt.values()]
            return "_vs_".join(roles)
        return "unknown"
