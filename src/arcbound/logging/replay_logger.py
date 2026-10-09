"""Game replay logging system.

Creates and manages replay files for completed games.
Each game produces a JSON file with metadata, decklists, and decision history.
"""

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from arcbound.logging.replay_codec import write_replay

# A game counts as "active" for the GUI status label if a decision was logged
# within this window. The Java client never calls /game/end, so a replay log
# stays open indefinitely after a game finishes; recency is what tells the GUI
# whether a game is actually in progress right now.
ACTIVE_WINDOW = timedelta(seconds=30)


class ReplayLogger:
    """Logs game decisions to replay files."""

    def __init__(self, replays_dir: Path, write_files: bool = True):
        self.replays_dir = replays_dir
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: Optional[Dict[str, Any]] = None
        self._decisions: List[Dict[str, Any]] = []
        self._decklists: Dict[str, List[str]] = {}
        self._active: bool = False
        self._game_id: Optional[str] = None
        self._filename: Optional[str] = None
        # When False, decisions are still tracked in memory (for the GUI's live
        # monitoring endpoints) but no per-game replay file is written to disk.
        # The match file (MatchLogger) is the single source of truth for
        # training, so the per-game file is redundant and can be disabled.
        self._write_files: bool = write_files
        # Monotonic per-game sequence number assigned to each logged decision.
        # Lets the GUI poll incrementally (only decisions with seq > last seen)
        # instead of offsetting into a capped "last N" list.
        self._seq: int = 0
        # Wall-clock time of the most recent logged decision (None if none yet).
        self._last_decision_time: Optional[datetime] = None

    def start_game(
        self,
        game_id: str,
        players: List[str],
        focal_player: str,
        format_name: str = "standard",
    ) -> str:
        """Start a new game log. Returns the replay filename."""
        now = datetime.now()
        safe_players = [p.replace(" ", "-").lower() for p in players]
        # Short UUID suffix guarantees uniqueness even when two games start
        # in the same second (e.g. rapid rollover on gameId change).
        filename = (
            f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'_vs_'.join(safe_players)}_"
            f"{format_name}_{uuid.uuid4().hex[:8]}.json.gz"
        )
        self._filename = filename

        self._metadata = {
            "timestamp": now.isoformat(),
            "game_id": game_id,
            "format": format_name,
            "players": players,
            "focal_player": focal_player,
            "result": None,
            "duration_seconds": 0,
            "total_turns": 0,
            "start_time": now.isoformat(),
        }
        self._decisions = []
        self._decklists = {}
        self._active = True
        self._game_id = game_id
        self._seq = 0
        return filename

    def set_decklist(self, player: str, cards: List[str]):
        """Record a player's decklist."""
        self._decklists[player] = cards

    def log_decision(self, decision: Dict[str, Any]):
        """Append a decision to the current game log."""
        if self._active:
            self._seq += 1
            decision["seq"] = self._seq
            self._decisions.append(decision)
            self._last_decision_time = datetime.now()

    def get_decisions_since(self, last_seq: int = 0) -> List[Dict[str, Any]]:
        """Return decisions with seq > last_seq (for incremental GUI polling)."""
        return [d for d in self._decisions if d.get("seq", 0) > last_seq]

    def end_game(self, result: str, total_turns: int = 0) -> Optional[Path]:
        """Finalize the game log and write to disk. Returns path to replay file.

        When file writing is disabled (``write_files=False``), the in-memory
        state is still finalized (so the GUI stops polling) but no file is
        written and ``None`` is returned.
        """
        if not self._metadata:
            raise RuntimeError("No active game. Call start_game() first.")

        self._metadata["result"] = result
        self._metadata["total_turns"] = total_turns
        if "start_time" in self._metadata:
            start = datetime.fromisoformat(self._metadata["start_time"])
            self._metadata["duration_seconds"] = (datetime.now() - start).total_seconds()

        # Finalize state regardless of whether we persist to disk.
        self._active = False

        if not self._write_files:
            return None

        replay = {
            "metadata": self._metadata,
            "decklists": self._decklists,
            "decisions": self._decisions,
        }

        # Reuse the filename chosen at start_game() so the file is unique and
        # consistent (regenerating it here could collide or mismatch).
        if not self._filename:
            now = datetime.now()
            safe_players = [p.replace(" ", "-").lower() for p in self._metadata["players"]]
            self._filename = (
                f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'_vs_'.join(safe_players)}_"
                f"{self._metadata['format']}.json.gz"
            )

        # Build the human-readable summary (embedded so it survives gzip).
        duration = self._metadata.get("duration_seconds", 0)
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        summary = "\n".join([
            "=== GAME SUMMARY ===",
            f"Date: {self._metadata['timestamp']}",
            f"Format: {self._metadata['format'].capitalize()}",
            f"Players: {' vs '.join(self._metadata['players'])}",
            f"Result: {result}",
            f"Duration: {hours}h {minutes}m {seconds}s",
            f"Total Turns: {total_turns}",
            f"Total Decisions: {len(self._decisions)}",
        ])

        # Write the lossless, gzip-compressed replay file.
        filepath = write_replay(self.replays_dir / self._filename, replay, summary=summary)
        return filepath

    def get_decisions(self) -> List[Dict[str, Any]]:
        return list(self._decisions)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_recently_active(self) -> bool:
        """True if a decision was logged within the last ACTIVE_WINDOW.

        Unlike :attr:`is_active` (which is True for the whole life of an open
        replay log), this reflects whether a game is actually in progress right
        now. The GUI uses it for its "Game active" status label.
        """
        if not self._active or self._last_decision_time is None:
            return False
        return datetime.now() - self._last_decision_time <= ACTIVE_WINDOW

    @property
    def current_game_id(self) -> Optional[str]:
        """The game_id of the in-progress replay, or None if no game is active."""
        return self._game_id if self._active else None

    @property
    def last_seq(self) -> int:
        """Highest sequence number assigned so far (for GUI startup sync)."""
        return self._seq

    @property
    def game_id(self) -> Optional[str]:
        """The game_id of the current (or most recent) replay log.

        Unlike :attr:`current_game_id` this is not gated on ``is_active`` so the
        GUI can tell which game the decisions it is about to poll belong to, even
        in the brief window between one game ending and the next starting.
        """
        return self._game_id

    def finalize_current_game(self, result: str = "abandoned") -> Optional[Path]:
        """Finalize the in-progress replay (e.g. on server shutdown).

        Returns the replay file path, or None if no game was active.
        """
        if not self._active or self._metadata is None:
            return None
        try:
            return self.end_game(result=result, total_turns=0)
        except Exception:
            return None
