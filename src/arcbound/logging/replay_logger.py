"""Game replay logging system.

Creates and manages replay files for completed games.
Each game produces a JSON file with metadata, decklists, and decision history.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReplayLogger:
    """Logs game decisions to replay files."""

    def __init__(self, replays_dir: Path):
        self.replays_dir = replays_dir
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: Optional[Dict[str, Any]] = None
        self._decisions: List[Dict[str, Any]] = []
        self._decklists: Dict[str, List[str]] = {}
        self._active: bool = False

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
        filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'_vs_'.join(safe_players)}_{format_name}.json"

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
        return filename

    def set_decklist(self, player: str, cards: List[str]):
        """Record a player's decklist."""
        self._decklists[player] = cards

    def log_decision(self, decision: Dict[str, Any]):
        """Append a decision to the current game log."""
        if self._active:
            self._decisions.append(decision)

    def end_game(self, result: str, total_turns: int = 0) -> Path:
        """Finalize the game log and write to disk. Returns path to replay file."""
        if not self._metadata:
            raise RuntimeError("No active game. Call start_game() first.")

        self._metadata["result"] = result
        self._metadata["total_turns"] = total_turns
        if "start_time" in self._metadata:
            start = datetime.fromisoformat(self._metadata["start_time"])
            self._metadata["duration_seconds"] = (datetime.now() - start).total_seconds()

        replay = {
            "metadata": self._metadata,
            "decklists": self._decklists,
            "decisions": self._decisions,
        }

        # Generate filename from metadata
        now = datetime.now()
        safe_players = [p.replace(" ", "-").lower() for p in self._metadata["players"]]
        filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'_vs_'.join(safe_players)}_{self._metadata['format']}.json"
        filepath = self.replays_dir / filename

        # Write JSON
        with open(filepath, "w") as f:
            json.dump(replay, f, indent=2)

        # Append human-readable summary
        with open(filepath, "a") as f:
            f.write("\n\n=== GAME SUMMARY ===\n")
            f.write(f"Date: {self._metadata['timestamp']}\n")
            f.write(f"Format: {self._metadata['format'].capitalize()}\n")
            f.write(f"Players: {' vs '.join(self._metadata['players'])}\n")
            f.write(f"Result: {result}\n")
            duration = self._metadata.get("duration_seconds", 0)
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            f.write(f"Duration: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Total Turns: {total_turns}\n")
            f.write(f"Total Decisions: {len(self._decisions)}\n")

        self._active = False
        return filepath

    def get_decisions(self) -> List[Dict[str, Any]]:
        return list(self._decisions)

    @property
    def is_active(self) -> bool:
        return self._active
