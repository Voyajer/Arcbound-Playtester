"""Replay file reader for loading and parsing game replays."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReplayReader:
    """Read and parse replay files."""

    @staticmethod
    def read(replay_path: Path) -> Dict[str, Any]:
        """Read a replay file and return its contents as a dictionary."""
        with open(replay_path, "r") as f:
            content = f.read()

        # Strip trailing summary (everything after === GAME SUMMARY ===)
        json_end = content.find("\n\n=== GAME SUMMARY ===")
        if json_end != -1:
            json_str = content[:json_end]
        else:
            json_str = content

        return json.loads(json_str)

    @staticmethod
    def list_replays(replays_dir: Path) -> List[Path]:
        """List all replay files in directory, sorted by name (timestamp)."""
        if not replays_dir.exists():
            return []
        return sorted(replays_dir.glob("*.json"))

    @staticmethod
    def get_summary(replay_path: Path) -> Dict[str, Any]:
        """Extract just the metadata summary from a replay file."""
        data = ReplayReader.read(replay_path)
        return data.get("metadata", {})

    @staticmethod
    def get_decisions(replay_path: Path) -> List[Dict[str, Any]]:
        """Extract decisions from a replay file."""
        data = ReplayReader.read(replay_path)
        return data.get("decisions", [])

    @staticmethod
    def get_decklists(replay_path: Path) -> Dict[str, List[str]]:
        """Extract decklists from a replay file."""
        data = ReplayReader.read(replay_path)
        return data.get("decklists", {})
