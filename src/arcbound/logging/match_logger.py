"""Match-level replay logging system.

Captures an entire match (e.g. a Best-of-3) as a SINGLE JSON file. The file
contains a top-level ``match`` object (match winner, final score, match reward)
and a ``games`` array where each game carries its own winner, game reward,
sideboard (games 2+), decklists, and the full per-action move list.

This is the training artifact for reward-based learning: the pipeline can use
either the match-level signal (``match.match_reward``) or the per-game signal
(``games[i].game_reward``).
"""

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from arcbound.logging.replay_codec import write_replay


class MatchLogger:
    """Logs an entire match (all games) to a single replay file."""

    def __init__(self, replays_dir: Path):
        self.replays_dir = replays_dir
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Active match state (one match at a time, mirroring ReplayLogger).
        self._match: Optional[Dict[str, Any]] = None
        self._games: Dict[str, Dict[str, Any]] = {}  # game_id -> game record
        self._game_order: List[str] = []  # preserves game_number ordering
        self._current_game_id: Optional[str] = None
        # Sideboard data that arrived before the game was registered (the
        # Forge side sends sideboard events during game setup, which can race
        # ahead of the /game/start/match POST). Keyed by game_id -> player -> data.
        self._pending_sideboards: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Decision records that arrived before the game was registered. The
        # /decision endpoint can fire before the /game/start/match POST is
        # processed (both are async HTTP), so decisions are buffered here and
        # flushed (in order) when the game is registered. Keyed by game_id.
        self._pending_decisions: Dict[str, List[Dict[str, Any]]] = {}
        self._filename: Optional[str] = None
        self._active: bool = False

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_draw(winner: Optional[str]) -> bool:
        if not winner:
            return True
        return winner.strip().lower() in ("draw", "tie", "none", "no winner", "")

    @staticmethod
    def _composition(player_types: Optional[Dict[str, str]]) -> str:
        """Derive a short label for the match composition from player roles.

        e.g. ``{"Jon": "human", "Ai": "ai"}`` -> ``"human_vs_ai"``. Players with
        an unknown/missing role are labelled ``unknown``. This makes every match
        composition (human vs human, human vs bot, bot vs bot, bot vs AI,
        AI vs AI, ...) self-describing in the replay file.
        """
        if not player_types:
            return "unknown"
        roles = [str(v).strip().lower() or "unknown" for v in player_types.values()]
        return "_vs_".join(roles)

    def _compute_reward(self, players: List[str], winner: Optional[str]) -> Dict[str, int]:
        """Winner +1, every other player -1. Draw -> all 0.

        If the winner name does not match any player (e.g. a team name), we
        cannot attribute the reward, so all players get 0.
        """
        if self._is_draw(winner):
            return {p: 0 for p in players}
        if winner not in players:
            return {p: 0 for p in players}
        return {p: (1 if p == winner else -1) for p in players}

    # ------------------------------------------------------------------
    # Match lifecycle
    # ------------------------------------------------------------------
    def start_match(
        self,
        match_id: str,
        players: List[str],
        format_name: str = "standard",
        games_to_win: int = 1,
        player_types: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new match log. Returns the replay filename.

        ``player_types`` maps each player name to their role ("human", "bot",
        "ai", or "unknown"). It is stored in the replay file (along with a
        derived ``composition`` label) so the trainer can tell which moves were
        made by a human, a Forge bot, or the external AI.
        """
        with self._lock:
            now = datetime.now()
            safe_players = [p.replace(" ", "-").lower() for p in players]
            filename = (
                f"match_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'_vs_'.join(safe_players)}_"
                f"{format_name}_{uuid.uuid4().hex[:8]}.json.gz"
            )
            self._filename = filename
            self._match = {
                "match_id": match_id,
                "players": players,
                "player_types": player_types or {},
                "composition": self._composition(player_types),
                "format": format_name,
                "games_to_win": games_to_win,
                "match_winner": None,
                "final_score": {p: 0 for p in players},
                "match_reward": None,
                "start_time": now.isoformat(),
                "end_time": None,
                "duration_seconds": 0,
                "result": None,
            }
            self._games = {}
            self._game_order = []
            self._current_game_id = None
            self._pending_sideboards = {}
            self._pending_decisions = {}
            self._active = True
            return filename

    def start_game(
        self,
        game_id: str,
        game_number: int,
        decklists: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Register a new game within the active match."""
        with self._lock:
            if not self._active or self._match is None:
                return
            if game_id in self._games:
                return
            self._games[game_id] = {
                "game_number": game_number,
                "game_id": game_id,
                "winner": None,
                "game_reward": None,
                "total_turns": 0,
                "sideboard": None,
                "decklists": decklists or {},
                "moves": [],
            }
            self._game_order.append(game_id)
            self._current_game_id = game_id
            # Apply any sideboard data that arrived before this game started.
            pending = self._pending_sideboards.pop(game_id, None)
            if pending:
                self._games[game_id]["sideboard"] = pending
            # Flush any decision records that arrived before this game started,
            # preserving their original order.
            pending_decisions = self._pending_decisions.pop(game_id, None)
            if pending_decisions:
                for record in pending_decisions:
                    record["kind"] = "decision"
                    self._games[game_id]["moves"].append(record)

    def log_sideboard(
        self,
        game_id: str,
        player: str,
        main_to_side: List[str],
        side_to_main: List[str],
    ) -> None:
        """Record a player's sideboard changes for a game (games 2+).

        If the game has not been registered yet (sideboard events can arrive
        before the /game/start/match POST), the data is buffered and applied
        when the game starts.
        """
        with self._lock:
            entry = {
                "main_to_side": list(main_to_side or []),
                "side_to_main": list(side_to_main or []),
            }
            game = self._games.get(game_id)
            if game is None:
                self._pending_sideboards.setdefault(game_id, {})[player] = entry
                return
            if game["sideboard"] is None:
                game["sideboard"] = {}
            game["sideboard"][player] = entry

    def log_move(self, game_id: str, move: Dict[str, Any]) -> None:
        """Append a single action (with full board state) to a game."""
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                return
            game["moves"].append(move)

    def log_decision(self, game_id: str, decision: Dict[str, Any]) -> None:
        """Append a decision-level record (with full board state + options) to a game.

        Decision records carry ``decision_request`` (the options that were
        offered) and ``action_taken`` (the option chosen), which is what the
        policy-learning pipeline needs. They are stored in the same ``moves``
        list as event-based moves, tagged with ``kind="decision"`` so the
        extractors can tell them apart.

        If the game has not been registered yet (the /game/start/match POST can
        race behind /decision requests), the record is buffered and flushed in
        order when the game is registered — decisions are never dropped.
        """
        with self._lock:
            game = self._games.get(game_id)
            if game is None:
                if self._active and self._match is not None:
                    self._pending_decisions.setdefault(game_id, []).append(dict(decision))
                return
            record = dict(decision)
            record["kind"] = "decision"
            game["moves"].append(record)

    def end_game(
        self,
        game_id: str,
        winner: Optional[str],
        total_turns: int = 0,
    ) -> None:
        """Finalize a game: set winner, compute game reward, update score."""
        with self._lock:
            if not self._active or self._match is None:
                return
            game = self._games.get(game_id)
            if game is None:
                return
            game["winner"] = winner
            game["total_turns"] = total_turns
            game["game_reward"] = self._compute_reward(self._match["players"], winner)

            # Update the running match score (games won per player).
            if not self._is_draw(winner) and winner in self._match["final_score"]:
                self._match["final_score"][winner] += 1

    def end_match(
        self,
        match_id: str,
        match_winner: Optional[str],
        final_score: Optional[Dict[str, int]] = None,
        result: str = "completed",
    ) -> Path:
        """Finalize the match and write the single JSON file to disk."""
        with self._lock:
            if not self._match:
                raise RuntimeError("No active match. Call start_match() first.")

            self._match["match_winner"] = match_winner
            if final_score is not None:
                self._match["final_score"] = final_score
            self._match["match_reward"] = self._compute_reward(
                self._match["players"], match_winner
            )
            self._match["result"] = result
            self._match["end_time"] = datetime.now().isoformat()
            if self._match.get("start_time"):
                start = datetime.fromisoformat(self._match["start_time"])
                self._match["duration_seconds"] = (
                    datetime.now() - start
                ).total_seconds()

            # Assemble games in game_number order.
            games = [self._games[gid] for gid in self._game_order if gid in self._games]

            replay = {
                "match": self._match,
                "games": games,
            }

            if not self._filename:
                now = datetime.now()
                safe_players = [p.replace(" ", "-").lower() for p in self._match["players"]]
                self._filename = (
                    f"match_{now.strftime('%Y-%m-%d_%H-%M-%S')}_{'_vs_'.join(safe_players)}_"
                    f"{self._match['format']}.json.gz"
                )

            # Build the human-readable summary (embedded in the file so it
            # survives gzip compression).
            lines = [
                "=== MATCH SUMMARY ===",
                f"Match ID: {self._match['match_id']}",
                f"Format: {self._match['format'].capitalize()}",
                f"Players: {' vs '.join(self._match['players'])}",
                f"Games to Win: {self._match['games_to_win']}",
                f"Match Winner: {match_winner or 'Draw'}",
                f"Final Score: {self._match['final_score']}",
                f"Match Reward: {self._match['match_reward']}",
                f"Games Played: {len(games)}",
            ]
            for g in games:
                lines.append(
                    f"  Game {g['game_number']}: winner={g['winner'] or 'Draw'}, "
                    f"turns={g['total_turns']}, moves={len(g['moves'])}, "
                    f"sideboard={'yes' if g['sideboard'] else 'no'}"
                )
            duration = self._match.get("duration_seconds", 0)
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            lines.append(f"Duration: {hours}h {minutes}m {seconds}s")
            summary = "\n".join(lines)

            # Write the lossless, gzip-compressed replay file.
            filepath = write_replay(self.replays_dir / self._filename, replay, summary=summary)

            self._active = False
            return filepath

    # ------------------------------------------------------------------
    # Introspection / shutdown
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_match_id(self) -> Optional[str]:
        if self._active and self._match:
            return self._match["match_id"]
        return None

    @property
    def current_game_id(self) -> Optional[str]:
        return self._current_game_id if self._active else None

    def get_move_count(self, game_id: Optional[str] = None) -> int:
        with self._lock:
            if game_id is not None:
                game = self._games.get(game_id)
                return len(game["moves"]) if game else 0
            return sum(len(g["moves"]) for g in self._games.values())

    def finalize_current_match(self, result: str = "abandoned") -> Optional[Path]:
        """Finalize the in-progress match (e.g. on server shutdown).

        Returns the match file path, or None if no match was active.
        """
        if not self._active or self._match is None:
            return None
        try:
            return self.end_match(
                match_id=self._match["match_id"],
                match_winner=None,
                result=result,
            )
        except Exception:
            return None
