"""Verify that ONLY match files are produced and that match replays yield
trainable experiences — including when /decision requests race ahead of the
/game/start/match registration (the bug that previously dropped every
decision and left match files with 0 experiences).
"""

import json
import sys
from pathlib import Path

# Make the package importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arcbound.logging.match_logger import MatchLogger
from arcbound.logging.replay_logger import ReplayLogger
from arcbound.logging.replay_codec import list_replays
from arcbound.training.trainer import extract_experiences, extract_value_experiences


def _board_state(game_id: str, turn: int) -> dict:
    return {
        "game_id": game_id,
        "active_player": "Ai",
        "turn": turn,
        "phase": "Main_1",
        "focal_player": "Ai",
        "players": [
            {
                "name": "Ai",
                "life": 20,
                "library_size": 20,
                "hand": [{"id": 1, "name": "Lightning Bolt", "mana_cost": "{R}", "types": ["Instant"]}],
                "battlefield": [{"id": 2, "name": "Mountain", "types": ["Land"]}],
                "graveyard": [],
                "exile": [],
                "command_zone": [],
                "decklist": [{"id": 1, "name": "Lightning Bolt", "count": 4}],
            },
            {
                "name": "Jon",
                "life": 20,
                "library_size": 20,
                "hand": [],
                "battlefield": [],
                "graveyard": [],
                "exile": [],
                "command_zone": [],
            },
        ],
        "stack": [],
    }


def _decision_request() -> dict:
    return {
        "type": "cards",
        "description": "Choose a card to play",
        "options": [
            {"id": "0", "type": "card", "label": "Lightning Bolt"},
            {"id": "1", "type": "card", "label": "Pass"},
        ],
        "constraints": {"minChoices": 0, "maxChoices": 1, "isOptional": True, "allowNone": True},
    }


def _decision_record(game_id: str, turn: int, action: str) -> dict:
    return {
        "turn": turn,
        "phase": "Main_1",
        "player": "Ai",
        "decision_type": "cards",
        "description": "Choose a card to play",
        "options_presented": ["Lightning Bolt", "Pass"],
        "action_taken": action,
        "action_reason": "test",
        "model_confidence": 0.9,
        "model_value_estimate": 0.5,
        "board_state": _board_state(game_id, turn),
        "decision_request": _decision_request(),
        "timestamp": "2026-09-09T00:00:00",
        "timeout": False,
    }


def test_no_per_game_files(tmp_path: Path) -> None:
    """ReplayLogger with write_files=False must never write a file."""
    rl = ReplayLogger(tmp_path, write_files=False)
    rl.start_game("g1", ["Ai", "Jon"], "Ai")
    rl.log_decision(_decision_record("g1", 1, "0"))
    result = rl.end_game(result="win")
    assert result is None, "ReplayLogger must not write a file when write_files=False"
    assert list_replays(tmp_path) == [], "No per-game file should exist"
    # In-memory state must still work for the monitoring endpoints.
    assert len(rl.get_decisions()) == 1
    print("PASS: no per-game files written; in-memory decisions intact")


def test_match_decisions_buffered_when_game_unregistered(tmp_path: Path) -> None:
    """Decisions arriving before /game/start/match must be buffered and flushed
    in order when the game is registered (the race that dropped all decisions)."""
    ml = MatchLogger(tmp_path)
    ml.start_match("m1", ["Jon", "Ai"], "constructed", 2)

    # Simulate the race: decisions for game "42" arrive BEFORE start_game.
    ml.log_decision("42", _decision_record("42", 1, "0"))
    ml.log_decision("42", _decision_record("42", 2, "1"))
    # An event-based move (no decision_request) also arrives early.
    ml.log_move("42", {"turn": 1, "phase": "Draw", "player": "Ai",
                       "action": "drew a card", "action_type": "draw",
                       "board_state": _board_state("42", 1)})

    # Now the /game/start/match POST is processed.
    ml.start_game(game_id="42", game_number=1, decklists={"Ai": ["Lightning Bolt"]})

    game = ml._games["42"]
    kinds = [m.get("kind") for m in game["moves"]]
    # The two buffered decisions must be present and in order, plus the move.
    assert kinds.count("decision") == 2, f"expected 2 decision records, got {kinds}"
    assert kinds[0] == "decision" and kinds[1] == "decision", f"decisions out of order: {kinds}"
    print("PASS: decisions buffered before game registration are flushed in order")


def test_match_file_yields_experiences(tmp_path: Path) -> None:
    """A completed match file must yield > 0 policy and value experiences."""
    ml = MatchLogger(tmp_path)
    ml.start_match("m1", ["Jon", "Ai"], "constructed", 2)
    ml.start_game(game_id="42", game_number=1, decklists={"Ai": ["Lightning Bolt"]})
    # Decisions AFTER registration (normal path).
    ml.log_decision("42", _decision_record("42", 1, "0"))
    ml.log_decision("42", _decision_record("42", 2, "1"))
    ml.log_move("42", {"turn": 3, "phase": "End", "player": "Ai",
                       "action": "ended turn", "action_type": "end",
                       "board_state": _board_state("42", 3)})
    ml.end_game(game_id="42", winner="Ai", total_turns=3)
    ml.end_match(match_id="m1", match_winner="Ai", final_score={"Jon": 0, "Ai": 1},
                 result="completed")

    files = [p for p in list_replays(tmp_path) if p.name.startswith("match_")]
    assert len(files) == 1, f"expected exactly one match file, got {files}"
    path = files[0]

    policy = extract_experiences(path)
    value = extract_value_experiences(path)
    assert len(policy) > 0, "match file must yield policy experiences"
    assert len(value) > 0, "match file must yield value experiences"
    print(f"PASS: match file yields {len(policy)} policy + {len(value)} value experiences")


def test_only_match_files_in_dir(tmp_path: Path) -> None:
    """After a full match, the replays dir must contain ONLY a match file."""
    ml = MatchLogger(tmp_path)
    rl = ReplayLogger(tmp_path, write_files=False)
    ml.start_match("m1", ["Jon", "Ai"], "constructed", 1)
    ml.start_game(game_id="7", game_number=1)
    ml.log_decision("7", _decision_record("7", 1, "0"))
    ml.end_game(game_id="7", winner="Jon", total_turns=2)
    ml.end_match(match_id="m1", match_winner="Jon", final_score={"Jon": 1, "Ai": 0},
                 result="completed")
    # The per-game logger is active in memory but writes nothing.
    rl.start_game("7", ["Jon", "Ai"], "Ai")
    rl.log_decision(_decision_record("7", 1, "0"))
    rl.end_game(result="win")

    all_files = sorted(p.name for p in list_replays(tmp_path))
    assert all_files, "expected at least the match file"
    assert all(name.startswith("match_") for name in all_files), \
        f"non-match files present: {all_files}"
    print(f"PASS: only match files present: {all_files}")


if __name__ == "__main__":
    import tempfile
    for fn in (test_no_per_game_files, test_match_decisions_buffered_when_game_unregistered,
               test_match_file_yields_experiences, test_only_match_files_in_dir):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("\nALL MATCH-ONLY TESTS PASSED")
