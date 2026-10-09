"""Sanity check for the per-perspective eval history feature.

Verifies:
1. EvalHistory stores per-player self-assessment values (no negation).
2. EvalHistory resets when the game id changes.
3. The /monitoring/eval_history endpoint returns per-player series.
4. The /game/move route records a value from EACH player's perspective.
5. The /decision route records the AI's decision value + the opponent's
   self-assessment (computed from the opponent's perspective).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from arcbound.logging.eval_history import EvalHistory


def test_per_player_values():
    eh = EvalHistory()
    eh.set_ai_player("Ai")
    # AI decision: AI self-assessment +0.4, opponent (Jon) self-assessment +0.3.
    eh.add(game_id="g1", turn=1, values={"Ai": 0.4, "Jon": 0.3}, actor="Ai", kind="decision")
    # AI move: AI self-assessment -0.2, Jon self-assessment +0.1.
    eh.add(game_id="g1", turn=2, values={"Ai": -0.2, "Jon": 0.1}, actor="Ai", kind="move")
    all_data = eh.get_all()
    # Both players get their own series (no negation).
    assert set(all_data["players"]) == {"Ai", "Jon"}, all_data["players"]
    assert all_data["ai_players"] == ["Ai"], all_data["ai_players"]
    ai_series = all_data["series"]["Ai"]
    jon_series = all_data["series"]["Jon"]
    assert len(ai_series) == 2 and len(jon_series) == 2
    assert abs(ai_series[0]["value"] - 0.4) < 1e-9, ai_series[0]
    assert abs(ai_series[1]["value"] - (-0.2)) < 1e-9, ai_series[1]
    assert abs(jon_series[0]["value"] - 0.3) < 1e-9, jon_series[0]
    assert abs(jon_series[1]["value"] - 0.1) < 1e-9, jon_series[1]
    # Backward-compatible get() returns the first AI's own series.
    pts = eh.get()
    assert len(pts) == 2 and abs(pts[0]["value"] - 0.4) < 1e-9, pts
    print("test_per_player_values OK")


def test_none_values_ignored():
    eh = EvalHistory()
    eh.set_ai_player("Ai")
    eh.add(game_id="g1", turn=1, values=None, actor="Ai", kind="decision")
    eh.add(game_id="g1", turn=1, values={"Ai": None, "Jon": None}, actor="Ai", kind="decision")
    assert eh.get() == []
    # "Ai" is registered (via set_ai_player) so it appears with an empty series;
    # "Jon" was never registered (its only value was None) so it is absent.
    series = eh.get_all()["series"]
    assert series.get("Ai") == [], series
    assert "Jon" not in series, series
    print("test_none_values_ignored OK")


def test_game_reset():
    eh = EvalHistory()
    eh.set_ai_player("Ai")
    eh.add(game_id="g1", turn=1, values={"Ai": 0.5}, actor="Ai", kind="decision")
    eh.add(game_id="g2", turn=1, values={"Ai": 0.1}, actor="Ai", kind="decision")
    pts = eh.get()
    assert len(pts) == 1, pts
    assert pts[0]["value"] == 0.1
    assert eh.game_id == "g2"
    print("test_game_reset OK")


def test_partial_values():
    # A point where only one player has a value: only that player's series
    # gets the point; the other player's series is unaffected.
    eh = EvalHistory()
    eh.set_ai_player("Ai")
    eh.add(game_id="g1", turn=1, values={"Ai": 0.5}, actor="Ai", kind="decision")
    eh.add(game_id="g1", turn=2, values={"Jon": 0.2}, actor="Jon", kind="move")
    all_data = eh.get_all()
    assert len(all_data["series"]["Ai"]) == 1
    assert len(all_data["series"]["Jon"]) == 1
    assert abs(all_data["series"]["Jon"][0]["value"] - 0.2) < 1e-9
    print("test_partial_values OK")


def _make_app(engine):
    from fastapi import FastAPI

    from arcbound.logging.eval_history import EvalHistory
    from arcbound.logging.match_logger import MatchLogger
    from arcbound.logging.replay_logger import ReplayLogger
    from arcbound.routes import match, monitoring

    app = FastAPI()
    app.state.eval_history = EvalHistory()
    app.state.match_logger = MatchLogger(Path("/tmp/arcbound-sanity-replays"))
    app.state.replay_logger = ReplayLogger(Path("/tmp/arcbound-sanity-replays"), write_files=False)
    app.state.inference_engine = engine
    app.include_router(monitoring.router)
    app.include_router(match.router)
    return app


def test_endpoint_and_routes():
    from fastapi.testclient import TestClient

    # A fake engine that returns a distinct value per perspective so we can
    # verify the route computes each player's self-assessment independently.
    class FakeEngine:
        def estimate_value(self, board_state, perspective=None):
            if not board_state:
                return None
            return {"Ai": 0.25, "Jon": 0.75}.get(perspective, 0.0)

    app = _make_app(FakeEngine())
    client = TestClient(app)

    client.post("/match/start", json={
        "matchId": "m1", "players": ["Jon", "Ai"],
        "playerTypes": {"Jon": "human", "Ai": "ai"},
    })
    assert app.state.eval_history.ai_player == "Ai"
    client.post("/game/start/match", json={"matchId": "m1", "gameId": "g1", "gameNumber": 1})

    # Opponent move: the route computes a value from EACH player's perspective.
    client.post("/game/move", json={
        "matchId": "m1", "gameId": "g1", "turn": 1, "phase": "MAIN",
        "player": "Jon", "action": "played a land", "actionType": "land",
        "boardState": {
            "focal_player": "Jon",
            "players": [{"name": "Jon"}, {"name": "Ai"}],
        },
    })

    r = client.get("/monitoring/eval_history")
    assert r.status_code == 200, r.text
    data = r.json()
    # Both players get a series; each from its own perspective (no negation).
    assert set(data["players"]) == {"Jon", "Ai"}, data["players"]
    assert abs(data["series"]["Jon"][0]["value"] - 0.75) < 1e-9, data["series"]["Jon"]
    assert abs(data["series"]["Ai"][0]["value"] - 0.25) < 1e-9, data["series"]["Ai"]
    assert data["series"]["Jon"][0]["player"] == "Jon"
    assert data["series"]["Jon"][0]["kind"] == "move"
    print("test_endpoint_and_routes OK")


def test_persist_until_new_game():
    """The eval history must NOT be cleared when a game or match ends. It
    persists so the GUI's graph keeps showing the finished game's curve, and is
    only reset when the NEXT game's first action arrives (auto-reset on a
    game_id change) or on a fresh server launch."""
    from fastapi.testclient import TestClient

    class FakeEngine:
        def estimate_value(self, board_state, perspective=None):
            if not board_state:
                return None
            return {"Ai": 0.5, "Jon": 0.6}.get(perspective, 0.0)

    app = _make_app(FakeEngine())
    client = TestClient(app)

    client.post("/match/start", json={
        "matchId": "m1", "players": ["Jon", "Ai"],
        "playerTypes": {"Jon": "human", "Ai": "ai"},
    })
    client.post("/game/start/match", json={"matchId": "m1", "gameId": "g1", "gameNumber": 1})
    client.post("/game/move", json={
        "matchId": "m1", "gameId": "g1", "turn": 1, "phase": "MAIN",
        "player": "Ai", "action": "cast", "actionType": "spell",
        "boardState": {"focal_player": "Ai", "players": [{"name": "Ai"}, {"name": "Jon"}]},
    })
    # Points exist while the game is in progress.
    data = client.get("/monitoring/eval_history").json()
    assert len(data["series"]["Ai"]) == 1

    # Game end -> eval history PERSISTS (not cleared).
    r = client.post("/game/end/match", json={
        "matchId": "m1", "gameId": "g1", "winner": "Ai", "totalTurns": 5,
    })
    assert r.status_code == 200, r.text
    data = client.get("/monitoring/eval_history").json()
    assert len(data["series"]["Ai"]) == 1, f"history must persist after game end, got {data}"
    assert app.state.eval_history.game_id == "g1"

    # A new game's first action resets the series (auto-reset on game_id change).
    client.post("/game/start/match", json={"matchId": "m1", "gameId": "g2", "gameNumber": 2})
    client.post("/game/move", json={
        "matchId": "m1", "gameId": "g2", "turn": 1, "phase": "MAIN",
        "player": "Jon", "action": "land", "actionType": "land",
        "boardState": {"focal_player": "Jon", "players": [{"name": "Jon"}, {"name": "Ai"}]},
    })
    data = client.get("/monitoring/eval_history").json()
    assert len(data["series"]["Jon"]) == 1, data
    # Jon's self-assessment from Jon's perspective (no negation).
    assert abs(data["series"]["Jon"][0]["value"] - 0.6) < 1e-9, data["series"]["Jon"]

    # Match end -> eval history PERSISTS (not cleared).
    r = client.post("/match/end", json={
        "matchId": "m1", "matchWinner": "Ai",
        "finalScore": {"Jon": 0, "Ai": 1}, "result": "completed",
    })
    assert r.status_code == 200, r.text
    data = client.get("/monitoring/eval_history").json()
    assert len(data["series"]["Jon"]) == 1, f"history must persist after match end, got {data}"
    print("test_persist_until_new_game OK")


def test_ai_player_from_decision_route():
    """The AI player must be identified from the /decision route's focal_player
    (always the external AI), and the opponent's self-assessment must be
    computed from the opponent's OWN perspective (not negated) — even when
    /match/start did not populate player_types.

    Scenario:
      - AI's turn: /decision, AI losing -> value -0.98 (AI perspective).
      - The route also computes the opponent's self-assessment from the
        opponent's perspective (+0.98, the opponent is winning).
    """
    from dataclasses import dataclass

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from arcbound.logging.eval_history import EvalHistory
    from arcbound.logging.match_logger import MatchLogger
    from arcbound.logging.replay_logger import ReplayLogger
    from arcbound.models.decision import AiDecisionResponse
    from arcbound.routes import decision, match, monitoring

    app = FastAPI()
    app.state.eval_history = EvalHistory()
    app.state.match_logger = MatchLogger(Path("/tmp/arcbound-sanity-replays"))
    app.state.replay_logger = ReplayLogger(Path("/tmp/arcbound-sanity-replays"), write_files=False)

    @dataclass
    class FakeResult:
        response: AiDecisionResponse
        confidence: float = None
        value_estimate: float = None
        used_model: bool = False

    class FakeEngine:
        # decide() returns the AI's decision value (AI perspective).
        # estimate_value() returns the given perspective's self-assessment.
        def __init__(self, decision_value, move_value):
            self._decision_value = decision_value
            self._move_value = move_value
            self.is_loaded = True

        def decide(self, request):
            resp = AiDecisionResponse(
                decisionId="d1", decisionType="boolean",
                selectedOptionId="yes", reasoning="fake",
            )
            return FakeResult(response=resp, value_estimate=self._decision_value)

        def estimate_value(self, board_state, perspective=None):
            return self._move_value if board_state else None

    # AI is losing (-0.98 from AI's view); the opponent's self-assessment is
    # +0.98 (the opponent is winning, from the opponent's own view).
    app.state.inference_engine = FakeEngine(decision_value=-0.98, move_value=0.98)
    app.include_router(monitoring.router)
    app.include_router(match.router)
    app.include_router(decision.router)
    client = TestClient(app)

    # Match start WITHOUT player_types (ai_player not set here).
    client.post("/match/start", json={"matchId": "m1", "players": ["Jon", "Ai"]})
    assert app.state.eval_history.ai_player is None, "precondition: ai_player unset"
    client.post("/game/start/match", json={"matchId": "m1", "gameId": "g1", "gameNumber": 1})

    # AI's turn: /decision. focal_player is always the external AI. The board
    # state includes both players so the route can compute the opponent's
    # self-assessment too.
    decision_payload = {
        "gameId": "ctrl-uuid",
        "boardState": {
            "game_id": "g1",
            "turn": 1,
            "phase": "MAIN",
            "focal_player": "Ai",
            "active_player": "Ai",
            "players": [{"name": "Ai"}, {"name": "Jon"}],
        },
        "decisionRequest": {
            "type": "boolean",
            "description": "Keep hand or mulligan?",
            "options": [
                {"id": "yes", "type": "boolean", "label": "yes"},
                {"id": "no", "type": "boolean", "label": "no"},
            ],
        },
    }
    r = client.post("/decision", json=decision_payload)
    assert r.status_code == 200, r.text
    # The /decision route must have identified the AI player from focal_player.
    assert app.state.eval_history.ai_player == "Ai", app.state.eval_history.ai_player

    data = client.get("/monitoring/eval_history").json()
    # AI decision: -0.98 (AI's own self-assessment).
    assert abs(data["series"]["Ai"][0]["value"] - (-0.98)) < 1e-9, data["series"]["Ai"]
    # Opponent's self-assessment: +0.98 from the opponent's OWN perspective
    # (NOT negated — with hidden information it is a different information set).
    assert abs(data["series"]["Jon"][0]["value"] - 0.98) < 1e-9, data["series"]["Jon"]
    print("test_ai_player_from_decision_route OK")


if __name__ == "__main__":
    test_per_player_values()
    test_none_values_ignored()
    test_game_reset()
    test_partial_values()
    test_endpoint_and_routes()
    test_persist_until_new_game()
    test_ai_player_from_decision_route()
    print("ALL SANITY CHECKS PASSED")
