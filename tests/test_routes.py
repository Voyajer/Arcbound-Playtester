"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient

from arcbound.models.decision import AiDecisionRequest, DecisionContext, Option
from arcbound.server import create_app


@pytest.fixture(scope="module")
def client():
    """TestClient used as a context manager so the app lifespan runs
    (sets up replay_logger, model_manager, and inference_engine)."""
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "modelLoaded" in data


class TestDecision:
    def test_decision_boolean(self, client):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="boolean",
                description="Pass?",
                options=[
                    Option(id="opt0", type="boolean", label="Yes"),
                    Option(id="opt1", type="boolean", label="No"),
                ],
            ),
        )
        resp = client.post("/decision", json=req.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisionType"] == "boolean"
        assert "decisionId" in data

    def test_decision_card(self, client):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="card",
                description="Choose a card",
                options=[
                    Option(id="card-1", type="card", label="Lightning Bolt"),
                    Option(id="card-2", type="card", label="Mountain"),
                ],
            ),
        )
        resp = client.post("/decision", json=req.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisionType"] == "card"


class TestGameLifecycle:
    def test_game_start(self, client):
        # Matches ExternalAiHttpClient.notifyGameStart(): JSON body with gameId + decklist
        resp = client.post(
            "/game/start",
            json={"gameId": "test-1", "decklist": {"Alice": ["Lightning Bolt"]}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_game_end(self, client):
        # Matches ExternalAiHttpClient.notifyGameEnd(): JSON body with gameId + result
        resp = client.post("/game/end", json={"gameId": "test-1", "result": "win"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestEvents:
    def test_events_endpoint_returns_list(self, client):
        resp = client.get("/monitoring/events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_fallback_emits_error_event(self, client):
        """A fallback decision (no model) emits a red 'error' event that the
        GUI can poll, stating why the fallback was used."""
        # Force the no-model path by clearing the loaded model.
        app = client.app
        app.state.model_manager.model = None
        app.state.inference_engine.model = None

        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="boolean",
                description="Pass?",
                options=[
                    Option(id="opt0", type="boolean", label="Yes"),
                    Option(id="opt1", type="boolean", label="No"),
                ],
            ),
        )
        client.post("/decision", json=req.model_dump())

        events = client.get("/monitoring/events").json()
        assert any(
            e["level"] == "error" and "FALLBACK" in e["message"] for e in events
        )


class TestModelReload:
    def test_reload_endpoint(self, client):
        """POST /model/reload returns a well-formed response.

        Whether it succeeds depends on whether a model checkpoint exists in the
        test environment; either way the endpoint must respond 200 with a
        ``success`` flag and, on success, the model name + action.
        """
        resp = client.post("/model/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
        if data["success"]:
            assert data["action"] in ("loaded", "reloaded")
            assert "modelName" in data
        else:
            assert "error" in data

    def test_reload_updates_health(self, client):
        """After a successful reload, /health reflects the loaded model."""
        resp = client.post("/model/reload")
        data = resp.json()
        if not data["success"]:
            pytest.skip("No model checkpoint available in test environment")
        health = client.get("/health").json()
        assert health["modelLoaded"] is True
        assert health["modelName"] == data["modelName"]


class TestSettings:
    def test_apply_endpoint_returns_applied(self, client):
        """POST /settings/apply re-reads config and reports the applied values."""
        resp = client.post("/settings/apply")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "epsilon" in data["applied"]
        assert 0.0 <= data["applied"]["epsilon"] <= 1.0

    def test_apply_updates_engine_epsilon(self, client):
        """Apply hot-swaps the engine's epsilon to the config value mid-run."""
        engine = client.app.state.inference_engine
        # Set a sentinel, then apply must restore the value from config.
        engine.set_epsilon(0.42)
        assert engine.epsilon == 0.42
        data = client.post("/settings/apply").json()
        assert data["success"] is True
        assert engine.epsilon == data["applied"]["epsilon"]
