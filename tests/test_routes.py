"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient

from arcbound.models.decision import AiDecisionRequest, DecisionContext, Option
from arcbound.server import create_app

app = create_app()
client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "modelLoaded" in data


class TestDecision:
    def test_decision_boolean(self):
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

    def test_decision_card(self):
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
    def test_game_start(self):
        resp = client.post("/game/start?game_id=test-1&game_type=commander")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_game_end(self):
        resp = client.post("/game/end?game_id=test-1&result=win")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
