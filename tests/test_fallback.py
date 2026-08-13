"""Tests for fallback decision logic."""

import pytest
from arcbound.decision.fallback import generate_fallback_decision
from arcbound.models.decision import (
    AiDecisionRequest,
    Constraints,
    DecisionContext,
    Option,
)


def _make_request(decision_type: str, options: list[Option] | None = None) -> AiDecisionRequest:
    return AiDecisionRequest(
        gameId="test-1",
        boardState={},
        decisionRequest=DecisionContext(
            type=decision_type,
            description="Test",
            options=options or [Option(id="opt0", type=decision_type, label="First")],
        ),
    )


class TestFallbackBoolean:
    def test_boolean_returns_false(self):
        req = _make_request("boolean")
        resp = generate_fallback_decision(req)
        assert resp.value["boolean"] is False
        assert resp.decisionType == "boolean"


class TestFallbackInteger:
    def test_integer_returns_zero(self):
        req = _make_request("integer")
        resp = generate_fallback_decision(req)
        assert resp.value["number"] == 0


class TestFallbackString:
    def test_string_returns_empty(self):
        req = _make_request("string")
        resp = generate_fallback_decision(req)
        assert resp.value["text"] == ""


class TestFallbackColor:
    def test_color_returns_white(self):
        req = _make_request("color")
        resp = generate_fallback_decision(req)
        assert resp.value["color"] == "White"


class TestFallbackCard:
    def test_card_picks_first_option(self):
        req = _make_request("card")
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "opt0"


class TestFallbackCards:
    def test_cards_empty_when_allowed(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="cards",
                description="Test",
                options=[Option(id="opt0", type="card", label="Card A")],
                constraints=Constraints(min_choices=0, max_choices=3, allow_none=True),
            ),
        )
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionIds == []

    def test_cards_picks_first_when_not_allowed(self):
        req = _make_request("cards")
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "opt0"


class TestFallbackUnknown:
    def test_unknown_type_returns_first_option(self):
        req = _make_request("unknownType")
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "opt0"
        assert resp.decisionType == "unknownType"


class TestFallbackNoOptions:
    def test_no_options_returns_none(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="boolean",
                description="Test",
                options=[],
            ),
        )
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId is None
