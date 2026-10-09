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

    def test_java_confirm_trigger_picks_no(self):
        """Java queryBoolean sends the prompt name as type and yes/no ids."""
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="CONFIRM_TRIGGER",
                description="Trigger X?",
                options=[
                    Option(id="yes", type="boolean", label="Play trigger"),
                    Option(id="no", type="boolean", label="Don't play trigger"),
                ],
            ),
        )
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "no"
        assert resp.value["boolean"] is False

    def test_java_mulligan_picks_keep(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="MULLIGAN",
                description="Keep hand or mulligan?",
                options=[
                    Option(id="keep", type="boolean", label="Keep hand"),
                    Option(id="mulligan", type="boolean", label="Mulligan"),
                ],
            ),
        )
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "keep"

    def test_java_binary_choice_picks_false(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="BINARY_CHOICE",
                description="Which?",
                options=[
                    Option(id="true", type="boolean", label="Yes/True"),
                    Option(id="false", type="boolean", label="No/False"),
                ],
            ),
        )
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "false"

    def test_boolean_unknown_ids_fall_back_to_last_option(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="SOME_PROMPT",
                description="Pick one",
                options=[
                    Option(id="a", type="boolean", label="First"),
                    Option(id="b", type="boolean", label="Second"),
                ],
            ),
        )
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionId == "b"


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


class TestFallbackEpsilon:
    """The fallback mixes random actions with pass-throughs, ratio set by
    epsilon. epsilon=0 (default) is fully conservative; epsilon=1 is fully
    random."""

    def test_epsilon_zero_is_conservative(self):
        # cards + allow_none with epsilon=0 -> no cards (the old behavior).
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
        resp = generate_fallback_decision(req, epsilon=0.0)
        assert resp.selectedOptionIds == []

    def test_epsilon_one_picks_a_random_card(self):
        # cards + allow_none with epsilon=1 -> a random card is picked
        # (not "no cards"), so the fallback can actually play things.
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
        resp = generate_fallback_decision(req, epsilon=1.0)
        assert resp.selectedOptionIds == ["opt0"]
        assert "random" in resp.reasoning.lower()

    def test_epsilon_one_boolean_picks_a_random_option(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="boolean",
                description="Test",
                options=[
                    Option(id="yes", type="boolean", label="Yes"),
                    Option(id="no", type="boolean", label="No"),
                ],
            ),
        )
        resp = generate_fallback_decision(req, epsilon=1.0)
        # A random option is picked; the value payload matches the pick.
        assert resp.selectedOptionId in ("yes", "no")
        assert resp.value["boolean"] == (resp.selectedOptionId == "yes")

    def test_epsilon_one_multi_picks_random_subset(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="cards",
                description="Test",
                options=[Option(id=f"opt{i}", type="card", label=f"C{i}") for i in range(5)],
                constraints=Constraints(min_choices=1, max_choices=3),
            ),
        )
        resp = generate_fallback_decision(req, epsilon=1.0)
        assert len(resp.selectedOptionIds) == 3
        assert set(resp.selectedOptionIds) <= {f"opt{i}" for i in range(5)}

    def test_epsilon_one_ordering_is_a_permutation(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="ORDER_BLOCKERS",
                description="Test",
                options=[Option(id=f"opt{i}", type="option", label=f"C{i}") for i in range(4)],
            ),
        )
        resp = generate_fallback_decision(req, epsilon=1.0)
        assert sorted(resp.selectedOptionIds) == ["opt0", "opt1", "opt2", "opt3"]
