"""Tests for multi-select and ordering decision responses.

Covers both the fallback path (no model) and the model path (InferenceEngine),
verifying that selectedOptionIds is populated correctly for:
  - multi-select decisions (constraints.max_choices > 1)
  - ordering decisions (decision type contains "ORDER")
and that single-choice decisions leave selectedOptionIds as None.
"""

import pytest

from arcbound.decision.engine import InferenceEngine
from arcbound.decision.fallback import generate_fallback_decision
from arcbound.encoder.transformer import create_model
from arcbound.models.decision import (
    AiDecisionRequest,
    Constraints,
    DecisionContext,
    Option,
)


def _board() -> dict:
    """A minimal valid board state for the tokenizer."""
    return {
        "game_id": "t", "active_player": "Alice", "turn": 1, "phase": "MAIN",
        "focal_player": "Alice",
        "players": [
            {"name": "Alice", "life": 40, "is_focal": True, "library_size": 25,
             "hand": [{"id": 1, "name": "Island", "converted_mana_cost": 0}],
             "battlefield": [], "graveyard": [], "exile": [], "command_zone": []},
            {"name": "Bob", "life": 40, "is_focal": False, "library_size": 25,
             "hand": [], "battlefield": [], "graveyard": [], "exile": [], "command_zone": []},
        ],
        "stack": [],
    }


def _request(decision_type: str, options: list[Option], constraints: Constraints | None = None) -> AiDecisionRequest:
    return AiDecisionRequest(
        gameId="t",
        boardState=_board(),
        decisionRequest=DecisionContext(
            type=decision_type,
            description="Test",
            options=options,
            constraints=constraints,
        ),
    )


def _opts(n: int, prefix: str = "opt") -> list[Option]:
    return [Option(id=f"{prefix}{i}", type="option", label=f"Label {i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------
class TestFallbackMultiSelect:
    def test_multi_select_returns_top_k(self):
        req = _request("cards", _opts(5), Constraints(min_choices=1, max_choices=3))
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionIds == ["opt0", "opt1", "opt2"]
        assert resp.selectedOptionId == "opt0"

    def test_multi_select_clamped_to_option_count(self):
        req = _request("cards", _opts(2), Constraints(min_choices=0, max_choices=5))
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionIds == ["opt0", "opt1"]

    def test_ordering_returns_full_permutation(self):
        req = _request("ORDER_BLOCKERS", _opts(4))
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionIds == ["opt0", "opt1", "opt2", "opt3"]
        assert resp.selectedOptionId == "opt0"

    def test_single_choice_has_no_ids(self):
        req = _request("boolean", [Option(id="yes", type="boolean", label="Yes"),
                                   Option(id="no", type="boolean", label="No")])
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionIds is None
        assert resp.selectedOptionId is not None

    def test_explicit_no_cards_preserved(self):
        # allow_none -> fallback returns an explicit empty selection.
        req = _request("cards", _opts(3), Constraints(min_choices=0, max_choices=3,
                                                      is_optional=True, allow_none=True))
        resp = generate_fallback_decision(req)
        assert resp.selectedOptionIds == []


# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
class TestEngineMultiSelect:
    @pytest.fixture
    def engine(self):
        eng = InferenceEngine(create_model())
        eng.set_model(eng.model)
        return eng

    def test_multi_select_returns_top_k(self, engine):
        req = _request("cards", _opts(5), Constraints(min_choices=1, max_choices=3))
        result = engine.decide(req)
        resp = result.response
        assert resp.selectedOptionIds is not None
        assert len(resp.selectedOptionIds) == 3
        # All ids must be valid, distinct options.
        valid = {o.id for o in req.decisionRequest.options}
        assert set(resp.selectedOptionIds) <= valid
        assert len(set(resp.selectedOptionIds)) == 3
        # Primary pick is the first of the list.
        assert resp.selectedOptionId == resp.selectedOptionIds[0]

    def test_ordering_returns_full_permutation(self, engine):
        req = _request("ORDER_BLOCKERS", _opts(4))
        resp = engine.decide(req).response
        assert resp.selectedOptionIds is not None
        # A full permutation: every option exactly once.
        assert sorted(resp.selectedOptionIds) == sorted(o.id for o in req.decisionRequest.options)
        assert resp.selectedOptionId == resp.selectedOptionIds[0]

    def test_single_choice_has_no_ids(self, engine):
        req = _request("boolean", [Option(id="yes", type="boolean", label="Yes"),
                                   Option(id="no", type="boolean", label="No")])
        resp = engine.decide(req).response
        assert resp.selectedOptionIds is None
        assert resp.selectedOptionId in ("yes", "no")

    def test_multi_select_respects_min_when_model_scores(self, engine):
        # Even with a low max, we always return at least 1 pick.
        req = _request("cards", _opts(4), Constraints(min_choices=1, max_choices=2))
        resp = engine.decide(req).response
        assert resp.selectedOptionIds is not None
        assert 1 <= len(resp.selectedOptionIds) <= 2
