"""Tests for Pydantic models."""

import pytest
from arcbound.models.board_state import BoardState, CardInfo, PlayerState
from arcbound.models.decision import (
    AiDecisionRequest,
    AiDecisionResponse,
    Constraints,
    DecisionContext,
    Option,
)


class TestCardInfo:
    def test_minimal_card(self):
        card = CardInfo(id=1, name="Mountain")
        assert card.id == 1
        assert card.name == "Mountain"
        assert card.tapped is False

    def test_full_card(self):
        card = CardInfo(
            id=42,
            name="Lightning Bolt",
            oracle_name="Lightning Bolt",
            mana_cost="{R}",
            converted_mana_cost=1,
            colors=["Red"],
            color_identity=["Red"],
            types=["Instant"],
            subtypes=[],
            text="Lightning Bolt deals 3 damage to target creature or player.",
            owner="player_0",
            controller="player_0",
            zone="Hand",
            keywords=[""],
        )
        assert card.converted_mana_cost == 1
        assert "Red" in card.colors

    def test_card_with_counters(self):
        card = CardInfo(
            id=1,
            name="Giant Spider",
            power=2,
            toughness=1,
            counters={"+1/+1": 2},
            keywords=["Flying"],
        )
        assert card.counters["+1/+1"] == 2


class TestPlayerState:
    def test_minimal_player(self):
        player = PlayerState(name="Alice")
        assert player.name == "Alice"
        assert player.life == 0
        assert player.hand_hidden is False

    def test_focal_player(self):
        player = PlayerState(
            name="Alice",
            life=20,
            starting_life=20,
            is_focal=True,
            library_size=25,
        )
        assert player.is_focal is True
        assert player.life == 20
        assert player.opponent_id is None

    def test_opponent_player(self):
        player = PlayerState(
            name="Bob",
            life=20,
            is_focal=False,
            opponent_id="bob",
        )
        assert player.opponent_id == "bob"
        assert player.is_focal is False

    def test_opponent_id_case_insensitive(self):
        player = PlayerState(
            name="Alice",
            opponent_id="alice",
        )
        assert player.opponent_id == "alice"


class TestBoardState:
    def test_basic_board(self):
        board = BoardState(
            game_id="test-1",
            active_player="Alice",
            turn=3,
            phase="Main",
            focal_player="Alice",
        )
        assert board.game_id == "test-1"
        assert board.turn == 3

    def test_board_with_players(self):
        board = BoardState(
            game_id="test-1",
            active_player="Alice",
            turn=1,
            phase="Main",
            focal_player="Alice",
            players=[
                PlayerState(name="Alice", life=20, is_focal=True),
                PlayerState(name="Bob", life=20, hand_hidden=True),
            ],
        )
        assert len(board.players) == 2
        assert board.players[0].is_focal is True


class TestDecisionModels:
    def test_decision_request(self):
        req = AiDecisionRequest(
            gameId="test-1",
            boardState={},
            decisionRequest=DecisionContext(
                type="boolean",
                description="Pass?",
                options=[Option(id="opt0", type="boolean", label="Yes")],
            ),
        )
        assert req.gameId == "test-1"
        assert req.decisionRequest.type == "boolean"

    def test_decision_response(self):
        resp = AiDecisionResponse(
            decisionId="resp-1",
            decisionType="boolean",
            value={"boolean": True},
            selectedOptionId="opt0",
        )
        assert resp.decisionId == "resp-1"
        assert resp.value["boolean"] is True

    def test_constraints(self):
        c = Constraints(min_choices=1, max_choices=3, is_optional=True)
        assert c.min_choices == 1
        assert c.is_optional is True
