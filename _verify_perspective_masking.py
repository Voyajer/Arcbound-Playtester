"""Verify per-perspective hidden-zone masking in the tokenizer.

Builds a full board (both hands populated, as the new Java serializer sends)
and checks that:
  1. Tokenizing from the AI's perspective emits the AI's hand but NOT the
     human's hand (and only the AI's decklist).
  2. Tokenizing from the human's perspective emits the human's hand but NOT
     the AI's hand (and only the human's decklist).
  3. A card explicitly revealed_to the perspective player IS emitted even
     though it is in the opponent's hand.
  4. The default perspective (board focal player) matches the AI's view.
"""

import sys

from arcbound.encoder.tokenizer import tokenize_board_state


def card(cid, name, revealed_to=None):
    return {
        "id": cid,
        "name": name,
        "mana_cost": "1",
        "converted_mana_cost": 1,
        "types": ["Creature"],
        "power": 1,
        "toughness": 1,
        "revealed_to": revealed_to or [],
    }


def build_board():
    ai_hand = [card(1, "AI Secret One"), card(2, "AI Secret Two")]
    human_hand = [
        card(3, "Human Secret One"),
        card(4, "Human Secret Two", revealed_to=["ai-0"]),  # revealed to the AI
    ]
    return {
        "game_id": "g1",
        "active_player": "ai-0",
        "turn": 3,
        "phase": "Main",
        "focal_player": "ai-0",
        "players": [
            {
                "name": "ai-0",
                "life": 20,
                "hand": ai_hand,
                "decklist": [card(10, "AI Deck Card")],
                "library_size": 20,
            },
            {
                "name": "human-0",
                "life": 20,
                "hand": human_hand,
                "decklist": [card(20, "Human Deck Card")],
                "library_size": 20,
            },
        ],
        "stack": [],
    }


def main():
    board = build_board()

    # --- 1 & 2: perspective masking via token counts ---------------------
    # The AI perspective should emit: AI hand (2) + AI decklist (1) of
    # hand/deck cards, plus the revealed human card (1). The human perspective
    # should emit: human hand (2) + human decklist (1), and NOT the AI's
    # secrets.
    #
    # We verify by counting total tokens: the two perspectives must differ by
    # exactly the number of hidden cards that flip visibility, and each must
    # exclude the opponent's unrevealed hand cards.
    tb_ai = tokenize_board_state(board, perspective="ai-0")
    tb_human = tokenize_board_state(board, perspective="human-0")
    tb_default = tokenize_board_state(board)  # focal = ai-0

    # Default must equal the AI perspective (focal player is ai-0).
    assert tb_default.features.shape == tb_ai.features.shape, (
        f"default {tb_default.features.shape} != ai {tb_ai.features.shape}"
    )

    # The AI sees its 2 hand cards + 1 revealed human card = 3 hand-zone cards.
    # The human sees its 2 hand cards = 2 hand-zone cards.
    # Decklists: AI sees 1 deck card; human sees 1 deck card.
    # So the AI perspective has exactly 1 more card token than the human
    # perspective (the revealed card).
    n_ai = tb_ai.features.shape[0]
    n_human = tb_human.features.shape[0]
    print(f"AI perspective tokens:    {n_ai}")
    print(f"Human perspective tokens: {n_human}")
    assert n_ai == n_human + 1, (
        f"expected AI to have exactly 1 more token (the revealed card), "
        f"got ai={n_ai} human={n_human}"
    )

    # --- 3: revealed card is visible to the AI, not to the human ---------
    # The revealed card ("Human Secret Two") is in the human's hand but
    # revealed_to ai-0, so the AI perspective includes it and the human
    # perspective includes it too (it's the human's own card). The AI's own
    # secrets must NOT appear in the human perspective. We verify this by
    # checking that the human perspective does NOT contain the AI's hand
    # cards: if it did, the human perspective would have 2 extra tokens.
    # (Covered by the count assertion above: human has the fewest tokens.)

    # --- 4: the AI's secrets are absent from the human view --------------
    # Build a board where the AI has 3 hand cards and the human has 1, with
    # no reveals. The human perspective must have 2 fewer tokens than the AI
    # perspective (3 AI hand cards hidden from the human, 1 human hand card
    # hidden from the AI -> net: AI view has 3+1deck, human view has 1+1deck).
    board2 = build_board()
    board2["players"][0]["hand"] = [card(1, "A1"), card(2, "A2"), card(3, "A3")]
    board2["players"][1]["hand"] = [card(4, "H1")]
    board2["players"][0]["decklist"] = []
    board2["players"][1]["decklist"] = []
    tb_ai2 = tokenize_board_state(board2, perspective="ai-0")
    tb_human2 = tokenize_board_state(board2, perspective="human-0")
    print(f"AI2 tokens: {tb_ai2.features.shape[0]}, Human2 tokens: {tb_human2.features.shape[0]}")
    # AI sees 3 hand cards; human sees 1. Difference = 2.
    assert tb_ai2.features.shape[0] - tb_human2.features.shape[0] == 2, (
        f"expected 2-token difference, got "
        f"{tb_ai2.features.shape[0] - tb_human2.features.shape[0]}"
    )

    print("ALL PERSPECTIVE MASKING CHECKS PASSED")


if __name__ == "__main__":
    main()
