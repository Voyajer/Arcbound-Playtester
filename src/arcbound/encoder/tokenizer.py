"""Decision tokenizer — converts an AiDecisionRequest into a fixed feature sequence.

The tokenizer produces a sequence of feature vectors (one per token) plus metadata
about which positions are option tokens. This allows the transformer to score each
option via attention, handling a variable number of options naturally.

Feature layout (122-dim per token):
  [0]  token_type        0=CLS 1=SEP 2=PLAYER 3=CARD 4=DECISION 5=OPTION 6=DECK
  [1]  is_focal          0/1
  [2]  is_active         0/1
  [3]  life / 50
  [4]  library / 100
  [5]  hand_size / 10
  [6]  battlefield_size / 32
  [7]  poison / 10
  [8]  cmc / 12
  [9]  power / 20
  [10] toughness / 20
  [11] damage / 20
  [12] tapped            0/1
  [13] attacking         0/1
  [14] blocking          0/1
  [15] zone_battlefield  0/1
  [16] zone_hand         0/1
  [17] zone_graveyard    0/1
  [18] zone_exile        0/1
  [19] zone_command      0/1
  [20] decision_type     normalized index / (len(DECISION_TYPES)-1)
  [21] num_options / 20
  [22] min_choices / 10
  [23] max_choices / 10
  [24] allow_none        0/1
  [25] option_type       normalized index / 10
  [26] option_index / 20
  [27] has_details       0/1
  [28] cards_to_return / 7    (mulligan: cards tuck-returned if mulliganing)
  [29] mulligan_count / 7     (mulligan: how many times already mulliganed)
  [30] zone_deck         0/1  (card is part of the focal player's decklist)
  [31] deck_count        min(count, 8) / 8  (copies of this card in the deck)
  --- card-level (card tokens only) ---
  [32] color_white       0/1
  [33] color_blue        0/1
  [34] color_black       0/1
  [35] color_red         0/1
  [36] color_green       0/1
  [37] loyalty / 20
  [38] counter_p1p1 / 10   (+1/+1 counters)
  [39] counter_m1m1 / 10   (-1/-1 counters)
  [40] counter_other / 10  (sum of all other counter types)
  [41] controller_is_focal 0/1  (card is controlled by the focal player)
  [42] owner_is_focal      0/1  (card is owned by the focal player)
  --- player-level (player tokens only) ---
  [43] mana_pool_total / 16
  [44] mana_pool_white / 16
  [45] mana_pool_blue  / 16
  [46] mana_pool_black / 16
  [47] mana_pool_red   / 16
  [48] mana_pool_green / 16
  [49] commander_damage_max / 21      (max damage from any single commander)
  [50] commander_damage_total / 21    (sum of all commander damage)
  [51] commander_damage_from_focal / 21 (damage from the focal player's commander)
  [52] commander_tax / 10             (times the player's commander has been cast)
  --- option-level (option tokens only) ---
  [53] option_detail_card    0/1  (details reference a card)
  [54] option_detail_player  0/1  (details reference a player)
  [55] option_detail_color   0/1  (details reference a color)
  [56] option_detail_number / 10  (numeric value in details)
  [57] option_detail_text_len / 50 (length of text in details)
  --- commander card-level (commander card tokens only) ---
  [58] commander_damage_to_focal / 21 (damage THIS commander dealt to the focal player)
  [59..66] commander_damage_to_player_i / 21 (damage THIS commander dealt to board.players[i], i=0..7)
  --- turn / spell-cast tracking (player + CLS tokens) ---
  [67] spells_cast_this_turn_total / 20 (all players; the Storm count)
  [68] spells_cast_this_turn_own / 20 (spells THIS player cast this turn)
  [69] spells_cast_this_turn_other / 20 (spells the other players cast this turn)
  [70] turn / 100 (game turn number)
  --- card type / identity (card tokens only) ---
  [71] type_creature     0/1
  [72] type_land         0/1
  [73] type_enchantment  0/1
  [74] type_instant      0/1
  [75] type_sorcery      0/1
  [76] type_artifact     0/1
  [77] is_token          0/1
  [78] is_legendary      0/1
  [79..93] keyword multi-hot (15 decision-relevant keywords, see KEYWORD_SLOTS)
  --- player-level aggregates (player tokens only) ---
  [94] graveyard_size / 20
  [95] exile_size / 20
  [96] land_count / 10        (lands on this player's battlefield)
  [97] creature_count / 16    (creatures on this player's battlefield)
  [98] total_power / 40       (sum of power on this player's battlefield)
  [99] total_toughness / 40   (sum of toughness on this player's battlefield)
  [100] untapped_lands / 10   (untapped lands = available mana sources)
  [101] hand_cmc_0_1 / 10     (hand cards with CMC 0-1)
  [102] hand_cmc_2_3 / 10     (hand cards with CMC 2-3)
  [103] hand_cmc_4plus / 10   (hand cards with CMC 4+)
  --- board-level (CLS token only) ---
  [104] phase_beginning  0/1  (untap / beginning of turn)
  [105] phase_main1      0/1
  [106] phase_combat     0/1  (any combat step)
  [107] phase_main2      0/1
  [108] phase_end        0/1  (end of turn)
  [109] phase_cleanup    0/1
  [110] phase_other      0/1  (mulligan / unknown)
  [111] stack_depth / 10
  [112] attackers_count / 10
  [113] blockers_count / 10
  [114] unblocked_attackers / 10
  [115] total_attacking_power / 40
  [116] lethal_damage    0/1  (unblocked attacking power >= opponent life)
  [117] avg_cmc_battlefield / 12 (avg CMC of all battlefield cards)
  [118] life_diff / 40   (focal life - opponent life, signed, clamped to [-1,1])
  [119] tempo_diff / 32  (focal battlefield size - opponent, signed, clamped)
  [120] card_advantage / 10 (focal hand - opponent hand, signed, clamped)
  --- decision-level (decision token only) ---
  [121] description_len / 100 (length of the decision prompt text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arcbound.models.board_state import BoardState
from arcbound.models.card import CardInfo
from arcbound.models.decision import AiDecisionRequest, DecisionContext, Option
from arcbound.encoder.vocabulary import (
    CardVocabulary,
    KeywordVocabulary,
    UNK_INDEX,
)

FEATURE_DIM = 122
MAX_SEQ_LEN = 2048

# Card identity embedding dimension. The model's input projection expects
# FEATURE_DIM + CARD_EMBED_DIM (+ KW_EMBED_DIM) input features per token (the
# card and keyword embeddings are concatenated onto the hand-crafted feature
# vector).
CARD_EMBED_DIM = 32
KW_EMBED_DIM = 16

# Module-level card vocabulary used to map card names -> embedding indices.
# Set by the trainer (from replays) and the server (from the saved vocab file)
# so tokenization is consistent between training and inference. When unset,
# every card maps to UNK_INDEX.
_vocab: Optional[CardVocabulary] = None
_kw_vocab: Optional[KeywordVocabulary] = None


def set_card_vocabulary(vocab: Optional[CardVocabulary]) -> None:
    """Set the active card vocabulary (None clears it)."""
    global _vocab
    _vocab = vocab


def get_card_vocabulary() -> Optional[CardVocabulary]:
    return _vocab


def card_index(name: Optional[str]) -> int:
    """Map a card name to its embedding index (UNK when no vocab is set)."""
    if _vocab is None or not name:
        return UNK_INDEX
    return _vocab.index_of(name)


def set_keyword_vocabulary(vocab: Optional[KeywordVocabulary]) -> None:
    """Set the active keyword vocabulary (None clears it)."""
    global _kw_vocab
    _kw_vocab = vocab


def get_keyword_vocabulary() -> Optional[KeywordVocabulary]:
    return _kw_vocab


def keyword_index(name: Optional[str]) -> int:
    """Map a keyword string to its embedding index (UNK when no vocab is set)."""
    if _kw_vocab is None or not name:
        return UNK_INDEX
    return _kw_vocab.index_of(name)


# Token type indices
TOK_CLS = 0
TOK_SEP = 1
TOK_PLAYER = 2
TOK_CARD = 3
TOK_DECISION = 4
TOK_OPTION = 5
TOK_DECK = 6
# Decision type → index mapping (for [20])
DECISION_TYPES = [
    "boolean", "integer", "string", "color", "card", "cards",
    "player", "damage", "shield", "manaCombo", "payment",
    "modes", "replacementEffect", "spellAbility",
    "MULLIGAN",
]

# Option type → index mapping (for [25])
OPTION_TYPES = [
    "card", "player", "color", "boolean", "integer",
    "string", "mode", "effect", "mana", "other",
]

# Zone → index mapping (for [15..19], stack at [21], plus deck at [30])
ZONE_MAP = {
    "battlefield": 15,
    "hand": 16,
    "graveyard": 17,
    "exile": 18,
    "command": 19,
    "stack": 21,
    "deck": 30,
}

# Color name → feature slot (for [32..36])
COLOR_SLOTS = {
    "white": 32,
    "blue": 33,
    "black": 34,
    "red": 35,
    "green": 36,
}

# Mana pool color → feature slot (for [44..48])
MANA_SLOTS = {
    "white": 44,
    "blue": 45,
    "black": 46,
    "red": 47,
    "green": 48,
}

# Commander damage matrix: max players in a commander game (bounded slots).
# Slots [CMDR_DMG_PLAYER_SLOT_BASE .. +MAX_PLAYERS-1] hold the damage THIS
# commander dealt to board.players[i] (i in 0..MAX_PLAYERS-1).
MAX_PLAYERS = 8
CMDR_DMG_PLAYER_SLOT_BASE = 59

# Card type → feature slot (for [71..76])
CARD_TYPE_SLOTS = {
    "creature": 71,
    "land": 72,
    "enchantment": 73,
    "instant": 74,
    "sorcery": 75,
    "artifact": 76,
}

# Decision-relevant keyword → feature slot (for [79..93]). Matched
# case-insensitively against the card's keyword list.
KEYWORD_SLOTS = {
    "flying": 79,
    "first strike": 80,
    "double strike": 81,
    "trample": 82,
    "haste": 83,
    "vigilance": 84,
    "reach": 85,
    "deathtouch": 86,
    "lifelink": 87,
    "defender": 88,
    "hexproof": 89,
    "shroud": 90,
    "ward": 91,
    "menace": 92,
    "toxic": 93,
}

# Phase name → feature slot (for [104..110]). Maps Forge PhaseType names to a
# coarse phase bucket so the model knows which actions are legal.
PHASE_SLOTS = {
    "UNTAP": 104,
    "BEGINNING": 104,
    "MAIN1": 105,
    "COMBAT_BEGIN": 106,
    "COMBAT_DECLARE_ATTACKERS": 106,
    "COMBAT_DECLARE_BLOCKERS": 106,
    "COMBAT_FIRST_STRIKE_DAMAGE": 106,
    "COMBAT_DAMAGE": 106,
    "COMBAT_END": 106,
    "MAIN2": 107,
    "END_OF_TURN": 108,
    "CLEANUP": 109,
}
PHASE_OTHER_SLOT = 110

_INT_RE = re.compile(r"-?\d+")


def _parse_int(value: Any) -> int:
    """Extract the first integer from a value (str or int), else 0."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    m = _INT_RE.search(str(value))
    return int(m.group()) if m else 0


@dataclass
class TokenizedBoard:
    """Result of tokenizing a bare board state (no decision/options)."""

    features: np.ndarray          # (L, FEATURE_DIM) float32
    card_indices: List[int]       # embedding index per position (0 for non-cards)
    keyword_indices: List[int] = field(default_factory=list)  # per-position kw idx


@dataclass
class TokenizedDecision:
    """Result of tokenizing a decision request."""

    features: np.ndarray          # (L, FEATURE_DIM) float32
    option_indices: List[int]     # positions of option tokens in the sequence
    num_options: int
    decision_type: str
    option_labels: List[str]      # labels in order, for mapping scores back
    card_indices: List[int] = field(default_factory=list)  # per-position card idx
    keyword_indices: List[int] = field(default_factory=list)  # per-position kw idx


def _new_token(token_type: int) -> np.ndarray:
    tok = np.zeros(FEATURE_DIM, dtype=np.float32)
    tok[0] = token_type
    return tok


def _card_features(
    card: CardInfo,
    zone: str,
    is_focal: bool,
    token_type: int = TOK_CARD,
    count: int = 1,
    focal_name: Optional[str] = None,
    commander_damage_to_focal: int = 0,
    commander_damage_to_players: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int, int]:
    """Return (feature vector, card embedding index, keyword index) for a card token.

    ``token_type`` defaults to TOK_CARD; pass TOK_DECK for cards that are part
    of the focal player's decklist (not currently on the board). ``count`` is
    the number of copies of this card in the deck (deck tokens only).
    ``focal_name`` is the focal player's name, used to encode ownership/control.
    """
    tok = _new_token(token_type)
    tok[1] = 1.0 if is_focal else 0.0
    cmc = card.converted_mana_cost or 0
    tok[8] = min(cmc, 12) / 12.0
    tok[9] = min(card.power or 0, 20) / 20.0
    tok[10] = min(card.toughness or 0, 20) / 20.0
    tok[11] = min(card.damage, 20) / 20.0
    tok[12] = 1.0 if card.tapped else 0.0
    tok[13] = 1.0 if card.attacking else 0.0
    tok[14] = 1.0 if card.blocking else 0.0
    zone_idx = ZONE_MAP.get(zone)
    if zone_idx is not None:
        tok[zone_idx] = 1.0
    if token_type == TOK_DECK:
        tok[31] = min(max(count, 1), 8) / 8.0

    # Colors / color identity (for [32..36])
    for color in card.colors:
        slot = COLOR_SLOTS.get(color.strip().lower())
        if slot is not None:
            tok[slot] = 1.0

    # Loyalty (for [37])
    if card.loyalty is not None:
        tok[37] = min(_parse_int(card.loyalty), 20) / 20.0

    # Counters (for [38..40])
    p1p1 = 0
    m1m1 = 0
    other = 0
    for name, amt in (card.counters or {}).items():
        key = name.strip()
        if key == "+1/+1":
            p1p1 += amt
        elif key == "-1/-1":
            m1m1 += amt
        else:
            other += amt
    tok[38] = min(p1p1, 10) / 10.0
    tok[39] = min(m1m1, 10) / 10.0
    tok[40] = min(other, 10) / 10.0

    # Ownership / control (for [41..42])
    if focal_name:
        if card.controller and card.controller == focal_name:
            tok[41] = 1.0
        if card.owner and card.owner == focal_name:
            tok[42] = 1.0

    # Card type one-hot (for [71..76]) + token/legendary flags (for [77..78])
    types_lower = {t.strip().lower() for t in card.types}
    for tname, slot in CARD_TYPE_SLOTS.items():
        if tname in types_lower:
            tok[slot] = 1.0
    supertypes_lower = {s.strip().lower() for s in card.supertypes}
    if "token" in supertypes_lower:
        tok[77] = 1.0
    if "legendary" in supertypes_lower:
        tok[78] = 1.0

    # Keyword multi-hot (for [79..93]). Forge emits parameterized keywords
    # (e.g. "Toxic 1", "Ward 3"), so match on the keyword name prefix as well
    # as the exact string.
    keywords_lower = [k.strip().lower() for k in card.keywords]
    for kw, slot in KEYWORD_SLOTS.items():
        for k in keywords_lower:
            if k == kw or k.startswith(kw + " "):
                tok[slot] = 1.0
                break

    # Per-commander damage (for [58]): how much damage THIS specific commander
    # has dealt to the focal player. The 21-damage loss rule is per-commander,
    # so the model must see each commander's individual total, not just a
    # max/sum aggregate on the player token.
    if commander_damage_to_focal:
        tok[58] = min(commander_damage_to_focal, 21) / 21.0

    # Full per-commander × per-player damage matrix (for [59..66]): how much
    # damage THIS commander dealt to each player (board.players order). This
    # lets the model see, for every commander, exactly which opponents are
    # close to the 21-damage loss threshold — not just the focal player.
    if commander_damage_to_players:
        for i, dmg in enumerate(commander_damage_to_players):
            if i >= MAX_PLAYERS:
                break
            if dmg:
                tok[CMDR_DMG_PLAYER_SLOT_BASE + i] = min(dmg, 21) / 21.0

    # Keyword index (first keyword, for the learned keyword embedding)
    kw = card.keywords[0] if card.keywords else None
    return tok, card_index(card.name), keyword_index(kw)


def _player_features(
    player_name: str,
    life: int,
    library: int,
    hand_size: int,
    battlefield_size: int,
    poison: int,
    is_active: bool,
    is_focal: bool,
    mana_pool: Optional[Dict[str, int]] = None,
    commander_damage: Optional[Dict[str, int]] = None,
    commander_tax: Optional[Dict[str, int]] = None,
    focal_commander_name: Optional[str] = None,
    spells_cast_this_turn: int = 0,
    spells_cast_this_turn_total: int = 0,
    turn: int = 0,
    graveyard_size: int = 0,
    exile_size: int = 0,
    battlefield: Optional[List[CardInfo]] = None,
    hand: Optional[List[CardInfo]] = None,
) -> np.ndarray:
    tok = _new_token(TOK_PLAYER)
    tok[1] = 1.0 if is_focal else 0.0
    tok[2] = 1.0 if is_active else 0.0
    tok[3] = min(life, 50) / 50.0
    tok[4] = min(library, 100) / 100.0
    tok[5] = min(hand_size, 10) / 10.0
    tok[6] = min(battlefield_size, 32) / 32.0
    tok[7] = min(poison, 10) / 10.0

    # Mana pool (for [43..48])
    if mana_pool:
        total = 0
        for color, amt in mana_pool.items():
            total += amt
            slot = MANA_SLOTS.get(color.strip().lower())
            if slot is not None:
                tok[slot] = min(amt, 16) / 16.0
        tok[43] = min(total, 16) / 16.0

    # Commander damage (for [49..51])
    if commander_damage:
        values = [v for v in commander_damage.values() if v]
        if values:
            tok[49] = min(max(values), 21) / 21.0
            tok[50] = min(sum(values), 21) / 21.0
        if focal_commander_name:
            from_focal = commander_damage.get(focal_commander_name, 0)
            tok[51] = min(from_focal, 21) / 21.0

    # Commander tax (for [52])
    if commander_tax:
        tok[52] = min(max(commander_tax.values()), 10) / 10.0

    # Turn / spell-cast tracking (for [67..70]). The total is the Storm count
    # (all players); "own" is this player's cast count; "other" is everyone
    # else's, so the focal player's token directly encodes the opponent's
    # activity this turn.
    tok[67] = min(spells_cast_this_turn_total, 20) / 20.0
    tok[68] = min(spells_cast_this_turn, 20) / 20.0
    tok[69] = min(max(spells_cast_this_turn_total - spells_cast_this_turn, 0), 20) / 20.0
    tok[70] = min(turn, 100) / 100.0

    # Zone sizes (for [94..95])
    tok[94] = min(graveyard_size, 20) / 20.0
    tok[95] = min(exile_size, 20) / 20.0

    # Battlefield aggregates (for [96..100])
    land_count = 0
    creature_count = 0
    total_power = 0
    total_toughness = 0
    untapped_lands = 0
    for c in (battlefield or []):
        ctypes = {t.strip().lower() for t in c.types}
        if "land" in ctypes:
            land_count += 1
            if not c.tapped:
                untapped_lands += 1
        if "creature" in ctypes:
            creature_count += 1
            total_power += c.power or 0
            total_toughness += c.toughness or 0
    tok[96] = min(land_count, 10) / 10.0
    tok[97] = min(creature_count, 16) / 16.0
    tok[98] = min(total_power, 40) / 40.0
    tok[99] = min(total_toughness, 40) / 40.0
    tok[100] = min(untapped_lands, 10) / 10.0

    # Hand mana curve (for [101..103])
    cmc_0_1 = 0
    cmc_2_3 = 0
    cmc_4plus = 0
    for c in (hand or []):
        cmc = c.converted_mana_cost or 0
        if cmc <= 1:
            cmc_0_1 += 1
        elif cmc <= 3:
            cmc_2_3 += 1
        else:
            cmc_4plus += 1
    tok[101] = min(cmc_0_1, 10) / 10.0
    tok[102] = min(cmc_2_3, 10) / 10.0
    tok[103] = min(cmc_4plus, 10) / 10.0

    return tok


def _decision_features(
    decision_type: str,
    num_options: int,
    min_choices: int,
    max_choices: int,
    allow_none: bool,
    cards_to_return: int = 0,
    mulligan_count: int = 0,
    description_len: int = 0,
) -> np.ndarray:
    tok = _new_token(TOK_DECISION)
    dt_idx = DECISION_TYPES.index(decision_type) if decision_type in DECISION_TYPES else 0
    tok[20] = dt_idx / max(len(DECISION_TYPES) - 1, 1)
    tok[21] = min(num_options, 20) / 20.0
    tok[22] = min(min_choices, 10) / 10.0
    tok[23] = min(max_choices, 10) / 10.0
    tok[24] = 1.0 if allow_none else 0.0
    # Mulligan context: how many cards would be tuck-returned if mulliganing,
    # and how many times the player has already mulliganed.
    tok[28] = min(cards_to_return, 7) / 7.0
    tok[29] = min(mulligan_count, 7) / 7.0
    # Decision prompt length (for [121])
    tok[121] = min(description_len, 100) / 100.0
    return tok


def _option_features(
    option: Option,
    option_index: int,
    num_options: int,
) -> np.ndarray:
    tok = _new_token(TOK_OPTION)
    ot = option.type.lower() if option.type else "other"
    ot_idx = OPTION_TYPES.index(ot) if ot in OPTION_TYPES else len(OPTION_TYPES) - 1
    tok[25] = ot_idx / max(len(OPTION_TYPES) - 1, 1)
    tok[26] = option_index / max(num_options, 1)
    tok[27] = 1.0 if option.details else 0.0

    # Option details content (for [53..57])
    if option.details:
        d = option.details
        if d.get("card"):
            tok[53] = 1.0
        if d.get("player"):
            tok[54] = 1.0
        if d.get("color"):
            tok[55] = 1.0
        if d.get("number") is not None:
            tok[56] = min(abs(_parse_int(d.get("number"))), 10) / 10.0
        text = d.get("text")
        if text:
            tok[57] = min(len(str(text)), 50) / 50.0
    return tok


def _board_level_features(board: BoardState) -> np.ndarray:
    """Build the CLS token carrying board-level features (for [70] and [104..120]).

    These are global game-state signals: the turn number, the current phase,
    stack depth, combat aggregates, average battlefield CMC, and the
    focal-vs-opponent relative features.
    """
    tok = _new_token(TOK_CLS)
    tok[70] = min(board.turn, 100) / 100.0

    # Phase one-hot (for [104..110])
    phase = (board.phase or "").strip().upper()
    slot = PHASE_SLOTS.get(phase)
    if slot is not None:
        tok[slot] = 1.0
    else:
        tok[PHASE_OTHER_SLOT] = 1.0

    # Stack depth (for [111])
    tok[111] = min(len(board.stack), 10) / 10.0

    # Combat aggregates (for [112..116])
    attackers: List[CardInfo] = []
    blockers: List[CardInfo] = []
    for player in board.players:
        for c in player.battlefield:
            if c.attacking:
                attackers.append(c)
            if c.blocking:
                blockers.append(c)
    tok[112] = min(len(attackers), 10) / 10.0
    tok[113] = min(len(blockers), 10) / 10.0
    unblocked = [c for c in attackers if not c.blocked]
    tok[114] = min(len(unblocked), 10) / 10.0
    tok[115] = min(sum(c.power or 0 for c in attackers), 40) / 40.0
    # Lethal: unblocked attacking power >= the minimum opponent life total.
    opponents = [p for p in board.players if p.name != board.focal_player]
    if unblocked and opponents:
        unblocked_power = sum(c.power or 0 for c in unblocked)
        if unblocked_power >= min(p.life for p in opponents):
            tok[116] = 1.0

    # Average battlefield CMC (for [117])
    all_bf = [c for p in board.players for c in p.battlefield]
    if all_bf:
        cmcs = [c.converted_mana_cost or 0 for c in all_bf]
        tok[117] = min(sum(cmcs) / len(cmcs), 12) / 12.0

    # Relative features (for [118..120])
    focal = next((p for p in board.players if p.name == board.focal_player), None)
    if focal and opponents:
        opp = opponents[0]
        tok[118] = max(-1.0, min(1.0, (focal.life - opp.life) / 40.0))
        tok[119] = max(-1.0, min(1.0, (len(focal.battlefield) - len(opp.battlefield)) / 32.0))
        tok[120] = max(-1.0, min(1.0, (len(focal.hand) - len(opp.hand)) / 10.0))

    return tok


def _build_board_tokens(
    board: BoardState, perspective: Optional[str] = None
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Build the CLS + player/card + stack tokens for a board state.

    ``perspective`` is the player whose point of view the tokens are built
    from. It controls hidden-information masking: only the perspective
    player's hand and decklist are emitted (plus any cards explicitly revealed
    to that player), and the "focal" features (is_focal, commander damage,
    relative features) are computed from that player. Defaults to the board's
    own ``focal_player``.

    Returns (tokens, card_indices, keyword_indices) where card_indices[i] and
    keyword_indices[i] are the card/keyword embedding indices for token i (0 for
    non-card tokens).

    Shared by :func:`tokenize_decision` (which appends decision/option tokens)
    and :func:`tokenize_board_state` (which does not).
    """
    # Re-focus the board on the perspective player so that every "focal"
    # feature and the hidden-zone masking below are applied from that player's
    # point of view. A shallow copy is enough: only the focal_player string
    # changes; the players list is shared.
    perspective_name = perspective or board.focal_player
    if perspective_name != board.focal_player:
        board = board.model_copy(update={"focal_player": perspective_name})

    tokens: List[np.ndarray] = []
    card_indices: List[int] = []
    keyword_indices: List[int] = []

    def _add(tok: np.ndarray, card_idx: int = 0, kw_idx: int = 0) -> None:
        tokens.append(tok)
        card_indices.append(card_idx)
        keyword_indices.append(kw_idx)

    # The focal player's commander name, used to encode "damage from the focal
    # player's commander" on every player token (the stolen-commander case).
    focal_player = next(
        (p for p in board.players if p.name == board.focal_player), None
    )
    focal_commander_name = None
    if focal_player and focal_player.commanders:
        focal_commander_name = focal_player.commanders[0].name
    # Damage each commander has dealt to the focal player (commander_name -> dmg).
    # Used to annotate each commander card token with its individual total, so
    # the model can tell which specific commander is the 21-damage threat.
    focal_commander_damage = focal_player.commander_damage if focal_player else {}

    # Full per-commander × per-player damage matrix. For a given commander name,
    # returns the damage it dealt to each player (board.players order), so each
    # commander card token can carry the complete "who is this commander about
    # to kill" picture, not just the focal player's column.
    def _cmdr_dmg_vec(card_name: str) -> List[int]:
        return [p.commander_damage.get(card_name, 0) for p in board.players]

    # CLS token (carries the turn number + board-level features so the value
    # head is turn- and phase-aware)
    _add(_board_level_features(board))
    # Player + card tokens
    for player in board.players:
        is_focal = player.name == board.focal_player
        is_active = player.name == board.active_player
        _add(_player_features(
            player_name=player.name,
            life=player.life,
            library=player.library_size,
            hand_size=len(player.hand),
            battlefield_size=len(player.battlefield),
            poison=player.poison_counters,
            is_active=is_active,
            is_focal=is_focal,
            mana_pool=player.mana_pool,
            commander_damage=player.commander_damage,
            commander_tax=player.commander_tax,
            focal_commander_name=focal_commander_name,
            spells_cast_this_turn=player.spells_cast_this_turn,
            spells_cast_this_turn_total=board.spells_cast_this_turn,
            turn=board.turn,
            graveyard_size=len(player.graveyard),
            exile_size=len(player.exile),
            battlefield=player.battlefield,
            hand=player.hand,
        ))
        for card in player.battlefield:
            tok, cidx, kidx = _card_features(card, "battlefield", is_focal, focal_name=board.focal_player,
                                             commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                                             commander_damage_to_players=_cmdr_dmg_vec(card.name))
            _add(tok, cidx, kidx)
        # Hand: hidden information. The perspective player sees its own hand;
        # any other player's hand is masked UNLESS a card is explicitly revealed
        # to the perspective player (e.g. by a card effect). The hand SIZE is
        # still visible via the player token's hand_size feature, so the model
        # knows how many cards are hidden, just not which ones.
        for card in player.hand:
            if not is_focal and board.focal_player not in (card.revealed_to or []):
                continue
            tok, cidx, kidx = _card_features(card, "hand", is_focal, focal_name=board.focal_player,
                                             commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                                             commander_damage_to_players=_cmdr_dmg_vec(card.name))
            _add(tok, cidx, kidx)
        for card in player.graveyard:
            tok, cidx, kidx = _card_features(card, "graveyard", is_focal, focal_name=board.focal_player,
                                             commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                                             commander_damage_to_players=_cmdr_dmg_vec(card.name))
            _add(tok, cidx, kidx)
        for card in player.exile:
            tok, cidx, kidx = _card_features(card, "exile", is_focal, focal_name=board.focal_player,
                                             commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                                             commander_damage_to_players=_cmdr_dmg_vec(card.name))
            _add(tok, cidx, kidx)
        for card in player.command_zone:
            tok, cidx, kidx = _card_features(card, "command", is_focal, focal_name=board.focal_player,
                                             commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                                             commander_damage_to_players=_cmdr_dmg_vec(card.name))
            _add(tok, cidx, kidx)
        # Decklist tokens (focal player only): every card in the deck that is
        # not already represented in a visible zone, so the model can attend to
        # cards it might draw in the future.
        if is_focal and player.decklist:
            seen = {c.name for zone in (
                player.battlefield, player.hand, player.graveyard,
                player.exile, player.command_zone,
            ) for c in zone}
            for card in player.decklist:
                if card.name in seen:
                    continue
                seen.add(card.name)
                tok, cidx, kidx = _card_features(
                    card, "deck", True, token_type=TOK_DECK, count=card.count,
                    focal_name=board.focal_player,
                    commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                    commander_damage_to_players=_cmdr_dmg_vec(card.name),
                )
                _add(tok, cidx, kidx)
    # Stack tokens. Labeled with the dedicated "stack" zone (not "battlefield")
    # so the model can tell a spell on the stack (counterable, about to resolve)
    # apart from a permanent sitting on the battlefield.
    for card in board.stack:
        tok, cidx, kidx = _card_features(card, "stack", False, focal_name=board.focal_player,
                                         commander_damage_to_focal=focal_commander_damage.get(card.name, 0),
                                         commander_damage_to_players=_cmdr_dmg_vec(card.name))
        _add(tok, cidx, kidx)
    return tokens, card_indices, keyword_indices


def tokenize_board_state(
    board_state: Dict[str, Any], perspective: Optional[str] = None
) -> TokenizedBoard:
    """Tokenize a raw board state dict into a feature sequence (no options).

    Used for move-based value learning: the model's value head reads the CLS
    token to estimate the board value for the focal player. This works for any
    recorded move (human, bot, or AI) because it only needs the board state,
    not the list of options that were offered.

    Args:
        board_state: Raw board state dict (as stored in a replay move).
        perspective: Player whose point of view to tokenize from (controls
            hidden-zone masking). Defaults to the board's focal player.

    Returns:
        TokenizedBoard with features and per-position card indices.
    """
    board = BoardState.model_validate(board_state)
    tokens, card_indices, keyword_indices = _build_board_tokens(board, perspective)
    tokens.append(_new_token(TOK_SEP))
    card_indices.append(0)
    keyword_indices.append(0)
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
        card_indices = card_indices[:MAX_SEQ_LEN]
        keyword_indices = keyword_indices[:MAX_SEQ_LEN]
    return TokenizedBoard(
        features=np.stack(tokens, axis=0),
        card_indices=card_indices,
        keyword_indices=keyword_indices,
    )


def tokenize_decision(
    request: AiDecisionRequest, perspective: Optional[str] = None
) -> TokenizedDecision:
    """Convert an AiDecisionRequest into a tokenized feature sequence.

    Args:
        request: The decision request from Forge.
        perspective: Player whose point of view to tokenize from (controls
            hidden-zone masking). Defaults to the board's focal player.

    Returns:
        TokenizedDecision with features, option indices, and metadata.
    """
    board = BoardState.model_validate(request.boardState)
    ctx = request.decisionRequest

    tokens, card_indices, keyword_indices = _build_board_tokens(board, perspective)
    option_indices: List[int] = []
    option_labels: List[str] = []

    # Decision context token
    constraints = ctx.constraints
    min_choices = constraints.min_choices if constraints else 0
    max_choices = constraints.max_choices if constraints else 1
    allow_none = constraints.allow_none if constraints else False
    # Mulligan context: cards tuck-returned if mulliganing, and how many times
    # the focal player has already mulliganed. Prefer the explicit value set on
    # the MULLIGAN decision context; fall back to the board state.
    cards_to_return = ctx.cards_to_return or 0
    if ctx.mulligan_count is not None:
        mulligan_count = ctx.mulligan_count
    else:
        focal = next(
            (p for p in board.players if p.name == board.focal_player), None
        )
        mulligan_count = focal.mulligan_count if focal else 0
    tokens.append(_decision_features(
        decision_type=ctx.type,
        num_options=len(ctx.options),
        min_choices=min_choices,
        max_choices=max_choices,
        allow_none=allow_none,
        cards_to_return=cards_to_return,
        mulligan_count=mulligan_count,
        description_len=len(ctx.description or ""),
    ))
    card_indices.append(0)
    keyword_indices.append(0)

    # Option tokens
    for i, opt in enumerate(ctx.options):
        option_indices.append(len(tokens))
        tokens.append(_option_features(opt, i, len(ctx.options)))
        card_indices.append(0)
        keyword_indices.append(0)
        option_labels.append(opt.label)

    # SEP token
    tokens.append(_new_token(TOK_SEP))
    card_indices.append(0)
    keyword_indices.append(0)

    # Truncate to max sequence length if needed
    if len(tokens) > MAX_SEQ_LEN:
        # Keep CLS, decision, options, SEP; truncate cards
        tokens = tokens[:MAX_SEQ_LEN]
        card_indices = card_indices[:MAX_SEQ_LEN]
        keyword_indices = keyword_indices[:MAX_SEQ_LEN]
        # Re-compute option indices (they may have been truncated)
        option_indices = [i for i in option_indices if i < len(tokens)]

    features = np.stack(tokens, axis=0)  # (L, FEATURE_DIM)

    return TokenizedDecision(
        features=features,
        option_indices=option_indices,
        num_options=len(option_indices),
        decision_type=ctx.type,
        option_labels=option_labels,
        card_indices=card_indices,
        keyword_indices=keyword_indices,
    )


def tokenize_from_dict(board_state: Dict[str, Any], decision_ctx: Dict[str, Any]) -> TokenizedDecision:
    """Tokenize from raw dicts (for replay training where we don't have pydantic models).

    Args:
        board_state: Raw board state dict.
        decision_ctx: Raw decision context dict.

    Returns:
        TokenizedDecision.
    """
    # Build a minimal AiDecisionRequest-compatible structure
    request = AiDecisionRequest(
        gameId="training",
        boardState=board_state,
        decisionRequest=DecisionContext.model_validate(decision_ctx),
    )
    return tokenize_decision(request)
