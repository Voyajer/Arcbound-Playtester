"""Training loop for the Arcbound AI model.

Reads replay files, extracts decision experiences (tokenized board states +
chosen actions), and trains the transformer model using PPO-style loss
(policy + value + entropy). Saves a checkpoint to models/<name>/model.pt
when training completes.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from arcbound.logging.replay_codec import list_replays

from arcbound.encoder.tokenizer import (
    CARD_EMBED_DIM,
    FEATURE_DIM,
    KW_EMBED_DIM,
    set_card_vocabulary,
    set_keyword_vocabulary,
    tokenize_board_state,
    tokenize_from_dict,
)
from arcbound.encoder.transformer import ArcboundModel, ModelConfig, create_model
from arcbound.encoder.vocabulary import CardVocabulary, KeywordVocabulary


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    # Weight of the TD reward (V(s') - V(s)) in the blended policy reward.
    # 0.0 = pure game-outcome reward (old behavior); 1.0 = pure TD reward.
    # The TD reward gives a dense per-step signal that correctly handles
    # "passing is sometimes the right thing to do": if passing keeps the
    # position the same, V(s') ≈ V(s) and the reward is ≈ 0 (neutral); if
    # passing makes things worse, the reward is negative; if playing a card
    # improves the position, the reward is positive.
    td_weight: float = 0.5


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    epoch: int = 0
    step: int = 0


@dataclass
class Experience:
    """A single training experience extracted from a replay decision.

    Stores the tokenized feature sequence, the option mask, the chosen action
    index, and the reward target.

    TD-reward fields: the value head is trained on *board-only* sequences
    (``tokenize_board_state``), so to compute a dense TD reward ``V(s') - V(s)``
    we store the board-only tokenization of the current state (``board_*``) and
    of the next state (``next_*``). Both are from the *same player's*
    perspective (the acting player), so the value difference is a clean
    per-step signal. This correctly handles "passing is sometimes the right
    thing to do": a good pass leaves the position unchanged, so ``V(s') ≈ V(s)``
    and the TD reward is ≈ 0 (neutral), not negative.
    """

    features: np.ndarray = field(default_factory=lambda: np.zeros((0, FEATURE_DIM), dtype=np.float32))
    option_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    action: int = 0
    reward: float = 0.0
    old_log_prob: float = 0.0
    num_options: int = 0
    card_indices: List[int] = field(default_factory=list)
    keyword_indices: List[int] = field(default_factory=list)
    # Dense TD reward V(s') - V(s), computed by compute_td_rewards() once the
    # value head is trained. 0.0 when the next state is unavailable (terminal).
    # Blended into the policy reward via TrainingConfig.td_weight.
    td_reward: float = 0.0
    # Board-only tokenization of the current state s (for V(s)).
    board_features: Optional[np.ndarray] = None
    board_card_indices: List[int] = field(default_factory=list)
    board_keyword_indices: List[int] = field(default_factory=list)
    # Board-only tokenization of the next state s' (for V(s')). Empty when the
    # decision is the last move of its game or the next state is unavailable.
    next_features: Optional[np.ndarray] = None
    next_card_indices: List[int] = field(default_factory=list)
    next_keyword_indices: List[int] = field(default_factory=list)


@dataclass
class ValueExperience:
    """A move-based value-learning experience.

    Pairs a tokenized board state (from the acting player's perspective) with a
    scalar outcome target: +1 if that player won the game, -1 if they lost,
    0 for a draw. This trains the model's value head to estimate board value
    and works for any recorded move (human, bot, or AI).
    """

    features: np.ndarray = field(default_factory=lambda: np.zeros((0, FEATURE_DIM), dtype=np.float32))
    target: float = 0.0
    card_indices: List[int] = field(default_factory=list)
    keyword_indices: List[int] = field(default_factory=list)


class ReplayBuffer:
    """Stores experiences extracted from replay files."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.experiences: List[Experience] = []

    def add(self, exp: Experience):
        if len(self.experiences) >= self.capacity:
            self.experiences.pop(0)
        self.experiences.append(exp)

    def add_many(self, experiences: List[Experience]):
        self.experiences.extend(experiences)
        if len(self.experiences) > self.capacity:
            self.experiences = self.experiences[-self.capacity:]

    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.experiences) < batch_size:
            batch_size = len(self.experiences)
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def __len__(self) -> int:
        return len(self.experiences)


def _read_replay(replay_path: Path) -> dict:
    """Read a replay file (new ``.json.gz`` or legacy ``.json``) and decode it.

    Delegates to :func:`arcbound.logging.replay_codec.read_replay`, which
    handles gzip decompression, card-table expansion, and the legacy trailing
    summary block.
    """
    from arcbound.logging.replay_codec import read_replay
    return read_replay(replay_path)


def _outcome_target(focal: Optional[str], winner: Optional[str]) -> float:
    """Map a game winner to a scalar target from ``focal``'s perspective."""
    if winner is None or focal is None:
        return 0.0
    return 1.0 if winner == focal else -1.0


def _winner_reward(
    focal: Optional[str],
    game_winner: Optional[str],
    match_weight: float,
    match_winner: Optional[str],
) -> float:
    """Blend the per-game and per-match outcome rewards for ``focal``.

    Returns a value in [-1, 1]: +1 if ``focal`` won, -1 if they lost, 0 for a
    draw or unknown. When a match winner is known, the reward is a weighted
    blend of the game outcome and the match outcome so the policy learns from
    both "did I win this game?" and "did I win the match?". When no match
    winner is available (e.g. a single-game match or legacy replay), it falls
    back to the game outcome alone.
    """
    game_r = _outcome_target(focal, game_winner)
    if match_winner is None:
        return game_r
    match_r = _outcome_target(focal, match_winner)
    return (1.0 - match_weight) * game_r + match_weight * match_r


def _player_type_allowed(
    player: Optional[str],
    player_types: Dict[str, str],
    allowed: Optional[Iterable[str]],
) -> bool:
    """Return True if ``player``'s role passes the optional type filter.

    ``allowed`` is a set of roles (e.g. ``{"ai"}`` or ``{"human", "bot"}``).
    When ``allowed`` is None/empty the filter is disabled and every player
    passes. Players with an unknown/missing role (legacy replays) pass only
    when the filter is disabled, so old data is never silently dropped.
    """
    if not allowed:
        return True
    if player is None:
        return False
    role = str(player_types.get(player, "")).strip().lower()
    return role in allowed


def _next_same_focal_board(
    moves: List[Dict[str, Any]],
    idx: int,
    focal: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return the board state of the next move (after ``idx``) in the same game
    whose focal player matches ``focal``, or None if there is none.

    The value head is trained on board-only sequences from a given player's
    perspective, and a player's hand is only visible to that player. So V(s) and
    V(s') must both be from the *same* player's perspective for the TD
    difference ``V(s') - V(s)`` to be a clean per-step signal. Moves alternate
    between players, so we skip ahead to the next state where ``focal`` is the
    focal player (the acting player's own next decision point).
    """
    if not focal:
        return None
    for j in range(idx + 1, len(moves)):
        bs = moves[j].get("board_state")
        if bs and bs.get("focal_player") == focal:
            return bs
    return None


def _aborted_cast_pick_indices(moves: List[Dict[str, Any]]) -> set:
    """Return the indices of the AI's own spell picks that were aborted.

    A pick is "aborted" when the AI chose a spell (not pass) in a
    ``SPELL_ABILITY_CHOICE`` decision and then, before any other spell pick,
    declined to pay for it via ``PAY_MANA_COST`` or ``APPLY_MANA_TO_COST``
    ("no"). This is the AI walking back its *own* cast — the exact
    "tried to cast, then decided against it" pattern that produces
    contradictory training pairs.

    Only the AI's own casts are matched:
      * Opponent effects (Rhystic Study, Mana Leak, ...) use ``PAY_TO_PREVENT``
        and are deliberately NOT matched, so the AI still learns from them.
      * The no-pay must be from the same focal player as the pick, so an
        opponent's payment decision can never abort one of the AI's picks.

    Returns a set of move indices (into ``moves``) to convert to "pass".
    """
    no_answers = {"no", "false"}
    pay_types = {"PAY_MANA_COST", "APPLY_MANA_TO_COST"}
    window = 8  # the payment decision lands a few moves after the pick
    aborted = set()
    n = len(moves)
    for i, move in enumerate(moves):
        if move.get("decision_type") != "SPELL_ABILITY_CHOICE":
            continue
        if move.get("action_taken") in (None, "", "pass"):
            continue  # already a pass; nothing to fix
        focal = move.get("player") or (move.get("board_state") or {}).get("focal_player")
        for j in range(i + 1, min(i + 1 + window, n)):
            nxt = moves[j]
            ndtype = nxt.get("decision_type")
            if ndtype == "SPELL_ABILITY_CHOICE":
                break  # a new pick happened before any payment -> not this cast
            if ndtype not in pay_types:
                continue  # skip ABILITY_CHOICE / events / other decisions
            nfocal = nxt.get("player") or (nxt.get("board_state") or {}).get("focal_player")
            if focal is not None and nfocal is not None and nfocal != focal:
                continue  # opponent's payment; keep looking for ours
            if nxt.get("action_taken") in no_answers:
                aborted.add(i)
            break
    return aborted


def _count_cast_patterns(moves: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Count cast-related patterns in a list of moves.

    Returns:
        (total_picks, total_no_pays, aborted_casts) where ``aborted_casts`` is
        the number of the AI's own spell picks that were walked back (see
        :func:`_aborted_cast_pick_indices`).
    """
    no_answers = {"no", "false"}
    pay_types = {"PAY_MANA_COST", "APPLY_MANA_TO_COST"}
    total_picks = sum(
        1 for m in moves
        if m.get("decision_type") == "SPELL_ABILITY_CHOICE"
        and m.get("action_taken") not in (None, "", "pass")
    )
    total_no_pays = sum(
        1 for m in moves
        if m.get("decision_type") in pay_types
        and m.get("action_taken") in no_answers
    )
    aborted_casts = len(_aborted_cast_pick_indices(moves))
    return total_picks, total_no_pays, aborted_casts


def extract_value_experiences(
    replay_path: Path,
    gamma: float = 0.99,
    player_types: Optional[Iterable[str]] = None,
) -> List[ValueExperience]:
    """Extract move-based value experiences from a replay file.

    Supports both the match format (``games[*].moves``) and the legacy per-game
    format (``decisions``). For each recorded move/decision that has a board
    state, the target is a discounted Monte-Carlo return of the game outcome
    from the acting player's perspective.

    ``player_types`` optionally restricts which acting players contribute
    experiences (e.g. ``{"ai"}`` to learn only from the external AI's moves, or
    ``{"human", "bot"}`` to learn from everyone else). When omitted, every
    recorded move is used — value learning works for any player type.

    The discount is applied **per game** (reset at each game boundary) so that
    the start of every game in a match carries the same weight — game 1 is not
    undervalued relative to game 3. Within a game, the last move gets the full
    outcome and earlier moves are discounted by ``gamma ** (distance_from_end)``.

    Sideboard decisions (``decision_type == "SIDEBOARD"``) happen at the start
    of games 2+ and are strategically more important than other early moves, so
    they receive the full (undiscounted) outcome rather than a discounted one.
    """
    data = _read_replay(replay_path)
    experiences: List[ValueExperience] = []
    skipped = 0
    match_meta = data.get("match") or {}
    type_map = match_meta.get("player_types") or {}
    allowed = {str(t).strip().lower() for t in player_types} if player_types else None

    def _target_for(focal: Optional[str], winner: Optional[str], t: int, n: int, is_sideboard: bool) -> float:
        outcome = _outcome_target(focal, winner)
        if is_sideboard or n <= 1:
            return outcome
        # Discount by distance from the end of the game: the last move (t = n-1)
        # gets the full outcome, earlier moves get less.
        return (gamma ** (n - 1 - t)) * outcome

    # Match format: games[*].moves, each game has its own winner.
    games = data.get("games")
    if games:
        for game in games:
            winner = game.get("winner")
            moves = game.get("moves", [])
            n = len(moves)
            for t, move in enumerate(moves):
                board_state = move.get("board_state")
                if not board_state:
                    skipped += 1
                    continue
                focal = move.get("player") or board_state.get("focal_player")
                if not _player_type_allowed(focal, type_map, allowed):
                    skipped += 1
                    continue
                try:
                    tb = tokenize_board_state(board_state)
                except Exception as e:
                    logger.debug("Failed to tokenize move in {}: {}", replay_path.name, e)
                    skipped += 1
                    continue
                is_sideboard = move.get("decision_type") == "SIDEBOARD"
                experiences.append(ValueExperience(
                    features=tb.features,
                    card_indices=tb.card_indices,
                    keyword_indices=tb.keyword_indices,
                    target=_target_for(focal, winner, t, n, is_sideboard),
                ))
        if skipped:
            logger.debug("{}: skipped {} moves without board state", replay_path.name, skipped)
        return experiences

    # Legacy per-game format: decisions, single winner in metadata.
    winner = data.get("metadata", {}).get("winner")
    decisions = data.get("decisions", [])
    n = len(decisions)
    for t, dec in enumerate(decisions):
        board_state = dec.get("board_state")
        if not board_state:
            skipped += 1
            continue
        focal = board_state.get("focal_player")
        if not _player_type_allowed(focal, type_map, allowed):
            skipped += 1
            continue
        try:
            tb = tokenize_board_state(board_state)
        except Exception as e:
            logger.debug("Failed to tokenize decision in {}: {}", replay_path.name, e)
            skipped += 1
            continue
        is_sideboard = dec.get("decision_type") == "SIDEBOARD"
        experiences.append(ValueExperience(
            features=tb.features,
            card_indices=tb.card_indices,
            target=_target_for(focal, winner, t, n, is_sideboard),
        ))
    if skipped:
        logger.debug("{}: skipped {} decisions without board state", replay_path.name, skipped)
    return experiences


def extract_experiences(
    replay_path: Path,
    match_weight: float = 0.5,
    player_types: Optional[Iterable[str]] = None,
) -> List[Experience]:
    """Extract policy training experiences from a replay file.

    Supports both the match format (``games[*].moves`` with decision-level
    records) and the legacy per-game format (top-level ``decisions``).

    Each decision record should contain:
      - board_state: the full board state dict at decision time
      - decision_request: the decision context dict (type, options, constraints)
      - action_taken: the option ID that was chosen
      - timeout: whether the decision timed out

    The reward is the winner-based outcome (per-game blended with per-match
    when a match winner is known), so the policy learns from who actually won
    rather than a flat constant. Decisions missing board_state/decision_request
    (event-based moves, older replays) are skipped.

    ``player_types`` optionally restricts which acting players contribute
    experiences (e.g. ``{"ai"}`` to learn only from the external AI's
    decisions). When omitted, every recorded decision is used.
    """
    data = _read_replay(replay_path)
    experiences: List[Experience] = []
    skipped = 0
    match_meta = data.get("match") or {}
    type_map = match_meta.get("player_types") or {}
    allowed = {str(t).strip().lower() for t in player_types} if player_types else None

    match_winner = None
    games = data.get("games")
    if games:
        match_winner = (data.get("match") or {}).get("match_winner")

    def _process(
        dec: Dict[str, Any],
        game_winner: Optional[str],
        next_board: Optional[Dict[str, Any]] = None,
        aborted_cast: bool = False,
    ) -> Optional[Experience]:
        board_state = dec.get("board_state")
        decision_request = dec.get("decision_request")
        if not board_state or not decision_request:
            return None

        options = decision_request.get("options", [])
        action_id = dec.get("action_taken")
        if not options or action_id is None:
            return None

        # Aborted-cast fix: when the AI picked a spell it then declined to pay
        # for (its OWN cast — opponent effects like Rhystic Study use
        # PAY_TO_PREVENT and are never flagged here), the recorded "pick spell"
        # action contradicts the recorded "don't pay" action. Train the pick as
        # a "pass" instead, so the model learns "don't pick a spell you won't
        # cast" rather than reproducing the pick-then-walk-back loop.
        if aborted_cast and action_id != "pass":
            action_id = "pass"

        # Find the index of the chosen option
        action_idx = None
        for i, opt in enumerate(options):
            if opt.get("id") == action_id:
                action_idx = i
                break
        if action_idx is None:
            return None

        try:
            tokenized = tokenize_from_dict(board_state, decision_request)
        except Exception as e:
            logger.debug("Failed to tokenize decision in {}: {}", replay_path.name, e)
            return None

        if tokenized.num_options == 0:
            return None

        # Build option mask
        L = tokenized.features.shape[0]
        option_mask = np.zeros(L, dtype=bool)
        for idx in tokenized.option_indices:
            if idx < L:
                option_mask[idx] = True

        # Reward: winner-based outcome (game blended with match when known).
        # A timeout is a failed decision, so it gets a negative reward.
        focal = dec.get("player") or board_state.get("focal_player")
        if not _player_type_allowed(focal, type_map, allowed):
            return None
        if dec.get("timeout", False):
            reward = -1.0
        else:
            reward = _winner_reward(focal, game_winner, match_weight, match_winner)

        # Old log-prob from recorded model confidence (if any)
        confidence = dec.get("model_confidence")
        if confidence is not None and confidence > 0:
            old_log_prob = float(np.log(max(confidence, 1e-7)))
        else:
            old_log_prob = float(np.log(1.0 / max(tokenized.num_options, 1)))

        # Board-only tokenization of the current state s (for V(s)). The value
        # head is trained on board-only sequences, so the TD reward must use the
        # same representation, not the decision sequence (which includes the
        # option tokens that are absent from value training).
        board_tb = None
        try:
            board_tb = tokenize_board_state(board_state)
        except Exception as e:
            logger.debug("Failed to tokenize board state in {}: {}", replay_path.name, e)

        # Board-only tokenization of the next state s' (for V(s')), from the
        # same player's perspective. None when this is the last same-focal
        # state in the game (terminal for TD purposes).
        next_tb = None
        if next_board is not None:
            try:
                next_tb = tokenize_board_state(next_board)
            except Exception as e:
                logger.debug("Failed to tokenize next board state in {}: {}", replay_path.name, e)

        return Experience(
            features=tokenized.features,
            option_mask=option_mask,
            action=action_idx,
            reward=reward,
            old_log_prob=old_log_prob,
            num_options=tokenized.num_options,
            card_indices=tokenized.card_indices,
            keyword_indices=tokenized.keyword_indices,
            board_features=board_tb.features if board_tb else None,
            board_card_indices=board_tb.card_indices if board_tb else [],
            board_keyword_indices=board_tb.keyword_indices if board_tb else [],
            next_features=next_tb.features if next_tb else None,
            next_card_indices=next_tb.card_indices if next_tb else [],
            next_keyword_indices=next_tb.keyword_indices if next_tb else [],
        )

    if games:
        # Match format: decision-level records live in games[*].moves.
        for game in games:
            game_winner = game.get("winner")
            moves = game.get("moves", [])
            # Aborted-cast fix: find the AI's own spell picks that were walked
            # back (pick then PAY_MANA_COST "no"). Those picks are retrained as
            # "pass" so the model stops learning the pick-then-refuse loop.
            # Opponent effects (PAY_TO_PREVENT) are never flagged.
            aborted_picks = _aborted_cast_pick_indices(moves)
            picks, no_pays, aborted = _count_cast_patterns(moves)
            if aborted:
                logger.info(
                    "{}: game {} has {} aborted cast(s) "
                    "(spell picked then mana cost declined) out of {} spell picks / {} no-pays; "
                    "those picks are retrained as 'pass'",
                    replay_path.name, game.get("game_number"), aborted, picks, no_pays,
                )
            for i, move in enumerate(moves):
                # Find the next same-focal board state for the TD reward.
                focal = move.get("player") or (move.get("board_state") or {}).get("focal_player")
                next_board = _next_same_focal_board(moves, i, focal)
                exp = _process(move, game_winner, next_board, aborted_cast=(i in aborted_picks))
                if exp is None:
                    skipped += 1
                else:
                    experiences.append(exp)
    else:
        # Legacy per-game format: top-level decisions, single winner in metadata.
        game_winner = data.get("metadata", {}).get("winner")
        decisions = data.get("decisions", [])
        aborted_picks = _aborted_cast_pick_indices(decisions)
        for i, dec in enumerate(decisions):
            exp = _process(dec, game_winner, aborted_cast=(i in aborted_picks))
            if exp is None:
                skipped += 1
            else:
                experiences.append(exp)

    if skipped:
        logger.debug("{}: skipped {} records without full decision state", replay_path.name, skipped)
    return experiences


def _collate_batch(batch: List[Experience]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a batch of variable-length experiences into padded tensors.

    Returns:
        features: (B, L_max, F) padded feature sequences
        option_mask: (B, L_max) bool mask of option positions
        actions: (B,) chosen action index per experience
        rewards: (B,) game-outcome reward targets (used for value + Q loss)
        old_log_probs: (B,) old log-probabilities
        card_indices: (B, L_max) per-position card vocabulary indices (0 = UNK)
        keyword_indices: (B, L_max) per-position keyword vocabulary indices (0 = UNK)
        td_rewards: (B,) dense TD reward V(s')-V(s) per experience (0.0 when
            the next state is unavailable). Blended into the policy reward via
            TrainingConfig.td_weight.
    """
    B = len(batch)
    L_max = max(e.features.shape[0] for e in batch)
    F_dim = batch[0].features.shape[1]

    features = torch.zeros(B, L_max, F_dim, dtype=torch.float32)
    option_mask = torch.zeros(B, L_max, dtype=torch.bool)
    actions = torch.zeros(B, dtype=torch.long)
    rewards = torch.zeros(B, dtype=torch.float32)
    old_log_probs = torch.zeros(B, dtype=torch.float32)
    card_indices = torch.zeros(B, L_max, dtype=torch.long)
    keyword_indices = torch.zeros(B, L_max, dtype=torch.long)
    td_rewards = torch.zeros(B, dtype=torch.float32)

    for b, e in enumerate(batch):
        L = e.features.shape[0]
        features[b, :L] = torch.from_numpy(e.features)
        option_mask[b, :L] = torch.from_numpy(e.option_mask)
        actions[b] = e.action
        rewards[b] = e.reward
        old_log_probs[b] = e.old_log_prob
        td_rewards[b] = e.td_reward
        if e.card_indices:
            card_indices[b, : len(e.card_indices)] = torch.tensor(e.card_indices[:L], dtype=torch.long)
        if e.keyword_indices:
            keyword_indices[b, : len(e.keyword_indices)] = torch.tensor(e.keyword_indices[:L], dtype=torch.long)

    return features, option_mask, actions, rewards, old_log_probs, card_indices, keyword_indices, td_rewards


class Trainer:
    """Manages the training loop.

    Trains an ArcboundModel on experiences extracted from replays and saves
    a checkpoint when done. Callbacks allow the GUI to receive progress updates.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model: Optional[ArcboundModel] = None,
        model_config: Optional[ModelConfig] = None,
        vocab: Optional[CardVocabulary] = None,
        kw_vocab: Optional[KeywordVocabulary] = None,
        on_progress: Optional[Callable[[int, float, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_metrics: Optional[Callable[[Dict[str, float]], None]] = None,
        on_load_progress: Optional[Callable[[int, int, str], None]] = None,
    ):
        self.config = config or TrainingConfig()
        self.buffer = ReplayBuffer()
        self.value_buffer: List[ValueExperience] = []
        self.on_progress = on_progress
        self.on_log = on_log
        # Per-batch metrics callback (policy + value + entropy + total). The GUI
        # uses it to display running weighted averages live during training.
        self.on_metrics = on_metrics
        # Per-file progress callback for the (slow) replay-loading phase:
        # (current_index, total_files, filename). Lets the GUI show a live
        # "Loading replays (i/n)" indicator instead of freezing.
        self.on_load_progress = on_load_progress
        self._running = False
        # Set by stop() so the (slow) replay-loading loops can bail out early
        # when the user clicks Stop during the prep phase. Independent of
        # _running (which train()/train_value() manage themselves).
        self._stop_requested = False
        self._metrics: Optional[TrainingMetrics] = None
        self.vocab = vocab
        self.kw_vocab = kw_vocab
        if vocab is not None:
            set_card_vocabulary(vocab)
            if model_config is not None:
                model_config.card_vocab_size = len(vocab)
                model_config.card_embed_dim = CARD_EMBED_DIM
        if kw_vocab is not None:
            set_keyword_vocabulary(kw_vocab)
            if model_config is not None:
                model_config.kw_vocab_size = len(kw_vocab)
                model_config.kw_embed_dim = KW_EMBED_DIM
        self.model = model or create_model(model_config)
        self.total_steps = 0
        self._model_dir: Optional[Path] = None
        # Cumulative game tracking: replay filenames trained on across runs,
        # persisted in the model's config.json as "trained_games".
        self._trained_games: set = set()
        self._loaded_games: set = set()
        # Set to True once the value head has been trained (train_value) or a
        # checkpoint with a trained value head has been loaded. compute_td_rewards
        # is only meaningful when the value head is trained, so train() uses this
        # flag to decide whether to compute TD rewards before policy training.
        self._value_trained: bool = False

    def _log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def load_replays(
        self,
        replay_paths: List[Path],
        match_weight: float = 0.5,
        player_types: Optional[Iterable[str]] = None,
    ):
        """Load experiences from replay files into the buffer.

        ``player_types`` optionally restricts which acting players contribute
        experiences (e.g. ``{"ai"}`` to learn only from the external AI's
        decisions). When omitted, every recorded decision is used.

        Emits per-file progress via ``on_load_progress`` (if set) and stops
        early if ``stop()`` is called mid-load.
        """
        total = 0
        n = len(replay_paths)
        for i, rp in enumerate(replay_paths):
            if self._stop_requested:
                self._log("Replay loading stopped by user.")
                break
            if self.on_load_progress:
                self.on_load_progress(i + 1, n, rp.name)
            self._loaded_games.add(rp.name)
            exps = extract_experiences(rp, match_weight=match_weight, player_types=player_types)
            self.buffer.add_many(exps)
            total += len(exps)
            self._log(f"Loaded {len(exps)} experiences from {rp.name}")
        self._log(f"Total experiences in buffer: {len(self.buffer)}")

    def load_replay_dir(
        self,
        replay_dir: Path,
        match_weight: float = 0.5,
        player_types: Optional[Iterable[str]] = None,
    ):
        """Load all replay files from a directory."""
        paths = list_replays(replay_dir)
        if not paths:
            self._log(f"No replay files found in {replay_dir}")
            return
        self.load_replays(paths, match_weight=match_weight, player_types=player_types)

    def load_replay_dir_all(
        self,
        replay_dir: Path,
        match_weight: float = 0.5,
        gamma: float = 0.99,
        player_types: Optional[Iterable[str]] = None,
    ):
        """Load BOTH policy and value experiences from all replay files in a directory.

        Emits a single combined per-file and total summary so the caller can see
        how many policy and value experiences were loaded side by side. Policy
        experiences require decision-level records (only present when the external
        AI played); value experiences work for any recorded move (human, bot, or
        AI), so a human-vs-bot replay yields 0 policy but N value experiences.
        """
        paths = list_replays(replay_dir)
        if not paths:
            self._log(f"No replay files found in {replay_dir}")
            return
        total_policy = 0
        total_value = 0
        for rp in paths:
            self._loaded_games.add(rp.name)
            policy = extract_experiences(rp, match_weight=match_weight, player_types=player_types)
            value = extract_value_experiences(rp, gamma=gamma, player_types=player_types)
            self.buffer.add_many(policy)
            self.value_buffer.extend(value)
            total_policy += len(policy)
            total_value += len(value)
            self._log(f"Loaded {len(policy)} policy + {len(value)} value experiences from {rp.name}")
        self._log(f"Total: {total_policy} policy + {total_value} value experiences in buffer")

    def _estimate_values(self, exps: List[Experience]) -> List[Optional[float]]:
        """Run the value head over a list of experiences' board states.

        Returns a list of V(s) estimates (one per experience, None when the
        experience has no board tokenization). Uses the model in eval mode with
        no grad, so it does not disturb the training state.
        """
        self.model.eval()
        results: List[Optional[float]] = []
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(exps), batch_size):
                chunk = exps[start:start + batch_size]
                # Collect the board states that have a tokenization.
                valid = [(i, e) for i, e in enumerate(chunk) if e.board_features is not None]
                chunk_vals: List[Optional[float]] = [None] * len(chunk)
                if not valid:
                    results.extend(chunk_vals)
                    continue
                L_max = max(e.board_features.shape[0] for _, e in valid)
                F_dim = valid[0][1].board_features.shape[1]
                feats = torch.zeros(len(valid), L_max, F_dim, dtype=torch.float32)
                cidx = torch.zeros(len(valid), L_max, dtype=torch.long)
                kidx = torch.zeros(len(valid), L_max, dtype=torch.long)
                for b, (_, e) in enumerate(valid):
                    L = e.board_features.shape[0]
                    feats[b, :L] = torch.from_numpy(e.board_features)
                    if e.board_card_indices:
                        cidx[b, : min(len(e.board_card_indices), L)] = torch.tensor(
                            e.board_card_indices[:L], dtype=torch.long
                        )
                    if e.board_keyword_indices:
                        kidx[b, : min(len(e.board_keyword_indices), L)] = torch.tensor(
                            e.board_keyword_indices[:L], dtype=torch.long
                        )
                _, value, _ = self.model(feats, None, cidx, kidx)
                for b, (i, _) in enumerate(valid):
                    chunk_vals[i] = float(value[b, 0].item())
                results.extend(chunk_vals)
        self.model.train()
        return results

    def compute_td_rewards(self) -> int:
        """Compute dense TD rewards V(s') - V(s) for all policy experiences.

        Requires the value head to be trained (``self._value_trained``). For each
        experience that has both a current and a next board tokenization, the TD
        reward is the change in the value head's estimate between the two states
        (both from the same player's perspective). Experiences without a next
        state (terminal) get a TD reward of 0.0.

        The TD reward is stored on each experience (``exp.td_reward``) and is
        blended into the policy reward in ``train()`` via
        ``TrainingConfig.td_weight``. This gives the policy a dense per-step
        signal that correctly handles "passing is sometimes the right thing to
        do": a good pass leaves the position unchanged, so the TD reward is ≈ 0
        (neutral), not negative.

        Returns:
            The number of experiences that received a non-zero TD reward.
        """
        if not self._value_trained:
            self._log("compute_td_rewards: value head not trained; skipping.")
            return 0
        exps = self.buffer.experiences
        if not exps:
            return 0

        # Estimate V(s) for every experience that has a board tokenization.
        v_s = self._estimate_values(exps)

        # Estimate V(s') for every experience that has a next tokenization.
        # Build a flat list of (exp_index, next_tb) and batch them.
        next_items: List[Tuple[int, np.ndarray, List[int], List[int]]] = []
        for i, e in enumerate(exps):
            if e.next_features is not None:
                next_items.append((i, e.next_features, e.next_card_indices, e.next_keyword_indices))

        v_sp: Dict[int, float] = {}
        if next_items:
            self.model.eval()
            batch_size = 64
            with torch.no_grad():
                for start in range(0, len(next_items), batch_size):
                    chunk = next_items[start:start + batch_size]
                    L_max = max(item[1].shape[0] for item in chunk)
                    F_dim = chunk[0][1].shape[1]
                    feats = torch.zeros(len(chunk), L_max, F_dim, dtype=torch.float32)
                    cidx = torch.zeros(len(chunk), L_max, dtype=torch.long)
                    kidx = torch.zeros(len(chunk), L_max, dtype=torch.long)
                    for b, (_, nf, nci, nki) in enumerate(chunk):
                        L = nf.shape[0]
                        feats[b, :L] = torch.from_numpy(nf)
                        if nci:
                            cidx[b, : min(len(nci), L)] = torch.tensor(nci[:L], dtype=torch.long)
                        if nki:
                            kidx[b, : min(len(nki), L)] = torch.tensor(nki[:L], dtype=torch.long)
                    _, value, _ = self.model(feats, None, cidx, kidx)
                    for b, (exp_i, _, _, _) in enumerate(chunk):
                        v_sp[exp_i] = float(value[b, 0].item())
            self.model.train()

        # Compute the TD reward for each experience.
        nonzero = 0
        for i, e in enumerate(exps):
            vs = v_s[i]
            vsp = v_sp.get(i)
            if vs is not None and vsp is not None:
                e.td_reward = vsp - vs
                if abs(e.td_reward) > 1e-6:
                    nonzero += 1
            else:
                e.td_reward = 0.0
        self._log(
            f"TD rewards computed: {nonzero}/{len(exps)} experiences have a non-zero "
            f"V(s')-V(s) signal."
        )
        return nonzero

    def train(self) -> Optional[Path]:
        """Run the training loop.

        If the value head is trained (``self._value_trained``) and
        ``TrainingConfig.td_weight > 0``, computes dense TD rewards
        ``V(s') - V(s)`` for all experiences first, then blends them into the
        policy reward. This gives the policy a per-step signal that correctly
        handles "passing is sometimes the right thing to do."

        Returns:
            Path to the saved checkpoint, or None if training did not run.
        """
        if len(self.buffer) == 0:
            self._log("No experiences in buffer. Load replays first.")
            return None

        # Compute dense TD rewards before policy training (requires a trained
        # value head). Blended into the reward via cfg.td_weight in the loop.
        if self._value_trained and self.config.td_weight > 0:
            self.compute_td_rewards()

        self._running = True
        cfg = self.config
        num_batches = max(1, len(self.buffer) // cfg.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)

        self._log(
            f"Training started: {len(self.buffer)} experiences, {cfg.epochs} epochs, "
            f"batch_size={cfg.batch_size}, model params={self.model.num_parameters():,}"
        )

        global_step = 0
        best_loss = float("inf")
        ckpt_path: Optional[Path] = None

        for epoch in range(cfg.epochs):
            if not self._running:
                self._log("Training stopped by user.")
                break

            epoch_loss = 0.0
            epoch_policy = 0.0
            epoch_value = 0.0
            epoch_entropy = 0.0

            for batch_idx in range(num_batches):
                if not self._running:
                    break

                batch = self.buffer.sample(cfg.batch_size)
                features, option_mask, actions, rewards, old_log_probs, card_indices, keyword_indices, td_rewards = _collate_batch(batch)

                optimizer.zero_grad()
                option_logits, value, q_values = self.model(features, option_mask, card_indices, keyword_indices)

                # Mask out padded option positions (beyond each experience's real count)
                # Build per-row valid-option mask
                B = features.shape[0]
                valid = torch.zeros(B, option_logits.shape[1], dtype=torch.bool)
                for b, e in enumerate(batch):
                    valid[b, :e.num_options] = True

                # Blend the dense TD reward into the policy reward. The game
                # outcome (``rewards``) is the sparse terminal signal; the TD
                # reward (``td_rewards`` = V(s')-V(s)) is a dense per-step signal
                # that correctly handles "passing is sometimes the right thing to
                # do" (a good pass leaves the position unchanged, so its TD reward
                # is ≈ 0, not negative). td_weight=0.0 recovers the old behavior.
                policy_reward = (1.0 - cfg.td_weight) * rewards + cfg.td_weight * td_rewards

                log_probs = F.log_softmax(option_logits, dim=-1)
                # Gather the chosen action's log-prob
                action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(action_log_probs - old_log_probs)
                surr1 = ratio * policy_reward
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * policy_reward
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (value is (B, 1, 1) -> reshape to (B,)). The value
                # head predicts the game outcome, so it stays on the sparse
                # terminal reward (not the TD blend).
                value_loss = F.mse_loss(value.reshape(B), rewards)

                # Q loss: the option-conditioned Q estimate of the chosen
                # action should match the outcome reward. This is what lets
                # the value signal influence option selection at inference.
                # Also stays on the game outcome (the Q head is an outcome
                # estimator, not a per-step TD estimator).
                q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                q_loss = F.mse_loss(q_action, rewards)

                # Entropy bonus (only over valid options)
                prob = log_probs.exp()
                entropy_per = -(prob * log_probs).sum(dim=-1)
                entropy = entropy_per[valid.any(dim=1)].mean() if valid.any() else torch.tensor(0.0)

                total_loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.value_coef * q_loss
                    - cfg.entropy_coef * entropy
                )
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                m = TrainingMetrics(
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    entropy=entropy.item(),
                    total_loss=total_loss.item(),
                    epoch=epoch,
                    step=global_step,
                )
                self._metrics = m
                epoch_loss += m.total_loss
                epoch_policy += m.policy_loss
                epoch_value += m.value_loss
                epoch_entropy += m.entropy
                global_step += 1
                self.total_steps += 1

                # Emit per-batch metrics so the GUI can show running weighted
                # averages live (not just once per epoch).
                if self.on_metrics:
                    self.on_metrics({
                        "phase": "policy",
                        "policy_loss": m.policy_loss,
                        "value_loss": m.value_loss,
                        "entropy": m.entropy,
                        "total_loss": m.total_loss,
                        "epoch": epoch + 1,
                        "step": global_step,
                    })

            avg_loss = epoch_loss / max(num_batches, 1)
            progress = (epoch + 1) / cfg.epochs * 100
            msg = (
                f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | "
                f"Policy: {epoch_policy/max(num_batches,1):.4f} | "
                f"Value: {epoch_value/max(num_batches,1):.4f} | "
                f"Entropy: {epoch_entropy/max(num_batches,1):.4f}"
            )
            self._log(msg)

            if self.on_progress:
                self.on_progress(progress, avg_loss, msg)

            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = self._save_checkpoint(tag="best_model.pt")

        self._running = False
        self._log(f"Training complete. Final step: {global_step}")

        # Save final checkpoint
        final_path = self._save_checkpoint(tag="model.pt")
        return final_path or ckpt_path

    def train_all(self) -> Optional[Path]:
        """Run value training (if any value experiences) then policy training
        (if any policy experiences). Mirrors the CLI's ``--mode both``.

        Value is trained first so the value head is available to compute dense
        TD rewards (V(s') - V(s)) for the policy training that follows. Each
        head is skipped when its buffer is empty, so a human-vs-bot replay
        (0 policy, N value) still trains the value head, and an AI replay with
        decision records trains both. Returns the last saved checkpoint path, or
        None if no model directory is configured (training still runs).
        """
        result: Optional[Path] = None
        trained_any = False
        if self.value_buffer:
            v = self.train_value()
            if v is not None:
                result = v
            trained_any = True
        if len(self.buffer) > 0:
            p = self.train()
            if p is not None:
                result = p
            trained_any = True
        if not trained_any:
            self._log("Nothing to train: no policy or value experiences loaded.")
        return result

    def load_value_replays(
        self,
        replay_paths: List[Path],
        gamma: float = 0.99,
        player_types: Optional[Iterable[str]] = None,
    ):
        """Load move-based value experiences from replay files.

        ``player_types`` optionally restricts which acting players contribute
        experiences (e.g. ``{"ai"}`` to learn only from the external AI's
        moves). When omitted, every recorded move is used.

        Emits per-file progress via ``on_load_progress`` (if set) and stops
        early if ``stop()`` is called mid-load.
        """
        total = 0
        n = len(replay_paths)
        for i, rp in enumerate(replay_paths):
            if self._stop_requested:
                self._log("Replay loading stopped by user.")
                break
            if self.on_load_progress:
                self.on_load_progress(i + 1, n, rp.name)
            self._loaded_games.add(rp.name)
            exps = extract_value_experiences(rp, gamma=gamma, player_types=player_types)
            self.value_buffer.extend(exps)
            total += len(exps)
            self._log(f"Loaded {len(exps)} value experiences from {rp.name}")
        self._log(f"Total value experiences: {len(self.value_buffer)}")

    def load_value_replay_dir(
        self,
        replay_dir: Path,
        gamma: float = 0.99,
        player_types: Optional[Iterable[str]] = None,
    ):
        """Load move-based value experiences from all replay files in a directory."""
        paths = list_replays(replay_dir)
        if not paths:
            self._log(f"No replay files found in {replay_dir}")
            return
        self.load_value_replays(paths, gamma=gamma, player_types=player_types)

    def train_value(self) -> Optional[Path]:
        """Train the value head on move-based experiences.

        Each experience is (board_state, outcome) where outcome is +1/-1/0 from
        the acting player's perspective. The model's value head (CLS token) is
        trained with MSE to predict the outcome. This teaches the model to
        evaluate board positions and works for any recorded move (human, bot,
        or AI), including games the AI did not play.

        Returns:
            Path to the saved checkpoint, or None if training did not run.
        """
        if not self.value_buffer:
            self._log("No value experiences in buffer. Load replays first.")
            return None

        self._running = True
        cfg = self.config
        num_batches = max(1, len(self.value_buffer) // cfg.batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)

        self._log(
            f"Value training started: {len(self.value_buffer)} experiences, "
            f"{cfg.epochs} epochs, batch_size={cfg.batch_size}"
        )

        global_step = 0
        best_loss = float("inf")
        ckpt_path: Optional[Path] = None

        for epoch in range(cfg.epochs):
            if not self._running:
                self._log("Training stopped by user.")
                break

            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                if not self._running:
                    break

                indices = np.random.choice(len(self.value_buffer), cfg.batch_size, replace=True)
                batch = [self.value_buffer[i] for i in indices]

                B = len(batch)
                L_max = max(e.features.shape[0] for e in batch)
                F_dim = batch[0].features.shape[1]
                features = torch.zeros(B, L_max, F_dim, dtype=torch.float32)
                targets = torch.zeros(B, dtype=torch.float32)
                card_indices = torch.zeros(B, L_max, dtype=torch.long)
                keyword_indices = torch.zeros(B, L_max, dtype=torch.long)
                for b, e in enumerate(batch):
                    L = e.features.shape[0]
                    features[b, :L] = torch.from_numpy(e.features)
                    targets[b] = e.target
                    if e.card_indices:
                        card_indices[b, : len(e.card_indices)] = torch.tensor(e.card_indices[:L], dtype=torch.long)
                    if e.keyword_indices:
                        keyword_indices[b, : len(e.keyword_indices)] = torch.tensor(e.keyword_indices[:L], dtype=torch.long)

                with torch.set_grad_enabled(True):
                    self.model.train()
                    _, value, _ = self.model(features, None, card_indices, keyword_indices)
                    loss = F.mse_loss(value.reshape(B), targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1
                self.total_steps += 1

                # Emit per-batch metrics so the GUI can show running weighted
                # averages live (not just once per epoch).
                if self.on_metrics:
                    self.on_metrics({
                        "phase": "value",
                        "value_loss": loss.item(),
                        "total_loss": loss.item(),
                        "epoch": epoch + 1,
                        "step": global_step,
                    })

            avg_loss = epoch_loss / max(num_batches, 1)
            progress = (epoch + 1) / cfg.epochs * 100
            msg = f"Value epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f}"
            self._log(msg)
            if self.on_progress:
                self.on_progress(progress, avg_loss, msg)

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = self._save_checkpoint(tag="best_model.pt")

        self._running = False
        self._log(f"Value training complete. Final step: {global_step}")
        # The value head is now trained, so TD rewards can be computed for
        # subsequent policy training runs.
        self._value_trained = True
        final_path = self._save_checkpoint(tag="model.pt")
        return final_path or ckpt_path

    def _save_checkpoint(self, tag: str = "model.pt") -> Optional[Path]:
        """Save the current model state to a checkpoint file.

        The checkpoint is written to models/<model_name>/<tag> if a model
        directory is configured, otherwise returned as None.
        """
        if not self.model_dir:
            return None
        self.model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.model_dir / tag

        # Merge this run's replay files into the cumulative trained-games set.
        # Idempotent across the multiple checkpoint saves within one run.
        self._trained_games.update(self._loaded_games)

        checkpoint = {
            "model_config": self.model.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "total_steps": self.total_steps,
            "games_trained": len(self._trained_games),
            "saved_at": datetime.now().isoformat(),
        }
        torch.save(checkpoint, ckpt_path)

        # Save the card and keyword vocabularies alongside the checkpoint so
        # the server can load them at startup for inference.
        if self.vocab is not None:
            try:
                self.vocab.save(self.model_dir / "vocab.json")
            except Exception as e:
                self._log(f"Warning: could not save vocabulary: {e}")
        if self.kw_vocab is not None:
            try:
                self.kw_vocab.save(self.model_dir / "kw_vocab.json")
            except Exception as e:
                self._log(f"Warning: could not save keyword vocabulary: {e}")

        # Update config.json with training stats
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                cfg["modified"] = datetime.now().isoformat()
                cfg["total_steps"] = self.total_steps
                cfg["encoder"] = self.model.config.to_dict()
                cfg["games_trained"] = len(self._trained_games)
                cfg["trained_games"] = sorted(self._trained_games)
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                self._log(
                    f"Stats updated: games_trained={cfg['games_trained']}, total_steps={cfg['total_steps']}"
                )
            except Exception as e:
                self._log(f"Warning: could not update config.json: {e}")

        self._log(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path

    @property
    def model_dir(self) -> Optional[Path]:
        """Directory where checkpoints are saved (set via set_model_dir)."""
        return self._model_dir

    def set_model_dir(self, model_dir: Path):
        """Set the directory where checkpoints will be saved.

        Also restores cumulative stats (total_steps, trained games) from the
        model's config.json so they keep accumulating across training runs.
        """
        self._model_dir = model_dir
        self._restore_stats()

    def _restore_stats(self):
        """Restore cumulative training stats from the model's config.json."""
        if not self._model_dir:
            return
        config_path = self._model_dir / "config.json"
        if not config_path.exists():
            return
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            self.total_steps = int(cfg.get("total_steps") or 0)
            self._trained_games = set(cfg.get("trained_games") or [])
            if self.total_steps or self._trained_games:
                self._log(
                    f"Restored stats from {config_path.name}: "
                    f"total_steps={self.total_steps}, games_trained={len(self._trained_games)}"
                )
        except Exception as e:
            self._log(f"Warning: could not restore stats from {config_path}: {e}")

    def stop(self):
        self._running = False
        # Also signal the (slow) replay-loading loops to bail out early, in case
        # Stop is clicked during the prep phase before training has started.
        self._stop_requested = True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def metrics(self) -> Optional[TrainingMetrics]:
        return self._metrics


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Arcbound AI model")
    parser.add_argument("--replays", type=str, help="Path to replays directory")
    parser.add_argument("--model", type=str, default="Default", help="Model name (directory under models/)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--mode",
        choices=["policy", "value", "both"],
        default="both",
        help="policy: option-selection PPO; value: move-based board evaluation; both: run both",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for per-game Monte-Carlo value returns (applied within each game)",
    )
    parser.add_argument(
        "--match-weight",
        type=float,
        default=0.5,
        help="weight of the match outcome in the policy reward (0 = game only, 1 = match only)",
    )
    parser.add_argument(
        "--player-types",
        type=str,
        default=None,
        help=(
            "comma-separated player roles to learn from (e.g. 'ai', 'human,bot'). "
            "Defaults to all players. Roles: human, bot, ai."
        ),
    )
    args = parser.parse_args()

    player_types = None
    if args.player_types:
        player_types = {t.strip().lower() for t in args.player_types.split(",") if t.strip()}

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    models_dir = project_root / "models"

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    replay_dir = Path(args.replays) if args.replays else project_root / "replays"
    if not replay_dir.exists():
        logger.warning("No replays directory found. Nothing to train on.")
        return

    # Build the card vocabulary from the replays so the model can learn a
    # per-card identity embedding. An empty vocab (no cards found) disables it.
    vocab = CardVocabulary.build_from_replays(replay_dir)
    if len(vocab) <= 1:
        logger.warning("No card names found in replays; card identity embedding disabled.")
        vocab = None
    else:
        logger.info(f"Card vocabulary: {len(vocab) - 1} unique cards")

    # Build the keyword vocabulary from the replays so the model can learn a
    # per-keyword embedding (Flying, Trample, ...). An empty vocab disables it.
    kw_vocab = KeywordVocabulary.build_from_replays(replay_dir)
    if len(kw_vocab) <= 1:
        logger.warning("No keywords found in replays; keyword embedding disabled.")
        kw_vocab = None
    else:
        logger.info(f"Keyword vocabulary: {len(kw_vocab) - 1} unique keywords")

    # Load existing model config if present
    model_config = None
    config_path = models_dir / args.model / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            raw = json.load(f)
        model_config = ModelConfig.from_dict(raw.get("encoder", {}))

    trainer = Trainer(config=cfg, model_config=model_config, vocab=vocab, kw_vocab=kw_vocab)
    trainer.set_model_dir(models_dir / args.model)

    if args.mode in ("policy", "both"):
        trainer.load_replay_dir(replay_dir, match_weight=args.match_weight, player_types=player_types)
    if args.mode in ("value", "both"):
        trainer.load_value_replay_dir(replay_dir, gamma=args.gamma, player_types=player_types)

    # Value is trained FIRST so the value head is available to compute dense
    # TD rewards (V(s') - V(s)) for the policy training that follows.
    if args.mode in ("value", "both"):
        trainer.train_value()
    if args.mode in ("policy", "both"):
        trainer.train()


if __name__ == "__main__":
    main()
