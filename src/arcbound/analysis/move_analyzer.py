"""Move quality analysis for replayed games.

Rates every recorded move — the AI's decisions *and* the human player's
event-based moves (spells, lands, attackers, blockers, mulligans, zone
changes) — so a replay analysis shows how well each player is doing relative
to the model.

Scoring sources (in priority order):
- ``timeout``: the decision timed out and a fallback was used -> score 0.
- ``confidence``: the decision record carries the model's own confidence
  (AI decisions made with a loaded model) -> score = confidence * 100.
- ``model`` (decision): a loaded model is available and the record has a
  decision request -> score = the probability the model assigns to the
  option that was actually chosen.
- ``model`` (value): a loaded model is available and the record has a board
  state (event-based moves, e.g. the human's) -> score = the model's value
  estimate of the position from the acting player's perspective, mapped from
  [-1, 1] to [0, 100].
- ``neutral``: nothing else applies -> score 50 (unknown quality).

Numeric score -> rating:
- 95-100: Best
- 80-95:  Good
- 60-80:  Inaccuracy
- 30-60:  Mistake
- 0-30:   Blunder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from arcbound.encoder.tokenizer import tokenize_board_state, tokenize_decision
from arcbound.models.decision import AiDecisionRequest, DecisionContext


@dataclass
class MoveAnalysis:
    """Result of analyzing a single move."""

    turn: int
    phase: str
    player: str
    decision_type: str
    description: str
    action_taken: str
    score: int  # 0-100
    rating: str  # "Blunder", "Mistake", "Inaccuracy", "Good", "Best"
    confidence: Optional[float] = None
    value_estimate: Optional[float] = None
    timeout: bool = False
    source: str = "neutral"  # "timeout" | "confidence" | "model" | "neutral"


class MoveAnalyzer:
    """Analyze move quality from replay moves.

    Works on both decision records (``kind="decision"``, with
    ``decision_request`` + ``action_taken``) and event-based moves (the
    human player's actions, with ``board_state`` + ``action``). When a model
    is supplied, moves without a recorded confidence are scored by the model
    itself, which is what lets the tool rate the human player's play.
    """

    @staticmethod
    def rate_score(score: int) -> str:
        """Convert numeric score to rating label."""
        if score >= 95:
            return "Best"
        elif score >= 80:
            return "Good"
        elif score >= 60:
            return "Inaccuracy"
        elif score >= 30:
            return "Mistake"
        else:
            return "Blunder"

    # ------------------------------------------------------------------
    # Model-based scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _indices_tensors(tb) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (card_indices, keyword_indices) (1, L) tensors from a tokenized board."""
        L = tb.features.shape[0]
        card_idx = torch.zeros(1, L, dtype=torch.long)
        kw_idx = torch.zeros(1, L, dtype=torch.long)
        for pos, i in enumerate(tb.card_indices):
            if pos < L and i > 0:
                card_idx[0, pos] = i
        for pos, i in enumerate(tb.keyword_indices):
            if pos < L and i > 0:
                kw_idx[0, pos] = i
        return card_idx, kw_idx

    @staticmethod
    def estimate_board_value(
        model, board_state: Dict[str, Any], perspective: Optional[str] = None
    ) -> Optional[float]:
        """Return the model's raw value estimate for ``board_state``.

        The value is from ``perspective``'s point of view (default: the board's
        focal/acting player) in [-1, 1] (+1 = that player wins). The board is
        re-focused on ``perspective`` so hidden zones are masked from that
        player's information set. Returns None when no model is available or
        tokenization/inference fails. Used to fill in value estimates for moves
        that were not recorded with one (e.g. the opponent's event-based moves)
        so the "Evaluation Over Time" graph can show every action.
        """
        if model is None or not board_state:
            return None
        try:
            tb = tokenize_board_state(board_state, perspective)
            features = torch.from_numpy(tb.features).unsqueeze(0)
            card_idx, kw_idx = MoveAnalyzer._indices_tensors(tb)
            with torch.no_grad():
                _, value, _ = model(features, None, card_idx, kw_idx)
            return float(value[0, 0].item())
        except Exception:
            return None

    @staticmethod
    def _model_value_score(model, board_state: Dict[str, Any]) -> Optional[int]:
        """Score a position via the model's value head (acting player's perspective).

        The value head is trained on (board_state, outcome) pairs where the
        board state is serialized from the acting player's perspective, so the
        CLS value is from that player's point of view. Maps [-1, 1] -> [0, 100].
        """
        try:
            tb = tokenize_board_state(board_state)
            features = torch.from_numpy(tb.features).unsqueeze(0)
            card_idx, kw_idx = MoveAnalyzer._indices_tensors(tb)
            with torch.no_grad():
                _, value, _ = model(features, None, card_idx, kw_idx)
            v = float(value[0, 0].item())
            return max(0, min(100, int(round(50 + v * 50))))
        except Exception:
            return None

    @staticmethod
    def _model_decision_score(
        model,
        board_state: Dict[str, Any],
        decision_request: Dict[str, Any],
        action_id: Optional[str],
    ) -> Optional[int]:
        """Score a decision by the probability the model assigns to the chosen option."""
        try:
            request = AiDecisionRequest(
                gameId="analysis",
                boardState=board_state,
                decisionRequest=DecisionContext.model_validate(decision_request),
            )
            td = tokenize_decision(request)
            if td.num_options == 0:
                return None
            action_idx = None
            for i, opt in enumerate(request.decisionRequest.options):
                if opt.id == action_id:
                    action_idx = i
                    break
            if action_idx is None:
                return None
            features = torch.from_numpy(td.features).unsqueeze(0)
            L = features.shape[1]
            option_mask = torch.zeros(1, L, dtype=torch.bool)
            for idx in td.option_indices:
                if idx < L:
                    option_mask[0, idx] = True
            card_idx = torch.zeros(1, L, dtype=torch.long)
            kw_idx = torch.zeros(1, L, dtype=torch.long)
            for pos, i in enumerate(td.card_indices):
                if pos < L and i > 0:
                    card_idx[0, pos] = i
            for pos, i in enumerate(td.keyword_indices):
                if pos < L and i > 0:
                    kw_idx[0, pos] = i
            with torch.no_grad():
                logits, _, _ = model(features, option_mask, card_idx, kw_idx)
            probs = torch.softmax(logits[0, :td.num_options], dim=-1)
            return int(round(float(probs[action_idx].item()) * 100))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def analyze_moves(
        moves: List[Dict],
        model=None,
    ) -> List[MoveAnalysis]:
        """Analyze a list of move dicts from a replay (any kind).

        ``model`` is an optional loaded ArcboundModel. When provided, moves
        that lack a recorded confidence (the human player's event-based moves,
        and AI decisions made without a model) are scored by the model.
        """
        analyses: List[MoveAnalysis] = []
        for move in moves:
            turn = move.get("turn", 0)
            phase = move.get("phase", "")
            player = move.get("player", "")
            decision_type = (
                move.get("decision_type") or move.get("action_type") or "move"
            )
            description = move.get("description") or move.get("action") or ""
            action = move.get("action_taken") or move.get("action") or ""
            confidence = move.get("model_confidence")
            value_est = move.get("model_value_estimate")
            timeout = move.get("timeout", False)
            board_state = move.get("board_state")
            decision_request = move.get("decision_request")

            score: Optional[int] = None
            source = "neutral"
            if timeout:
                score, source = 0, "timeout"
            elif confidence is not None:
                score, source = int(confidence * 100), "confidence"
            elif model is not None and board_state and decision_request:
                s = MoveAnalyzer._model_decision_score(
                    model, board_state, decision_request, action
                )
                if s is not None:
                    score, source = s, "model"
            elif model is not None and board_state:
                s = MoveAnalyzer._model_value_score(model, board_state)
                if s is not None:
                    score, source = s, "model"
            if score is None:
                score, source = 50, "neutral"

            analyses.append(MoveAnalysis(
                turn=turn,
                phase=phase,
                player=player,
                decision_type=decision_type,
                description=description,
                action_taken=action,
                score=score,
                rating=MoveAnalyzer.rate_score(score),
                confidence=confidence,
                value_estimate=value_est,
                timeout=timeout,
                source=source,
            ))
        return analyses

    @staticmethod
    def analyze_decisions(
        decisions: List[Dict],
        model=None,
    ) -> List[MoveAnalysis]:
        """Analyze a list of decision dicts from a replay (legacy entry point)."""
        return MoveAnalyzer.analyze_moves(decisions, model=model)

    @staticmethod
    def summarize(analyses: List[MoveAnalysis]) -> Dict[str, Any]:
        """Produce a summary of move analyses."""
        if not analyses:
            return {
                "total": 0,
                "best": 0,
                "good": 0,
                "inaccuracy": 0,
                "mistake": 0,
                "blunder": 0,
                "average_score": 0.0,
            }

        total = len(analyses)
        counts: Dict[str, int] = {"Best": 0, "Good": 0, "Inaccuracy": 0, "Mistake": 0, "Blunder": 0}
        for a in analyses:
            counts[a.rating] = counts.get(a.rating, 0) + 1

        avg = sum(a.score for a in analyses) / total

        return {
            "total": total,
            "best": counts["Best"],
            "good": counts["Good"],
            "inaccuracy": counts["Inaccuracy"],
            "mistake": counts["Mistake"],
            "blunder": counts["Blunder"],
            "average_score": round(avg, 1),
        }

    @staticmethod
    def summarize_by_player(analyses: List[MoveAnalysis]) -> Dict[str, Dict[str, Any]]:
        """Produce a per-player summary (e.g. human vs AI) of move analyses."""
        by_player: Dict[str, List[MoveAnalysis]] = {}
        for a in analyses:
            by_player.setdefault(a.player or "unknown", []).append(a)
        return {
            player: MoveAnalyzer.summarize(moves)
            for player, moves in by_player.items()
        }
