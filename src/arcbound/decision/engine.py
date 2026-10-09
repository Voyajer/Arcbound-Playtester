"""Inference engine — runs the trained model on a decision request.

Falls back to heuristic decisions when the model is unavailable or produces
no valid option, so the server never crashes Forge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from loguru import logger

from arcbound.decision.fallback import generate_fallback_decision
from arcbound.encoder.tokenizer import tokenize_board_state, tokenize_decision
from arcbound.encoder.transformer import ArcboundModel
from arcbound.models.decision import AiDecisionRequest, AiDecisionResponse


@dataclass
class InferenceResult:
    """Result of a model inference."""

    response: AiDecisionResponse
    confidence: Optional[float] = None
    value_estimate: Optional[float] = None
    used_model: bool = False


class InferenceEngine:
    """Runs model inference on decision requests.

    The engine holds a reference to the loaded model (set by ModelManager)
    and produces decisions. If the model is None or inference fails, it
    falls back to heuristics.
    """

    def __init__(
        self,
        model: Optional[ArcboundModel] = None,
        epsilon: float = 0.15,
        event_bus=None,
    ):
        self.device = torch.device("cpu")
        self.model = None
        # Optional EventBus for live GUI notifications (fallback / epsilon).
        self.event_bus = event_bus
        if model is not None:
            # Put the model in eval mode so dropout is disabled during
            # inference (a model passed to the constructor may still be in
            # train mode, which would make inference non-deterministic).
            model.to(self.device)
            model.eval()
        self.model = model
        # Blend weight for the option-conditioned Q estimate when selecting an
        # option: score = (1 - q_weight) * policy_prob + q_weight * sigmoid(Q).
        # 0.0 = pure policy (old behavior); 1.0 = pure value.
        self.q_weight = 0.5
        # Epsilon-greedy exploration: with probability ``epsilon`` pick a random
        # option instead of the model's top pick. This keeps the replay data
        # diverse and breaks the pass-mode collapse (the model otherwise learns
        # to always pass, because its own pass-heavy replays are all it trains
        # on). 0.0 disables exploration (pure exploitation).
        self.epsilon = max(0.0, min(1.0, epsilon))

    def set_model(self, model: Optional[ArcboundModel]):
        """Set or clear the active model."""
        self.model = model
        if model is not None:
            model.to(self.device)
            model.eval()

    def set_epsilon(self, epsilon: float):
        """Update the exploration rate at runtime (hot-swap, no restart).

        Clamped to [0, 1] like the constructor. Takes effect on the next
        decision, so it can be changed mid-match.
        """
        self.epsilon = max(0.0, min(1.0, float(epsilon)))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def estimate_value(
        self, board_state: dict, perspective: Optional[str] = None
    ) -> Optional[float]:
        """Estimate the board value for ``perspective`` (default: the focal player).

        Runs the model's value head on the board state (no options) and returns
        the CLS value in [-1, 1] from ``perspective``'s point of view. The board
        is re-focused on ``perspective`` so hidden zones are masked from that
        player's information set (e.g. the human-perspective value sees the
        human's hand but not the AI's). Returns None when no model is loaded or
        tokenization/inference fails.

        This is used to score moves so the live eval graph can show how the
        position shifts after *every* player's action, from each player's own
        perspective.
        """
        if self.model is None or not board_state:
            return None
        try:
            tb = tokenize_board_state(board_state, perspective)
            features = torch.from_numpy(tb.features).unsqueeze(0).to(self.device)
            card_indices = torch.zeros(1, features.shape[1], dtype=torch.long, device=self.device)
            for pos, idx in enumerate(tb.card_indices):
                if pos < features.shape[1] and idx > 0:
                    card_indices[0, pos] = idx
            keyword_indices = torch.zeros(1, features.shape[1], dtype=torch.long, device=self.device)
            for pos, idx in enumerate(tb.keyword_indices):
                if pos < features.shape[1] and idx > 0:
                    keyword_indices[0, pos] = idx
            with torch.no_grad():
                _, value, _ = self.model(features, None, card_indices, keyword_indices)
            return float(value[0, 0].item())
        except Exception as e:
            logger.debug("Value estimate failed: {}", e)
            return None

    def _emit(self, level: str, message: str) -> None:
        """Emit a live event to the GUI terminal (if an event bus is set)."""
        if self.event_bus is not None:
            try:
                self.event_bus.emit(level, message)
            except Exception:
                pass

    def _fallback(self, request: AiDecisionRequest, reason: str) -> InferenceResult:
        """Run the fallback heuristic and announce it loudly.

        A fallback means the model could not be used — that is a problem the
        user needs to know about, so it is emitted to the GUI terminal as an
        ``error`` (red) event stating *why*, in addition to the loguru log.
        """
        logger.warning("FALLBACK decision: {} — {}", reason, request.decisionRequest.type)
        self._emit("error", f"FALLBACK (model unavailable): {reason}")
        # Pass epsilon so the fallback mixes random actions with pass-throughs
        # (ratio set by epsilon) instead of always doing nothing.
        response = generate_fallback_decision(request, epsilon=self.epsilon)
        response.reasoning = f"Fallback: {reason}"
        return InferenceResult(response=response, used_model=False)

    def decide(self, request: AiDecisionRequest) -> InferenceResult:
        """Produce a decision for the given request.

        Args:
            request: The decision request from Forge.

        Returns:
            InferenceResult with the response, confidence, and value estimate.
        """
        if self.model is None:
            return self._fallback(
                request, "no model is loaded — create and train a model in the GUI"
            )

        try:
            return self._model_decide(request)
        except Exception as e:
            return self._fallback(request, f"model inference failed: {e}")

    def _model_decide(self, request: AiDecisionRequest) -> InferenceResult:
        """Run model inference with fallback on failure."""
        # Tokenize
        tokenized = tokenize_decision(request)

        if tokenized.num_options == 0:
            # No options to score — use fallback (nothing to pick, so this is a
            # no-op selection; still pass epsilon for consistency).
            response = generate_fallback_decision(request, epsilon=self.epsilon)
            return InferenceResult(response=response, used_model=False)

        # Build tensors
        features = torch.from_numpy(tokenized.features).unsqueeze(0).to(self.device)  # (1, L, F)
        L = features.shape[1]
        option_mask = torch.zeros(1, L, dtype=torch.bool, device=self.device)
        for idx in tokenized.option_indices:
            if idx < L:
                option_mask[0, idx] = True

        # Card identity indices (per position); 0 for non-card tokens.
        card_indices = torch.zeros(1, L, dtype=torch.long, device=self.device)
        for pos, idx in enumerate(tokenized.card_indices):
            if pos < L and idx > 0:
                card_indices[0, pos] = idx

        # Keyword indices (per position); 0 for tokens without a keyword.
        keyword_indices = torch.zeros(1, L, dtype=torch.long, device=self.device)
        for pos, idx in enumerate(tokenized.keyword_indices):
            if pos < L and idx > 0:
                keyword_indices[0, pos] = idx

        # Forward pass
        with torch.no_grad():
            option_logits, value, q_values = self.model(
                features, option_mask, card_indices, keyword_indices
            )

        option_logits = option_logits[0, :tokenized.num_options]  # (num_options,)
        q_values = q_values[0, :tokenized.num_options]  # (num_options,)
        value_est = value[0, 0].item()

        # Blend the policy's option probabilities with the option-conditioned
        # Q estimate so the trained value signal influences which option is
        # picked (not just reported in reasoning). Q is squashed to [0, 1]
        # with a sigmoid so it is comparable to the policy probabilities.
        probs = torch.softmax(option_logits, dim=-1)
        q_norm = torch.sigmoid(q_values)
        scores = (1.0 - self.q_weight) * probs + self.q_weight * q_norm

        # Rank all options by blended score, best first.
        order = [int(i) for i in torch.argsort(scores, descending=True).tolist()]

        # Epsilon-greedy exploration: with probability ``self.epsilon``, shuffle
        # the ranking so the primary pick (and, for multi/ordering decisions, the
        # full pick list) is random rather than the model's top choice. This keeps
        # the replay data diverse and breaks the pass-mode collapse — without it,
        # the model only ever sees its own pass-heavy decisions in training and
        # learns to always pass. Shuffling ``order`` handles all three modes
        # uniformly: single -> random pick, multi -> random top-k, ordering ->
        # random permutation.
        if self.epsilon > 0.0 and len(order) > 1 and random.random() < self.epsilon:
            random.shuffle(order)
            # Announce exploration so the user can see epsilon is active.
            self._emit(
                "warning",
                f"EPSILON exploration (p={self.epsilon:.2f}): picked a random option",
            )

        best_idx = order[0]
        confidence = float(probs[best_idx].item())

        ctx = request.decisionRequest
        mode = self._classify_mode(ctx)
        if mode == "ordering":
            # Return every option, best-first (a full permutation). Forge's
            # queryOrder reads selectedOptionIds in exactly this order.
            selected_indices = order
        elif mode == "multi":
            # Return the model's top-k picks, where k is the max allowed.
            # Forge's queryMulti/queryCards read selectedOptionIds as a subset.
            max_k = ctx.constraints.max_choices if ctx.constraints else len(ctx.options)
            k = max(1, min(max_k, len(ctx.options)))
            selected_indices = order[:k]
        else:
            selected_indices = [best_idx]

        response = self._build_response(request, selected_indices, confidence, value_est)
        return InferenceResult(
            response=response,
            confidence=confidence,
            value_estimate=value_est,
            used_model=True,
        )

    def _classify_mode(self, ctx) -> str:
        """Classify a decision as 'ordering', 'multi', or 'single'.

        - 'ordering': the decision type is an ordering prompt (e.g.
          ORDER_BLOCKERS, ORDER_ATTACKERS, ZONE_ORDER). Forge's queryOrder
          expects a full permutation in selectedOptionIds.
        - 'multi': constraints allow more than one choice. Forge's
          queryMulti/queryCards expect a subset in selectedOptionIds.
        - 'single': everything else. Forge reads selectedOptionId.
        """
        if ctx.type and "ORDER" in ctx.type.upper():
            return "ordering"
        if ctx.constraints is not None and ctx.constraints.max_choices > 1:
            return "multi"
        return "single"

    def _build_response(
        self,
        request: AiDecisionRequest,
        selected_indices: list[int],
        confidence: float,
        value_est: float,
    ) -> AiDecisionResponse:
        """Build an AiDecisionResponse from the selected option indices.

        ``selected_indices`` are indices into ``request.decisionRequest.options``,
        ordered best-first. The first is the primary pick (selectedOptionId);
        for multi-select and ordering decisions the full ordered list is also
        provided as selectedOptionIds, which is what Forge's queryMulti /
        queryCards / queryOrder read.
        """
        import uuid

        ctx = request.decisionRequest
        decision_type = ctx.type
        decision_id = str(uuid.uuid4())
        options = ctx.options

        # Guard: ensure at least one valid index.
        if not selected_indices or selected_indices[0] >= len(options):
            selected_indices = [0]
        selected_option = options[selected_indices[0]]

        # Determine the value payload based on decision type (top pick).
        value: dict = {}
        if decision_type == "boolean":
            value = {"boolean": selected_option.label.lower() in ("yes", "true", "1")}
        elif decision_type == "integer":
            try:
                value = {"number": int(selected_option.label)}
            except (ValueError, TypeError):
                value = {"number": 0}
        elif decision_type == "string":
            value = {"text": selected_option.label}
        elif decision_type == "color":
            value = {"color": selected_option.label}
        elif decision_type == "modes":
            try:
                value = {"modes": [int(selected_option.label)]}
            except (ValueError, TypeError):
                value = {"modes": [0]}
        elif decision_type == "replacementEffect":
            try:
                value = {"index": int(selected_option.label)}
            except (ValueError, TypeError):
                value = {"index": 0}
        # card, cards, player, damage, shield, manaCombo, payment, spellAbility:
        # value is empty; the selectedOptionId(s) carry the choice

        # Multi-select and ordering decisions expose the full ordered pick list.
        mode = self._classify_mode(ctx)
        if mode in ("ordering", "multi"):
            selected_option_ids = [
                options[i].id for i in selected_indices if 0 <= i < len(options)
            ]
        else:
            selected_option_ids = None

        if mode == "single":
            reasoning = (
                f"Model selected '{selected_option.label}' "
                f"(confidence={confidence:.3f}, value={value_est:.3f})"
            )
        else:
            reasoning = (
                f"Model selected {len(selected_indices)} option(s), "
                f"first='{selected_option.label}' "
                f"(confidence={confidence:.3f}, value={value_est:.3f})"
            )

        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType=decision_type,
            value=value,
            reasoning=reasoning,
            selectedOptionId=selected_option.id,
            selectedOptionIds=selected_option_ids,
        )
