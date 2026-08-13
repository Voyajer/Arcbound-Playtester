"""Fallback decision logic when the model is not yet trained."""

import random
import uuid

from arcbound.models.decision import AiDecisionRequest, AiDecisionResponse


def generate_fallback_decision(request: AiDecisionRequest) -> AiDecisionResponse:
    """Generate a fallback decision when the model is unavailable.

    Uses simple heuristics based on decision type to return valid responses
    that won't crash Forge.
    """
    ctx = request.decisionRequest
    decision_type = ctx.type

    decision_id = str(uuid.uuid4())

    # Boolean decisions: default to False/No
    if decision_type == "boolean":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="boolean",
            value={"boolean": False},
            reasoning="Fallback: default to false",
            selectedOptionId=_find_option_by_label(ctx, "No") or _first_option_id(ctx),
        )

    # Integer decisions: pick the minimum or middle value
    if decision_type == "integer":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="integer",
            value={"number": 0},
            reasoning="Fallback: default to 0",
            selectedOptionId=_first_option_id(ctx),
        )

    # String decisions: pick first option
    if decision_type == "string":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="string",
            value={"text": ""},
            reasoning="Fallback: empty string",
            selectedOptionId=_first_option_id(ctx),
        )

    # Color decisions: pick White (first color)
    if decision_type == "color":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="color",
            value={"color": "White"},
            reasoning="Fallback: default to White",
            selectedOptionId=_first_option_id(ctx),
        )

    # Card decisions: pick first card
    if decision_type == "card":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="card",
            value={},
            reasoning="Fallback: first card",
            selectedOptionId=_first_option_id(ctx),
        )

    # Cards decisions: pick no cards (if allowed) or first
    if decision_type == "cards":
        if ctx.constraints and ctx.constraints.allow_none:
            return AiDecisionResponse(
                decisionId=decision_id,
                decisionType="cards",
                value={},
                reasoning="Fallback: no cards selected",
                selectedOptionIds=[],
            )
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="cards",
            value={},
            reasoning="Fallback: first card",
            selectedOptionId=_first_option_id(ctx),
        )

    # Player decisions: pick first player
    if decision_type == "player":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="player",
            value={},
            reasoning="Fallback: first player",
            selectedOptionId=_first_option_id(ctx),
        )

    # Damage decisions: assign to first target
    if decision_type == "damage":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="damage",
            value={},
            reasoning="Fallback: first target",
            selectedOptionId=_first_option_id(ctx),
        )

    # Shield decisions: divide to first
    if decision_type == "shield":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="shield",
            value={},
            reasoning="Fallback: first target",
            selectedOptionId=_first_option_id(ctx),
        )

    # Mana combo: use first option
    if decision_type == "manaCombo":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="manaCombo",
            value={},
            reasoning="Fallback: first mana combo",
            selectedOptionId=_first_option_id(ctx),
        )

    # Payment decisions: first option
    if decision_type == "payment":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="payment",
            value={},
            reasoning="Fallback: first payment",
            selectedOptionId=_first_option_id(ctx),
        )

    # Mode decisions: first mode
    if decision_type == "modes":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="modes",
            value={"modes": [0]},
            reasoning="Fallback: first mode",
            selectedOptionId=_first_option_id(ctx),
        )

    # Replacement effect: first effect
    if decision_type == "replacementEffect":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="replacementEffect",
            value={"index": 0},
            reasoning="Fallback: first replacement effect",
            selectedOptionId=_first_option_id(ctx),
        )

    # Spell ability: first ability
    if decision_type == "spellAbility":
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="spellAbility",
            value={},
            reasoning="Fallback: first spell ability",
            selectedOptionId=_first_option_id(ctx),
        )

    # Default: pick first option or pass
    return AiDecisionResponse(
        decisionId=decision_id,
        decisionType=decision_type,
        value={},
        reasoning=f"Fallback: unknown decision type '{decision_type}', picking first option",
        selectedOptionId=_first_option_id(ctx),
    )


def _first_option_id(ctx) -> str | None:
    """Return the ID of the first option, or None if no options."""
    if ctx.options:
        return ctx.options[0].id
    return None


def _find_option_by_label(ctx, label: str) -> str | None:
    """Find an option by its label text (case-insensitive)."""
    label_lower = label.lower()
    for opt in ctx.options:
        if opt.label and opt.label.lower() == label_lower:
            return opt.id
    return None
