"""Fallback decision logic when the model is not yet trained."""

import random
import uuid

from loguru import logger

from arcbound.models.decision import AiDecisionRequest, AiDecisionResponse


def generate_fallback_decision(
    request: AiDecisionRequest, epsilon: float = 0.0
) -> AiDecisionResponse:
    """Generate a fallback decision when the model is unavailable.

    The fallback is a last resort, not a strategy. With probability
    ``epsilon`` it takes a *random* action (a random option / subset /
    ordering); otherwise it takes the conservative "do nothing" default
    (the safe option for yes/no, no cards when allowed, first option
    otherwise). ``epsilon`` therefore sets the ratio of random actions to
    pass-throughs, mirroring the exploration the model path uses.

    For multi-select and ordering decisions, selectedOptionIds is populated
    so Forge's queryMulti/queryOrder can act on it. Logs the chosen option
    (what) and the rationale (why) to the console.
    """
    ctx = request.decisionRequest
    if epsilon > 0.0 and ctx.options and random.random() < epsilon:
        response = _random_action(request)
    else:
        response = _fallback_impl(request)
        _apply_multi_select(response, ctx)
    _log_fallback(request, response)
    return response


def _random_action(request: AiDecisionRequest) -> AiDecisionResponse:
    """Pick a random valid action for the decision.

    - ordering: a random permutation of all options.
    - multi: a random subset of the allowed size.
    - single: a random option (with a type-appropriate value payload).
    """
    ctx = request.decisionRequest
    decision_id = str(uuid.uuid4())
    options = ctx.options
    mode = _mode(ctx)

    if mode == "ordering":
        ids = [o.id for o in options]
        random.shuffle(ids)
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType=ctx.type,
            value={},
            reasoning="Fallback (random): random ordering",
            selectedOptionId=ids[0] if ids else None,
            selectedOptionIds=ids,
        )

    if mode == "multi":
        max_k = ctx.constraints.max_choices if ctx.constraints else len(options)
        k = max(1, min(max_k, len(options)))
        chosen = random.sample(options, k)
        ids = [o.id for o in chosen]
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType=ctx.type,
            value={},
            reasoning=f"Fallback (random): {k} random option(s)",
            selectedOptionId=ids[0] if ids else None,
            selectedOptionIds=ids,
        )

    opt = random.choice(options)
    return AiDecisionResponse(
        decisionId=decision_id,
        decisionType=ctx.type,
        value=_value_for(ctx.type, opt),
        reasoning=f"Fallback (random): '{opt.label}'",
        selectedOptionId=opt.id,
    )


def _value_for(decision_type: str, option) -> dict:
    """Build a type-appropriate value payload for a chosen option.

    Mirrors the engine's response-building so a random single-pick carries a
    valid value (e.g. a random boolean pick sets the boolean to match).
    """
    if decision_type == "boolean":
        return {"boolean": option.label.lower() in ("yes", "true", "1")}
    if decision_type == "integer":
        try:
            return {"number": int(option.label)}
        except (ValueError, TypeError):
            return {"number": 0}
    if decision_type == "string":
        return {"text": option.label}
    if decision_type == "color":
        return {"color": option.label}
    if decision_type == "modes":
        try:
            return {"modes": [int(option.label)]}
        except (ValueError, TypeError):
            return {"modes": [0]}
    if decision_type == "replacementEffect":
        try:
            return {"index": int(option.label)}
        except (ValueError, TypeError):
            return {"index": 0}
    return {}


def _log_fallback(request: AiDecisionRequest, response: AiDecisionResponse) -> None:
    """Log a fallback decision: what was chosen and why.

    Resolves the selected option id(s) to their human-readable labels so the
    console shows the actual choice, not just an opaque id.
    """
    ctx = request.decisionRequest
    labels = {opt.id: opt.label for opt in ctx.options}

    if response.selectedOptionIds:
        what = ", ".join(labels.get(i, i) for i in response.selectedOptionIds)
    elif response.selectedOptionId is not None:
        what = labels.get(response.selectedOptionId, response.selectedOptionId)
    else:
        what = "(none)"

    logger.warning(
        "FALLBACK decision [{}]: chose '{}' — {}",
        ctx.type,
        what,
        response.reasoning or "no reason given",
    )


def _mode(ctx) -> str:
    """Classify a decision as 'ordering', 'multi', or 'single' (mirrors engine)."""
    if ctx.type and "ORDER" in ctx.type.upper():
        return "ordering"
    if ctx.constraints is not None and ctx.constraints.max_choices > 1:
        return "multi"
    return "single"


def _apply_multi_select(response: AiDecisionResponse, ctx) -> None:
    """Fill in selectedOptionIds for multi-select / ordering fallbacks.

    - ordering: all option ids in natural order (a full permutation).
    - multi: the first ``max_choices`` option ids (a valid subset).
    An explicit empty selection (e.g. 'no cards') is left untouched.
    """
    if not ctx.options:
        return
    mode = _mode(ctx)
    if mode == "ordering":
        response.selectedOptionIds = [o.id for o in ctx.options]
    elif mode == "multi":
        if response.selectedOptionIds is not None and len(response.selectedOptionIds) == 0:
            return
        max_k = ctx.constraints.max_choices if ctx.constraints else len(ctx.options)
        k = max(1, min(max_k, len(ctx.options)))
        response.selectedOptionIds = [o.id for o in ctx.options[:k]]


def _fallback_impl(request: AiDecisionRequest) -> AiDecisionResponse:
    """Heuristic fallback (single-pick). See generate_fallback_decision."""
    ctx = request.decisionRequest
    decision_type = ctx.type

    decision_id = str(uuid.uuid4())

    # Boolean decisions: default to the conservative option (No/Keep/False).
    # Java sends the prompt name as the decision type (e.g. CONFIRM_TRIGGER,
    # MULLIGAN, BINARY_CHOICE), so detect boolean decisions by option shape
    # rather than ctx.type, and key on the conservative option *id* (labels
    # vary: "No", "Cancel", "Don't play trigger", ...).
    if decision_type == "boolean" or _is_boolean_decision(ctx):
        return AiDecisionResponse(
            decisionId=decision_id,
            decisionType="boolean",
            value={"boolean": False},
            reasoning="Fallback: default to the conservative option",
            selectedOptionId=_conservative_boolean_id(ctx),
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


def _last_option_id(ctx) -> str | None:
    """Return the ID of the last option, or None if no options."""
    if ctx.options:
        return ctx.options[-1].id
    return None


# Conservative option IDs for boolean decisions. Java emits boolean options
# with these IDs for the "safe" choice:
#   queryBoolean / playTrigger / confirmAction -> "no"
#   mulliganKeepHand                           -> "keep"
#   chooseBinary                               -> "false"
_CONSERVATIVE_BOOLEAN_IDS = {"no", "keep", "false"}


def _is_boolean_decision(ctx) -> bool:
    """Detect a boolean decision by option shape.

    Java sends the prompt name as the decision type (e.g. CONFIRM_TRIGGER,
    MULLIGAN, BINARY_CHOICE), so ctx.type can't be relied on. A boolean
    decision is one whose options are all typed "boolean".
    """
    return bool(ctx.options) and all(o.type == "boolean" for o in ctx.options)


def _conservative_boolean_id(ctx) -> str | None:
    """Return the conservative option id for a boolean decision.

    Keys on the option *id* rather than the label (labels vary: "No",
    "Cancel", "Don't play trigger", ...). Falls back to the last option if
    no known conservative id is present.
    """
    for opt in ctx.options:
        if opt.id in _CONSERVATIVE_BOOLEAN_IDS:
            return opt.id
    return _last_option_id(ctx)
