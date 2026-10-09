"""Decision route — main AI endpoint with replay logging."""

from datetime import datetime

from fastapi import APIRouter, Request
from loguru import logger

from arcbound.logging.new_card_logger import report_new_cards
from arcbound.logging.replay_logger import ReplayLogger
from arcbound.models.decision import AiDecisionRequest, AiDecisionResponse

router = APIRouter()

# Live diagnostic: remember the most recent SPELL_ABILITY_CHOICE pick per game
# so we can flag an "aborted cast" when a PAY_MANA_COST "no" arrives right after.
# Keyed by game_id -> (spell_label, seq). Bounded to avoid unbounded growth.
_last_spell_pick: dict = {}


def _get_logger(req: Request) -> ReplayLogger:
    return req.app.state.replay_logger  # type: ignore[union-attr]


def _ensure_game_logged(replay_logger: ReplayLogger, request_body: AiDecisionRequest) -> None:
    """Ensure a replay log is active for the request's game.

    The Java client does not call /game/start, so we auto-start a replay log
    on the first decision of a game and roll over to a new one when the game
    id changes.

    The game key is the Forge game id (``boardState.game_id``), NOT the
    per-controller UUID in ``request_body.gameId``. Each
    PlayerControllerExternalAi generates its own UUID, so in an AI-vs-AI game
    the two controllers would alternate "new game" rollovers on every single
    decision, wiping the in-memory decision list (and the GUI's live
    terminal/graph) on every move. The Forge game id is identical for both
    controllers, so both AIs' decisions land in one continuous replay.
    """
    game_id = request_body.boardState.get("game_id") or request_body.gameId
    if not game_id:
        return

    if not replay_logger.is_active:
        focal = request_body.boardState.get("focal_player") or "ai"
        active = request_body.boardState.get("active_player") or focal
        players = [focal, active] if active != focal else [focal]
        filename = replay_logger.start_game(
            game_id=game_id,
            players=players,
            focal_player=focal,
        )
        logger.info("Auto-started replay log for game {} (replay={})", game_id, filename)
    elif replay_logger.current_game_id != game_id:
        # New game began — finalize the previous replay, then start a new one.
        prev = replay_logger.finalize_current_game(result="ended")
        if prev:
            logger.info("Finalized previous replay: {}", prev)
        focal = request_body.boardState.get("focal_player") or "ai"
        active = request_body.boardState.get("active_player") or focal
        players = [focal, active] if active != focal else [focal]
        filename = replay_logger.start_game(
            game_id=game_id,
            players=players,
            focal_player=focal,
        )
        logger.info("Rolled over to new replay log for game {} (replay={})", game_id, filename)


@router.post("/decision")
async def post_decision(req: Request, request_body: AiDecisionRequest) -> AiDecisionResponse:
    """Main decision endpoint. Called by ExternalAiHttpClient.requestDecision().

    Logs each decision to the active replay file, then returns a decision from
    the loaded model if available, otherwise a fallback decision.
    """
    replay_logger = _get_logger(req)
    engine = req.app.state.inference_engine  # type: ignore[union-attr]

    logger.info(
        "Decision request: game_id={}, type={}, model_loaded={}",
        request_body.gameId,
        request_body.decisionRequest.type,
        engine.is_loaded,
    )

    # Log any cards the AI is seeing for the first time (full card info).
    new_cards = report_new_cards(request_body.boardState)
    if new_cards:
        logger.info(
            "First-time cards this decision ({}): {}",
            len(new_cards), ", ".join(new_cards),
        )

    result = engine.decide(request_body)
    response = result.response

    # Ensure a replay log is active for this game (Java never calls /game/start)
    _ensure_game_logged(replay_logger, request_body)

    # Build the decision record once (full state + options so it is trainable),
    # then log it to whichever loggers are active for this game.
    decision_log = {
        "turn": request_body.boardState.get("turn"),
        "phase": request_body.boardState.get("phase"),
        "player": request_body.boardState.get("focal_player"),
        "decision_type": request_body.decisionRequest.type,
        "description": request_body.decisionRequest.description,
        "options_presented": [opt.label for opt in request_body.decisionRequest.options],
        "action_taken": response.selectedOptionId or "",
        "action_reason": response.reasoning or "fallback",
        "model_confidence": result.confidence,
        "model_value_estimate": result.value_estimate,
        "board_state": request_body.boardState,
        "decision_request": request_body.decisionRequest.model_dump(),
        "timestamp": datetime.now().isoformat(),
        "timeout": False,
    }

    if replay_logger.is_active:
        replay_logger.log_decision(decision_log)

    # Also capture the decision into the active match file (when this game is
    # part of a logged match) so policy learning has decision-level data with
    # the options that were offered and the option that was chosen.
    #
    # The match logger keys games by the Forge game id (sent to
    # /game/start/match), which is carried in boardState.game_id — NOT the
    # per-controller UUID in request_body.gameId.
    #
    # The /game/start/match POST is async and can race behind the first
    # /decision requests, so the game may not be registered yet. log_decision()
    # buffers the record and flushes it (in order) when the game is registered,
    # so no decision is ever dropped from the match file.
    match_game_id = request_body.boardState.get("game_id")
    match_logger = req.app.state.match_logger  # type: ignore[union-attr]
    if match_logger.is_active and match_game_id:
        match_logger.log_decision(match_game_id, decision_log)

    # DIAGNOSTIC (aborted-cast detection): track the last spell pick per game
    # and flag when a PAY_MANA_COST "no" arrives right after it. This is the
    # "AI tried to cast, then walked it back" pattern the user is watching for.
    # It is logged loudly (warning) so it is visible in the GUI terminal, and it
    # confirms whether the replay captures the attempt + the walk-back as two
    # separate decision records (which is what feeds the trainer).
    if match_game_id:
        dtype = request_body.decisionRequest.type
        selected = response.selectedOptionId or ""
        if dtype == "SPELL_ABILITY_CHOICE" and selected not in ("", "pass"):
            # Resolve the picked option's label for a readable log line.
            spell_label = selected
            for opt in request_body.decisionRequest.options:
                if opt.id == selected:
                    spell_label = opt.label
                    break
            _last_spell_pick[match_game_id] = (spell_label, decision_log["seq"])
            if len(_last_spell_pick) > 64:  # bound the cache
                _last_spell_pick.pop(next(iter(_last_spell_pick)))
        elif dtype == "PAY_MANA_COST" and selected in ("no", "false"):
            prev = _last_spell_pick.pop(match_game_id, None)
            if prev is not None:
                spell_label, prev_seq = prev
                logger.warning(
                    "ABORTED CAST: game={} picked '{}' (seq={}) then declined the "
                    "mana cost (seq={}) — the cast was walked back. Both decisions "
                    "are logged as separate training records.",
                    match_game_id, spell_label, prev_seq, decision_log["seq"],
                )
                self_emit = getattr(req.app.state, "event_bus", None)
                if self_emit is not None:
                    try:
                        self_emit.emit(
                            "warning",
                            f"ABORTED CAST: picked '{spell_label}' then declined mana cost",
                        )
                    except Exception:
                        pass

    # Record the decision's value in the unified eval history so the GUI's
    # live graph updates after the AI's own actions too. The board state is the
    # FULL board (both hands), so we compute a value from EACH player's own
    # perspective: the AI's self-assessment (its hand visible, opponent's
    # masked) and the opponent's self-assessment (e.g. the human's hand
    # visible, the AI's masked). Each is stored in that player's own series.
    eval_history = req.app.state.eval_history  # type: ignore[union-attr]
    if eval_history is not None:
        # The focal player of a /decision request is ALWAYS the external AI
        # (Forge only asks the external AI for decisions). Use it to identify
        # the AI player. This is a bulletproof signal that does not depend on
        # /match/start's player_types being populated.
        focal = request_body.boardState.get("focal_player")
        if focal:
            eval_history.set_ai_player(focal)
        values: dict = {}
        if result.value_estimate is not None:
            # The decision's value estimate is already from the AI's (focal)
            # perspective.
            values[focal] = result.value_estimate
        # Also compute the opponent's self-assessment from the full board, so
        # the GUI can show "how the AI thinks the opponent is doing".
        for p in request_body.boardState.get("players", []) or []:
            pname = p.get("name") if isinstance(p, dict) else None
            if not pname or pname == focal:
                continue
            eval_history.set_player(pname)
            opp_value = engine.estimate_value(request_body.boardState, perspective=pname)
            if opp_value is not None:
                values[pname] = opp_value
        if values:
            eval_history.add(
                game_id=match_game_id or request_body.gameId,
                turn=request_body.boardState.get("turn"),
                values=values,
                actor=focal,
                kind="decision",
            )

    logger.info(
        "Decision response: type={}, selectedOptionId={}, used_model={}",
        response.decisionType,
        response.selectedOptionId,
        result.used_model,
    )

    return response
