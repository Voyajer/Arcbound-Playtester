"""Match lifecycle routes — match start/end, sideboard, and per-action move logging.

These endpoints are called by the Forge-side ``ExternalMatchLogger`` and
``GameEventRecorder`` so that ANY match (human vs human, human vs bot, AI vs AI)
is captured as a single match replay file with per-game and match-level winners.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import BaseModel, Field

from arcbound.logging.match_logger import MatchLogger

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models matching the JSON sent by the Forge-side logger
# ---------------------------------------------------------------------------

class MatchStartRequest(BaseModel):
    """POST /match/start — a new match is beginning."""
    model_config = {"populate_by_name": True}

    matchId: str = Field(alias="matchId")
    players: List[str] = Field(default_factory=list)
    format: Optional[str] = Field(default=None)
    games_to_win: Optional[int] = Field(default=None, alias="gamesToWin")
    player_types: Optional[Dict[str, str]] = Field(default=None, alias="playerTypes")


class MatchEndRequest(BaseModel):
    """POST /match/end — the match is over; record the match winner."""
    model_config = {"populate_by_name": True}

    matchId: str = Field(alias="matchId")
    match_winner: Optional[str] = Field(default=None, alias="matchWinner")
    final_score: Optional[Dict[str, int]] = Field(default=None, alias="finalScore")
    result: Optional[str] = Field(default=None)


class GameStartRequest(BaseModel):
    """POST /game/start (match context) — a new game within the active match."""
    model_config = {"populate_by_name": True}

    matchId: str = Field(alias="matchId")
    gameId: str = Field(alias="gameId")
    game_number: Optional[int] = Field(default=None, alias="gameNumber")
    decklists: Optional[Dict[str, List[str]]] = Field(default=None)


class SideboardRequest(BaseModel):
    """POST /game/sideboard — a player finished sideboarding for a game."""
    model_config = {"populate_by_name": True}

    matchId: str = Field(alias="matchId")
    gameId: str = Field(alias="gameId")
    player: str
    main_to_side: List[str] = Field(default_factory=list, alias="mainToSide")
    side_to_main: List[str] = Field(default_factory=list, alias="sideToMain")


class MoveRequest(BaseModel):
    """POST /game/move — a single action with the full board state.

    ``board_state`` is the state BEFORE the action (the (state, action) pair
    used for training). ``action`` describes what happened.
    """
    model_config = {"populate_by_name": True}

    matchId: str = Field(alias="matchId")
    gameId: str = Field(alias="gameId")
    turn: Optional[int] = Field(default=None)
    phase: Optional[str] = Field(default=None)
    player: Optional[str] = Field(default=None)
    action: Optional[str] = Field(default=None)
    action_type: Optional[str] = Field(default=None, alias="actionType")
    details: Optional[Dict] = Field(default=None)
    board_state: Optional[Dict] = Field(default=None, alias="boardState")


class GameEndRequest(BaseModel):
    """POST /game/end (match context) — a game within the match is over."""
    model_config = {"populate_by_name": True}

    matchId: str = Field(alias="matchId")
    gameId: str = Field(alias="gameId")
    winner: Optional[str] = Field(default=None)
    total_turns: Optional[int] = Field(default=None, alias="totalTurns")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _get_logger(req: Request) -> MatchLogger:
    return req.app.state.match_logger  # type: ignore[union-attr]


@router.post("/match/start")
async def post_match_start(req: Request, body: MatchStartRequest):
    """Notify the AI server that a new match is starting."""
    match_logger = _get_logger(req)

    # Remember which player is the external AI so the eval history can
    # normalize every action's value to the AI's perspective (opponent moves
    # are negated, since the value head is trained from the actor's view).
    eval_history = req.app.state.eval_history  # type: ignore[union-attr]
    if eval_history is not None and body.player_types:
        for name, role in body.player_types.items():
            if str(role).strip().lower() == "ai":
                eval_history.set_ai_player(name)
                break

    # If a previous match is still active (e.g. the client never sent /match/end),
    # finalize it so we don't lose data.
    if match_logger.is_active:
        prev = match_logger.finalize_current_match(result="abandoned")
        if prev:
            logger.info("Finalized previous match on new match start: {}", prev)

    format_name = (body.format or "standard").lower()
    games_to_win = body.games_to_win or 1
    players = body.players or ["player-1", "player-2"]

    filename = match_logger.start_match(
        match_id=body.matchId,
        players=players,
        format_name=format_name,
        games_to_win=games_to_win,
        player_types=body.player_types,
    )
    logger.info(
        "Match start: match_id={}, format={}, games_to_win={}, players={}, player_types={}, replay={}",
        body.matchId, format_name, games_to_win, players, body.player_types, filename,
    )
    return {"status": "ok", "replay": filename}


@router.post("/match/end")
async def post_match_end(req: Request, body: MatchEndRequest):
    """Notify the AI server that the match is over; write the match file."""
    match_logger = _get_logger(req)

    try:
        filepath = match_logger.end_match(
            match_id=body.matchId,
            match_winner=body.match_winner,
            final_score=body.final_score,
            result=body.result or "completed",
        )
        # NOTE: the eval history is intentionally NOT cleared here. It persists
        # after the match ends and is only reset when a NEW game's first action
        # arrives (EvalHistory.add() auto-resets on a game_id change) or on a
        # fresh server launch. This keeps the GUI's graph populated until the
        # next game actually begins.
        logger.info(
            "Match end: match_id={}, winner={}, score={}, moves={}, replay={}",
            body.matchId, body.match_winner, body.final_score,
            match_logger.get_move_count(), filepath,
        )
        return {"status": "ok", "replay": str(filepath)}
    except RuntimeError as e:
        logger.warning("Match end without start: match_id={}, error={}", body.matchId, e)
        return {"status": "error", "message": str(e)}


@router.post("/game/start/match")
async def post_game_start_match(req: Request, body: GameStartRequest):
    """Register a new game within the active match (match-scoped game start)."""
    match_logger = _get_logger(req)
    # game_number is provided by the Forge side; fall back to 1 if absent.
    match_logger.start_game(
        game_id=body.gameId,
        game_number=body.game_number or 1,
        decklists=body.decklists,
    )
    logger.info(
        "Game start (match): match_id={}, game_id={}, game_number={}",
        body.matchId, body.gameId, body.game_number,
    )
    return {"status": "ok"}


@router.post("/game/sideboard")
async def post_sideboard(req: Request, body: SideboardRequest):
    """Record a player's sideboard changes for a game."""
    match_logger = _get_logger(req)
    match_logger.log_sideboard(
        game_id=body.gameId,
        player=body.player,
        main_to_side=body.main_to_side,
        side_to_main=body.side_to_main,
    )
    logger.debug(
        "Sideboard: match_id={}, game_id={}, player={}, main_to_side={}, side_to_main={}",
        body.matchId, body.gameId, body.player,
        len(body.main_to_side), len(body.side_to_main),
    )
    return {"status": "ok"}


@router.post("/game/move")
async def post_move(req: Request, body: MoveRequest):
    """Record a single action (with full board state) for a game.

    Also computes the model's value estimate for the position from *each*
    player's own perspective and records it in the unified eval history so the
    GUI's live "Evaluation History" graph updates after *every* player's
    action — not just the external AI's decisions. Each player's value is its
    own self-assessment (its hidden info visible, the opponent's masked) and is
    stored in that player's own series; no zero-sum negation is applied.
    """
    match_logger = _get_logger(req)
    move = {
        "turn": body.turn,
        "phase": body.phase,
        "player": body.player,
        "action": body.action,
        "action_type": body.action_type,
        "details": body.details,
        "board_state": body.board_state,
    }
    match_logger.log_move(game_id=body.gameId, move=move)

    # Estimate the position's value from EACH player's own perspective. The
    # board state is the FULL board (both hands), so we compute a value for
    # every player: the actor's self-assessment (its hand visible, opponent's
    # masked) and the opponent's self-assessment (e.g. the human's hand
    # visible, the AI's masked). Each is stored in that player's own series.
    # Best-effort: if no model is loaded or inference fails, the move is simply
    # not plotted.
    eval_history = req.app.state.eval_history  # type: ignore[union-attr]
    if eval_history is not None and body.board_state:
        engine = req.app.state.inference_engine  # type: ignore[union-attr]
        values: dict = {}
        for p in body.board_state.get("players", []) or []:
            pname = p.get("name") if isinstance(p, dict) else None
            if not pname:
                continue
            eval_history.set_player(pname)
            v = engine.estimate_value(body.board_state, perspective=pname)
            if v is not None:
                values[pname] = v
        if values:
            eval_history.add(
                game_id=body.gameId,
                turn=body.turn,
                values=values,
                actor=body.player,
                kind="move",
            )
    return {"status": "ok"}


@router.post("/game/end/match")
async def post_game_end_match(req: Request, body: GameEndRequest):
    """Finalize a game within the match: record winner + game reward."""
    match_logger = _get_logger(req)
    match_logger.end_game(
        game_id=body.gameId,
        winner=body.winner,
        total_turns=body.total_turns or 0,
    )
    # NOTE: the eval history is intentionally NOT cleared when a game ends. It
    # persists so the GUI's graph keeps showing the finished game's curve, and
    # is only reset when the NEXT game's first action arrives (EvalHistory.add()
    # auto-resets on a game_id change) or on a fresh server launch.
    logger.info(
        "Game end (match): match_id={}, game_id={}, winner={}, turns={}",
        body.matchId, body.gameId, body.winner, body.total_turns,
    )
    return {"status": "ok"}
