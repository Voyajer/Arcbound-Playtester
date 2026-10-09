"""Game lifecycle routes — start and end notifications with replay logging."""

from typing import List, Optional

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import BaseModel, Field

from arcbound.logging.replay_logger import ReplayLogger

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models matching the Java JSON sent by ExternalAiHttpClient
# ---------------------------------------------------------------------------

class GameStartRequest(BaseModel):
    """Matches the JSON body from ExternalAiHttpClient.notifyGameStart().

    Java sends: {\"gameId\":\"...\",\"decklist\":...}
    """
    model_config = {"populate_by_name": True}

    gameId: str = Field(alias="gameId")
    game_type: Optional[str] = Field(default=None, alias="gameType")
    players: Optional[List[str]] = Field(default=None, alias="players")
    focal_player: Optional[str] = Field(default=None, alias="focalPlayer")
    decklists: Optional[dict[str, List[str]]] = Field(default=None, alias="decklists")
    # Java sends "decklist" (singular) — accept both forms
    decklist: Optional[dict[str, List[str]]] = Field(default=None, alias="decklist")


class GameEndRequest(BaseModel):
    """Matches the JSON body from ExternalAiHttpClient.notifyGameEnd().

    Java sends: {\"gameId\":\"...\",\"result\":\"...\"}
    """
    model_config = {"populate_by_name": True}

    gameId: str = Field(alias="gameId")
    result: Optional[str] = Field(default=None, alias="result")
    total_turns: Optional[int] = Field(default=None, alias="totalTurns")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _get_logger(req: Request) -> ReplayLogger:
    return req.app.state.replay_logger  # type: ignore[union-attr]


@router.post("/game/start")
async def post_game_start(req: Request, body: GameStartRequest):
    """Notify AI that a new game is starting. Called by ExternalAiHttpClient.notifyGameStart().

    Creates a new replay log file and stores metadata.
    """
    replay_logger = _get_logger(req)

    # Merge decklist (singular) into decklists if only the singular form was sent
    effective_decklists = body.decklists or body.decklist

    format_name = (body.game_type or "standard").lower()

    # Default players if not provided
    players = body.players or ["player-1", "player-2"]
    focal_player = body.focal_player or players[0]

    filename = replay_logger.start_game(
        game_id=body.gameId,
        players=players,
        focal_player=focal_player,
        format_name=format_name,
    )

    # Store decklists
    for player, cards in (effective_decklists or {}).items():
        replay_logger.set_decklist(player, cards)

    logger.info(
        "Game start: game_id={}, game_type={}, players={}, replay={}",
        body.gameId, format_name, players, filename,
    )
    return {"status": "ok", "replay": filename}


@router.post("/game/end")
async def post_game_end(req: Request, body: GameEndRequest):
    """Notify AI that a game has ended. Called by ExternalAiHttpClient.notifyGameEnd().

    Finalizes the replay log file and writes to disk.
    """
    replay_logger = _get_logger(req)
    result_str = body.result or "unknown"
    turns = body.total_turns or 0

    try:
        filepath = replay_logger.end_game(result=result_str, total_turns=turns)
        decision_count = len(replay_logger.get_decisions())
        logger.info(
            "Game end: game_id={}, result={}, turns={}, decisions={}, replay={}",
            body.gameId, result_str, turns, decision_count, filepath,
        )
        return {"status": "ok", "replay": str(filepath), "decisions_logged": decision_count}
    except RuntimeError as e:
        logger.warning("Game end without start: game_id={}, error={}", body.gameId, e)
        return {"status": "error", "message": str(e)}
