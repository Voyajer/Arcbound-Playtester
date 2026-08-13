"""Game lifecycle routes — start and end notifications with replay logging."""

from typing import List

from fastapi import APIRouter, Request
from loguru import logger

from arcbound.logging.replay_logger import ReplayLogger

router = APIRouter()


def _get_logger(req: Request) -> ReplayLogger:
    return req.app.state.replay_logger  # type: ignore[union-attr]


class _GameStartBody:
    """Thin wrapper for the game/start POST body."""
    def __init__(self, game_id: str, game_type: str | None = None, players: List[str] | None = None, focal_player: str | None = None, decklists: dict[str, List[str]] | None = None):
        self.game_id = game_id
        self.game_type = game_type
        self.players = players or []
        self.focal_player = focal_player
        self.decklists = decklists or {}


class _GameEndBody:
    """Thin wrapper for the game/end POST body."""
    def __init__(self, game_id: str, result: str | None = None, total_turns: int | None = None):
        self.game_id = game_id
        self.result = result
        self.total_turns = total_turns or 0


@router.post("/game/start")
async def post_game_start(req: Request, game_id: str, game_type: str | None = None, players: List[str] | None = None, focal_player: str | None = None, decklists: dict[str, List[str]] | None = None):
    """Notify AI that a new game is starting. Called by ExternalAiHttpClient.notifyGameStart().

    Creates a new replay log file and stores metadata.
    """
    replay_logger = _get_logger(req)
    format_name = (game_type or "standard").lower()

    # Default players if not provided
    if not players:
        players = ["player-1", "player-2"]
    if not focal_player:
        focal_player = players[0]

    filename = replay_logger.start_game(
        game_id=game_id,
        players=players,
        focal_player=focal_player,
        format_name=format_name,
    )

    # Store decklists
    for player, cards in (decklists or {}).items():
        replay_logger.set_decklist(player, cards)

    logger.info("Game start: game_id={}, game_type={}, players={}, replay={}", game_id, game_type, players, filename)
    return {"status": "ok", "replay": filename}


@router.post("/game/end")
async def post_game_end(req: Request, game_id: str, result: str | None = None, total_turns: int | None = None):
    """Notify AI that a game has ended. Called by ExternalAiHttpClient.notifyGameEnd().

    Finalizes the replay log file and writes to disk.
    """
    replay_logger = _get_logger(req)
    result_str = result or "unknown"
    turns = total_turns or 0

    try:
        filepath = replay_logger.end_game(result=result_str, total_turns=turns)
        decision_count = len(replay_logger.get_decisions())
        logger.info("Game end: game_id={}, result={}, turns={}, decisions={}, replay={}", game_id, result_str, turns, decision_count, filepath)
        return {"status": "ok", "replay": str(filepath), "decisions_logged": decision_count}
    except RuntimeError as e:
        logger.warning("Game end without start: game_id={}, error={}", game_id, e)
        return {"status": "error", "message": str(e)}
