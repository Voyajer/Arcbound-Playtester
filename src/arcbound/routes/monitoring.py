"""Monitoring routes for live GUI connection.

Provides endpoints for the GUI to poll recent decisions and game state.
"""

from fastapi import APIRouter, Request
from loguru import logger

from arcbound.logging.replay_logger import ReplayLogger

router = APIRouter()


def _get_logger(req: Request) -> ReplayLogger:
    return req.app.state.replay_logger  # type: ignore[union-attr]


@router.get("/monitoring/status")
async def get_status(req: Request):
    """Return current server status for the GUI."""
    replay_logger = _get_logger(req)
    return {
        "active": replay_logger.is_active,
        "decisions_logged": len(replay_logger.get_decisions()),
    }


@router.get("/monitoring/decisions")
async def get_recent_decisions(req: Request, limit: int = 50):
    """Return the most recent decisions for the GUI action terminal."""
    replay_logger = _get_logger(req)
    decisions = replay_logger.get_decisions()
    return decisions[-limit:]


@router.get("/monitoring/evaluations")
async def get_evaluations(req: Request):
    """Return evaluation history for the eval graph.

    Extracts model_value_estimate from decisions, paired with decision index as turn.
    """
    replay_logger = _get_logger(req)
    decisions = replay_logger.get_decisions()
    evals = []
    for i, dec in enumerate(decisions):
        val = dec.get("model_value_estimate")
        if val is not None:
            evals.append({"turn": dec.get("turn", i), "value": val})
    return evals
