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
    """Return current server status for the GUI.

    ``active`` is recency-based (a decision within the last 30s) so the GUI does
    not show "Game active" when no game is actually in progress — the Java client
    never calls /game/end, so a replay log stays open after a game finishes.
    ``last_seq`` and ``game_id`` let the GUI (a) skip decisions left over from a
    previous session on startup, and (b) detect when a new game begins in a match
    so it can reset its terminal/graph and seq cursor.
    """
    replay_logger = _get_logger(req)
    return {
        "active": replay_logger.is_recently_active,
        "decisions_logged": len(replay_logger.get_decisions()),
        "last_seq": replay_logger.last_seq,
        "game_id": replay_logger.game_id,
    }


@router.get("/monitoring/events")
async def get_events(req: Request, since: int = 0):
    """Return live notification events for the GUI terminal.

    The decision engine emits events here when something the user needs to see
    happens: a fallback was used (level ``error`` — something is wrong) or
    epsilon exploration triggered (level ``warning``). The GUI polls this with
    an incremental ``since`` seq, exactly like ``/monitoring/decisions``.
    """
    event_bus = getattr(req.app.state, "event_bus", None)
    if event_bus is None:
        return []
    return event_bus.since(since)


@router.get("/monitoring/decisions")
async def get_recent_decisions(req: Request, limit: int = 50, since: int = 0):
    """Return decisions for the GUI action terminal.

    If `since` > 0, return only decisions with seq > since (incremental
    polling). Otherwise return the most recent `limit` decisions.
    """
    replay_logger = _get_logger(req)
    if since > 0:
        return replay_logger.get_decisions_since(since)
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


@router.get("/monitoring/eval_history")
async def get_eval_history(req: Request):
    """Return the current game's per-action evaluation history, per player.

    Unlike ``/monitoring/evaluations`` (AI decisions only), this includes a
    value point for *every* player's action — the external AI's decisions and
    the opponent's event-based moves. Each player (AI *and* human) gets its own
    series of *its own* self-assessments, computed from that player's own
    perspective (its own hidden info visible, the opponent's masked). No
    zero-sum negation is applied: with hidden information, the opponent's
    self-assessment is a different information set and is shown in the
    opponent's own series instead.

    This is what powers the GUI's per-player tabs, including "how the external
    AI thinks the human is doing" (the human's self-assessment series).

    Shape::

        {
          "players": ["ai-0", "human-0"],  # all players seen, in order
          "ai_players": ["ai-0"],          # the external AIs
          "active": "ai-0",                # most recent AI to act (or null)
          "series": {"ai-0": [...], "human-0": [...]},
        }

    Each series is a list of ``{seq, turn, value, player, kind}`` points.
    """
    eval_history = req.app.state.eval_history  # type: ignore[union-attr]
    if eval_history is None:
        return {"players": [], "active": None, "series": {}}
    # The eval history is already scoped to the current game (it resets when the
    # game id changes), so return its per-player series directly. We do NOT
    # filter by the replay logger's game_id here: the eval history is keyed by
    # the Forge game id (consistent across /decision and /game/move), whereas
    # the replay logger uses the per-controller UUID, so the two would never
    # match.
    return eval_history.get_all()
