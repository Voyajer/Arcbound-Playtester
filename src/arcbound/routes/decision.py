"""Decision route — main AI endpoint with replay logging."""

from datetime import datetime

from fastapi import APIRouter, Request
from loguru import logger

from arcbound.decision.fallback import generate_fallback_decision
from arcbound.logging.replay_logger import ReplayLogger
from arcbound.models.decision import AiDecisionRequest, AiDecisionResponse

router = APIRouter()


def _get_logger(req: Request) -> ReplayLogger:
    return req.app.state.replay_logger  # type: ignore[union-attr]


@router.post("/decision")
async def post_decision(req: Request, request_body: AiDecisionRequest) -> AiDecisionResponse:
    """Main decision endpoint. Called by ExternalAiHttpClient.requestDecision().

    Logs each decision to the active replay file, then returns a fallback decision.
    Will be replaced with transformer inference once the model is trained.
    """
    replay_logger = _get_logger(req)

    logger.info(
        "Decision request: game_id={}, type={}",
        request_body.gameId,
        request_body.decisionRequest.type,
    )

    # TODO: Replace with transformer inference once model is available
    response = generate_fallback_decision(request_body)

    # Log decision to replay
    if replay_logger.is_active:
        decision_log = {
            "turn": None,  # Not provided by Forge; could be extracted from boardState
            "phase": None,
            "player": None,
            "decision_type": request_body.decisionRequest.type,
            "description": request_body.decisionRequest.description,
            "options_presented": [opt.label for opt in request_body.decisionRequest.options],
            "action_taken": response.selectedOptionId or "",
            "action_reason": response.reasoning or "fallback",
            "model_confidence": None,  # Will be populated by model inference
            "model_value_estimate": None,  # Will be populated by model value head
            "timestamp": datetime.now().isoformat(),
            "timeout": False,
        }
        replay_logger.log_decision(decision_log)

    logger.info(
        "Decision response: type={}, selectedOptionId={}",
        response.decisionType,
        response.selectedOptionId,
    )

    return response
