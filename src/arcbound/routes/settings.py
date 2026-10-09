"""Settings hot-apply route.

Provides a ``/settings/apply`` endpoint so the GUI can push freshly-saved
settings to the *running* server — without a restart. The server re-reads
``config/default.yaml`` and applies every runtime-mutable setting immediately
(epsilon today; host/port/timeout require a restart and are reported as such).
"""

from fastapi import APIRouter, Request
from loguru import logger

router = APIRouter()


@router.post("/settings/apply")
async def apply_settings(req: Request):
    """Re-read config/default.yaml and hot-apply runtime-mutable settings.

    Called by the GUI's Apply button after it has persisted the settings, so
    the running server picks them up mid-match. Returns the applied values so
    the GUI can confirm what took effect.
    """
    # Imported lazily to avoid a circular import (server.py imports this
    # module's router at startup, before load_config is defined).
    from arcbound.server import load_config

    config = load_config()
    req.app.state.config = config

    inference_cfg = config.get("inference") or {}
    epsilon = float(inference_cfg.get("epsilon", 0.15))
    req.app.state.inference_engine.set_epsilon(epsilon)
    logger.info("Settings applied via /settings/apply: epsilon={}", epsilon)

    return {
        "success": True,
        "applied": {"epsilon": req.app.state.inference_engine.epsilon},
        "restart_required": ["host", "port", "timeout_ms"],
    }
