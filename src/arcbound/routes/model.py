"""Model management routes.

Provides a ``/model/reload`` endpoint so the GUI can make the running server
pick up a freshly-saved checkpoint (after training) or switch to a different
model — without a full server restart.
"""

from typing import Optional

from fastapi import APIRouter, Request
from loguru import logger

router = APIRouter()


@router.post("/model/reload")
async def reload_model(req: Request, model_name: Optional[str] = None):
    """(Re)load a model checkpoint into the running server.

    Called by the GUI after training completes or when a different model is
    selected, so the server uses the latest checkpoint on disk. ``model_name``
    is optional; if omitted, the last-used model is (re)loaded.

    Returns ``action`` = "loaded" (the server had no model before) or
    "reloaded" (it already had a model and swapped in a new one), plus the
    model name so the GUI can log it to the terminal.
    """
    model_manager = req.app.state.model_manager
    inference_engine = req.app.state.inference_engine

    was_loaded = model_manager.is_loaded
    ok = model_manager.load(model_name)
    if not ok:
        return {
            "success": False,
            "error": f"Could not load model '{model_name or 'last-used'}'",
        }

    # Point the inference engine at the fresh model so new decisions use it.
    inference_engine.set_model(model_manager.model)

    action = "reloaded" if was_loaded else "loaded"
    logger.info(
        "Model {} via /model/reload: '{}' from {}",
        action,
        model_manager.model_name,
        model_manager.checkpoint_path,
    )
    return {
        "success": True,
        "action": action,
        "modelName": model_manager.model_name,
        "checkpointPath": str(model_manager.checkpoint_path)
        if model_manager.checkpoint_path
        else None,
    }
