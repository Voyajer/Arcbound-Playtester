"""Health check route."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(req: Request):
    """Health check endpoint. Called by ExternalAiHttpClient.isServerAvailable()."""
    model_status = {}
    model_manager = getattr(req.app.state, "model_manager", None)
    if model_manager is not None:
        model_status = model_manager.status()
    return {
        "status": "healthy",
        "modelLoaded": model_status.get("modelLoaded", False),
        "modelName": model_status.get("modelName"),
        "version": "0.1.0",
    }
