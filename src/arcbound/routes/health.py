"""Health check route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint. Called by ExternalAiHttpClient.isServerAvailable()."""
    return {
        "status": "healthy",
        "modelLoaded": False,
        "version": "0.1.0",
    }
