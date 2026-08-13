"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from loguru import logger

from arcbound.logging.replay_logger import ReplayLogger
from arcbound.routes import decision, game, health, monitoring


def load_config() -> dict:
    """Load configuration from config/default.yaml."""
    import os

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        logger.warning("Config file not found at {}, using defaults", config_path)
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    config = load_config()
    app.state.config = config

    # Setup replay logger
    data_dir = Path.home() / ".arcbound"
    replays_dir = data_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    app.state.replay_logger = ReplayLogger(replays_dir)

    logger.info("Arcbound AI server starting")
    logger.info("Config: {}", config)
    logger.info("Replays directory: {}", replays_dir)
    yield
    # Shutdown
    logger.info("Arcbound AI server shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Arcbound MTG AI Server",
        description="Transformer-based AI for Magic: The Gathering",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(health.router)
    app.include_router(decision.router)
    app.include_router(game.router)
    app.include_router(monitoring.router)

    return app


def main():
    """Main entry point for running the server."""
    import os

    host = os.getenv("ARCBOUND_HOST", "0.0.0.0")
    port = int(os.getenv("ARCBOUND_PORT", "8090"))

    import uvicorn

    logger.info("Starting server on {}:{}", host, port)
    uvicorn.run("arcbound.server:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":
    main()
