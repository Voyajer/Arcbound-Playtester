"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from loguru import logger

from arcbound.decision.engine import InferenceEngine
from arcbound.logging.eval_history import EvalHistory
from arcbound.logging.event_bus import EventBus
from arcbound.logging.match_logger import MatchLogger
from arcbound.logging.replay_logger import ReplayLogger
from arcbound.model_manager import ModelManager
from arcbound.routes import decision, game, health, match, model, monitoring, settings


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

    # Data directories live inside the project root (self-contained)
    project_root = Path(__file__).resolve().parent.parent.parent
    replays_dir = project_root / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)

    # Logs are split per server launch: logs/server-<timestamp>.log. Each
    # launch gets its own file so a long-running server's history is not
    # mixed with a previous run's. The GUI may pass ARCBOUND_LOG_FILE so the
    # server's loguru output and the subprocess stdout land in the SAME
    # per-launch file; otherwise the server picks its own timestamped path.
    import os
    from datetime import datetime

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env_log = os.getenv("ARCBOUND_LOG_FILE")
    if env_log:
        log_file = Path(env_log)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = logs_dir / f"server-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="50 MB",
        retention=14,
        encoding="utf-8",
    )
    logger.info("Log file: {}", log_file)

    # In-memory event bus for live GUI notifications (fallback / epsilon).
    app.state.event_bus = EventBus()

    # The match file (MatchLogger) is the ONLY replay artifact. Per-game replay
    # files are deprecated and are NEVER written — the ReplayLogger is kept
    # purely for in-memory decision tracking that the GUI's live monitoring
    # endpoints (status/decisions/evaluations) depend on.
    app.state.replay_logger = ReplayLogger(replays_dir, write_files=False)
    logger.info("Per-game replay files are disabled; only match files are written")

    # Match-level logger — writes ONE file per match (Bo3) with per-game +
    # match winners/rewards. Used by the Forge-side ExternalMatchLogger.
    app.state.match_logger = MatchLogger(replays_dir)

    # Unified per-action evaluation history (AI decisions + opponent moves),
    # normalized to the external AI's perspective. Backs the GUI's live
    # "Evaluation History" graph so it updates after every player's action.
    app.state.eval_history = EvalHistory()

    # Setup model manager — loads the last-used model checkpoint if available
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    app.state.model_manager = ModelManager(models_dir)
    app.state.model_manager.load()

    # Setup inference engine — wraps the loaded model (or falls back to heuristics)
    inference_cfg = config.get("inference") or {}
    epsilon = float(inference_cfg.get("epsilon", 0.15))
    app.state.inference_engine = InferenceEngine(
        app.state.model_manager.model,
        epsilon=epsilon,
        event_bus=app.state.event_bus,
    )
    logger.info("Inference epsilon-greedy exploration: {}", epsilon)

    logger.info("Arcbound AI server starting")
    logger.info("Config: {}", config)
    logger.info("Replays directory: {}", replays_dir)
    logger.info("Models directory: {}", models_dir)
    if app.state.model_manager.is_loaded:
        logger.info(
            "Model loaded: '{}' — inference will use the trained model",
            app.state.model_manager.model_name,
        )
    else:
        logger.warning(
            "No model loaded — all decisions will use fallback heuristics. "
            "Create a model in the GUI and train it to enable real inference."
        )
    yield
    # Shutdown
    # Finalize any in-progress replay so the game's decisions are written to disk
    # (the Java client does not call /game/end).
    if app.state.replay_logger.is_active:
        replay_path = app.state.replay_logger.finalize_current_game(result="ended")
        if replay_path:
            logger.info("Finalized in-progress replay on shutdown: {}", replay_path)
    if app.state.match_logger.is_active:
        match_path = app.state.match_logger.finalize_current_match(result="abandoned")
        if match_path:
            logger.info("Finalized in-progress match on shutdown: {}", match_path)
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
    app.include_router(match.router)
    app.include_router(model.router)
    app.include_router(settings.router)
    app.include_router(monitoring.router)

    return app


def main():
    """Main entry point for running the server."""
    import os

    host = os.getenv("ARCBOUND_HOST", "0.0.0.0")
    port = int(os.getenv("ARCBOUND_PORT", "8080"))

    import uvicorn

    logger.info("Starting server on {}:{}", host, port)
    uvicorn.run("arcbound.server:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":
    main()
