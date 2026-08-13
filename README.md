# Arcbound MTG AI

Transformer-based AI server for Magic: The Gathering (Forge).

## Overview

This project implements a transformer encoder-based AI that learns to play Magic: The Gathering using [card-forge](https://github.com/magefree/mage) as the rules engine. The AI server communicates with Forge via HTTP REST API.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
uvicorn arcbound.server:create_app --factory --host 0.0.0.0 --port 8090

# Run tests
pytest tests/ -v
```

## Project Structure

- `src/arcbound/` - Main package
  - `models/` - Pydantic models for API schemas
  - `routes/` - FastAPI route handlers
  - `decision/` - Decision logic (fallback + transformer)
  - `encoder/` - Transformer encoder implementation
  - `training/` - Training pipeline
  - `utils/` - Utility functions

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.
