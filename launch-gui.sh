#!/usr/bin/env bash
# Launch the Arcbound AI Server GUI.
# Usage: ./launch-gui.sh
#
# Automatically creates a virtual environment and installs dependencies
# if they don't already exist.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/update dependencies (editable install picks up pyproject.toml)
echo "Installing dependencies..."
pip install -q -e "$SCRIPT_DIR"

# Launch the GUI
cd "$SCRIPT_DIR"
arcbound-gui
