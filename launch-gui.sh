#!/usr/bin/env bash
# Launch the Arcbound AI Server GUI.
# Usage: ./launch-gui.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Launch the GUI
cd "$SCRIPT_DIR"
arcbound-gui
