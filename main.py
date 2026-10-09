#!/usr/bin/env python3
"""Entry point for Arcbound AI Server GUI.

Launches the Tkinter GUI application.
Usage:
    python main.py
    # Or after installation:
    arcbound-gui
"""

import sys
from pathlib import Path

# Ensure the src directory is on the path when running from repo root
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))


def main():
    from arcbound.gui.app import ArcboundApp

    app = ArcboundApp()
    app.mainloop()


if __name__ == "__main__":
    main()
