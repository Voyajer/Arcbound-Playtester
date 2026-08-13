"""Main Tkinter application window for Arcbound AI Server playtester.

Layout:
+----------------------------------------------------------+
|  Arcbound AI Server v0.1.0                               |
+------------------+  +-----------------------------------+
|                  |  |                                   |
|  Model List      |  |         Content Area              |
|  (Left Panel)    |  |    (Tabbed interface)             |
|                  |  |                                   |
|  [New] [Load]    |  |  Tabs: Game | Training | Analysis |
|  [Rename] [Del]  |  |  Settings                               |
|                  |  |                                   |
+------------------+  +-----------------------------------+
|  Server: [Running]  Port: 8090  |  [Start] [Stop]       |
+----------------------------------------------------------+
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Optional

from arcbound.gui.components.model_list import ModelList
from arcbound.gui.tabs.model_tab import ModelTab
from arcbound.gui.tabs.training_tab import TrainingTab
from arcbound.gui.tabs.game_tab import GameTab
from arcbound.gui.tabs.analysis_tab import AnalysisTab
from arcbound.gui.tabs.settings_tab import SettingsTab


class ArcboundApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Arcbound AI Server v0.1.0")
        self.geometry("1250x1250")
        self.minsize(800, 500)

        # Data directories
        self.data_dir = Path.home() / ".arcbound"
        self.models_dir = self.data_dir / "models"
        self.replays_dir = self.data_dir / "replays"
        self.config_path = self.data_dir / "config.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        # Server state
        self._server_running: bool = False
        self._current_model: Optional[str] = None

        self._build_ui()

    def _build_ui(self):
        # Configure styles
        self.configure(bg="#1e1e1e")

        # Top title bar
        title = ttk.Frame(self, padding=(10, 5))
        title.pack(fill=tk.X)
        ttk.Label(title, text="Arcbound AI Server v0.1.0", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        # Main content: left panel + right tabs
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - model list
        left = ttk.Frame(main, width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)

        self.model_list = ModelList(left, self.models_dir, on_select=self._on_model_select)
        self.model_list.pack(fill=tk.BOTH, expand=True)

        # Right panel - notebook tabs
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.game_tab = GameTab(self.notebook)
        self.training_tab = TrainingTab(self.notebook, self.replays_dir)
        self.analysis_tab = AnalysisTab(self.notebook, self.replays_dir)
        self.model_tab = ModelTab(self.notebook, self.models_dir)
        self.settings_tab = SettingsTab(self.notebook, self.models_dir, self.config_path)

        self.notebook.add(self.game_tab, text="Game")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.settings_tab, text="Settings")

        # Bottom status bar
        status = ttk.Frame(self, padding=(10, 5))
        status.pack(fill=tk.X)

        self.lbl_server = ttk.Label(status, text="Server: Stopped", foreground="#ff6666")
        self.lbl_server.pack(side=tk.LEFT, padx=5)
        self.lbl_port = ttk.Label(status, text="Port: 8090")
        self.lbl_port.pack(side=tk.LEFT, padx=5)
        self.lbl_model = ttk.Label(status, text="Model: —")
        self.lbl_model.pack(side=tk.LEFT, padx=20)

        ttk.Button(status, text="Start Server", command=self._toggle_server).pack(side=tk.RIGHT, padx=5)
        ttk.Button(status, text="Quit", command=self.quit).pack(side=tk.RIGHT, padx=5)

    def _on_model_select(self, name: str):
        """Called when a model is selected in the model list."""
        self._current_model = name
        self.lbl_model.config(text=f"Model: {name}")
        # Notify all tabs
        self.model_tab.select_model(name)
        self.training_tab.select_model(name)
        self.settings_tab.select_model(name)

    def _toggle_server(self):
        """Toggle server start/stop."""
        if self._server_running:
            self._server_running = False
            self.lbl_server.config(text="Server: Stopped", foreground="#ff6666")
            self.game_tab.update_status("Server stopped")
            self.game_tab.log_event("Server stopped.")
        else:
            self._server_running = True
            self.lbl_server.config(text="Server: Running", foreground="#66ff66")
            self.game_tab.update_status("Server running on port 8090")
            self.game_tab.log_event("Server started on port 8090.")

    def log_decision(self, description: str, chosen: str, reason: str = "", confidence: float = None, value: float = None, timeout: bool = False):
        """Log a decision to the game tab terminal."""
        self.game_tab.log_decision(description, chosen, reason, confidence, value, timeout)

    def log_event(self, message: str):
        """Log an event to the game tab terminal."""
        self.game_tab.log_event(message)

    def log_error(self, message: str):
        """Log an error to the game tab terminal."""
        self.game_tab.log_error(message)


def main():
    """Entry point for arcbound-gui command."""
    app = ArcboundApp()
    app.mainloop()
