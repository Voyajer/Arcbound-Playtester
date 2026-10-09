"""Main Tkinter application window for Arcbound AI Server playtester.

Layout:
+----------------------------------------------------------+
|  Arcbound AI Server v0.1.1                               |
+------------------+  +-----------------------------------+
|                  |  |                                   |
|  Model List      |  |         Content Area              |
|  (Left Panel)    |  |    (Tabbed interface)             |
|                  |  |                                   |
|  [New] [Load]    |  |  Tabs: Game | Training | Analysis |
|  [Rename] [Del]  |  |  Settings                               |
|                  |  |                                   |
+------------------+  +-----------------------------------+
|  Server: [Running]  Port: <dynamic>  |  [Start] [Stop]       |
+----------------------------------------------------------+
"""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Optional

import httpx
import yaml

from arcbound.gui.components.model_list import ModelList
from arcbound.gui.tabs.model_tab import ModelTab
from arcbound.gui.tabs.training_tab import TrainingTab
from arcbound.gui.tabs.game_tab import GameTab
from arcbound.gui.tabs.analysis_tab import AnalysisTab
from arcbound.gui.tabs.settings_tab import SettingsTab


def _load_server_config() -> dict:
    """Load server config from config/default.yaml, falling back to defaults."""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("server", {})
        except Exception:
            pass
    return {}


class ArcboundApp(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Arcbound AI Server v0.1.1")

        # Data directories (self-contained inside the project root). Loaded
        # before geometry so the saved window size/position can be restored.
        self.data_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.models_dir = self.data_dir / "models"
        self.replays_dir = self.data_dir / "replays"
        self.config_path = self.data_dir / "config.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self._app_config = self._load_app_config()

        self._set_initial_geometry()
        self.minsize(800, 500)

        # Load server config (port, host, timeout)
        self._server_config = _load_server_config()
        self.server_port = int(os.getenv("ARCBOUND_PORT", self._server_config.get("port", 8080)))
        self.server_host = os.getenv("ARCBOUND_HOST", self._server_config.get("host", "0.0.0.0"))

        # Server state
        self._server_running: bool = False
        self._server_process: Optional[subprocess.Popen] = None
        self._server_log_file = None
        self._current_model: Optional[str] = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-select the last used model so the server loads it on startup
        last_model = self._app_config.get("last_model")
        if last_model and (self.models_dir / last_model).is_dir():
            self.model_list.select_model(last_model)

    def _set_initial_geometry(self):
        """Restore the saved window geometry, or default to 1250x1000 centered.

        The window remembers its size and position across launches (saved in
        config.json). On first run it opens at 1250x1000, clamped to the screen
        so the whole window (including the status bar) stays visible.
        """
        saved = self._app_config.get("geometry")
        if saved:
            try:
                self.geometry(saved)
                return
            except Exception:
                pass
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        # Leave room for the OS taskbar and window decorations.
        margin = 60
        w = min(1250, max(800, screen_w - margin))
        h = min(1000, max(500, screen_h - margin))
        x = max(0, (screen_w - w) // 2)
        y = max(0, (screen_h - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # Configure styles
        self.configure(bg="#1e1e1e")

        # Pack order matters: the status bar (side=BOTTOM) and title bar
        # (side=TOP) must be packed *before* the expanding main frame, so
        # their space is reserved first. If the main frame (expand=True)
        # were packed first it would consume all remaining vertical space
        # and squeeze the status bar down to 1px.

        # Bottom status bar (packed first so its height is reserved).
        status = ttk.Frame(self, padding=(10, 5))
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_server = ttk.Label(status, text="Server: Stopped", foreground="#ff6666")
        self.lbl_server.pack(side=tk.LEFT, padx=5)
        self.lbl_port = ttk.Label(status, text=f"Port: {self.server_port}")
        self.lbl_port.pack(side=tk.LEFT, padx=5)
        self.lbl_model = ttk.Label(status, text="Model: —")
        self.lbl_model.pack(side=tk.LEFT, padx=20)

        self.btn_server = ttk.Button(status, text="Start Server", command=self._toggle_server)
        self.btn_server.pack(side=tk.RIGHT, padx=5)
        ttk.Button(status, text="Quit", command=self.quit).pack(side=tk.RIGHT, padx=5)

        # Top title bar
        title = ttk.Frame(self, padding=(10, 5))
        title.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(title, text="Arcbound AI Server v0.1.1", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        # Main content: left panel + right tabs (expands to fill the rest)
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - model list
        left = ttk.Frame(main, width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)

        self.model_list = ModelList(
            left,
            self.models_dir,
            on_select=self._on_model_select,
            on_load=self._on_model_load,
        )
        self.model_list.pack(fill=tk.BOTH, expand=True)

        # Right panel - notebook tabs
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Pass server URL to game tab so polling uses the correct port
        server_url = f"http://localhost:{self.server_port}"
        self.game_tab = GameTab(self.notebook, server_url=server_url)
        self.training_tab = TrainingTab(
            self.notebook, self.replays_dir, self.models_dir, on_stop=self._on_training_finished
        )
        self.analysis_tab = AnalysisTab(self.notebook, self.replays_dir, self.models_dir)
        self.model_tab = ModelTab(self.notebook, self.models_dir, on_model_loaded=self._on_model_load)
        self.settings_tab = SettingsTab(
            self.notebook, self.models_dir, self.config_path, on_apply=self._on_settings_apply
        )

        self.notebook.add(self.game_tab, text="Game")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.settings_tab, text="Settings")

    def _load_app_config(self) -> dict:
        """Load app-level config (last_model, etc.) from <project>/config.json."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_app_config(self):
        """Persist app-level config to <project>/config.json."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self._app_config, f, indent=2)
        except Exception:
            pass

    def _on_model_select(self, name: str):
        """Called when a model is clicked in the model list.

        Only updates the info panels and remembers the selection — it does NOT
        switch the active model on a running server. Use the Load button
        (``_on_model_load``) for that.
        """
        self._current_model = name
        self.lbl_model.config(text=f"Model: {name}")
        # Remember as last used model so the server auto-loads it next time
        self._app_config["last_model"] = name
        self._save_app_config()
        # Notify all tabs
        self.model_tab.select_model(name)
        self.training_tab.select_model(name)
        self.settings_tab.select_model(name)

    def _on_model_load(self, name: str):
        """Called by the Load button — actually switch the active model.

        Updates the selection (info panels, last-used) and, if the server is
        running, makes it switch to this model now so the next game uses it.
        """
        self._on_model_select(name)
        self._reload_model(name)

    def _on_training_finished(self):
        """Refresh the model tab after a training run so the displayed
        Games Trained / Total Steps reflect the checkpoint just saved."""
        if self._current_model:
            self.model_tab.select_model(self._current_model)
        # If the server is running, make it pick up the checkpoint that was
        # just saved so the next game uses the newly trained model.
        self._reload_model(self._current_model)

    def _reload_model(self, model_name: Optional[str]):
        """Ask the running server to (re)load a model checkpoint.

        No-op when the server isn't running (it will load the model on
        startup). Runs in a background thread because loading a checkpoint
        can take a few seconds and must not block the UI. The outcome is
        logged to the Game tab terminal with the model name.
        """
        if not model_name or not self._server_running:
            return

        def _do_reload():
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"http://127.0.0.1:{self.server_port}/model/reload",
                        params={"model_name": model_name},
                    )
                    data = resp.json()
            except Exception as e:
                self.after(0, self.game_tab.log_error, f"Model reload failed: {e}")
                return
            if data.get("success"):
                action = data.get("action", "loaded")
                name = data.get("modelName", model_name)
                msg = f"Model {action}: '{name}'"
                self.after(0, self.game_tab.log_model, msg)
            else:
                self.after(
                    0,
                    self.game_tab.log_error,
                    f"Model reload failed: {data.get('error', 'unknown error')}",
                )

        threading.Thread(target=_do_reload, daemon=True).start()

    def _on_settings_apply(self):
        """Push the just-saved settings to the running server (hot-apply).

        Called by the Settings tab's Apply button after it has persisted the
        values to config/default.yaml. The server re-reads the file and applies
        every runtime-mutable setting (epsilon) immediately — no restart, so it
        works mid-match. No-op when the server isn't running (it reads the
        config on startup). Runs in a background thread so the UI stays
        responsive; the outcome is logged to the Game tab terminal.
        """
        if not self._server_running:
            return

        def _do_apply():
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"http://127.0.0.1:{self.server_port}/settings/apply"
                    )
                    data = resp.json()
            except Exception as e:
                self.after(0, self.game_tab.log_error, f"Settings apply failed: {e}")
                return
            if data.get("success"):
                applied = data.get("applied", {})
                parts = ", ".join(f"{k}={v}" for k, v in applied.items())
                self.after(0, self.game_tab.log_event, f"Settings applied: {parts}")
            else:
                self.after(
                    0,
                    self.game_tab.log_error,
                    f"Settings apply failed: {data.get('error', 'unknown error')}",
                )

        threading.Thread(target=_do_apply, daemon=True).start()

    def _toggle_server(self):
        """Toggle server start/stop."""
        if self._server_running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        """Launch the uvicorn server as a subprocess and wait for readiness."""
        if self._server_running:
            return
        self.lbl_server.config(text="Server: Starting...", foreground="#ffaa44")
        self.game_tab.update_status("Starting server...")
        self.game_tab.log_event(f"Starting server on {self.server_host}:{self.server_port}...")

        # One log file per launch: logs/server-<timestamp>.log. The server's
        # loguru output and the subprocess stdout both go here (the server is
        # told the path via ARCBOUND_LOG_FILE so they share the same file).
        from datetime import datetime

        logs_dir = self.data_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"server-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        try:
            self._server_log_file = open(log_path, "ab")
            env = os.environ.copy()
            env["ARCBOUND_LOG_FILE"] = str(log_path)
            self._server_process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "arcbound.server:create_app", "--factory",
                    "--host", self.server_host,
                    "--port", str(self.server_port),
                ],
                cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
                stdout=self._server_log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except Exception as e:
            self._server_process = None
            self._close_server_log()
            self._set_server_stopped(f"Failed to start server: {e}")
            return

        self._wait_for_ready()

    def _wait_for_ready(self, attempts: int = 60):
        """Poll /health (non-blocking) until the server responds or fails."""
        proc = self._server_process
        if self._check_health():
            if proc is not None and proc.poll() is None:
                # Our subprocess is serving.
                self._server_running = True
                self.lbl_server.config(text="Server: Running", foreground="#66ff66")
                self.btn_server.config(text="Stop Server")
                self.lbl_port.config(text=f"Port: {self.server_port}")
                self.game_tab.update_status(f"Server running on port {self.server_port}")
                self.game_tab.log_event(f"Server started on port {self.server_port}.")
            else:
                # An external server is already listening on this port.
                self._server_process = None
                self._close_server_log()
                self._server_running = True
                self.lbl_server.config(text="Server: Running", foreground="#66ff66")
                self.btn_server.config(text="Stop Server")
                self.lbl_port.config(text=f"Port: {self.server_port}")
                self.game_tab.update_status(f"Server running on port {self.server_port} (external)")
                self.game_tab.log_event(f"Server already running on port {self.server_port} (external process).")
            return

        if proc is not None and proc.poll() is not None:
            # Process died during startup.
            self._server_process = None
            tail = self._read_log_tail()
            self._close_server_log()
            self._set_server_stopped(f"Server process exited (code {proc.returncode}).\n{tail}")
            return

        if attempts <= 0:
            self._stop_server_process()
            self._set_server_stopped("Server did not become ready within 30s. See the newest log in the logs/ folder")
            return

        self.after(500, self._wait_for_ready, attempts - 1)

    def _check_health(self) -> bool:
        """Return True if the server's /health endpoint responds 200."""
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"http://127.0.0.1:{self.server_port}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def _stop_server(self):
        """Stop the server subprocess (if we started it) and update UI."""
        self._stop_server_process()
        self._server_running = False
        self.lbl_server.config(text="Server: Stopped", foreground="#ff6666")
        self.btn_server.config(text="Start Server")
        self.game_tab.update_status("Server stopped")
        self.game_tab.log_event("Server stopped.")

    def _stop_server_process(self):
        """Terminate the server subprocess and release the log file."""
        proc = self._server_process
        self._server_process = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        self._close_server_log()

    def _close_server_log(self):
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None

    def _read_log_tail(self, n: int = 2000) -> str:
        """Read the last n bytes of the newest server log for diagnostics.

        Logs are split per launch under logs/ (server-<timestamp>.log); this
        reads the most recent one.
        """
        logs_dir = self.data_dir / "logs"
        try:
            if logs_dir.is_dir():
                files = sorted(logs_dir.glob("server-*.log"))
                if files:
                    log_path = files[-1]
                else:
                    return ""
            else:
                return ""
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - n))
                return f.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _set_server_stopped(self, message: str):
        """Reset UI to stopped state and log an error message."""
        self._server_running = False
        self.lbl_server.config(text="Server: Stopped", foreground="#ff6666")
        self.btn_server.config(text="Start Server")
        self.game_tab.update_status("Server stopped")
        self.game_tab.log_error(message)

    def _on_close(self):
        """Clean up the server subprocess when the window closes."""
        self._save_geometry()
        self._stop_server_process()
        self.quit()

    def _save_geometry(self):
        """Persist the window geometry so it is restored on next launch."""
        try:
            self._app_config["geometry"] = self.geometry()
            self._save_app_config()
        except Exception:
            pass

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
