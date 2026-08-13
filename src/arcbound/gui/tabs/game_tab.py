"""Game monitoring tab with action terminal, evaluation bar, and eval graph."""

import tkinter as tk
from tkinter import ttk
from typing import Optional

import httpx

from arcbound.gui.components.terminal import ActionTerminal
from arcbound.gui.components.eval_bar import EvaluationBar
from arcbound.gui.components.eval_graph import EvalGraph


class GameTab(ttk.Frame):
    """Tab for live game monitoring."""

    def __init__(self, master: tk.Misc, server_url: str = "http://localhost:8090", **kwargs):
        super().__init__(master, **kwargs)
        self._player_name = "AI"
        self._opponent_name = "Opponent"
        self.server_url = server_url.rstrip("/")
        self._polling = False
        self._last_count = 0

        self._build_ui()

    def _build_ui(self):
        # Vertical PanedWindow: terminal+eval-bar on top, eval graph below
        # NOTE: tk.PanedWindow (not ttk) is required for minsize support in Tk 8.6
        paned = tk.PanedWindow(self, orient=tk.VERTICAL, showhandle=True)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top pane: terminal (left) + eval bar (right)
        top = ttk.Frame(paned, padding=5)
        paned.add(top, minsize=150)

        # Left: terminal
        term_frame = ttk.LabelFrame(top, text="Action Terminal", padding=5)
        term_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.terminal = ActionTerminal(term_frame)
        self.terminal.pack(fill=tk.BOTH, expand=True)

        # Right: eval bar + controls
        right = ttk.Frame(top, width=120)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="Evaluation", font=("TkDefaultFont", 9, "bold")).pack(anchor="n", pady=5)

        self.eval_bar = EvaluationBar(right, player_name=self._player_name, opponent_name=self._opponent_name)
        self.eval_bar.pack(fill=tk.Y, pady=5)

        # Controls
        ctrl = ttk.Frame(right)
        ctrl.pack(fill=tk.X, pady=10)

        self.lbl_status = ttk.Label(ctrl, text="No game active", foreground="#888888")
        self.lbl_status.pack(anchor="w")

        ttk.Button(ctrl, text="Poll Server", command=self._toggle_polling).pack(anchor="w", pady=2)
        ttk.Button(ctrl, text="Clear Terminal", command=self.terminal.clear).pack(anchor="w", pady=2)

        # Bottom pane: eval graph
        graph_frame = ttk.LabelFrame(paned, text="Evaluation History", padding=5)
        paned.add(graph_frame, minsize=80)

        self.eval_graph = EvalGraph(graph_frame, height=120)
        self.eval_graph.pack(fill=tk.X)

    def set_players(self, player: str, opponent: str):
        self._player_name = player
        self._opponent_name = opponent
        self.eval_bar.set_names(player, opponent)

    def log_decision(self, description: str, chosen: str, reason: str = "", confidence: float = None, value: float = None, timeout: bool = False):
        self.terminal.log_decision(description, chosen, reason, confidence, value, timeout)
        if value is not None:
            self.eval_bar.set_eval(value)

    def log_event(self, message: str):
        self.terminal.log_event(message)

    def log_error(self, message: str):
        self.terminal.log_error(message)

    def update_status(self, text: str):
        self.lbl_status.config(text=text)

    def _toggle_polling(self):
        """Start/stop polling the server for new decisions."""
        if self._polling:
            self._polling = False
            self.lbl_status.config(text="Polling stopped", foreground="#ffaa44")
            return

        self._polling = True
        self.lbl_status.config(text="Polling server...", foreground="#66ff66")
        self._poll()

    def _poll(self):
        """Poll server for new decisions and eval data."""
        if not self._polling:
            return
        try:
            with httpx.Client(timeout=2.0) as client:
                # Get recent decisions
                resp = client.get(f"{self.server_url}/monitoring/decisions?limit=20")
                if resp.status_code == 200:
                    decisions = resp.json()
                    current_count = len(decisions)
                    # Only process new decisions
                    new = decisions[self._last_count:]
                    for dec in new:
                        self.log_decision(
                            description=dec.get("description", "Unknown"),
                            chosen=dec.get("action_taken", "?"),
                            reason=dec.get("action_reason", ""),
                            confidence=dec.get("model_confidence"),
                            value=dec.get("model_value_estimate"),
                            timeout=dec.get("timeout", False),
                        )
                        # Update eval graph
                        val = dec.get("model_value_estimate")
                        if val is not None:
                            self.eval_graph.add_point(dec.get("turn", 0), val)
                    self._last_count = current_count

                # Get status
                resp = client.get(f"{self.server_url}/monitoring/status")
                if resp.status_code == 200:
                    status = resp.json()
                    active = status.get("active", False)
                    cnt = status.get("decisions_logged", 0)
                    if active:
                        self.lbl_status.config(text=f"Game active — {cnt} decisions", foreground="#66ff66")
                    else:
                        self.lbl_status.config(text=f"No game — {cnt} decisions logged", foreground="#888888")
        except Exception as e:
            if self._polling:
                self.log_error(f"Poll error: {e}")
                self.lbl_status.config(text="Server unreachable", foreground="#ff4444")

        # Schedule next poll (1 second interval)
        self.after(1000, self._poll)
