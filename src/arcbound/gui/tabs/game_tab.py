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

    def __init__(self, master: tk.Misc, server_url: str = "http://localhost:8080", **kwargs):
        super().__init__(master, **kwargs)
        self._player_name = "AI"
        self._opponent_name = "Opponent"
        self.server_url = server_url.rstrip("/")
        self._polling = False
        # Last decision seq seen; used for incremental polling (seq > last).
        self._last_seq = 0
        # Last event seq seen; used for incremental polling of live
        # notifications (fallback / epsilon) from the server's event bus.
        self._last_event_seq = 0
        # Startup sync: on the first successful poll we adopt the server's
        # current seq/game so decisions left over from a previous session are
        # not replayed into the terminal and graph.
        self._synced = False
        # The game_id we are currently displaying (None until first sync).
        self._current_game_id = None

        self._build_ui()

        # Polling is on by default so the terminal shows live decisions
        # as soon as the server is reachable.
        self.after(1000, self._start_polling)

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
        self.eval_bar.pack(fill=tk.BOTH, expand=True, pady=5)

        # Controls
        ctrl = ttk.Frame(right)
        ctrl.pack(fill=tk.X, pady=10)

        self.lbl_status = ttk.Label(ctrl, text="No game active", foreground="#888888")
        self.lbl_status.pack(anchor="w")

        ttk.Button(ctrl, text="Poll Server", command=self._toggle_polling).pack(anchor="w", pady=2)
        ttk.Button(ctrl, text="Clear Terminal", command=self.terminal.clear).pack(anchor="w", pady=2)

        # Bottom pane: eval graph — one tab per player (external AI *and*
        # human), each showing that player's own self-assessment (its own
        # hidden info visible, the opponent's masked). This includes a tab for
        # "how the external AI thinks the human is doing" (the human's
        # self-assessment), and tracks AI-vs-AI matches independently.
        graph_frame = ttk.LabelFrame(paned, text="Evaluation History (every action)", padding=5)
        paned.add(graph_frame, minsize=80)

        # Controls row: auto-switch checkbox.
        graph_ctrl = ttk.Frame(graph_frame)
        graph_ctrl.pack(fill=tk.X, pady=(0, 4))
        self.var_auto_switch = tk.BooleanVar(value=True)
        self.chk_auto_switch = ttk.Checkbutton(
            graph_ctrl,
            text="Auto-switch to active AI",
            variable=self.var_auto_switch,
        )
        self.chk_auto_switch.pack(side=tk.LEFT)

        # One tab per external AI player; tabs are created as the server
        # reports which AIs are playing.
        self.eval_notebook = ttk.Notebook(graph_frame)
        self.eval_notebook.pack(fill=tk.BOTH, expand=True)
        self._eval_graphs: dict = {}  # player name -> EvalGraph
        self._eval_active: Optional[str] = None

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

    def log_model(self, message: str):
        self.terminal.log_model(message)

    def log_warning(self, message: str):
        self.terminal.log_warning(message)

    def log_error(self, message: str):
        self.terminal.log_error(message)

    def update_status(self, text: str):
        self.lbl_status.config(text=text)

    def _start_polling(self):
        """Start polling (used for the default-on behavior at startup)."""
        if not self._polling:
            self._polling = True
            self.lbl_status.config(text="Polling server...", foreground="#66ff66")
            self._poll()

    def _toggle_polling(self):
        """Start/stop polling the server for new decisions."""
        if self._polling:
            self._polling = False
            self.lbl_status.config(text="Polling stopped", foreground="#ffaa44")
            return

        self._start_polling()

    def _poll(self):
        """Poll server for new decisions and eval data."""
        if not self._polling:
            return
        try:
            with httpx.Client(timeout=2.0) as client:
                # Status first: it tells us the current game and the server's
                # latest seq, which we need to (a) skip stale decisions left
                # over from a previous session on startup, and (b) detect when
                # a new game begins in a match so we can reset the display.
                resp = client.get(f"{self.server_url}/monitoring/status")
                if resp.status_code == 200:
                    status = resp.json()
                    game_id = status.get("game_id")
                    if not self._synced:
                        # First successful poll: adopt the server's current
                        # state so old decisions are not replayed, and clear
                        # any stale display left over from a previous session.
                        self._synced = True
                        self._last_seq = status.get("last_seq", 0)
                        self._current_game_id = game_id
                        self.terminal.clear()
                        self._clear_eval_graphs()
                        self.eval_bar.reset()
                        # Adopt the server's current event seq so events from a
                        # previous session are not replayed into the terminal.
                        ev_resp = client.get(f"{self.server_url}/monitoring/events?since=0")
                        if ev_resp.status_code == 200:
                            evs = ev_resp.json()
                            if evs:
                                self._last_event_seq = max(e.get("seq", 0) for e in evs)
                        self._update_status_label(status)
                        self.after(1000, self._poll)
                        return
                    if game_id is not None and game_id != self._current_game_id:
                        # A new game started (e.g. the next game of a match):
                        # clear the terminal/graph and reset the seq cursor
                        # (the server resets its seq counter per game).
                        self._reset_for_new_game(game_id)
                    self._update_status_label(status)

                # Incrementally fetch only decisions newer than the last seen seq.
                # (Offset-based slicing breaks once the server caps the list.)
                resp = client.get(f"{self.server_url}/monitoring/decisions?since={self._last_seq}")
                if resp.status_code == 200:
                    decisions = resp.json()
                    if decisions and decisions[0].get("seq", 0) < self._last_seq:
                        # The server's seq counter was reset between the status
                        # and decisions calls (a new game started): reset the
                        # UI and reprocess the batch from the start.
                        self._reset_for_new_game(self._current_game_id)
                        self._last_seq = 0
                    for dec in decisions:
                        self.log_decision(
                            description=dec.get("description", "Unknown"),
                            chosen=dec.get("action_taken", "?"),
                            reason=dec.get("action_reason", ""),
                            confidence=dec.get("model_confidence"),
                            value=dec.get("model_value_estimate"),
                            timeout=dec.get("timeout", False),
                        )
                        self._last_seq = max(self._last_seq, dec.get("seq", self._last_seq))

                # Poll live notification events (fallback / epsilon) and render
                # them in the terminal with the appropriate color.
                resp = client.get(f"{self.server_url}/monitoring/events?since={self._last_event_seq}")
                if resp.status_code == 200:
                    for ev in resp.json():
                        level = ev.get("level", "info")
                        msg = ev.get("message", "")
                        if level == "error":
                            self.log_error(msg)
                        elif level == "warning":
                            self.log_warning(msg)
                        else:
                            self.log_event(msg)
                        self._last_event_seq = max(self._last_event_seq, ev.get("seq", self._last_event_seq))

                # Sync the eval graphs from the unified per-action eval history.
                # The server returns one series per player (AI *and* human):
                # each is that player's own self-assessment (its own hidden info
                # visible, the opponent's masked). No zero-sum negation — with
                # hidden information the opponent's self-assessment is a
                # different information set and is shown in the opponent's own
                # tab. The x-axis is the action index.
                resp = client.get(f"{self.server_url}/monitoring/eval_history")
                if resp.status_code == 200:
                    self._sync_eval_graphs(resp.json())
        except Exception as e:
            if self._polling:
                self.log_error(f"Poll error: {e}")
                self.lbl_status.config(text="Server unreachable", foreground="#ff4444")

        # Schedule next poll (1 second interval)
        self.after(1000, self._poll)

    def _update_status_label(self, status: dict):
        """Update the status label from a /monitoring/status payload."""
        active = status.get("active", False)
        cnt = status.get("decisions_logged", 0)
        if active:
            self.lbl_status.config(text=f"Game active — {cnt} decisions", foreground="#66ff66")
        else:
            self.lbl_status.config(text=f"No game — {cnt} decisions logged", foreground="#888888")

    def _reset_for_new_game(self, game_id):
        """Clear the terminal, graphs, and eval bar when a new game starts."""
        self._current_game_id = game_id
        self._last_seq = 0
        self.terminal.clear()
        self._clear_eval_graphs()
        self.eval_bar.reset()
        self.terminal.log_event(f"New game started ({game_id})")

    # ------------------------------------------------------------------
    # Per-AI evaluation graphs
    # ------------------------------------------------------------------
    def _sync_eval_graphs(self, payload: dict):
        """Update the per-player eval graphs from a /monitoring/eval_history payload.

        ``payload`` is ``{"players": [...], "ai_players": [...], "active":
        name|None, "series": {name: [points]}}``. Creates a tab for each player
        the server reports (external AIs *and* the human), feeds each its own
        self-assessment series, and — when auto-switch is on — jumps to the tab
        of the AI that most recently acted.
        """
        players = payload.get("players") or []
        series = payload.get("series") or {}
        active = payload.get("active")

        # Create tabs for any AI players we haven't seen yet.
        for name in players:
            if name not in self._eval_graphs:
                graph = EvalGraph(self.eval_notebook, height=120, x_label="Action")
                self.eval_notebook.add(graph, text=name)
                self._eval_graphs[name] = graph

        # Feed each graph its own series (x = action index).
        for name in players:
            points = series.get(name) or []
            data = [(i + 1, p.get("value", 0.0)) for i, p in enumerate(points)]
            self._eval_graphs[name].set_data(data)

        # Auto-switch to the active AI's tab (the one that most recently acted).
        if self.var_auto_switch.get() and active in self._eval_graphs:
            if active != self._eval_active:
                self.eval_notebook.select(self._eval_graphs[active])
                self._eval_active = active

    def _clear_eval_graphs(self):
        """Clear all per-AI eval graphs (e.g. on a new game or startup sync)."""
        for graph in self._eval_graphs.values():
            graph.clear()
        self._eval_active = None
