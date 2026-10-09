"""Move analysis tab for post-game replay analysis.

Analyzes every recorded move in a replay — the AI's decisions *and* the human
player's event-based moves — and rates each qualitatively (Best/Good/
Inaccuracy/Mistake/Blunder). When a model is loaded, moves without a recorded
confidence (the human's moves) are scored by the model itself, so the tab
shows how well the human is doing relative to the AI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Optional

from arcbound.analysis.move_analyzer import MoveAnalyzer, MoveAnalysis
from arcbound.gui.components.eval_graph import EvalGraph
from arcbound.gui.tabs.settings_tab import _Tooltip
from arcbound.logging.replay_reader import ReplayReader


# Hover tooltip descriptions for the analysis tab's sections.
ANALYSIS_DESCRIPTIONS = {
    "load_replay": (
        "Pick a match replay file (match_*.json in the replays folder) to analyze. "
        "Each replay contains every recorded move — the AI's decisions and the "
        "opponent's event-based moves — with the full board state at each step."
    ),
    "analyze": (
        "Score every move in the loaded replay. AI moves are rated from the "
        "recorded model confidence; opponent moves are rated by the loaded model "
        "(if any) from the board state. Ratings: Best / Good / Inaccuracy / "
        "Mistake / Blunder."
    ),
    "export_report": (
        "Save the full analysis (summary, per-player breakdown, and every move's "
        "rating) as a plain-text report."
    ),
    "model": (
        "Load a model checkpoint so the opponent's moves (which have no recorded "
        "confidence) can be scored by the model itself. Without a model, only the "
        "AI's own decisions are rated. Use a model trained on similar data for "
        "meaningful scores."
    ),
    "match": (
        "Who played this match and with what roles (human / bot / ai), as recorded "
        "when the game was logged. This tells you which moves came from the "
        "external AI and which from opponents — important for interpreting the "
        "per-player summary."
    ),
    "summary": (
        "Overall move quality across ALL players in the replay. Percentages show "
        "how often each rating occurred. A high Blunder/Mistake share means the "
        "model (or player) made many position-damaging moves; compare against the "
        "per-player summary to see who made them."
    ),
    "per_player": (
        "Move quality broken down by player, so you can compare the external AI "
        "against humans or bots. Each line shows total moves, average score, and "
        "the count of each rating. A well-trained AI should show fewer "
        "Inaccuracies/Mistakes/Blunders than untrained opponents."
    ),
    "eval_graph": (
        "Each player's own self-assessment after every action, one tab per "
        "player (external AI *and* human). Each value is that player's estimate "
        "from its own perspective, using its own hidden information (its hand "
        "visible, the opponent's masked). No zero-sum negation — the opponent's "
        "self-assessment is a different information set and is shown in its own "
        "tab. The human's tab shows how the external AI thinks the human is "
        "doing. Rising = that player thinks it's doing better; falling = worse. "
        "Steep drops mark the moves it judged as hurting most — cross-reference "
        "them with the move table."
    ),
    "move_history": (
        "Every recorded move in the replay, in order. Columns: turn number, the "
        "player who acted, the move description, the model's score for it, and the "
        "qualitative rating. Click a row to see details."
    ),
    "move_detail": (
        "Details for the selected move: turn, phase, player, decision type, the "
        "action taken, the score and rating, where the score came from (recorded "
        "confidence vs. model estimate), and whether the decision timed out."
    ),
}


class AnalysisTab(ttk.Frame):
    """Tab for analyzing replayed games."""

    def __init__(self, master: tk.Misc, replays_dir: Path, models_dir: Optional[Path] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.replays_dir = replays_dir
        self.models_dir = models_dir
        self._analyses: List[MoveAnalysis] = []
        self._model = None
        self._model_name: Optional[str] = None

        self._build_ui()

    def _build_ui(self):
        # Vertical PanedWindow: controls+summary+graph on top, table+detail below
        paned = tk.PanedWindow(self, orient=tk.VERTICAL, showhandle=True)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top pane: controls, summary, eval graph
        top = ttk.Frame(paned, padding=15)
        paned.add(top, minsize=120)

        # Load controls
        ctrl = ttk.Frame(top)
        ctrl.pack(fill=tk.X, pady=(0, 10))

        lbl_load = ttk.Label(ctrl, text="Load Replay:", font=("TkDefaultFont", 9, "bold"))
        lbl_load.pack(side=tk.LEFT)
        self.lbl_replay = ttk.Label(ctrl, text="No replay loaded")
        self.lbl_replay.pack(side=tk.LEFT, padx=10)
        btn_browse = ttk.Button(ctrl, text="Browse...", command=self._browse)
        btn_browse.pack(side=tk.LEFT, padx=5)
        btn_analyze = ttk.Button(ctrl, text="Analyze", command=self._analyze)
        btn_analyze.pack(side=tk.LEFT, padx=5)
        btn_export = ttk.Button(ctrl, text="Export Report", command=self._export_report)
        btn_export.pack(side=tk.LEFT, padx=5)
        _Tooltip(lbl_load, ANALYSIS_DESCRIPTIONS["load_replay"])
        _Tooltip(self.lbl_replay, ANALYSIS_DESCRIPTIONS["load_replay"])
        _Tooltip(btn_browse, ANALYSIS_DESCRIPTIONS["load_replay"])
        _Tooltip(btn_analyze, ANALYSIS_DESCRIPTIONS["analyze"])
        _Tooltip(btn_export, ANALYSIS_DESCRIPTIONS["export_report"])

        # Model controls (used to score the human's moves)
        model_ctrl = ttk.Frame(top)
        model_ctrl.pack(fill=tk.X, pady=(0, 10))
        lbl_model = ttk.Label(model_ctrl, text="Model:", font=("TkDefaultFont", 9, "bold"))
        lbl_model.pack(side=tk.LEFT)
        self.lbl_model = ttk.Label(model_ctrl, text="No model loaded (AI moves use recorded confidence)")
        self.lbl_model.pack(side=tk.LEFT, padx=10)
        btn_load_model = ttk.Button(model_ctrl, text="Load Model...", command=self._load_model)
        btn_load_model.pack(side=tk.LEFT, padx=5)
        _Tooltip(lbl_model, ANALYSIS_DESCRIPTIONS["model"])
        _Tooltip(self.lbl_model, ANALYSIS_DESCRIPTIONS["model"])
        _Tooltip(btn_load_model, ANALYSIS_DESCRIPTIONS["model"])

        # Match composition (who is playing: human / bot / ai)
        match_frame = ttk.LabelFrame(top, text="Match", padding=10)
        match_frame.pack(fill=tk.X, pady=(0, 10))
        self.lbl_composition = ttk.Label(match_frame, text="Composition: —")
        self.lbl_composition.pack(anchor="w", pady=2)
        self.lbl_player_types = ttk.Label(match_frame, text="Players: —")
        self.lbl_player_types.pack(anchor="w", pady=2)
        _Tooltip(match_frame, ANALYSIS_DESCRIPTIONS["match"])
        _Tooltip(self.lbl_composition, ANALYSIS_DESCRIPTIONS["match"])
        _Tooltip(self.lbl_player_types, ANALYSIS_DESCRIPTIONS["match"])

        # Summary
        summary_frame = ttk.LabelFrame(top, text="Summary (all players)", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_total = ttk.Label(summary_frame, text="Total Moves: 0")
        self.lbl_total.pack(anchor="w", pady=2)
        self.lbl_best = ttk.Label(summary_frame, text="Best Moves: 0 (0%)")
        self.lbl_best.pack(anchor="w", pady=2)
        self.lbl_good = ttk.Label(summary_frame, text="Good Moves: 0 (0%)")
        self.lbl_good.pack(anchor="w", pady=2)
        self.lbl_inacc = ttk.Label(summary_frame, text="Inaccuracies: 0 (0%)")
        self.lbl_inacc.pack(anchor="w", pady=2)
        self.lbl_mistake = ttk.Label(summary_frame, text="Mistakes: 0 (0%)")
        self.lbl_mistake.pack(anchor="w", pady=2)
        self.lbl_blunder = ttk.Label(summary_frame, text="Blunders: 0 (0%)")
        self.lbl_blunder.pack(anchor="w", pady=2)
        self.lbl_avg = ttk.Label(summary_frame, text="Average Score: 0.0")
        self.lbl_avg.pack(anchor="w", pady=2)
        for _w in (summary_frame, self.lbl_total, self.lbl_best, self.lbl_good,
                   self.lbl_inacc, self.lbl_mistake, self.lbl_blunder, self.lbl_avg):
            _Tooltip(_w, ANALYSIS_DESCRIPTIONS["summary"])

        # Per-player summary (human vs AI)
        self.player_frame = ttk.LabelFrame(top, text="Per-Player Summary", padding=10)
        self.player_frame.pack(fill=tk.X, pady=(0, 10))
        self.lbl_players = ttk.Label(self.player_frame, text="Analyze a replay to see per-player ratings.")
        self.lbl_players.pack(anchor="w")
        _Tooltip(self.player_frame, ANALYSIS_DESCRIPTIONS["per_player"])
        _Tooltip(self.lbl_players, ANALYSIS_DESCRIPTIONS["per_player"])

        # Eval graphs — one tab per player (external AI *and* human), each
        # showing that player's own self-assessment (its own hidden info
        # visible, the opponent's masked). This includes "how the external AI
        # thinks the human is doing" (the human's self-assessment).
        graph_frame = ttk.LabelFrame(top, text="Evaluation Over Time (every action)", padding=5)
        graph_frame.pack(fill=tk.X, pady=(0, 10))
        _Tooltip(graph_frame, ANALYSIS_DESCRIPTIONS["eval_graph"])
        # The x-axis is the action index (every player's action), not the turn,
        # because the graph now shows a value point for each action.
        self.eval_notebook = ttk.Notebook(graph_frame)
        self.eval_notebook.pack(fill=tk.BOTH, expand=True)
        self._eval_graphs: dict = {}  # player name -> EvalGraph

        # Bottom pane: move history table + detail
        bottom = ttk.Frame(paned, padding=(15, 10))
        paned.add(bottom, minsize=100)

        # Move history table
        table_frame = ttk.LabelFrame(bottom, text="Move History (all players)", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        _Tooltip(table_frame, ANALYSIS_DESCRIPTIONS["move_history"])

        columns = ("turn", "player", "move", "score", "rating")
        # Make the move-history rows ~20% taller than the default so the text
        # is not vertically clipped. The default row height tracks the font's
        # linespace, so we derive a taller height from it (Tk 8.6+ rowheight).
        import tkinter.font as tkfont
        _linespace = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        ttk.Style().configure(
            "Tall.Treeview", rowheight=int(_linespace * 1.2) + 4
        )
        self.tree = ttk.Treeview(
            table_frame, style="Tall.Treeview", columns=columns, show="headings", height=10
        )
        for col in columns:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=80)
        self.tree.column("move", width=250)
        self.tree.column("player", width=100)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Detail
        detail_frame = ttk.LabelFrame(bottom, text="Move Detail", padding=10)
        detail_frame.pack(fill=tk.X)
        _Tooltip(detail_frame, ANALYSIS_DESCRIPTIONS["move_detail"])

        self.lbl_detail = ttk.Label(detail_frame, text="Select a move to see details.", wraplength=600)
        self.lbl_detail.pack(anchor="w")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _load_model(self):
        """Load a model checkpoint so the human's moves can be scored by it."""
        if not self.models_dir or not self.models_dir.exists():
            messagebox.showwarning("No models", "No models directory found.")
            return
        path = filedialog.askdirectory(
            initialdir=self.models_dir,
            title="Select Model Directory",
        )
        if not path:
            return
        model_dir = Path(path)
        try:
            from arcbound.encoder.transformer import load_model_from_checkpoint
            from arcbound.encoder.vocabulary import CardVocabulary, KeywordVocabulary
            from arcbound.encoder.tokenizer import set_card_vocabulary, set_keyword_vocabulary

            ckpt = None
            for candidate in ("model.pt", "best_model.pt", "checkpoint.pt"):
                p = model_dir / candidate
                if p.exists():
                    ckpt = p
                    break
            if ckpt is None:
                messagebox.showwarning("No checkpoint", f"No checkpoint found in {model_dir}")
                return

            self._model = load_model_from_checkpoint(str(ckpt))
            self._model_name = model_dir.name

            # Load vocabularies so tokenization matches training.
            vocab_path = model_dir / "vocab.json"
            if vocab_path.exists():
                set_card_vocabulary(CardVocabulary.load(vocab_path))
            kw_vocab_path = model_dir / "kw_vocab.json"
            if kw_vocab_path.exists():
                set_keyword_vocabulary(KeywordVocabulary.load(kw_vocab_path))

            self.lbl_model.config(text=f"Loaded '{self._model_name}' (human moves scored by model)")
        except Exception as e:
            self._model = None
            self._model_name = None
            self.lbl_model.config(text="Failed to load model")
            messagebox.showerror("Load failed", f"Could not load model:\n{e}")

    def _browse(self):
        path = filedialog.askopenfilename(
            initialdir=self.replays_dir,
            title="Select Replay File",
            filetypes=[("Replay files", "*.json.gz *.json"), ("All files", "*.*")],
        )
        if path:
            self.lbl_replay.config(text=Path(path).name)
            self._replay_path = Path(path)

    def _analyze(self):
        if not hasattr(self, "_replay_path"):
            return

        # Analyze ALL moves (AI decisions + human event-based moves).
        moves = ReplayReader.get_moves(self._replay_path)
        self._analyses = MoveAnalyzer.analyze_moves(moves, model=self._model)
        summary = MoveAnalyzer.summarize(self._analyses)

        # Match composition + player types (who is playing).
        self._player_types = ReplayReader.get_player_types(self._replay_path)
        composition = ReplayReader.get_composition(self._replay_path)
        self.lbl_composition.config(text=f"Composition: {composition}")
        if self._player_types:
            parts = [f"{name} ({role})" for name, role in self._player_types.items()]
            self.lbl_player_types.config(text="Players: " + ", ".join(parts))
        else:
            self.lbl_player_types.config(text="Players: (not recorded)")

        total = summary["total"]
        self.lbl_total.config(text=f"Total Moves: {total}")
        self.lbl_best.config(text=f"Best Moves: {summary['best']} ({summary['best']/total*100:.0f}%)") if total else None
        self.lbl_good.config(text=f"Good Moves: {summary['good']} ({summary['good']/total*100:.0f}%)") if total else None
        self.lbl_inacc.config(text=f"Inaccuracies: {summary['inaccuracy']} ({summary['inaccuracy']/total*100:.0f}%)") if total else None
        self.lbl_mistake.config(text=f"Mistakes: {summary['mistake']} ({summary['mistake']/total*100:.0f}%)") if total else None
        self.lbl_blunder.config(text=f"Blunders: {summary['blunder']} ({summary['blunder']/total*100:.0f}%)") if total else None
        self.lbl_avg.config(text=f"Average Score: {summary['average_score']}")

        # Per-player summary (human vs AI)
        by_player = MoveAnalyzer.summarize_by_player(self._analyses)
        lines = []
        for player, s in by_player.items():
            lines.append(
                f"{player}: {s['total']} moves | avg {s['average_score']} | "
                f"Best {s['best']} / Good {s['good']} / Inacc {s['inaccuracy']} / "
                f"Mistake {s['mistake']} / Blunder {s['blunder']}"
            )
        self.lbl_players.config(text="\n".join(lines) if lines else "No moves found.")

        # Populate table
        for item in self.tree.get_children():
            self.tree.delete(item)

        for a in self._analyses:
            self.tree.insert(
                "", tk.END,
                values=(a.turn, a.player, f"{a.description}: {a.action_taken}", a.score, a.rating),
            )

        # Update eval graph with a value point for EVERY action (AI decisions +
        # opponent moves), normalized to the external AI's perspective.
        self._update_eval_graph(moves)

    def _update_eval_graph(self, moves: List[Dict]):
        """Populate the per-player eval graphs with each player's self-assessment.

        For each move, a value is computed from EACH player's own perspective
        (its own hidden info visible, the opponent's masked): the recorded
        ``model_value_estimate`` is used for the acting player when present
        (the AI's decisions), otherwise the loaded model computes it from the
        move's board state. No zero-sum negation is applied: with hidden
        information, the opponent's self-assessment is a different information
        set and is shown in the opponent's own tab. The x-axis is the action
        index.
        """
        # Clear any tabs left over from a previously analyzed replay so the
        # notebook reflects only the current replay's players.
        for graph in self._eval_graphs.values():
            self.eval_notebook.forget(graph)
        self._eval_graphs.clear()

        # Determine the set of players to show: the external AI(s) plus any
        # human/bot opponents. Fall back to the players seen in the moves.
        player_names: List[str] = []
        for name, role in (self._player_types or {}).items():
            if name not in player_names:
                player_names.append(name)
        for move in moves:
            actor = move.get("player") or (move.get("board_state") or {}).get("focal_player")
            if actor and actor not in player_names:
                player_names.append(actor)

        # Build each player's own self-assessment series.
        series: Dict[str, List] = {name: [] for name in player_names}
        for i, move in enumerate(moves):
            board_state = move.get("board_state")
            actor = move.get("player") or (board_state or {}).get("focal_player")
            for name in player_names:
                value = None
                if name == actor:
                    # The acting player's value may already be recorded (the
                    # AI's decisions carry model_value_estimate).
                    value = move.get("model_value_estimate")
                if value is None and self._model is not None and board_state:
                    value = MoveAnalyzer.estimate_board_value(
                        self._model, board_state, perspective=name
                    )
                if value is not None:
                    series[name].append((i + 1, value))

        # Create a tab per player and feed it its series.
        for name in player_names:
            if name not in self._eval_graphs:
                graph = EvalGraph(self.eval_notebook, height=120, x_label="Action")
                self.eval_notebook.add(graph, text=name)
                self._eval_graphs[name] = graph
            self._eval_graphs[name].set_data(series[name])

    def _export_report(self):
        if not self._analyses:
            messagebox.showinfo("Info", "Analyze a replay first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="analysis-report.txt",
            title="Export Analysis Report",
        )
        if not path:
            return

        summary = MoveAnalyzer.summarize(self._analyses)
        by_player = MoveAnalyzer.summarize_by_player(self._analyses)
        player_types = getattr(self, "_player_types", None) or {}
        composition = ReplayReader.get_composition(self._replay_path)
        lines = [
            "=== ARCBOUND MOVE ANALYSIS REPORT ===",
            f"Replay: {getattr(self, '_replay_path', 'unknown').name}",
            f"Generated: {__import__('datetime').datetime.now().isoformat()}",
            f"Model: {self._model_name or 'none (recorded confidence only)'}",
            f"Composition: {composition}",
        ]
        if player_types:
            lines.append("Players: " + ", ".join(f"{n} ({r})" for n, r in player_types.items()))
        lines += [
            "",
            "--- Summary (all players) ---",
            f"Total Moves:     {summary['total']}",
            f"Best Moves:      {summary['best']} ({summary['best']/max(summary['total'],1)*100:.0f}%)",
            f"Good Moves:      {summary['good']} ({summary['good']/max(summary['total'],1)*100:.0f}%)",
            f"Inaccuracies:    {summary['inaccuracy']} ({summary['inaccuracy']/max(summary['total'],1)*100:.0f}%)",
            f"Mistakes:        {summary['mistake']} ({summary['mistake']/max(summary['total'],1)*100:.0f}%)",
            f"Blunders:        {summary['blunder']} ({summary['blunder']/max(summary['total'],1)*100:.0f}%)",
            f"Average Score:   {summary['average_score']}",
            "",
            "--- Per-Player Summary ---",
        ]
        for player, s in by_player.items():
            lines.append(
                f"{player}: {s['total']} moves | avg {s['average_score']} | "
                f"Best {s['best']} / Good {s['good']} / Inacc {s['inaccuracy']} / "
                f"Mistake {s['mistake']} / Blunder {s['blunder']}"
            )
        lines += [
            "",
            "--- Move Details ---",
            f"{'Turn':<6}{'Player':<16}{'Rating':<12}{'Score':<7}{'Action'}",
            "-" * 70,
        ]
        for a in self._analyses:
            lines.append(f"{a.turn:<6}{a.player:<16}{a.rating:<12}{a.score:<7}{a.action_taken}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        messagebox.showinfo("Export", f"Report saved to {path}")

    def _on_select(self, event=None):
        selection = self.tree.selection()
        if not selection:
            return
        item = self.tree.item(selection[0])
        idx = self.tree.index(selection[0])
        if 0 <= idx < len(self._analyses):
            a = self._analyses[idx]
            detail = (
                f"Turn {a.turn} | {a.phase} | Player: {a.player}\n"
                f"Type: {a.decision_type}\n"
                f"Action: {a.action_taken}\n"
                f"Score: {a.score} ({a.rating})\n"
                f"Source: {a.source}"
            )
            if a.confidence is not None:
                detail += f"\nConfidence: {a.confidence:.2%}"
            if a.value_estimate is not None:
                detail += f"\nValue: {a.value_estimate:+.2f}"
            if a.timeout:
                detail += "\n(TIMEOUT — fallback used)"
            self.lbl_detail.config(text=detail)
