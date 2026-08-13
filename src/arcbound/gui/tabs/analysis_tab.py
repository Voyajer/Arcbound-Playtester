"""Move analysis tab for post-game replay analysis."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional

from arcbound.analysis.move_analyzer import MoveAnalyzer, MoveAnalysis
from arcbound.gui.components.eval_graph import EvalGraph
from arcbound.logging.replay_reader import ReplayReader


class AnalysisTab(ttk.Frame):
    """Tab for analyzing replayed games."""

    def __init__(self, master: tk.Misc, replays_dir: Path, **kwargs):
        super().__init__(master, **kwargs)
        self.replays_dir = replays_dir
        self._analyses: List[MoveAnalysis] = []

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

        ttk.Label(ctrl, text="Load Replay:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_replay = ttk.Label(ctrl, text="No replay loaded")
        self.lbl_replay.pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text="Browse...", command=self._browse).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Analyze", command=self._analyze).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Export Report", command=self._export_report).pack(side=tk.LEFT, padx=5)

        # Summary
        summary_frame = ttk.LabelFrame(top, text="Summary", padding=10)
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

        # Eval graph
        graph_frame = ttk.LabelFrame(top, text="Evaluation Over Time", padding=5)
        graph_frame.pack(fill=tk.X, pady=(0, 10))
        self.eval_graph = EvalGraph(graph_frame, height=120)
        self.eval_graph.pack(fill=tk.X)

        # Bottom pane: move history table + detail
        bottom = ttk.Frame(paned, padding=(15, 10))
        paned.add(bottom, minsize=100)

        # Move history table
        table_frame = ttk.LabelFrame(bottom, text="Move History", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        columns = ("turn", "move", "score", "rating")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=80)
        self.tree.column("move", width=250)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Detail
        detail_frame = ttk.LabelFrame(bottom, text="Move Detail", padding=10)
        detail_frame.pack(fill=tk.X)

        self.lbl_detail = ttk.Label(detail_frame, text="Select a move to see details.", wraplength=600)
        self.lbl_detail.pack(anchor="w")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _browse(self):
        path = filedialog.askopenfilename(
            initialdir=self.replays_dir,
            title="Select Replay File",
            filetypes=[("Replay files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.lbl_replay.config(text=Path(path).name)
            self._replay_path = Path(path)

    def _analyze(self):
        if not hasattr(self, "_replay_path"):
            return

        decisions = ReplayReader.get_decisions(self._replay_path)
        self._analyses = MoveAnalyzer.analyze_decisions(decisions)
        summary = MoveAnalyzer.summarize(self._analyses)

        total = summary["total"]
        self.lbl_total.config(text=f"Total Moves: {total}")
        self.lbl_best.config(text=f"Best Moves: {summary['best']} ({summary['best']/total*100:.0f}%)") if total else None
        self.lbl_good.config(text=f"Good Moves: {summary['good']} ({summary['good']/total*100:.0f}%)") if total else None
        self.lbl_inacc.config(text=f"Inaccuracies: {summary['inaccuracy']} ({summary['inaccuracy']/total*100:.0f}%)") if total else None
        self.lbl_mistake.config(text=f"Mistakes: {summary['mistake']} ({summary['mistake']/total*100:.0f}%)") if total else None
        self.lbl_blunder.config(text=f"Blunders: {summary['blunder']} ({summary['blunder']/total*100:.0f}%)") if total else None
        self.lbl_avg.config(text=f"Average Score: {summary['average_score']}")

        # Populate table
        for item in self.tree.get_children():
            self.tree.delete(item)

        for a in self._analyses:
            self.tree.insert("", tk.END, values=(a.turn, f"{a.description}: {a.action_taken}", a.score, a.rating))

        # Update eval graph with value estimates
        self.eval_graph.clear()
        for a in self._analyses:
            if a.value_estimate is not None:
                self.eval_graph.add_point(a.turn, a.value_estimate)

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
        lines = [
            "=== ARCBOUND MOVE ANALYSIS REPORT ===",
            f"Replay: {getattr(self, '_replay_path', 'unknown').name}",
            f"Generated: {__import__('datetime').datetime.now().isoformat()}",
            "",
            "--- Summary ---",
            f"Total Moves:     {summary['total']}",
            f"Best Moves:      {summary['best']} ({summary['best']/max(summary['total'],1)*100:.0f}%)",
            f"Good Moves:      {summary['good']} ({summary['good']/max(summary['total'],1)*100:.0f}%)",
            f"Inaccuracies:    {summary['inaccuracy']} ({summary['inaccuracy']/max(summary['total'],1)*100:.0f}%)",
            f"Mistakes:        {summary['mistake']} ({summary['mistake']/max(summary['total'],1)*100:.0f}%)",
            f"Blunders:        {summary['blunder']} ({summary['blunder']/max(summary['total'],1)*100:.0f}%)",
            f"Average Score:   {summary['average_score']}",
            "",
            "--- Move Details ---",
            f"{'Turn':<6}{'Rating':<12}{'Score':<7}{'Action'}",
            "-" * 60,
        ]
        for a in self._analyses:
            lines.append(f"{a.turn:<6}{a.rating:<12}{a.score:<7}{a.action_taken}")

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
                f"Score: {a.score} ({a.rating})"
            )
            if a.confidence is not None:
                detail += f"\nConfidence: {a.confidence:.2%}"
            if a.value_estimate is not None:
                detail += f"\nValue: {a.value_estimate:+.2f}"
            if a.timeout:
                detail += "\n(TIMEOUT — fallback used)"
            self.lbl_detail.config(text=detail)
