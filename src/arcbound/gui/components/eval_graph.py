"""Evaluation history graph widget.

Displays a line chart of evaluation values over time (turns/decisions).
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple


class EvalGraph(ttk.Frame):
    """Canvas-based line graph showing eval over time."""

    def __init__(
        self,
        master: tk.Misc,
        width: int = 500,
        height: int = 150,
        x_label: str = "Turn",
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._data: List[Tuple[int, float]] = []  # (x, eval_value)
        self._last_w = 0
        self._last_h = 0
        self._x_label = x_label

        self.canvas = tk.Canvas(
            self, width=width, height=height, bg="#1e1e1e", highlightthickness=1, highlightbackground="#444444"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Re-render whenever the canvas is resized so the chart fills the
        # allocated area (winfo_width/height only become valid after layout).
        self.canvas.bind("<Configure>", self._on_configure)

        self._draw()

    def _on_configure(self, event=None):
        # Only redraw when the size actually changes (avoids redundant work).
        if event is not None and (event.width != self._last_w or event.height != self._last_h):
            self._draw()

    def add_point(self, turn: int, value: float):
        """Add a data point."""
        self._data.append((turn, max(-1.0, min(1.0, value))))
        self._draw()

    def set_data(self, data: List[Tuple[int, float]]):
        """Replace all data points."""
        self._data = [(t, max(-1.0, min(1.0, v))) for t, v in data]
        self._draw()

    def clear(self):
        self._data.clear()
        self._draw()

    def _draw(self):
        # Use the *actual* allocated size (not the requested size) so the chart
        # fills the canvas even when packed with fill=BOTH, expand=True. Before
        # the first layout pass the allocated size is 0; fall back to the
        # requested size so the chart still renders (it is redrawn at the real
        # size once <Configure> fires after layout).
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w = self.canvas.winfo_reqwidth()
        if h <= 1:
            h = self.canvas.winfo_reqheight()
        if w <= 1 or h <= 1:
            return
        self._last_w = w
        self._last_h = h

        self.canvas.delete("all")

        margin_left = 40
        margin_right = 10
        margin_top = 10
        margin_bottom = 25
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        # Grid lines and labels
        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            y = margin_top + plot_h * (1 - (val + 1.0) / 2.0)
            color = "#555555" if val == 0.0 else "#333333"
            self.canvas.create_line(margin_left, y, margin_left + plot_w, y, fill=color)
            self.canvas.create_text(margin_left - 5, y, anchor="e", text=f"{val:+.1f}", fill="#888888", font=("TkFixedFont", 7))

        # Zero line
        zero_y = margin_top + plot_h * 0.5
        self.canvas.create_line(margin_left, zero_y, margin_left + plot_w, zero_y, fill="#666666", dash=(4, 4))

        if len(self._data) < 2:
            return

        # X range
        turns = [t for t, _ in self._data]
        min_turn, max_turn = min(turns), max(turns)
        turn_range = max(max_turn - min_turn, 1)

        # Draw line
        points = []
        for turn, val in self._data:
            x = margin_left + (turn - min_turn) / turn_range * plot_w
            y = margin_top + plot_h * (1 - (val + 1.0) / 2.0)
            points.extend([x, y])

        # Color based on final value
        final_val = self._data[-1][1]
        if final_val >= 0.5:
            line_color = "#00cc00"
        elif final_val >= 0.1:
            line_color = "#88cc44"
        elif final_val >= -0.1:
            line_color = "#aaaaaa"
        elif final_val >= -0.5:
            line_color = "#cc8844"
        else:
            line_color = "#cc4444"

        self.canvas.create_line(points, fill=line_color, width=2)

        # Draw dots at each point
        for turn, val in self._data:
            x = margin_left + (turn - min_turn) / turn_range * plot_w
            y = margin_top + plot_h * (1 - (val + 1.0) / 2.0)
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=line_color, outline="")

        # X-axis labels
        self.canvas.create_text(margin_left, h - 5, anchor="sw", text=str(min_turn), fill="#888888", font=("TkFixedFont", 7))
        self.canvas.create_text(margin_left + plot_w, h - 5, anchor="se", text=str(max_turn), fill="#888888", font=("TkFixedFont", 7))
        self.canvas.create_text(margin_left + plot_w / 2, h - 5, anchor="s", text=self._x_label, fill="#888888", font=("TkFixedFont", 7))
