"""Chess-style evaluation bar widget showing game advantage."""

import tkinter as tk
from tkinter import ttk


class EvaluationBar(ttk.Frame):
    """Vertical evaluation bar widget.

    Displays a bar from 0 (bottom, opponent advantage) to 100 (top, focal player advantage).
    The bar fill position corresponds to the model's value estimate mapped from [-1.0, +1.0] to [0, 100].
    """

    def __init__(
        self,
        master: tk.Misc,
        player_name: str = "AI",
        opponent_name: str = "Opponent",
        width: int = 30,
        height: int = 200,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.player_name = player_name
        self.opponent_name = opponent_name
        self._eval_value: float = 0.5  # 50% = equal
        self._last_w = 0
        self._last_h = 0

        self.canvas = tk.Canvas(
            self, width=width, height=height, bg="#1e1e1e",
            highlightthickness=1, highlightbackground="#666666"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Re-render whenever the canvas is resized so the bar fills the
        # allocated area (winfo_width/height only become valid after layout).
        self.canvas.bind("<Configure>", self._on_configure)

        self._draw()

    def _on_configure(self, event=None):
        # Only redraw when the size actually changes (avoids redundant work).
        if event is not None and (event.width != self._last_w or event.height != self._last_h):
            self._draw()

    def set_eval(self, value: float):
        """Set evaluation from -1.0 (opponent winning) to +1.0 (player winning)."""
        clamped = max(-1.0, min(1.0, value))
        self._eval_value = (clamped + 1.0) / 2.0  # Map to 0.0-1.0
        self._draw()

    def reset(self):
        """Reset to a neutral (0.0) evaluation, e.g. when a new game starts."""
        self._eval_value = 0.5
        self._draw()

    def set_names(self, player: str, opponent: str):
        self.player_name = player
        self.opponent_name = opponent
        self._draw()

    def _get_color(self) -> str:
        """Return color based on current evaluation."""
        raw = self._eval_value * 2.0 - 1.0  # Back to -1..+1
        if raw >= 0.5:
            return "#00cc00"  # Green - decisive advantage
        elif raw >= 0.1:
            return "#88cc44"  # Light green - slight advantage
        elif raw >= -0.1:
            return "#aaaaaa"  # Gray - equal
        elif raw >= -0.5:
            return "#cc8844"  # Light red/orange - slight disadvantage
        else:
            return "#cc4444"  # Red - decisive disadvantage

    def _draw(self):
        # Use the *actual* allocated size (not the requested size) so the bar
        # fills the canvas even when packed with fill=BOTH, expand=True.
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return  # Not laid out yet
        self._last_w = w
        self._last_h = h

        self.canvas.delete("all")
        # Reserve room on the right for the scale labels so they are not
        # clipped outside the canvas.
        label_space = 30
        bar_inner_w = max(10, w - 6 - label_space)
        bar_inner_h = max(10, h - 40)
        x1 = 3
        y1 = 20
        bar_cx = x1 + bar_inner_w / 2

        # Background
        self.canvas.create_rectangle(x1, y1, x1 + bar_inner_w, y1 + bar_inner_h, fill="#333333", outline="")

        # Fill
        fill_height = int(bar_inner_h * self._eval_value)
        fill_y = y1 + bar_inner_h - fill_height
        color = self._get_color()
        self.canvas.create_rectangle(x1, fill_y, x1 + bar_inner_w, y1 + bar_inner_h, fill=color, outline="")

        # Scale markings
        for pct in [0, 0.25, 0.5, 0.75, 1.0]:
            y = y1 + bar_inner_h * (1 - pct)
            label = int(pct * 100)
            self.canvas.create_line(x1 - 3, y, x1, y, fill="#888888")
            self.canvas.create_text(x1 + bar_inner_w + 12, y, anchor="w", text=str(label), fill="#cccccc", font=("TkFixedFont", 6))

        # Player names (centered over the bar, not the whole canvas)
        self.canvas.create_text(bar_cx, 8, anchor="center", text=self.player_name, fill="#ffffff", font=("TkFixedFont", 7))
        self.canvas.create_text(bar_cx, y1 + bar_inner_h + 14, anchor="center", text=self.opponent_name, fill="#ffffff", font=("TkFixedFont", 7))
