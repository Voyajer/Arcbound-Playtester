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

        self.canvas = tk.Canvas(
            self, width=width, height=height, highlightthickness=1, highlightbackground="#666666"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._draw()

    def set_eval(self, value: float):
        """Set evaluation from -1.0 (opponent winning) to +1.0 (player winning)."""
        clamped = max(-1.0, min(1.0, value))
        self._eval_value = (clamped + 1.0) / 2.0  # Map to 0.0-1.0
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
        w, h = self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()
        if w == 1 or h == 1:
            return  # Not laid out yet

        self.canvas.delete("all")
        bar_inner_w = w - 6
        bar_inner_h = h - 40
        x1 = 3
        y1 = 20

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

        # Player names
        self.canvas.create_text(w / 2, 8, anchor="center", text=self.player_name, fill="#ffffff", font=("TkFixedFont", 7))
        self.canvas.create_text(w / 2, y1 + bar_inner_h + 14, anchor="center", text=self.opponent_name, fill="#ffffff", font=("TkFixedFont", 7))
