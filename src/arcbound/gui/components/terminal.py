"""Action terminal widget for real-time decision logging."""

import tkinter as tk
from tkinter import ttk
from datetime import datetime


class ActionTerminal(ttk.Frame):
    """Scrolled text terminal with color-coded log levels."""

    COLORS = {
        "default": "#e0e0e0",
        "timestamp": "#888888",
        "decision": "#e0e0e0",
        "chosen": "#66ff66",
        "timeout": "#ff6666",
        "error": "#ff4444",
        "event": "#66ff66",
        "model": "#6699ff",
        "info": "#cccccc",
    }

    def __init__(self, master: tk.Misc, **kwargs):
        super().__init__(master, **kwargs)

        self.text = tk.Text(self, wrap=tk.WORD, bg="#1e1e1e", fg="#e0e0e0", font=("TkFixedFont", 9), state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure tags
        for tag, color in self.COLORS.items():
            self.text.tag_configure(tag, foreground=color)

    def _append(self, message: str, tag: str = "default"):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, message, tag)
        self.text.configure(state=tk.DISABLED)
        self.text.see(tk.END)

    def log(self, message: str, level: str = "default"):
        """Log a message with timestamp and color level."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._append(f"[{ts}] ", "timestamp")
        self._append(message + "\n", level)

    def log_decision(self, description: str, chosen: str, reason: str = "", confidence: float = None, value: float = None, timeout: bool = False):
        """Log a single AI decision."""
        prefix = "TIMEOUT: " if timeout else "DECISION: "
        level = "timeout" if timeout else "decision"
        self._append(f"[{datetime.now().strftime('%H:%M:%S')}] ", "timestamp")
        self._append(f"{prefix}{description}\n", level)
        self._append(f"  CHOSEN: {chosen}\n", "chosen")
        if reason:
            self._append(f"  Reason: {reason}\n", "info")
        if confidence is not None:
            self._append(f"  Confidence: {confidence:.0%}\n", "info")
        if value is not None:
            sign = "+" if value >= 0 else ""
            self._append(f"  Value: {sign}{value:.2f}\n", "model")

    def log_event(self, message: str):
        self.log(message, "event")

    def log_error(self, message: str):
        self.log(message, "error")

    def log_model(self, message: str):
        self.log(message, "model")

    def clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)
