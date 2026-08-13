"""Training tab with replay selection and training controls."""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable, Optional

from arcbound.training.trainer import Trainer, TrainingConfig

# Reuse descriptions from settings tab
from arcbound.gui.tabs.settings_tab import PARAM_DESCRIPTIONS, _Tooltip, _add_tooltip


class TrainingTab(ttk.Frame):
    """Tab for configuring and running training."""

    def __init__(
        self,
        master: tk.Misc,
        replays_dir: Path,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.replays_dir = replays_dir
        self.on_start = on_start
        self.on_stop = on_stop
        self._training = False
        self._selected_model: Optional[str] = None
        self._entries: dict = {}

        self._build_ui()

    def _build_ui(self):
        # Vertical PanedWindow: controls on top, training log below
        paned = tk.PanedWindow(self, orient=tk.VERTICAL, showhandle=True)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top pane: model, data, params, buttons, progress
        top = ttk.Frame(paned, padding=15)
        paned.add(top, minsize=180)

        # Model selector label
        ttk.Label(top, text="Model:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.lbl_model = ttk.Label(top, text="—")
        self.lbl_model.pack(anchor="w", pady=(0, 10))

        # Training data
        ttk.Label(top, text="Training Data:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        data_frame = ttk.Frame(top)
        data_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_data_path = ttk.Label(data_frame, text="No data selected")
        self.lbl_data_path.pack(side=tk.LEFT, padx=5)
        ttk.Button(data_frame, text="Browse...", command=self._browse_data).pack(side=tk.RIGHT)

        self.lbl_file_count = ttk.Label(data_frame, text="")
        self.lbl_file_count.pack(anchor="w", pady=(0, 10))

        # Parameters
        param_frame = ttk.LabelFrame(top, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        self._add_int_field(param_frame, "epochs", "Epochs", 100, 1, 10000)
        self._add_int_field(param_frame, "batch_size", "Batch Size", 32, 1, 4096)
        self._add_float_field(param_frame, "learning_rate", "Learning Rate", 0.001)
        self._add_float_field(param_frame, "entropy_coef", "Entropy Coef", 0.01)
        self._add_float_field(param_frame, "value_coef", "Value Coef", 0.5)
        self._add_float_field(param_frame, "gamma", "Gamma", 0.99)

        # Buttons
        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=10)
        self.btn_start = ttk.Button(btn_frame, text="Start Training", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Progress
        prog_frame = ttk.Frame(top)
        prog_frame.pack(fill=tk.X, pady=10)

        self.progress = ttk.Progressbar(prog_frame, mode="determinate", length=400)
        self.progress.pack(fill=tk.X)

        self.lbl_progress = ttk.Label(prog_frame, text="")
        self.lbl_progress.pack(anchor="w", pady=5)

        # Bottom pane: training log
        log_frame = ttk.LabelFrame(paned, text="Training Log", padding=5)
        paned.add(log_frame, minsize=60)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, bg="#1e1e1e", fg="#e0e0e0", font=("TkFixedFont", 9), state=tk.DISABLED, height=10)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _add_int_field(self, parent, key: str, label: str, default: int, lo: int, hi: int):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=label, width=14)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Spinbox(frame, from_=lo, to=hi, width=10, value=default)
        entry.pack(side=tk.RIGHT)
        self._entries[key] = (entry, "int")
        _add_tooltip(lbl, key)
        _add_tooltip(entry, key)

    def _add_float_field(self, parent, key: str, label: str, default: float):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=label, width=14)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(frame, width=12)
        entry.insert(0, str(default))
        entry.pack(side=tk.RIGHT)
        self._entries[key] = (entry, "float")
        _add_tooltip(lbl, key)
        _add_tooltip(entry, key)

    def select_model(self, name: str):
        self._selected_model = name
        self.lbl_model.config(text=name if name else "—")

    def _browse_data(self):
        path = filedialog.askdirectory(initialdir=self.replays_dir, title="Select Replay Directory")
        if path:
            self.lbl_data_path.config(text=path)
            replay_path = Path(path)
            count = len(list(replay_path.glob("*.json")))
            self.lbl_file_count.config(text=f"Files found: {count} games")

    def _start(self):
        if not self._selected_model:
            messagebox.showinfo("Info", "Select a model first.")
            return

        # Determine replay directory
        data_path_str = self.lbl_data_path.cget("text")
        if data_path_str == "No data selected":
            replay_dir = self.replays_dir
        else:
            replay_dir = Path(data_path_str)

        params = self.get_params()
        cfg = TrainingConfig(
            epochs=params.get("epochs", 100),
            batch_size=params.get("batch_size", 32),
            learning_rate=params.get("learning_rate", 0.001),
            entropy_coef=params.get("entropy_coef", 0.01),
            value_coef=params.get("value_coef", 0.5),
            gamma=params.get("gamma", 0.99),
        )

        self._trainer = Trainer(
            config=cfg,
            on_progress=self._on_progress,
            on_log=self._log,
        )

        # Load replays
        if replay_dir.exists():
            self._trainer.load_replay_dir(replay_dir)
        else:
            self._log(f"Replay directory not found: {replay_dir}")
            return

        if len(self._trainer.buffer) == 0:
            messagebox.showinfo("Info", "No replay data found. Play some games first.")
            return

        self._training = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress["value"] = 0

        # Run training in background thread
        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

        if self.on_start:
            self.on_start()

    def _run_training(self):
        self._trainer.train()
        # Update UI on main thread after training finishes
        self.after(0, self._training_finished)

    def _training_finished(self):
        self._training = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress["value"] = 100
        self._log("Training complete.")
        if self.on_stop:
            self.on_stop()

    def _stop(self):
        if hasattr(self, "_trainer"):
            self._trainer.stop()
        self._training = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._log("Training stopped by user.")
        if self.on_stop:
            self.on_stop()

    def _on_progress(self, pct: float, loss: float, msg: str):
        # Called from training thread; schedule on main thread
        self.after(0, self._update_progress_safe, pct, msg)

    def _update_progress_safe(self, pct: float, msg: str):
        self.progress["value"] = pct
        self.lbl_progress.config(text=msg)

    def update_progress(self, value: float, message: str = ""):
        self.progress["value"] = value
        self.lbl_progress.config(text=message)

    def _log(self, message: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def get_params(self) -> dict:
        params = {}
        for key, (widget, vtype) in self._entries.items():
            try:
                raw = widget.get().strip()
                params[key] = int(raw) if vtype == "int" else float(raw)
            except (ValueError, tk.TclError):
                pass
        return params
