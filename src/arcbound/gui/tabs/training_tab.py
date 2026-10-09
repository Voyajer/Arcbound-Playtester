"""Training tab with replay selection and training controls."""

import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Callable, Optional

from arcbound.training.trainer import Trainer, TrainingConfig

# Reuse descriptions from settings tab
from arcbound.gui.tabs.settings_tab import (
    PARAM_DESCRIPTIONS,
    _Tooltip,
    _add_tooltip,
    _add_metric_tooltip,
)


class TrainingTab(ttk.Frame):
    """Tab for configuring and running training."""

    def __init__(
        self,
        master: tk.Misc,
        replays_dir: Path,
        models_dir: Optional[Path] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.replays_dir = replays_dir
        self.models_dir = models_dir
        self.on_start = on_start
        self.on_stop = on_stop
        self._training = False
        self._selected_model: Optional[str] = None
        self._entries: dict = {}
        # Guards the run lifecycle so the background thread's finalization and
        # the Stop button don't both finalize (double on_stop / conflicting logs).
        self._run_done = False
        self._stop_requested_by_user = False
        # Parameters captured in _start() and consumed by the background thread.
        self._pending: dict = {}

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

        # Player-type filter: which players' moves/decisions to learn from.
        filter_frame = ttk.Frame(top)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        lbl_learn = ttk.Label(filter_frame, text="Learn From:")
        lbl_learn.pack(side=tk.LEFT)
        self.cmb_player_types = ttk.Combobox(
            filter_frame,
            values=["All players", "AI only", "Human + Bot", "Human only", "Bot only"],
            state="readonly",
            width=16,
        )
        self.cmb_player_types.set("All players")
        self.cmb_player_types.pack(side=tk.LEFT, padx=(8, 0))
        _add_tooltip(lbl_learn, "learn_from")
        _add_tooltip(self.cmb_player_types, "learn_from")

        # Head selection: which model head(s) to train this run.
        head_frame = ttk.Frame(top)
        head_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(head_frame, text="Train Heads:").pack(side=tk.LEFT)
        self.var_policy = tk.BooleanVar(value=True)
        self.var_value = tk.BooleanVar(value=True)
        chk_policy = ttk.Checkbutton(head_frame, text="Policy", variable=self.var_policy)
        chk_policy.pack(side=tk.LEFT, padx=(8, 4))
        chk_value = ttk.Checkbutton(head_frame, text="Value", variable=self.var_value)
        chk_value.pack(side=tk.LEFT, padx=4)
        _add_tooltip(chk_policy, "train_policy")
        _add_tooltip(chk_value, "train_value")

        # Parameters
        param_frame = ttk.LabelFrame(top, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        self._add_int_field(param_frame, "epochs", "Epochs", 5, 1, 10000)
        self._add_int_field(param_frame, "batch_size", "Batch Size", 32, 1, 4096)
        self._add_float_field(param_frame, "learning_rate", "Learning Rate", 0.001)
        self._add_float_field(param_frame, "entropy_coef", "Entropy Coef", 0.01)
        self._add_float_field(param_frame, "value_coef", "Value Coef", 0.5)
        self._add_float_field(param_frame, "gamma", "Gamma", 0.99)
        self._add_float_field(param_frame, "td_weight", "TD Weight", 0.5)

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

        # Live training metrics: running weighted averages (exponential moving
        # averages) of the per-batch values, updated as training progresses.
        # Hover any label for a tooltip explaining what the value means and
        # what healthy vs. warning behavior looks like.
        self._build_metrics_panel(top)

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
        # ttk.Spinbox's value= option is silently ignored on some Tcl builds
        # (the field renders blank), so set the displayed text explicitly.
        entry.delete(0, tk.END)
        entry.insert(0, str(default))
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

    # ------------------------------------------------------------------
    # Live training metrics (weighted averages)
    # ------------------------------------------------------------------
    # (key, display label, format) for each metric shown in the panel.
    _METRIC_FIELDS = [
        ("total_loss", "Total", "{:.4f}"),
        ("policy_loss", "Policy", "{:.4f}"),
        ("value_loss", "Value", "{:.4f}"),
        ("entropy", "Entropy", "{:.4f}"),
    ]
    # Smoothing factor for the exponential moving average. 0.9 keeps ~10% of
    # the newest batch and ~90% of the previous average, so the displayed value
    # tracks the trend without jittering on every batch.
    _EMA_ALPHA = 0.1

    def _build_metrics_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Training Metrics (weighted avg)", padding=8)
        frame.pack(fill=tk.X, pady=(0, 10))

        self._metric_labels: dict = {}
        self._metric_ema: dict = {}
        self._metric_count: int = 0

        grid = ttk.Frame(frame)
        grid.pack(fill=tk.X)
        for col, (key, label, _fmt) in enumerate(self._METRIC_FIELDS):
            head = ttk.Label(grid, text=label, font=("TkDefaultFont", 9, "bold"))
            head.grid(row=0, column=col, padx=12, sticky="w")
            _add_metric_tooltip(head, key)
            val = ttk.Label(grid, text="—", font=("TkFixedFont", 10), width=12, anchor="w")
            val.grid(row=1, column=col, padx=12, sticky="w")
            _add_metric_tooltip(val, key)
            self._metric_labels[key] = val

        # A small caption so the user knows these are smoothed, live values.
        cap = ttk.Label(
            frame,
            text="Running weighted averages of per-batch values (hover a value for details).",
            foreground="#666666",
        )
        cap.pack(anchor="w", pady=(4, 0))
        _add_metric_tooltip(cap, "total_loss")

    def _on_metrics(self, metrics: dict):
        """Called from the training thread with per-batch metrics.

        Updates the exponential moving averages and schedules a UI refresh on
        the main thread.
        """
        alpha = self._EMA_ALPHA
        for key, _label, _fmt in self._METRIC_FIELDS:
            v = metrics.get(key)
            if v is None:
                continue
            v = float(v)
            if key in self._metric_ema:
                self._metric_ema[key] = (1 - alpha) * self._metric_ema[key] + alpha * v
            else:
                self._metric_ema[key] = v
        self._metric_count += 1
        # Snapshot for the main thread.
        snapshot = dict(self._metric_ema)
        count = self._metric_count
        self.after(0, self._update_metrics_safe, snapshot, count)

    def _update_metrics_safe(self, snapshot: dict, count: int):
        for key, _label, fmt in self._METRIC_FIELDS:
            if key in snapshot:
                self._metric_labels[key].config(text=fmt.format(snapshot[key]))

    def select_model(self, name: str):
        self._selected_model = name
        self.lbl_model.config(text=name if name else "—")

    def _browse_data(self):
        path = filedialog.askdirectory(initialdir=self.replays_dir, title="Select Replay Directory")
        if path:
            self.lbl_data_path.config(text=path)
            from arcbound.logging.replay_codec import list_replays
            count = len(list_replays(Path(path)))
            self.lbl_file_count.config(text=f"Files found: {count} games (including subfolders)")

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
            epochs=params.get("epochs", 5),
            batch_size=params.get("batch_size", 32),
            learning_rate=params.get("learning_rate", 0.001),
            entropy_coef=params.get("entropy_coef", 0.01),
            value_coef=params.get("value_coef", 0.5),
            gamma=params.get("gamma", 0.99),
            td_weight=params.get("td_weight", 0.5),
        )

        # Load the model's architecture config so training continues from the
        # existing checkpoint (or initializes a fresh model with the right shape).
        model_config = None
        model_dir = None
        if self.models_dir:
            model_dir = self.models_dir / self._selected_model
            config_path = model_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        raw = json.load(f)
                    from arcbound.encoder.transformer import ModelConfig
                    model_config = ModelConfig.from_dict(raw.get("encoder", {}))
                except Exception as e:
                    self._log(f"Warning: could not read model config: {e}")

        # Validate head selection on the main thread (needs a messagebox).
        self._train_policy = self.var_policy.get()
        self._train_value = self.var_value.get()
        if not self._train_policy and not self._train_value:
            messagebox.showinfo("Info", "Select at least one head to train (Policy and/or Value).")
            return
        if not replay_dir.exists():
            self._log(f"Replay directory not found: {replay_dir}")
            return

        player_types = self._selected_player_types()
        if player_types:
            self._log(f"Learning from: {', '.join(sorted(player_types))}")

        # Everything slow (vocab build + replay loading + training) runs in a
        # background thread so the UI stays responsive and the progress bar can
        # show live "Loading replays (i/n)" feedback. The trainer is created in
        # the thread because the vocab (built there) is needed at construction.
        self._pending = {
            "replay_dir": replay_dir,
            "cfg": cfg,
            "model_config": model_config,
            "model_dir": model_dir,
            "params": params,
            "player_types": player_types,
        }

        self._training = True
        self._run_done = False
        self._stop_requested_by_user = False
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress["value"] = 0
        self.lbl_progress.config(text="Preparing…")
        # Reset the live metrics panel for the new run.
        self._metric_ema = {}
        self._metric_count = 0
        for key, _label, _fmt in self._METRIC_FIELDS:
            self._metric_labels[key].config(text="—")

        # Run the whole run (prep + training) in a background thread.
        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

        if self.on_start:
            self.on_start()

    def _run_training(self):
        """Background thread: build vocab, load replays, then train.

        Runs off the main thread so the slow prep phase (which re-reads every
        replay file) doesn't freeze the UI. Progress is reported via
        ``on_load_progress`` (per file) and ``on_progress`` (per epoch).
        """
        p = self._pending
        replay_dir: Path = p["replay_dir"]
        cfg = p["cfg"]
        params = p["params"]
        player_types = p["player_types"]

        # Build the card/keyword vocabularies from the replay files so the model
        # can learn per-card / per-keyword identity embeddings.
        vocab = None
        kw_vocab = None
        if replay_dir.exists():
            from arcbound.encoder.vocabulary import CardVocabulary, KeywordVocabulary
            self._log("Building card vocabulary…")
            vocab = CardVocabulary.build_from_replays(replay_dir)
            if len(vocab) <= 1:
                self._log("No card names found in replays; card identity embedding disabled.")
                vocab = None
            else:
                self._log(f"Card vocabulary: {len(vocab) - 1} unique cards")
            self._log("Building keyword vocabulary…")
            kw_vocab = KeywordVocabulary.build_from_replays(replay_dir)
            if len(kw_vocab) <= 1:
                self._log("No keywords found in replays; keyword embedding disabled.")
                kw_vocab = None
            else:
                self._log(f"Keyword vocabulary: {len(kw_vocab) - 1} unique keywords")

        self._trainer = Trainer(
            config=cfg,
            model_config=p["model_config"],
            vocab=vocab,
            kw_vocab=kw_vocab,
            on_progress=self._on_progress,
            on_log=self._log,
            on_metrics=self._on_metrics,
            on_load_progress=self._on_load_progress,
        )

        # Point the trainer at the model directory so it saves checkpoints there
        model_dir = p["model_dir"]
        if model_dir:
            self._trainer.set_model_dir(model_dir)
            # Resume from an existing checkpoint if present
            ckpt = model_dir / "model.pt"
            if ckpt.exists():
                try:
                    from arcbound.encoder.transformer import load_model_from_checkpoint
                    self._trainer.model = load_model_from_checkpoint(str(ckpt))
                    # A resumed checkpoint has a trained value head, so TD
                    # rewards can be computed for policy training.
                    self._trainer._value_trained = True
                    self._log(f"Resuming from checkpoint: {ckpt}")
                except Exception as e:
                    self._log(f"Warning: could not load checkpoint ({e}); starting fresh")

        # Load replays (optionally restricted to specific player roles) for the
        # head(s) the user selected. Policy needs decision-level records (only
        # present when the external AI played); value experiences work for any
        # recorded move (human, bot, or AI).
        if self._train_policy:
            self._log("Loading policy experiences…")
            self._trainer.load_replay_dir(replay_dir, player_types=player_types)
        if self._train_value:
            self._log("Loading value experiences…")
            self._trainer.load_value_replay_dir(
                replay_dir, gamma=params.get("gamma", 0.99), player_types=player_types
            )

        if self._train_policy and len(self._trainer.buffer) == 0:
            self._log("Warning: no policy experiences found (decision records only exist when the external AI played).")
        if self._train_value and not self._trainer.value_buffer:
            self._log("Warning: no value experiences found.")
        if (self._train_policy and len(self._trainer.buffer) == 0) and (
            self._train_value and not self._trainer.value_buffer
        ):
            # No data: report on the main thread and end the run.
            self.after(0, self._no_data_finished)
            return

        # Train only the head(s) the user selected. Each head is skipped when
        # its buffer is empty, so a human-vs-bot replay (0 policy, N value)
        # still trains the value head when Value is checked.
        #
        # Value is trained FIRST so the value head is available to compute
        # dense TD rewards (V(s') - V(s)) for the policy training that follows.
        # Without a trained value head, the policy falls back to the sparse
        # game-outcome reward only.
        if self._train_value and self._trainer.value_buffer:
            self._trainer.train_value()
        if self._train_policy and len(self._trainer.buffer) > 0:
            self._trainer.train()
        # Update UI on main thread after training finishes
        self.after(0, self._training_finished)

    def _no_data_finished(self):
        """Main-thread finalizer for the 'no replay data' case."""
        if self._run_done:
            return
        self._run_done = True
        self._training = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_progress.config(text="No replay data found.")
        messagebox.showinfo("Info", "No replay data found for the selected head(s). Play some games first.")
        if self.on_stop:
            self.on_stop()

    def _training_finished(self):
        # Guard against double-finalization (e.g. Stop clicked just as training
        # finished on the background thread).
        if self._run_done:
            return
        self._run_done = True
        self._training = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress["value"] = 100
        self._log("Training complete.")
        if self.on_stop:
            self.on_stop()

    def _stop(self):
        self._stop_requested_by_user = True
        if hasattr(self, "_trainer"):
            self._trainer.stop()
        # If the run has already finished (or never started training), finalize
        # now. Otherwise the background thread's finalizer will do it.
        if self._run_done:
            return
        self._run_done = True
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

    def _on_load_progress(self, current: int, total: int, filename: str):
        """Called from the background thread as each replay file is loaded.

        Maps the (slow) loading phase onto the progress bar so the user sees
        live "Loading replays (i/n)" feedback instead of a frozen UI. The bar
        fills from 0% to ~95% across the load; the remaining 5% is reserved for
        the training phase (which reports its own 0-100% via on_progress).
        """
        if total <= 0:
            return
        frac = current / total
        # Reserve the top 5% for training so the bar never looks "done" early.
        pct = frac * 95.0
        self.after(0, self._update_load_progress_safe, pct, current, total, filename)

    def _update_load_progress_safe(self, pct: float, current: int, total: int, filename: str):
        self.progress["value"] = pct
        self.lbl_progress.config(text=f"Loading replays ({current}/{total}): {filename}")

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

    def _selected_player_types(self):
        """Map the 'Learn From' combobox to a set of player roles (or None)."""
        choice = self.cmb_player_types.get()
        mapping = {
            "All players": None,
            "AI only": {"ai"},
            "Human + Bot": {"human", "bot"},
            "Human only": {"human"},
            "Bot only": {"bot"},
        }
        return mapping.get(choice)
