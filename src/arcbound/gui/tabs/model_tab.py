"""Model management tab for creating, loading, renaming, and deleting models."""

import json
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from arcbound.gui.tabs.settings_tab import _Tooltip


# Hover tooltip descriptions for the model tab's sections.
MODEL_DESCRIPTIONS = {
    "name": (
        "The model's name — the directory under models/ where its checkpoint "
        "(model.pt), config, and vocabularies live. Select a model in the list "
        "to see its details."
    ),
    "created": (
        "When this model was first created. A model's architecture (encoder "
        "parameters) is fixed at creation; training only updates its weights."
    ),
    "modified": (
        "When the model was last trained or its config was last saved. Compare "
        "with 'Created' to see how recently it has been updated."
    ),
    "games_trained": (
        "How many distinct replay files this model has been trained on (cumulative "
        "across all training runs). A low number means the model has seen little "
        "data — expect weak play and noisy evaluations. Play and log more games, "
        "then train again to grow this."
    ),
    "total_steps": (
        "Total gradient updates applied to this model across all training runs. "
        "Roughly (experiences / batch_size) per epoch. Very low step counts "
        "(hundreds) mean the model is barely trained; thousands to tens of "
        "thousands is more typical for a usable model."
    ),
    "d_model": (
        "Transformer hidden dimension size — the width of every token's internal "
        "representation. Larger = more capacity but slower. Changing this "
        "requires retraining from scratch (the checkpoint shape changes)."
    ),
    "nhead": (
        "Number of attention heads — parallel 'lenses' the model uses to relate "
        "tokens. Must divide d_model evenly. Changing this requires retraining "
        "from scratch."
    ),
    "num_layers": (
        "Number of transformer layers — how many times tokens are re-mixed before "
        "the model decides. More layers = deeper reasoning, slower inference. "
        "Changing this requires retraining from scratch."
    ),
    "dim_feedforward": (
        "Inner feed-forward network width — the per-token 'thinking space' "
        "between attention steps. Typically 4x d_model. Changing this requires "
        "retraining from scratch."
    ),
    "dropout": (
        "Fraction of neurons randomly zeroed during training to prevent "
        "overfitting. Only active while training (no effect at inference). "
        "Higher values help when you have little data."
    ),
    "max_seq_len": (
        "Maximum token sequence length the model can process. Must cover your "
        "largest board states or they get truncated. Changing this requires "
        "retraining from scratch."
    ),
    "load": (
        "Load the selected model into the AI server so it uses this model for "
        "live decisions and evaluations. The server falls back to heuristics "
        "when no model is loaded."
    ),
    "create_config": (
        "Create a default config.json for the selected model (required before "
        "training). This sets the architecture (encoder) and training "
        "hyperparameters; you can tune them in the Settings tab afterwards."
    ),
}


DEFAULT_ENCODER = {
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "max_seq_len": 2048,
}

DEFAULT_TRAINING = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "gamma": 0.99,
    "max_epochs": 100,
}


class ModelTab(ttk.Frame):
    """Tab for model management and info display."""

    def __init__(
        self,
        master: tk.Misc,
        models_dir: Path,
        on_model_loaded: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.models_dir = models_dir
        self.on_model_loaded = on_model_loaded
        self._current_model: Optional[str] = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Model info section
        info_frame = ttk.LabelFrame(main, text="Model Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_name = ttk.Label(info_frame, text="Name: —")
        self.lbl_name.pack(anchor="w", pady=2)
        self.lbl_created = ttk.Label(info_frame, text="Created: —")
        self.lbl_created.pack(anchor="w", pady=2)
        self.lbl_modified = ttk.Label(info_frame, text="Modified: —")
        self.lbl_modified.pack(anchor="w", pady=2)
        self.lbl_games = ttk.Label(info_frame, text="Games Trained: 0")
        self.lbl_games.pack(anchor="w", pady=2)
        self.lbl_steps = ttk.Label(info_frame, text="Total Steps: 0")
        self.lbl_steps.pack(anchor="w", pady=2)
        _Tooltip(self.lbl_name, MODEL_DESCRIPTIONS["name"])
        _Tooltip(self.lbl_created, MODEL_DESCRIPTIONS["created"])
        _Tooltip(self.lbl_modified, MODEL_DESCRIPTIONS["modified"])
        _Tooltip(self.lbl_games, MODEL_DESCRIPTIONS["games_trained"])
        _Tooltip(self.lbl_steps, MODEL_DESCRIPTIONS["total_steps"])

        # Encoder params
        enc_frame = ttk.LabelFrame(main, text="Encoder Parameters", padding=10)
        enc_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_d_model = ttk.Label(enc_frame, text="d_model: —")
        self.lbl_d_model.pack(anchor="w", pady=2)
        self.lbl_nhead = ttk.Label(enc_frame, text="nhead: —")
        self.lbl_nhead.pack(anchor="w", pady=2)
        self.lbl_num_layers = ttk.Label(enc_frame, text="num_layers: —")
        self.lbl_num_layers.pack(anchor="w", pady=2)
        self.lbl_dim_ff = ttk.Label(enc_frame, text="dim_feedforward: —")
        self.lbl_dim_ff.pack(anchor="w", pady=2)
        self.lbl_dropout = ttk.Label(enc_frame, text="dropout: —")
        self.lbl_dropout.pack(anchor="w", pady=2)
        self.lbl_max_seq = ttk.Label(enc_frame, text="max_seq_len: —")
        self.lbl_max_seq.pack(anchor="w", pady=2)
        _Tooltip(self.lbl_d_model, MODEL_DESCRIPTIONS["d_model"])
        _Tooltip(self.lbl_nhead, MODEL_DESCRIPTIONS["nhead"])
        _Tooltip(self.lbl_num_layers, MODEL_DESCRIPTIONS["num_layers"])
        _Tooltip(self.lbl_dim_ff, MODEL_DESCRIPTIONS["dim_feedforward"])
        _Tooltip(self.lbl_dropout, MODEL_DESCRIPTIONS["dropout"])
        _Tooltip(self.lbl_max_seq, MODEL_DESCRIPTIONS["max_seq_len"])

        # Actions
        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=10)
        btn_load = ttk.Button(action_frame, text="Load Model", command=self._load)
        btn_load.pack(side=tk.LEFT, padx=5)
        btn_create = ttk.Button(action_frame, text="Create Default Config", command=self._create_config)
        btn_create.pack(side=tk.LEFT, padx=5)
        _Tooltip(btn_load, MODEL_DESCRIPTIONS["load"])
        _Tooltip(btn_create, MODEL_DESCRIPTIONS["create_config"])

    def select_model(self, name: str):
        """Called when a model is selected in the model list."""
        self._current_model = name
        self._refresh_info()

    def _refresh_info(self):
        if not self._current_model:
            self._clear_info()
            return

        config_path = self.models_dir / self._current_model / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.lbl_name.config(text=f"Name: {cfg.get('name', self._current_model)}")
            self.lbl_created.config(text=f"Created: {cfg.get('created', '—')}")
            self.lbl_modified.config(text=f"Modified: {cfg.get('modified', '—')}")
            self.lbl_games.config(text=f"Games Trained: {cfg.get('games_trained', 0)}")
            self.lbl_steps.config(text=f"Total Steps: {cfg.get('total_steps', 0)}")

            enc = cfg.get("encoder", {})
            self.lbl_d_model.config(text=f"d_model: {enc.get('d_model', '—')}")
            self.lbl_nhead.config(text=f"nhead: {enc.get('nhead', '—')}")
            self.lbl_num_layers.config(text=f"num_layers: {enc.get('num_layers', '—')}")
            self.lbl_dim_ff.config(text=f"dim_feedforward: {enc.get('dim_feedforward', '—')}")
            self.lbl_dropout.config(text=f"dropout: {enc.get('dropout', '—')}")
            self.lbl_max_seq.config(text=f"max_seq_len: {enc.get('max_seq_len', '—')}")
        else:
            self.lbl_name.config(text=f"Name: {self._current_model}")
            self.lbl_created.config(text="Created: —")
            self.lbl_modified.config(text="Modified: —")
            self.lbl_games.config(text="Games Trained: 0")
            self.lbl_steps.config(text="Total Steps: 0")
            self.lbl_d_model.config(text="d_model: —")
            self.lbl_nhead.config(text="nhead: —")
            self.lbl_num_layers.config(text="num_layers: —")
            self.lbl_dim_ff.config(text="dim_feedforward: —")
            self.lbl_dropout.config(text="dropout: —")
            self.lbl_max_seq.config(text="max_seq_len: —")

    def _clear_info(self):
        self.lbl_name.config(text="Name: —")
        self.lbl_created.config(text="Created: —")
        self.lbl_modified.config(text="Modified: —")
        self.lbl_games.config(text="Games Trained: 0")
        self.lbl_steps.config(text="Total Steps: 0")
        self.lbl_d_model.config(text="d_model: —")
        self.lbl_nhead.config(text="nhead: —")
        self.lbl_num_layers.config(text="num_layers: —")
        self.lbl_dim_ff.config(text="dim_feedforward: —")
        self.lbl_dropout.config(text="dropout: —")
        self.lbl_max_seq.config(text="max_seq_len: —")

    def _load(self):
        if not self._current_model:
            messagebox.showinfo("Info", "Select a model first.")
            return
        if self.on_model_loaded:
            self.on_model_loaded(self._current_model)

    def _create_config(self):
        if not self._current_model:
            messagebox.showinfo("Info", "Select a model first.")
            return

        config_path = self.models_dir / self._current_model / "config.json"
        if config_path.exists():
            if not messagebox.askyesno("Overwrite", "Config already exists. Overwrite?"):
                return

        now = datetime.now().isoformat()
        cfg = {
            "name": self._current_model,
            "created": now,
            "modified": now,
            "games_trained": 0,
            "total_steps": 0,
            "encoder": dict(DEFAULT_ENCODER),
            "training": dict(DEFAULT_TRAINING),
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        self._refresh_info()
        messagebox.showinfo("Success", f"Created config for '{self._current_model}'.")
