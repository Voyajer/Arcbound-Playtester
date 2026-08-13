"""Settings tab with hyperparameter editor."""

import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Callable, Optional


# Hover tooltip descriptions for each parameter
PARAM_DESCRIPTIONS = {
    "d_model": "Transformer hidden dimension size. Larger = more capacity but slower. Must be divisible by nhead.",
    "nhead": "Number of attention heads. Splits d_model into parallel attention streams.",
    "num_layers": "Number of transformer encoder/decoder layers. More layers = deeper model.",
    "dim_feedforward": "Inner feed-forward network dimension. Typically 4x d_model.",
    "max_seq_len": "Maximum token sequence length the model can process.",
    "batch_size": "Number of training samples per gradient update step.",
    "dropout": "Fraction of neurons randomly zeroed during training to prevent overfitting.",
    "learning_rate": "Step size for gradient descent optimizer. Too high = unstable, too low = slow.",
    "entropy_coef": "Weight of the entropy bonus term. Encourages exploration during training.",
    "value_coef": "Weight of the value loss term relative to policy loss.",
    "gamma": "Discount factor for future rewards. 1.0 = no discount, 0.0 = only immediate rewards.",
    "clip_epsilon": "PPO clipping range. Limits how much policy can change per update step.",
    "max_grad_norm": "Maximum gradient norm for clipping. Prevents exploding gradients.",
    "host": "Network interface the AI server binds to. 0.0.0.0 = all interfaces.",
    "port": "TCP port the AI server listens on. Forge must match this value.",
    "timeout_ms": "Milliseconds to wait for AI decision before fallback. Higher = more think time.",
    "epochs": "Number of full passes through the training dataset.",
}


class _Tooltip:
    """Simple hover tooltip for Tkinter widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tooltip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("TkDefaultFont", 8), wraplength=350)
        label.pack()

    def _hide(self, _event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def _add_tooltip(widget, key):
    """Attach a hover tooltip to a widget based on parameter key."""
    desc = PARAM_DESCRIPTIONS.get(key)
    if desc:
        _Tooltip(widget, desc)


class SettingsTab(ttk.Frame):
    """Tab for editing model architecture and training hyperparameters."""

    # Integer fields
    INT_FIELDS = [
        ("d_model", "d_model", 512, 64, 4096),
        ("nhead", "nhead", 8, 1, 64),
        ("num_layers", "num_layers", 6, 1, 128),
        ("dim_feedforward", "dim_feedforward", 2048, 128, 32768),
        ("max_seq_len", "max_seq_len", 2048, 256, 65536),
        ("batch_size", "Batch Size", 32, 1, 4096),
    ]

    # Float fields
    FLOAT_FIELDS = [
        ("dropout", "Dropout", 0.1, 0.0, 1.0),
        ("learning_rate", "Learning Rate", 0.001, 0.0, 1.0),
        ("entropy_coef", "Entropy Coef", 0.01, 0.0, 1.0),
        ("value_coef", "Value Coef", 0.5, 0.0, 10.0),
        ("gamma", "Gamma", 0.99, 0.0, 1.0),
        ("clip_epsilon", "Clip Epsilon", 0.2, 0.0, 1.0),
        ("max_grad_norm", "Max Grad Norm", 1.0, 0.0, 100.0),
    ]

    # Server fields
    SERVER_FIELDS = [
        ("host", "Host", "0.0.0.0"),
        ("port", "Port", 8090, 1, 65535),
        ("timeout_ms", "Timeout (ms)", 5000, 100, 60000),
    ]

    def __init__(
        self,
        master: tk.Misc,
        models_dir: Path,
        config_path: Path,
        on_apply: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.models_dir = models_dir
        self.config_path = config_path
        self.on_apply = on_apply
        self._current_model: Optional[str] = None

        # Store entry widgets for value retrieval
        self._entries: dict = {}

        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        padding = ttk.Frame(scroll_frame, padding=20)
        padding.pack(fill=tk.BOTH, expand=True)

        # Model Architecture
        arch = ttk.LabelFrame(padding, text="Model Architecture", padding=10)
        arch.pack(fill=tk.X, pady=(0, 10))

        for key, label, default, lo, hi in self.INT_FIELDS[:6]:
            self._add_int_field(arch, key, label, default, lo, hi)

        # Training Hyperparameters
        train = ttk.LabelFrame(padding, text="Training Hyperparameters", padding=10)
        train.pack(fill=tk.X, pady=(0, 10))

        for key, label, default, lo, hi in self.FLOAT_FIELDS:
            self._add_float_field(train, key, label, default, lo, hi)

        # Server
        srv = ttk.LabelFrame(padding, text="Server", padding=10)
        srv.pack(fill=tk.X, pady=(0, 10))

        self._add_string_field(srv, "host", "Host", "0.0.0.0")
        self._add_int_field(srv, "port", "Port", 8090, 1, 65535)
        self._add_int_field(srv, "timeout_ms", "Timeout (ms)", 5000, 100, 60000)

        # Buttons
        btn_frame = ttk.Frame(padding)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self._load_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side=tk.LEFT, padx=5)

    def _add_int_field(self, parent, key: str, label: str, default: int, lo: int, hi: int):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=label, width=18)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Spinbox(frame, from_=lo, to=hi, width=12, value=default)
        entry.pack(side=tk.RIGHT)
        self._entries[key] = (entry, "int", lo, hi)
        _add_tooltip(lbl, key)
        _add_tooltip(entry, key)

    def _add_float_field(self, parent, key: str, label: str, default: float, lo: float, hi: float):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=label, width=18)
        lbl.pack(side=tk.LEFT)
        step = max(0.001, default / 100)
        entry = ttk.Entry(frame, width=14)
        entry.insert(0, str(default))
        entry.pack(side=tk.RIGHT)
        self._entries[key] = (entry, "float", lo, hi)
        _add_tooltip(lbl, key)
        _add_tooltip(entry, key)

    def _add_string_field(self, parent, key: str, label: str, default: str):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        lbl = ttk.Label(frame, text=label, width=18)
        lbl.pack(side=tk.LEFT)
        entry = ttk.Entry(frame, width=14)
        entry.insert(0, default)
        entry.pack(side=tk.RIGHT)
        self._entries[key] = (entry, "string", None, None)
        _add_tooltip(lbl, key)
        _add_tooltip(entry, key)

    def _get_value(self, key: str):
        entry, vtype, lo, hi = self._entries[key]
        raw = entry.get().strip()
        if vtype == "int":
            val = int(raw)
            return max(lo, min(hi, val))
        elif vtype == "float":
            val = float(raw)
            return max(lo, min(hi, val))
        else:
            return raw

    def _load_defaults(self):
        for key, meta in self._entries.items():
            widget, vtype, _, _ = meta
            if vtype == "string":
                continue
            # Reset to default from field definitions
            for defn in self.INT_FIELDS + self.FLOAT_FIELDS + self.SERVER_FIELDS:
                if defn[0] == key:
                    if vtype == "int":
                        widget.delete(0, tk.END)
                        widget.insert(0, str(defn[2]))
                    elif vtype == "float":
                        widget.delete(0, tk.END)
                        widget.insert(0, str(defn[2]))
                    break

    def select_model(self, name: str):
        """Load settings from model config when selected."""
        self._current_model = name
        if name:
            config_path = self.models_dir / name / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                enc = cfg.get("encoder", {})
                trn = cfg.get("training", {})
                for key, meta in self._entries.items():
                    widget, vtype, _, _ = meta
                    val = None
                    if key in enc:
                        val = enc[key]
                    elif key in trn:
                        val = trn[key]
                    if val is not None:
                        widget.delete(0, tk.END)
                        widget.insert(0, str(val))

    def _save(self):
        """Save current values to model config."""
        if not self._current_model:
            messagebox.showinfo("Info", "Select a model first.")
            return

        cfg = {"encoder": {}, "training": {}}
        enc_keys = {d[0] for d in self.INT_FIELDS[:6]}
        enc_keys.add("dropout")
        trn_keys = {d[0] for d in self.FLOAT_FIELDS[1:]}

        for key, meta in self._entries.items():
            try:
                val = self._get_value(key)
            except (ValueError, tk.TclError):
                messagebox.showerror("Validation", f"Invalid value for '{key}'.")
                return
            if key in enc_keys:
                cfg["encoder"][key] = val
            elif key in trn_keys:
                cfg["training"][key] = val

        config_path = self.models_dir / self._current_model / "config.json"
        # Merge with existing
        if config_path.exists():
            with open(config_path) as f:
                existing = json.load(f)
            existing["encoder"] = cfg["encoder"]
            existing["training"] = cfg["training"]
            existing["modified"] = __import__("datetime").datetime.now().isoformat()
            cfg = existing

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        messagebox.showinfo("Success", "Settings saved.")

    def _apply(self):
        self._save()
        if self.on_apply:
            self.on_apply()
