"""Settings tab with hyperparameter editor."""

import json
import re
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Callable, Optional


# Hover tooltip descriptions for each parameter.
#
# Each entry explains (1) what the parameter is, (2) what it does, and
# (3) how to adjust it based on the amount and quality of the replay data
# you have. Keep them short enough to read in a tooltip.
PARAM_DESCRIPTIONS = {
    # --- Model architecture (Settings tab) ---
    "d_model": (
        "Transformer hidden dimension size — the width of every token's internal "
        "representation. Larger = more capacity to memorize card interactions, but "
        "slower training and inference, and it must be divisible by nhead. "
        "With little data (< ~50 games) keep it small (256-512) so the model can't "
        "memorize noise; with lots of data (hundreds of games) you can grow it "
        "(512-1024) to capture more nuance."
    ),
    "nhead": (
        "Number of attention heads — parallel 'lenses' the model uses to relate "
        "tokens (e.g. one head may track mana, another threats). Splits d_model "
        "into nhead equal streams, so d_model must be divisible by nhead. "
        "Rarely needs changing: 4-8 is a good range. More heads help when your "
        "data contains many distinct interaction types; with small data, fewer "
        "heads (4) concentrate capacity better."
    ),
    "num_layers": (
        "Number of transformer layers — how many times tokens are re-mixed before "
        "the model decides. More layers = deeper reasoning but slower and harder "
        "to train. With small data, 3-4 layers is usually enough; with large, "
        "diverse data, 6-8 can help. Watch for the loss plateauing or rising — "
        "that's a sign of too many layers for your data."
    ),
    "dim_feedforward": (
        "Inner feed-forward network width — the per-token 'thinking space' between "
        "attention steps. Typically 4x d_model. Larger = more capacity, slower "
        "training. Scale it with d_model (keep ~4x). With limited data, a smaller "
        "value (2x d_model) reduces overfitting."
    ),
    "max_seq_len": (
        "Maximum token sequence length the model can process — a hard cap on how "
        "many board tokens (cards, players, options) fit in one input. Must be at "
        "least as large as your biggest board state or those states get truncated. "
        "2048 covers most games; raise it only if you see truncation warnings, "
        "since longer sequences cost quadratically more compute."
    ),
    "dropout": (
        "Fraction of neurons randomly zeroed during training — the main defense "
        "against overfitting (memorizing your replays instead of generalizing). "
        "With little data, raise it (0.1-0.2) so the model can't memorize; with "
        "lots of high-quality data, lower it (0.05-0.1) so it can learn fine "
        "details. 0.0 disables it and will overfit on small datasets."
    ),
    "learning_rate": (
        "Step size for the gradient optimizer — how boldly the model updates its "
        "weights each batch. Too high = loss spikes or diverges; too low = slow, "
        "stuck learning. Start at 0.001. If the loss jumps around or explodes, "
        "halve it; if it barely moves, double it. Small datasets tolerate lower "
        "rates better (0.0003-0.001)."
    ),
    "entropy_coef": (
        "Weight of the entropy bonus — encourages the policy to keep its option "
        "probabilities spread out rather than collapsing onto one choice. "
        "Useful early in training (keeps exploration) but too high makes the AI "
        "play randomly. With small data, keep it modest (0.01); if the AI always "
        "picks the same option, raise it slightly; if it plays erratically, lower "
        "it toward 0."
    ),
    "value_coef": (
        "Weight of the value/Q loss relative to the policy loss — how hard the "
        "model is pushed to estimate board value (win/lose) alongside choosing "
        "options. Higher = better position evaluation but the value signal can "
        "dominate. With lots of games (value data is plentiful), 0.5-1.0 is fine; "
        "with few games, keep it at 0.5 or lower so the sparse value targets "
        "don't destabilize policy learning."
    ),
    "gamma": (
        "Discount factor for value targets — how much earlier moves in a game are "
        "credited with the final outcome. 0.99 (default) means a move 10 turns "
        "from the end gets ~90% of the outcome's weight. Lower (0.9) makes the "
        "value head focus on late-game positions; higher (0.999) spreads credit "
        "further back. Keep 0.99 unless you have a specific reason; it matters "
        "most when you have many long games."
    ),
    "td_weight": (
        "Blend weight for the dense TD reward (V(s') - V(s)) in policy training. "
        "The policy reward becomes (1 - td_weight) * game_outcome + td_weight * "
        "TD_reward. The TD reward is a per-step signal from the value head: it "
        "rewards moves that improve the position and — crucially — treats a good "
        "pass as neutral (the position is unchanged, so the reward is ~0, not "
        "negative). This is what stops the AI learning that 'always pass' is "
        "safe. 0.5 (default) balances the sparse win/lose signal with the dense "
        "per-step signal. 0.0 = old behavior (game outcome only). Only used when "
        "the value head has been trained (value is trained first automatically)."
    ),
    "clip_epsilon": (
        "PPO clipping range — limits how much the policy can change per update "
        "(the ratio of new to old action probabilities is clamped to "
        "[1-eps, 1+eps]). Smaller = safer, slower policy updates; larger = faster "
        "but riskier. 0.2 is the standard. With small or noisy data, use 0.1-0.15 "
        "to avoid the policy swinging wildly between batches."
    ),
    "max_grad_norm": (
        "Maximum gradient norm for clipping — a safety brake that rescales "
        "gradients if they exceed this value, preventing exploding-gradient "
        "crashes. 1.0 is the standard and rarely needs changing. Lower it (0.5) "
        "if training diverges; raise it only if training is suspiciously slow."
    ),
    # --- Server (Settings tab) ---
    "host": (
        "Network interface the AI server binds to. 0.0.0.0 = all interfaces "
        "(needed if Forge runs on another machine); 127.0.0.1 = local only. "
        "Does not affect training or model quality."
    ),
    "port": (
        "TCP port the AI server listens on. Forge's external-AI config must use "
        "the same port or it can't reach the server. Does not affect training or "
        "model quality."
    ),
    "timeout_ms": (
        "Milliseconds to wait for the AI's decision before falling back to "
        "heuristics. Higher = more think time (better play, slower game); lower = "
        "faster but more fallbacks. 5000 is a good default; raise to 10000+ if "
        "you see many 'timeout' entries in the action terminal, especially with a "
        "large model."
    ),
    "epsilon": (
        "Epsilon-greedy exploration for inference — with this probability the AI "
        "picks a random option instead of the model's top pick. This keeps the "
        "replay data diverse and breaks the pass-mode collapse (without it the AI "
        "only ever trains on its own pass-heavy decisions and learns to always "
        "pass). 0.15 (default) is a good starting point; raise to 0.2-0.3 if the "
        "AI still rarely plays cards, lower toward 0.0 once it plays well and you "
        "want pure exploitation. Requires a server restart to take effect."
    ),
    # --- Training (Training tab) ---
    "epochs": (
        "Number of full passes through your replay data. More epochs = more "
        "learning, but past a point the model just memorizes (overfits). With "
        "little data, use fewer epochs (20-50) and watch the loss; with lots of "
        "data, 100+ is fine. Stop early if the loss stops improving."
    ),
    "batch_size": (
        "Number of training samples per gradient update. Larger batches = smoother, "
        "more stable gradients but slower per-epoch progress and more memory; "
        "smaller batches = noisier but more updates per epoch. 32 is a good "
        "default. With small datasets, 16-32 keeps updates frequent; with large "
        "datasets, 64-128 is more stable."
    ),
    # --- Training tab: data filters ---
    "learn_from": (
        "Which players' moves to learn from. 'AI only' trains purely on the "
        "external AI's own decisions (best for imitating a strong model); "
        "'Human + Bot' or 'Human only' learns from your own play (useful when you "
        "have few AI games but many human games); 'All players' uses everything. "
        "Policy learning only sees decisions where the external AI played; value "
        "learning works for any player type."
    ),
    "train_policy": (
        "Train the policy head — learns WHICH option to pick (cast, target, "
        "block, ...) from decision records. Only available when the external AI "
        "actually played (its decisions are recorded with the options offered). "
        "Uncheck it to skip policy training, e.g. when you only have human-vs-bot "
        "replays (0 policy experiences) and just want to improve board evaluation."
    ),
    "train_value": (
        "Train the value head — learns to estimate how good a board position is "
        "(win/lose) from the outcome of every recorded move, for any player type "
        "(human, bot, or AI). This is the head that powers the evaluation graph "
        "and move analysis. Uncheck it to skip value training, e.g. when you have "
        "plenty of value data but want to focus a run on the policy."
    ),
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


# Hover tooltip descriptions for the live training metrics (Training tab).
#
# Each entry explains (1) what the metric means, (2) what healthy behavior
# looks like, and (3) warning signs with a quick fix. The values shown next to
# these tooltips are exponential moving averages (weighted averages) of the
# per-batch values, so they smooth out batch-to-batch noise while still
# tracking the trend.
METRIC_DESCRIPTIONS = {
    "total_loss": (
        "Total Loss — the overall error the model is minimizing. It combines "
        "Policy Loss, Value Loss, and an Entropy bonus.\n\n"
        "Healthy: fluctuates dynamically and generally trends downward over "
        "long training cycles.\n"
        "Warning: if it stays perfectly flat across multiple epochs, your "
        "gradients are dead or stuck (try a higher learning rate or fresh "
        "replay data)."
    ),
    "policy_loss": (
        "Policy Loss (Actor) — how well the AI is choosing actions: whether a "
        "chosen action was better or worse than the critic expected.\n\n"
        "Healthy: dynamic and often negative (e.g. -0.15 to -0.25). A negative "
        "value means the agent is successfully maximizing rewards.\n"
        "Warning: if it flatlines at a hard ceiling (like exactly 0.8000 or "
        "1.2000), your updates are too aggressive — the policy is hitting the "
        "PPO clipping wall and freezing. Fix: lower the Learning Rate or "
        "reduce the total Epochs."
    ),
    "value_loss": (
        "Value Loss (Critic) — how accurately the AI predicts who will win "
        "from the current board state (Mean Squared Error).\n\n"
        "Healthy: starts higher and smoothly descends toward a very low "
        "number (e.g. 0.005 down to 0.0001).\n"
        "Warning: if it drops to zero almost instantly on a tiny dataset, the "
        "critic is overfitting (memorizing the specific games instead of "
        "learning general MTG rules). Fix: increase your sample size (load "
        "more game replays)."
    ),
    "entropy": (
        "Entropy (Exploration Rate) — the randomness of the AI's choices. High "
        "entropy = unpredictable exploration; low entropy = confident "
        "execution.\n\n"
        "Healthy: starts high (e.g. 1.5 to 2.0+, depending on available "
        "actions) and very gradually decreases over thousands of steps as the "
        "AI masters the game.\n"
        "Warning: if it drops to near zero instantly, the policy has collapsed "
        "into a repetitive loop (e.g. constantly passing the turn) — raise the "
        "Entropy Coef to force exploration. If it stays completely flat and "
        "high, the AI is just guessing randomly and isn't learning."
    ),
}


def _add_metric_tooltip(widget, key):
    """Attach a hover tooltip to a widget based on a training-metric key."""
    desc = METRIC_DESCRIPTIONS.get(key)
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
        ("td_weight", "TD Weight", 0.5, 0.0, 1.0),
        ("clip_epsilon", "Clip Epsilon", 0.2, 0.0, 1.0),
        ("max_grad_norm", "Max Grad Norm", 1.0, 0.0, 100.0),
    ]

    # Server fields
    SERVER_FIELDS = [
        ("host", "Host", "0.0.0.0"),
        ("port", "Port", 8080, 1, 65535),
        ("timeout_ms", "Timeout (ms)", 5000, 100, 60000),
        ("epsilon", "Epsilon", 0.15, 0.0, 1.0),
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
        # Overlay the live server values (host/port/timeout/epsilon) from
        # config/default.yaml so the Server section reflects reality, not just
        # the built-in defaults.
        self._load_server_config()

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
        self._add_int_field(srv, "port", "Port", 8080, 1, 65535)
        self._add_int_field(srv, "timeout_ms", "Timeout (ms)", 5000, 100, 60000)
        self._add_float_field(srv, "epsilon", "Epsilon", 0.15, 0.0, 1.0)

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
        # ttk.Spinbox's value= option is silently ignored on some Tcl builds
        # (the field renders blank), so set the displayed text explicitly.
        entry.delete(0, tk.END)
        entry.insert(0, str(default))
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

    def _server_yaml_path(self) -> Path:
        """Path to config/default.yaml (the file the server actually reads).

        This file lives at <project_root>/config/default.yaml. settings_tab.py
        is at <project_root>/src/arcbound/gui/tabs/, so five parents up.
        """
        return (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "config"
            / "default.yaml"
        )

    def _load_server_config(self):
        """Populate the Server section from config/default.yaml.

        The server reads host/port/timeout from the ``server:`` block and
        epsilon from the ``inference:`` block. Values not present in the file
        keep their built-in defaults.
        """
        path = self._server_yaml_path()
        if not path.exists():
            return
        try:
            import yaml

            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return
        server = cfg.get("server") or {}
        inference = cfg.get("inference") or {}
        values = {
            "host": server.get("host"),
            "port": server.get("port"),
            "timeout_ms": server.get("timeout_ms"),
            "epsilon": inference.get("epsilon"),
        }
        for key, val in values.items():
            if val is None or key not in self._entries:
                continue
            widget, vtype, _, _ = self._entries[key]
            widget.delete(0, tk.END)
            widget.insert(0, str(val))

    def _save_server_config(self):
        """Persist the Server section to config/default.yaml.

        Uses targeted line replacement (rather than a full YAML dump) so the
        file's comments and layout are preserved. host/port/timeout go under
        ``server:``; epsilon goes under ``inference:``.
        """
        path = self._server_yaml_path()
        if not path.exists():
            return
        try:
            text = path.read_text()
        except Exception:
            return

        def fmt(val):
            return f'"{val}"' if isinstance(val, str) else str(val)

        def set_key(text: str, key: str, val, section: str) -> str:
            # Match the key anywhere in the file, capturing its leading
            # indentation so it can be preserved on replacement.
            pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*.*$", re.MULTILINE)
            line = f"{key}: {fmt(val)}"
            m = pattern.search(text)
            if m:
                # Preserve the original indentation of the key.
                return pattern.sub(lambda mm: mm.group(1) + line, text, count=1)
            # Key missing — append it under the section header, indented.
            sec = re.search(rf"^{re.escape(section)}\s*:\s*$", text, re.MULTILINE)
            if sec:
                insert_at = sec.end()
                return text[:insert_at] + "\n  " + line + text[insert_at:]
            return text

        try:
            host = self._get_value("host")
            port = self._get_value("port")
            timeout_ms = self._get_value("timeout_ms")
            epsilon = self._get_value("epsilon")
        except (ValueError, tk.TclError):
            return

        text = set_key(text, "host", host, "server")
        text = set_key(text, "port", port, "server")
        text = set_key(text, "timeout_ms", timeout_ms, "server")
        text = set_key(text, "epsilon", epsilon, "inference")
        path.write_text(text)

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
        """Save current values.

        Server fields (host/port/timeout/epsilon) are written to
        config/default.yaml — they are server-level, not model-level, and the
        server reads them there. Model architecture + training fields are
        written to the selected model's config.json.
        """
        # 1. Persist the Server section to config/default.yaml (no model needed).
        self._save_server_config()

        # 2. Persist model architecture + training fields to the model config.
        if not self._current_model:
            messagebox.showinfo("Info", "Server settings saved. Select a model to save model settings.")
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
