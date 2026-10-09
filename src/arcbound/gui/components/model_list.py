"""Model selection listbox with CRUD buttons."""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Callable, Optional


class ModelList(ttk.Frame):
    """Left panel for model management with listbox and action buttons."""

    def __init__(
        self,
        master: tk.Misc,
        models_dir: Path,
        on_select: Optional[Callable[[str], None]] = None,
        on_load: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.models_dir = models_dir
        # on_select: fired when a model is clicked (updates info panels only —
        # does NOT switch the active model on a running server).
        self.on_select = on_select
        # on_load: fired by the Load button (actually switches the active model).
        self.on_load = on_load
        self._selected_model: Optional[str] = None

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Listbox with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(list_frame, bg="#2d2d2d", fg="#e0e0e0", font=("TkFixedFont", 9), selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="New", command=self._new_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Load", command=self._load_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Rename", command=self._rename_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self._delete_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Copy", command=self._copy_model).pack(side=tk.LEFT, padx=2)

        self.refresh()

    def refresh(self):
        """Reload model list from disk."""
        self.listbox.delete(0, tk.END)
        models = sorted([d.name for d in self.models_dir.iterdir() if d.is_dir()])
        for name in models:
            self.listbox.insert(tk.END, name)
        # Restore selection
        if self._selected_model and self._selected_model in models:
            idx = models.index(self._selected_model)
            self.listbox.selection_set(idx)

    def _on_select(self, event=None):
        selection = self.listbox.curselection()
        if selection:
            self._selected_model = self.listbox.get(selection[0])
            if self.on_select:
                self.on_select(self._selected_model)

    def get_selected(self) -> Optional[str]:
        return self._selected_model

    def select_model(self, name: str):
        """Programmatically select a model in the listbox (e.g. last used on startup)."""
        models = [self.listbox.get(i) for i in range(self.listbox.size())]
        if name in models:
            idx = models.index(name)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
            self._selected_model = name
            if self.on_select:
                self.on_select(name)

    def _new_model(self):
        name = tk.simpledialog.askstring("New Model", "Model name:", initialvalue="model_1")
        if name and name.strip():
            name = name.strip().replace(" ", "-")
            model_path = self.models_dir / name
            if model_path.exists():
                messagebox.showerror("Error", f"Model '{name}' already exists.")
                return
            model_path.mkdir()
            self._init_model(model_path, name)
            self.refresh()
            # Auto-select
            idx = self.listbox.size() - 1
            self.listbox.selection_set(idx)
            self._selected_model = name
            if self.on_select:
                self.on_select(name)

    def _init_model(self, model_path: Path, name: str):
        """Create config.json and an initial (randomly-initialized) model.pt checkpoint."""
        import json
        from datetime import datetime

        now = datetime.now().isoformat()
        encoder = {
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "max_seq_len": 2048,
        }
        training = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "max_grad_norm": 1.0,
        }
        cfg = {
            "name": name,
            "created": now,
            "modified": now,
            "games_trained": 0,
            "total_steps": 0,
            "encoder": encoder,
            "training": training,
        }
        with open(model_path / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # Create an initial randomly-initialized checkpoint so the model can be
        # loaded and used immediately (inference will be near-random until trained).
        try:
            import torch
            from arcbound.encoder.transformer import ModelConfig, create_model

            model = create_model(ModelConfig.from_dict(encoder))
            ckpt = {
                "model_config": model.config.to_dict(),
                "state_dict": model.state_dict(),
                "total_steps": 0,
                "saved_at": now,
            }
            torch.save(ckpt, model_path / "model.pt")
        except Exception as e:
            from loguru import logger
            logger.warning("Could not create initial checkpoint for {}: {}", name, e)

    def _load_model(self):
        if not self._selected_model:
            messagebox.showinfo("Info", "Select a model to load.")
            return
        # The Load button is the only way to actually switch the active model
        # on a running server. Clicking a model in the list only updates the
        # info panels (via on_select).
        if self.on_load:
            self.on_load(self._selected_model)
        elif self.on_select:
            self.on_select(self._selected_model)

    def _rename_model(self):
        old_name = self._selected_model
        if not old_name:
            messagebox.showinfo("Info", "Select a model to rename.")
            return
        new_name = tk.simpledialog.askstring("Rename Model", "New name:", initialvalue=old_name)
        if new_name and new_name.strip() and new_name.strip() != old_name:
            new_name = new_name.strip().replace(" ", "-")
            old_path = self.models_dir / old_name
            new_path = self.models_dir / new_name
            if new_path.exists():
                messagebox.showerror("Error", f"Model '{new_name}' already exists.")
                return
            old_path.rename(new_path)
            self._selected_model = new_name
            self.refresh()

    def _delete_model(self):
        name = self._selected_model
        if not name:
            messagebox.showinfo("Info", "Select a model to delete.")
            return
        confirm = messagebox.askyesno("Confirm Delete", f"Delete model '{name}'? This cannot be undone.")
        if confirm:
            import shutil
            model_path = self.models_dir / name
            shutil.rmtree(model_path)
            self._selected_model = None
            self.refresh()

    def _copy_model(self):
        src_name = self._selected_model
        if not src_name:
            messagebox.showinfo("Info", "Select a model to copy.")
            return
        dst_name = tk.simpledialog.askstring("Copy Model", "New model name:", initialvalue=f"{src_name}-copy")
        if dst_name and dst_name.strip():
            dst_name = dst_name.strip().replace(" ", "-")
            import shutil
            src = self.models_dir / src_name
            dst = self.models_dir / dst_name
            if dst.exists():
                messagebox.showerror("Error", f"Model '{dst_name}' already exists.")
                return
            shutil.copytree(src, dst)
            self.refresh()
