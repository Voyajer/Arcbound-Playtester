"""Model manager — loads, saves, and tracks the active model checkpoint.

On startup the server calls :meth:`load`, which instantiates the transformer
from the last-used model's checkpoint (recorded by the GUI in
``<project>/config.json``) and logs the outcome explicitly so it is easy to
verify whether a model was actually instantiated.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from arcbound.encoder.tokenizer import set_card_vocabulary, set_keyword_vocabulary
from arcbound.encoder.transformer import ArcboundModel, ModelConfig, load_model_from_checkpoint
from arcbound.encoder.vocabulary import CardVocabulary, KeywordVocabulary

CHECKPOINT_CANDIDATES = ("model.pt", "best_model.pt", "checkpoint.pt")


class ModelManager:
    """Manages loading and tracking of the AI model checkpoint.

    On startup the server calls :meth:`load`, which looks for a checkpoint
    in the last-used model directory (recorded by the GUI in
    ``<project>/config.json``) and logs the outcome explicitly so it is
    easy to verify whether a model was actually instantiated.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model = None
        self.model_name: Optional[str] = None
        self.checkpoint_path: Optional[Path] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, model_name: Optional[str] = None) -> bool:
        """Attempt to load a model checkpoint.

        Args:
            model_name: Explicit model directory name. If None, the
                last-used model from config.json is used.

        Returns:
            True if a model was loaded, False otherwise (fallback mode).
        """
        if model_name:
            model_dir = self.models_dir / model_name
        else:
            model_dir = self._find_last_model_dir()

        if model_dir is None or not model_dir.exists():
            logger.warning(
                "No model directory found (models_dir={}) — server will use fallback decisions",
                self.models_dir,
            )
            return False

        for candidate in CHECKPOINT_CANDIDATES:
            ckpt = model_dir / candidate
            if ckpt.exists():
                if self._load_checkpoint(ckpt):
                    return True
                break

        logger.warning(
            "No checkpoint found in {} (looked for {}) — server will use fallback decisions",
            model_dir,
            ", ".join(CHECKPOINT_CANDIDATES),
        )
        return False

    def _find_last_model_dir(self) -> Optional[Path]:
        """Find the last-used model directory from config.json."""
        config_path = self.models_dir.parent / "config.json"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                last = cfg.get("last_model")
                if last:
                    return self.models_dir / last
        except Exception as e:
            logger.debug("Failed to read last_model from {}: {}", config_path, e)

        # Fall back to the only model directory if there is exactly one
        try:
            dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
            if len(dirs) == 1:
                return dirs[0]
        except Exception:
            pass
        return None

    def _load_checkpoint(self, ckpt_path: Path) -> bool:
        """Load a PyTorch checkpoint and instantiate the transformer model."""
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("PyTorch not available — cannot load model checkpoint {}", ckpt_path)
            return False

        try:
            model = load_model_from_checkpoint(str(ckpt_path))
            self.model = model
            self.model_name = ckpt_path.parent.name
            self.checkpoint_path = ckpt_path
            n_params = model.num_parameters()
            logger.info(
                "Model instantiated: '{}' from {} ({} parameters, d_model={}, layers={})",
                self.model_name,
                ckpt_path,
                f"{n_params:,}",
                model.config.d_model,
                model.config.num_layers,
            )

            # Load the card vocabulary (if present) so the tokenizer can map
            # card names to embedding indices during inference.
            vocab_path = ckpt_path.parent / "vocab.json"
            if vocab_path.exists():
                try:
                    vocab = CardVocabulary.load(vocab_path)
                    set_card_vocabulary(vocab)
                    logger.info(
                        "Card vocabulary loaded: {} unique cards from {}",
                        len(vocab) - 1,
                        vocab_path,
                    )
                except Exception as e:
                    logger.warning("Failed to load card vocabulary {}: {}", vocab_path, e)
            elif model.config.card_vocab_size > 0:
                logger.warning(
                    "Model '{}' expects a card vocabulary (size {}) but no vocab.json "
                    "was found next to the checkpoint — card identity embeddings will "
                    "be all-UNK during inference.",
                    self.model_name,
                    model.config.card_vocab_size,
                )

            # Load the keyword vocabulary (if present) so the tokenizer can map
            # keyword strings to embedding indices during inference.
            kw_vocab_path = ckpt_path.parent / "kw_vocab.json"
            if kw_vocab_path.exists():
                try:
                    kw_vocab = KeywordVocabulary.load(kw_vocab_path)
                    set_keyword_vocabulary(kw_vocab)
                    logger.info(
                        "Keyword vocabulary loaded: {} unique keywords from {}",
                        len(kw_vocab) - 1,
                        kw_vocab_path,
                    )
                except Exception as e:
                    logger.warning("Failed to load keyword vocabulary {}: {}", kw_vocab_path, e)
            elif model.config.kw_vocab_size > 0:
                logger.warning(
                    "Model '{}' expects a keyword vocabulary (size {}) but no "
                    "kw_vocab.json was found next to the checkpoint — keyword "
                    "embeddings will be all-UNK during inference.",
                    self.model_name,
                    model.config.kw_vocab_size,
                )

            return True
        except Exception as e:
            logger.error("Failed to load model checkpoint {}: {}", ckpt_path, e)
            self.model = None
            self.model_name = None
            self.checkpoint_path = None
            return False

    def save_checkpoint(
        self,
        model: ArcboundModel,
        model_name: str,
        extra: Optional[dict] = None,
    ) -> Path:
        """Save a model checkpoint to models/<model_name>/model.pt.

        Also writes the model config into the model's config.json so the
        architecture can be reconstructed on load.

        Args:
            model: The trained model to save.
            model_name: Model directory name (e.g. "Default").
            extra: Optional extra keys to include in the checkpoint
                (epoch, total_steps, games_trained, etc.).

        Returns:
            Path to the saved checkpoint file.
        """
        import torch

        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = model_dir / "model.pt"

        checkpoint = {
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
        }
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, ckpt_path)

        # Update config.json with the encoder architecture + training stats
        config_path = model_dir / "config.json"
        cfg = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        cfg["encoder"] = model.config.to_dict()
        if extra:
            for key in ("total_steps", "games_trained", "modified"):
                if key in extra:
                    cfg[key] = extra[key]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        logger.info("Checkpoint saved: {} ({} parameters)", ckpt_path, f"{model.num_parameters():,}")
        return ckpt_path

    def status(self) -> dict:
        """Return model status for health/monitoring endpoints."""
        return {
            "modelLoaded": self.is_loaded,
            "modelName": self.model_name,
            "checkpointPath": str(self.checkpoint_path) if self.checkpoint_path else None,
        }
