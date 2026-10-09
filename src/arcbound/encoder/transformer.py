"""Transformer encoder and decision model for Arcbound AI.

Architecture (per ARCHITECTURE.md):
  - Input: sequence of FEATURE_DIM feature vectors (one per token)
  - Embedding: Linear(FEATURE_DIM -> d_model) + LayerNorm + learned positional encoding
  - Encoder: N x TransformerEncoderLayer (pre-norm, GELU)
  - Policy head: Linear(d_model -> 1) applied to option-token positions -> logits
  - Value head: Linear(d_model -> 1) applied to CLS position -> scalar value

The model scores each option independently via its option-token embedding,
so it handles a variable number of options without a fixed action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcbound.encoder.tokenizer import CARD_EMBED_DIM, FEATURE_DIM, KW_EMBED_DIM, MAX_SEQ_LEN


@dataclass
class ModelConfig:
    """Architecture configuration for the Arcbound model."""

    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    # Card identity embedding. card_vocab_size is the number of card names
    # (incl. the <UNK> row at index 0); 0 means "no card embedding" (legacy
    # models). card_embed_dim is the learned vector size per card.
    card_embed_dim: int = CARD_EMBED_DIM
    card_vocab_size: int = 0
    # Keyword embedding. kw_vocab_size is the number of keyword strings
    # (incl. the <KW_UNK> row at index 0); 0 means "no keyword embedding"
    # (legacy models). kw_embed_dim is the learned vector size per keyword.
    kw_embed_dim: int = KW_EMBED_DIM
    kw_vocab_size: int = 0

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(
            d_model=int(d.get("d_model", 512)),
            nhead=int(d.get("nhead", 8)),
            num_layers=int(d.get("num_layers", 6)),
            dim_feedforward=int(d.get("dim_feedforward", 2048)),
            dropout=float(d.get("dropout", 0.1)),
            max_seq_len=int(d.get("max_seq_len", 2048)),
            card_embed_dim=int(d.get("card_embed_dim", CARD_EMBED_DIM)),
            card_vocab_size=int(d.get("card_vocab_size", 0)),
            kw_embed_dim=int(d.get("kw_embed_dim", KW_EMBED_DIM)),
            kw_vocab_size=int(d.get("kw_vocab_size", 0)),
        )

    def to_dict(self) -> Dict:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "card_embed_dim": self.card_embed_dim,
            "card_vocab_size": self.card_vocab_size,
            "kw_embed_dim": self.kw_embed_dim,
            "kw_vocab_size": self.kw_vocab_size,
        }


class ArcboundModel(nn.Module):
    """Transformer decision model with policy and value heads.

    Forward pass:
        features: (B, L, FEATURE_DIM) float tensor
        option_mask: (B, L) bool tensor — True at option-token positions

    Returns:
        option_logits: (B, num_options) — score per option (variable length handled
            by gathering option positions; for batching, options are right-padded
            and masked)
        value: (B, 1) — board value estimate from CLS token
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        # Card identity embedding (learned per-card vector, looked up by the
        # card name index). Concatenated onto the hand-crafted features before
        # the input projection.
        self.card_embedding = None
        if cfg.card_vocab_size > 0:
            self.card_embedding = nn.Embedding(
                cfg.card_vocab_size, cfg.card_embed_dim, padding_idx=0
            )

        # Keyword embedding (learned per-keyword vector, looked up by the
        # keyword index). Concatenated onto the hand-crafted features before
        # the input projection, alongside the card embedding.
        self.keyword_embedding = None
        if cfg.kw_vocab_size > 0:
            self.keyword_embedding = nn.Embedding(
                cfg.kw_vocab_size, cfg.kw_embed_dim, padding_idx=0
            )

        # Input projection: raw features (+ card + keyword embeddings) -> d_model
        input_dim = FEATURE_DIM
        if cfg.card_vocab_size > 0:
            input_dim += cfg.card_embed_dim
        if cfg.kw_vocab_size > 0:
            input_dim += cfg.kw_embed_dim
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        # Transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is incompatible
        # with pre-norm (norm_first=True) layers; set explicitly to avoid the
        # UserWarning (PyTorch disables it anyway).
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )

        # Heads
        self.policy_head = nn.Linear(cfg.d_model, 1)
        self.value_head = nn.Linear(cfg.d_model, 1)
        # Option-conditioned Q-head: scores each option token as the expected
        # outcome of taking that option from the current board state. Trained
        # on (state, chosen_option, outcome) so the value signal can influence
        # which option is picked at inference time.
        self.q_head = nn.Linear(cfg.d_model, 1)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        features: torch.Tensor,
        option_mask: Optional[torch.Tensor] = None,
        card_indices: Optional[torch.Tensor] = None,
        keyword_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass.

        Args:
            features: (B, L, FEATURE_DIM) input feature sequence.
            option_mask: (B, L) bool tensor marking option-token positions.
                If None, all positions are treated as options.
            card_indices: (B, L) long tensor of card embedding indices per
                position (0 for non-card tokens). Ignored when the model has
                no card embedding.
            keyword_indices: (B, L) long tensor of keyword embedding indices
                per position (0 for tokens without a keyword). Ignored when
                the model has no keyword embedding.

        Returns:
            option_logits: (B, max_options) — policy logits per option (padded,
                use option_counts to know valid length per batch element).
            value: (B, 1) — board value estimate from CLS token.
            q_values: (B, max_options) — option-conditioned Q estimate per
                option (same padding as option_logits).
        """
        B, L, _ = features.shape

        # Project and add positional encoding
        parts = [features]
        if self.card_embedding is not None:
            if card_indices is None:
                card_indices = torch.zeros(B, L, dtype=torch.long, device=features.device)
            else:
                # Clamp to valid range (vocabulary may have grown/shrunk).
                card_indices = card_indices.clamp(0, self.config.card_vocab_size - 1)
            parts.append(self.card_embedding(card_indices))  # (B, L, card_embed_dim)
        if self.keyword_embedding is not None:
            if keyword_indices is None:
                keyword_indices = torch.zeros(B, L, dtype=torch.long, device=features.device)
            else:
                keyword_indices = keyword_indices.clamp(0, self.config.kw_vocab_size - 1)
            parts.append(self.keyword_embedding(keyword_indices))  # (B, L, kw_embed_dim)
        x = self.input_proj(torch.cat(parts, dim=-1))
        x = self.input_norm(x)
        x = x + self.pos_encoding[:, :L, :]
        x = self.dropout(x)

        # Padding mask for transformer (all positions valid here; sequence is dense)
        x = self.encoder(x)

        # Value from CLS token (position 0)
        value = self.value_head(x[:, 0, :]).unsqueeze(-1)  # (B, 1)

        # Policy: score each option token
        if option_mask is None:
            option_mask = torch.ones(B, L, dtype=torch.bool, device=features.device)

        option_scores = self.policy_head(x).squeeze(-1)  # (B, L)
        # Mask non-option positions to -inf for softmax, 0 for gathering
        option_scores = option_scores.masked_fill(~option_mask, float("-inf"))

        # Gather option scores in order (right-padded to max options in batch)
        max_options = option_mask.sum(dim=1).max().item()
        if max_options == 0:
            # No options — return empty logits
            return (
                torch.zeros(B, 0, device=features.device),
                value,
                torch.zeros(B, 0, device=features.device),
            )

        # Build index tensor: for each batch element, positions of its options
        # (padded with 0 for elements with fewer options)
        indices = torch.zeros(B, max_options, dtype=torch.long, device=features.device)
        for b in range(B):
            pos = option_mask[b].nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                indices[b, :len(pos)] = pos

        option_logits = torch.gather(option_scores, 1, indices)  # (B, max_options)
        # Replace -inf (from padding positions) with a large negative number
        option_logits = option_logits.masked_fill(torch.isinf(option_logits), -1e4)

        # Option-conditioned Q values (same gather/mask as policy logits)
        q_scores = self.q_head(x).squeeze(-1)  # (B, L)
        q_scores = q_scores.masked_fill(~option_mask, float("-inf"))
        q_values = torch.gather(q_scores, 1, indices)  # (B, max_options)
        q_values = q_values.masked_fill(torch.isinf(q_values), -1e4)

        return option_logits, value, q_values

    def get_option_counts(self, option_mask: torch.Tensor) -> torch.Tensor:
        """Number of valid options per batch element: (B,)"""
        return option_mask.sum(dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Optional[ModelConfig] = None) -> ArcboundModel:
    """Factory for creating a new Arcbound model."""
    return ArcboundModel(config)


def load_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device = torch.device("cpu"),
) -> ArcboundModel:
    """Load a model from a checkpoint file.

    The checkpoint is a dict with keys:
        - model_config: ModelConfig dict
        - state_dict: model state dict
        - (optional) optimizer_state, epoch, etc.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        config = ModelConfig.from_dict(checkpoint["model_config"])
        state_dict = checkpoint["state_dict"]
    else:
        # Assume raw state dict — use default config
        config = ModelConfig()
        state_dict = checkpoint

    model = ArcboundModel(config)
    # Load leniently so checkpoints saved before a head was added (e.g. the
    # option-conditioned q_head) still load; any newly added head keeps its
    # fresh initialization until retrained.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        import logging
        logging.getLogger(__name__).info(
            "Checkpoint %s loaded with strict=False (missing=%s, unexpected=%s)",
            ckpt_path, missing, unexpected,
        )
    model.to(device)
    model.eval()
    return model
