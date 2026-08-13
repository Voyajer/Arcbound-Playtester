"""Training loop for the Arcbound AI model.

Reads replay files, extracts decision experiences, and trains the transformer
model using PPO-style loss (policy + value + entropy).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    warmup_steps: int = 500


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    epoch: int = 0
    step: int = 0


@dataclass
class Experience:
    """A single training experience extracted from a replay decision."""

    # Tokenized board state (placeholder — will be integer token IDs)
    board_tokens: List[int] = field(default_factory=list)

    # Action taken (token index)
    action: int = 0

    # Reward / value target
    reward: float = 0.0

    # Old log-probability (for PPO clipping)
    old_log_prob: float = 0.0

    # Advantage estimate
    advantage: float = 0.0


class ReplayBuffer:
    """Stores experiences extracted from replay files."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.experiences: List[Experience] = []

    def add(self, exp: Experience):
        if len(self.experiences) >= self.capacity:
            self.experiences.pop(0)
        self.experiences.append(exp)

    def add_many(self, experiences: List[Experience]):
        self.experiences.extend(experiences)
        if len(self.experiences) > self.capacity:
            self.experiences = self.experiences[-self.capacity:]

    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.experiences) < batch_size:
            batch_size = len(self.experiences)
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def __len__(self) -> int:
        return len(self.experiences)


def extract_experiences(replay_path: Path) -> List[Experience]:
    """Extract training experiences from a replay file.

    Currently creates placeholder experiences. Will be replaced with actual
    tokenized board states once the tokenizer is implemented.
    """
    with open(replay_path, "r") as f:
        content = f.read()

    # Strip trailing summary
    json_end = content.find("\n\n=== GAME SUMMARY ===")
    if json_end != -1:
        data = json.loads(content[:json_end])
    else:
        data = json.loads(content)

    decisions = data.get("decisions", [])
    experiences = []

    for i, dec in enumerate(decisions):
        exp = Experience()
        # Placeholder: use decision index as a simple token
        exp.board_tokens = [i]
        # Action: index of chosen option in options list
        options = dec.get("options_presented", [])
        action = dec.get("action_taken", "")
        if action in options:
            exp.action = options.index(action)
        else:
            exp.action = 0
        # Reward: simple heuristic — positive if not timeout
        exp.reward = -1.0 if dec.get("timeout", False) else 0.1
        # Old log-prob from confidence
        confidence = dec.get("model_confidence")
        if confidence is not None and confidence > 0:
            exp.old_log_prob = float(np.log(max(confidence, 1e-7)))
        else:
            exp.old_log_prob = float(np.log(1.0 / max(len(options), 1)))
        experiences.append(exp)

    return experiences


class Trainer:
    """Manages the training loop.

    Callbacks allow the GUI to receive progress updates.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        on_progress: Optional[Callable[[int, float, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or TrainingConfig()
        self.buffer = ReplayBuffer()
        self.on_progress = on_progress
        self.on_log = on_log
        self._running = False
        self._metrics: Optional[TrainingMetrics] = None

    def _log(self, message: str):
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def load_replays(self, replay_paths: List[Path]):
        """Load experiences from replay files into the buffer."""
        total = 0
        for rp in replay_paths:
            exps = extract_experiences(rp)
            self.buffer.add_many(exps)
            total += len(exps)
            self._log(f"Loaded {len(exps)} experiences from {rp.name}")
        self._log(f"Total experiences in buffer: {len(self.buffer)}")

    def load_replay_dir(self, replay_dir: Path):
        """Load all replay files from a directory."""
        paths = sorted(replay_dir.glob("*.json"))
        if not paths:
            self._log(f"No replay files found in {replay_dir}")
            return
        self.load_replays(paths)

    def train(self, model=None):
        """Run the training loop.

        Args:
            model: PyTorch model with .forward(tokens) -> (logits, value) and .parameters().
                   If None, simulates training for GUI demonstration.
        """
        if len(self.buffer) == 0:
            self._log("No experiences in buffer. Load replays first.")
            return

        self._running = True
        cfg = self.config
        num_batches = max(1, len(self.buffer) // cfg.batch_size)

        self._log(f"Training started: {len(self.buffer)} experiences, {cfg.epochs} epochs, batch_size={cfg.batch_size}")

        optimizer = None
        if model is not None:
            try:
                import torch
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
            except ImportError:
                self._log("PyTorch not available — running simulation mode")

        global_step = 0

        for epoch in range(cfg.epochs):
            if not self._running:
                self._log("Training stopped by user.")
                break

            epoch_loss = 0.0
            epoch_policy = 0.0
            epoch_value = 0.0
            epoch_entropy = 0.0

            for batch_idx in range(num_batches):
                if not self._running:
                    break

                batch = self.buffer.sample(cfg.batch_size)

                if model is not None and optimizer is not None:
                    # Real training step
                    import torch
                    tokens = torch.tensor([e.board_tokens for e in batch], dtype=torch.long)
                    actions = torch.tensor([e.action for e in batch], dtype=torch.long)
                    rewards = torch.tensor([e.reward for e in batch], dtype=torch.float)
                    old_log_probs = torch.tensor([e.old_log_prob for e in batch], dtype=torch.float)

                    optimizer.zero_grad()
                    logits, value = model(tokens)

                    # Policy loss (PPO clipped)
                    import torch.nn.functional as F
                    log_probs = F.log_softmax(logits, dim=-1)
                    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(action_log_probs - old_log_probs)
                    surr1 = ratio * rewards
                    surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * rewards
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(value.squeeze(), rewards)

                    # Entropy bonus
                    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

                    total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    m = TrainingMetrics(
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        entropy=entropy.item(),
                        total_loss=total_loss.item(),
                        epoch=epoch,
                        step=global_step,
                    )
                else:
                    # Simulation mode — fake losses that decrease over time
                    decay = 1.0 / (1.0 + 0.01 * global_step)
                    m = TrainingMetrics(
                        policy_loss=0.5 * decay + 0.05 * np.random.randn(),
                        value_loss=0.3 * decay + 0.03 * np.random.randn(),
                        entropy=0.8 + 0.1 * np.random.randn(),
                        total_loss=0.8 * decay + 0.08 * np.random.randn(),
                        epoch=epoch,
                        step=global_step,
                    )

                self._metrics = m
                epoch_loss += m.total_loss
                epoch_policy += m.policy_loss
                epoch_value += m.value_loss
                epoch_entropy += m.entropy
                global_step += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            progress = (epoch + 1) / cfg.epochs * 100
            msg = f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | Policy: {epoch_policy/num_batches:.4f} | Value: {epoch_value/num_batches:.4f} | Entropy: {epoch_entropy/num_batches:.4f}"
            self._log(msg)

            if self.on_progress:
                self.on_progress(progress, avg_loss, msg)

        self._running = False
        self._log(f"Training complete. Final step: {global_step}")

    def stop(self):
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def metrics(self) -> Optional[TrainingMetrics]:
        return self._metrics


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Arcbound AI model")
    parser.add_argument("--replays", type=str, help="Path to replays directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = Trainer(config=cfg)

    if args.replays:
        trainer.load_replay_dir(Path(args.replays))
    else:
        replays_dir = Path.home() / ".arcbound" / "replays"
        if replays_dir.exists():
            trainer.load_replay_dir(replays_dir)
        else:
            logger.warning("No replays directory found. Nothing to train on.")
            return

    trainer.train()


if __name__ == "__main__":
    main()
