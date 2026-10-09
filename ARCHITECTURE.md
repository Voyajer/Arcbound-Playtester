# Arcbound Playtester — Transformer-Based MTG AI Server

## Overview

This document plans a transformer encoder-based AI server that learns to play Magic: The Gathering. The server receives complete board state JSON from Card Forge via HTTP REST API, encodes the state through a transformer model, and returns decisions matching the Forge external AI contract.

**Key Design Goals:**
- Transformer encoder architecture for board state understanding
- Self-play training loop for unsupervised learning
- Per-perspective zone visibility: the server receives the full board (both hands and both decks) but the tokenizer masks each player's hidden zones (hand, decklist) to that player's point of view, so the AI only ever sees its own hand and deck
- Support for all decision types defined by Forge's [`PlayerController`](../forge/forge-game/src/main/java/forge/game/player/PlayerController.java)
- Fast inference (< 2 seconds per decision) to maintain gameplay flow

## System Architecture

```mermaid
flowchart TB
    subgraph Forge["Forge JVM"]
        PCE[PlayerControllerExternalAi]
        HC[ExternalAiHttpClient]
    end

    subgraph PythonServer["Python AI Server localhost:8090"]
        API[FastAPI Router]
        Preproc[State Preprocessor]
        Tokenizer[MTG Tokenizer]
        Transformer[Transformer Encoder]
        PolicyHead[Policy Head]
        ValueHead[Value Head]
        Decoder[Decision Decoder]
    end

    PCE -->|POST /decision| HC
    HC -->|JSON Request| API
    API --> Preproc
    Preproc --> Tokenizer
    Tokenizer --> Transformer
    Transformer --> PolicyHead
    Transformer --> ValueHead
    PolicyHead --> Decoder
    Decoder --> API
    API -->|JSON Response| HC
```

## Project Structure

```
Arcbound-Playtester/
├── pyproject.toml
├── README.md
├── config/
│   ├── default.yaml              # Default server and model config
│   ├── training.yaml             # Training-specific config
│   └── inference.yaml            # Inference-only config
├── src/
│   ├── arcbound/
│   │   ├── __init__.py
│   │   ├── server.py             # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── decision.py       # POST /decision endpoint
│   │   │   ├── game.py           # POST /game/start, /game/end
│   │   │   └── health.py         # GET /health
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── board_state.py    # Pydantic models matching Java BoardState
│   │   │   ├── decision.py       # Pydantic models matching AiDecisionRequest/Response
│   │   │   └── card.py           # Card representation
│   │   ├── encoder/
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py    # Transformer encoder model
│   │   │   ├── tokenization.py   # MTG-specific tokenization
│   │   │   └── features.py       # Feature extraction from board state
│   │   ├── decision/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py         # Decision-making engine
│   │   │   ├── policy.py         # Policy network head
│   │   │   └── value.py          # Value network head
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py        # Training loop
│   │   │   ├── self_play.py      # Self-play data generation
│   │   │   ├── replay_buffer.py  # Experience replay buffer
│   │   │   └── reward.py         # Reward computation
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py        # Structured logging
│   │       └── config.py         # Config loading
├── scripts/
│   ├── start_server.sh           # Start inference server
│   ├── start_training.sh         # Start training loop
│   └── generate_data.sh          # Generate self-play data
├── checkpoints/                  # Model checkpoints (gitignored)
├── data/                         # Training data (gitignored)
└── tests/
    ├── conftest.py
    ├── test_server.py
    ├── test_models.py
    ├── test_encoder.py
    ├── test_decision.py
    └── test_training.py
```

## API Contract

The Python server must match the Java [`ExternalAiHttpClient`](../forge/forge-external-ai/src/main/java/forge/externalai/http/ExternalAiHttpClient.java) endpoints exactly.

### Endpoints

| Method | Path | Purpose | Java Caller |
|--------|------|---------|-------------|
| POST | `/decision` | Main decision endpoint | [`ExternalAiHttpClient.requestDecision()`](../forge/forge-external-ai/src/main/java/forge/externalai/http/ExternalAiHttpClient.java:75) |
| POST | `/game/start` | Game start notification | [`ExternalAiHttpClient.notifyGameStart()`](../forge/forge-external-ai/src/main/java/forge/externalai/http/ExternalAiHttpClient.java:141) |
| POST | `/game/end` | Game end notification | [`ExternalAiHttpClient.notifyGameEnd()`](../forge/forge-external-ai/src/main/java/forge/externalai/http/ExternalAiHttpClient.java:165) |
| GET | `/health` | Health check | [`ExternalAiHttpClient.isServerAvailable()`](../forge/forge-external-ai/src/main/java/forge/externalai/http/ExternalAiHttpClient.java:191) |

### Request/Response Schema

The Python Pydantic models must match the Java DTOs exactly. Field names use snake_case in Python but serialize to the Java-expected names.

#### POST /decision Request

Matches [`AiDecisionRequest`](../forge/forge-external-ai/src/main/java/forge/externalai/AiDecisionRequest.java):

```python
class DecisionContext(BaseModel):
    type: str                          # e.g. "boolean", "card", "cards", "color"
    description: str                   # Human-readable description
    options: list[Option]              # Available options
    constraints: Constraints | None    # Choice constraints
    prompt: str | None                 # Prompt text

class Option(BaseModel):
    id: str                            # Option identifier
    type: str                          # Option type
    label: str                         # Display label
    details: dict[str, Any] | None     # Additional details

class Constraints(BaseModel):
    min_choices: int = 0
    max_choices: int = 1
    is_optional: bool = False
    allow_none: bool = False

class AiDecisionRequest(BaseModel):
    game_id: str
    board_state: dict[str, Any]        # Full BoardState as nested dict
    decision_context: DecisionContext
```

#### POST /decision Response

Matches [`AiDecisionResponse`](../forge/forge-external-ai/src/main/java/forge/externalai/AiDecisionResponse.java):

```python
class AiDecisionResponse(BaseModel):
    decision_id: str
    decision_type: str                 # Matches decision context type
    value: dict[str, Any]              # Decision value (type-dependent)
    reasoning: str | None              # Human-readable reasoning
    selected_option_id: str | None     # Single option selection
    selected_option_ids: list[str] | None  # Multiple option selection
```

#### BoardState Schema

Matches [`BoardState`](../forge/forge-external-ai/src/main/java/forge/externalai/boardstate/BoardState.java):

```python
class CardInfo(BaseModel):
    id: str
    name: str
    oracle_name: str | None
    mana_cost: str | None
    converted_mana_cost: int | None
    colors: list[str] | None
    color_identity: list[str] | None
    types: list[str] | None
    supertypes: list[str] | None
    subtypes: list[str] | None
    text: str | None
    power: int | None
    toughness: int | None
    loyalty: int | None
    damage: int | None
    counters: dict[str, int] | None
    tapped: bool | None
    attacking: bool | None
    blocking: bool | None
    flipped: bool | None
    face_down: bool | None
    owner: str | None
    controller: str | None
    zone: str | None
    abilities: list[str] | None
    keywords: list[str] | None
    revealed_to: list[str] | None

class PlayerState(BaseModel):
    name: str
    life: float
    starting_life: float = 20.0
    poison_counters: int = 0
    mana_pool: dict[str, int] | None
    commanders: list[CardInfo] | None
    # commander_damage maps each commander's name to the damage THAT commander
    # has dealt to THIS player (the 21-damage loss rule is per-commander).
    commander_damage: dict[str, int] | None
    # commander_tax maps each commander's name to how many times it has been
    # cast (each recast adds one mana of tax).
    commander_tax: dict[str, int] | None
    battlefield: list[CardInfo]
    hand: list[CardInfo]
    graveyard: list[CardInfo]
    exile: list[CardInfo]
    command_zone: list[CardInfo] | None
    library_size: int
    sideboard_size: int | None
    mulligan_count: int = 0
    hand_hidden: bool = False
    is_focal: bool = False
    opponent_id: str | None = None
    # Full decklist (main + sideboard). Forge sends it for EVERY player; the
    # tokenizer only emits decklist tokens for the perspective player, so the
    # AI still only ever sees its own deck.
    decklist: list[CardInfo] = []

class BoardState(BaseModel):
    game_id: str
    active_player: str
    turn: int
    phase: str
    focal_player: str
    players: list[PlayerState]
    stack: list[CardInfo] | None
```

## Transformer Encoder Architecture

### Overall Design

The transformer encoder processes the board state as a sequence of tokenized elements. Each element (card, player, game state) is converted to a token embedding, passed through transformer encoder layers, and the resulting representations feed into policy and value heads.

```mermaid
flowchart LR
    subgraph Input["Input Processing"]
        BS[BoardState JSON]
        PP[Preprocessor]
        TK[Tokenizer]
        PE[Positional Encoding]
    end

    subgraph Encoder["Transformer Encoder"]
        L1[Encoder Layer 1]
        L2[Encoder Layer 2]
        LN[Encoder Layer N]
    end

    subgraph Output["Output Heads"]
        CLS[CLS Token]
        PH[Policy Head]
        VH[Value Head]
    end

    BS --> PP --> TK --> PE
    PE --> L1 --> L2 --> LN
    LN --> CLS
    CLS --> PH
    CLS --> VH
    PH --> Decision
    VH --> Value
```

### Tokenization Strategy

Magic: The Gathering board states are heterogeneous and hierarchical. The tokenizer flattens the board state into a linear sequence of tokens while preserving structure through special tokens.

**Perspective-based hidden-zone masking:** Forge sends the full board (both
players' hands and both decklists), but the tokenizer builds tokens from a
single *perspective* (default: the board's `focal_player`). Only the
perspective player's hand and decklist are emitted (plus any cards explicitly
`revealed_to` that player), and the focal/relative features are computed from
that player. This guarantees the AI never sees the opponent's hidden cards, and
it is what lets the server compute a per-player self-assessment (e.g. "how the
AI thinks the human is doing") by re-tokenizing the same board from the
human's perspective.

**Token Categories:**

| Category | Tokens | Description |
|----------|--------|-------------|
| Structure | `[GAME]`, `[PLAYER]`, `[END_PLAYER]`, `[ZONE]`, `[CARD]`, `[END_CARD]`, `[STACK]`, `[END_GAME]` | Hierarchical delimiters |
| Card Identity | `[CARD:Name]` | Each unique card name gets a token |
| Mana | `[MANA:W]`, `[MANA:U]`, `[MANA:B]`, `[MANA:R]`, `[MANA:G]`, `[MANA:C]`, `[MANA:N]` | Mana symbols |
| Numbers | `[NUM:0]` through `[NUM:99+]` | Binned numeric values |
| Zones | `[ZONE:Hand]`, `[ZONE:Battlefield]`, `[ZONE:Graveyard]`, `[ZONE:Exile]`, `[ZONE:Library]`, `[ZONE:Command]` | Zone identifiers |
| Properties | `[TAPPED]`, `[UNTAPPED]`, `[ATTACKING]`, `[BLOCKING]`, `[FLYING]`, `[TRAMPLE]`, ... | Card states and keywords |
| Colors | `[COLOR:W]`, `[COLOR:U]`, `[COLOR:B]`, `[COLOR:R]`, `[COLOR:G]`, `[COLOR:C]` | Color identifiers |
| Types | `[TYPE:Creature]`, `[TYPE:Instant]`, `[TYPE:Sorcery]`, `[TYPE:Land]`, ... | Card types |
| Subtypes | `[SUBTYPE:Wizard]`, `[SUBTYPE:Mountain]`, ... | Card subtypes |
| Decision | `[DECISION]`, `[DECISION_TYPE:type]`, `[OPTION:id]` | Decision context |

**Tokenized Board State Example:**

For a simplified board state:
```
[GAME] [TURN:3] [PHASE:Main] [ACTIVE:Alice]
[PLAYER] [NAME:Alice] [FOCAL] [LIFE:20] [POISON:0]
  [ZONE:Hand]
    [CARD] [CARD:Lightning Bolt] [COLOR:R] [TYPE:Instant] [CMC:1] [END_CARD]
    [CARD] [CARD:Mountain] [TYPE:Land] [SUBTYPE:Mountain] [END_CARD]
  [ZONE:Battlefield]
    [CARD] [CARD:Mountain] [UNTAPPED] [TYPE:Land] [END_CARD]
    [CARD] [CARD:Giant Spider] [UNTAPPED] [TYPE:Creature] [SUBTYPE:Spider] [PT:2/1] [FLYING] [END_CARD]
  [ZONE:Graveyard]
  [LIBRARY:23]
[END_PLAYER]
[PLAYER] [NAME:Bob] [LIFE:18] [POISON:0]
  [ZONE:Hand] [HAND_HIDDEN]
  [ZONE:Battlefield]
    [CARD] [CARD:Island] [TAPPED] [TYPE:Land] [END_CARD]
  [LIBRARY:20]
[END_PLAYER]
[STACK]
[END_GAME]
[DECISION] [DECISION_TYPE:card] [OPTION:opt0] [OPTION:opt1]
```

**Vocabulary Size Estimate:**

| Category | Count | Notes |
|----------|-------|-------|
| Unique card names (oracle) | ~30,000 | One token per card name, NOT per printing |
| Structure tokens | ~20 | `[GAME]`, `[PLAYER]`, `[CARD]`, etc. |
| Mana tokens | 36 | WUBRG (5) + Colorless (1) + Generic (1) + Variable X (1) + Hybrid color/color (10) + 2brid (5) + Colorless-hybrid (5) + Phyrexian (6) + Snow (1) + Legendary (1) |
| Color tokens | 6 | `[COLOR:W]`, `[COLOR:U]`, `[COLOR:B]`, `[COLOR:R]`, `[COLOR:G]`, `[COLOR:C]` (colorless) |
| Number bins | ~100 | `[NUM:0]` through `[NUM:99+]` |
| Keywords/Abilities | ~500 | `[FLYING]`, `[TRAMPLE]`, etc. |
| Type/Subtype tokens | ~500 | `[TYPE:Creature]`, `[SUBTYPE:Wizard]`, etc. |
| Zone tokens | ~15 | `[ZONE:Hand]`, `[ZONE:Battlefield]`, etc. |
| Decision tokens | ~50 | Decision types and option markers |
| **Total** | **~31,200** | Manageable vocabulary size |

**Important: One Token Per Card Name, Not Per Printing.** "Lightning Bolt" has been printed in 50+ sets but gets a single token ID. All occurrences share the same token so the model learns from every game where any printing appears. Uses the `oracle_name` field from [`CardInfo`](../forge/forge-external-ai/src/main/java/forge/externalai/boardstate/BoardState.java) as the canonical identifier, not the printed set or collector number.

### Feature Layout (per token)

The implemented tokenizer ([`tokenizer.py`](src/arcbound/encoder/tokenizer.py)) produces a
fixed **122-dimensional** feature vector per token (`FEATURE_DIM = 122`), plus a learned
card-name embedding (32-dim) and keyword embedding (16-dim) concatenated on. The
authoritative slot map lives in the module docstring; the highlights:

| Slots | Meaning |
|-------|---------|
| `[0]` | token type (CLS / SEP / PLAYER / CARD / DECISION / OPTION / DECK) |
| `[1..7]` | player-level: focal, active, life, library, hand size, battlefield size, poison |
| `[8..14]` | card-level: CMC, power, toughness, damage, tapped, attacking, blocking |
| `[15..19]` | zone one-hot (battlefield / hand / graveyard / exile / command) |
| `[20..29]` | decision + option context (type, counts, mulligan info) |
| `[30..31]` | decklist membership + copy count |
| `[32..36]` | card color identity (W/U/B/R/G) |
| `[37]` | loyalty |
| `[38..40]` | counters (+1/+1, -1/-1, other) |
| `[41..42]` | controller/owner is focal |
| `[43..48]` | player mana pool (total + per color) |
| `[49..51]` | player-level commander-damage aggregates (max / total / from focal) |
| `[52]` | commander tax |
| `[53..57]` | option detail content |
| `[58]` | **commander card:** damage this commander dealt to the focal player |
| `[59..66]` | **commander card:** damage this commander dealt to each player (`board.players[i]`, i=0..7) |
| `[67..70]` | turn / spell-cast tracking (total, own, other, turn number) |
| `[71..76]` | **card:** type one-hot (creature / land / enchantment / instant / sorcery / artifact) |
| `[77..78]` | **card:** is-token / is-legendary flags |
| `[79..93]` | **card:** keyword multi-hot (15 decision-relevant keywords) |
| `[94..95]` | **player:** graveyard size, exile size |
| `[96..100]` | **player:** land count, creature count, total power, total toughness, untapped lands |
| `[101..103]` | **player:** hand mana curve (CMC 0–1 / 2–3 / 4+) |
| `[104..110]` | **board (CLS):** phase one-hot (beginning / main1 / combat / main2 / end / cleanup / other) |
| `[111]` | **board (CLS):** stack depth |
| `[112..116]` | **board (CLS):** combat aggregates (attackers, blockers, unblocked, attacking power, lethal flag) |
| `[117]` | **board (CLS):** average battlefield CMC |
| `[118..120]` | **board (CLS):** relative features (life diff, tempo diff, card advantage) |
| `[121]` | **decision:** prompt description length |

**Commander damage is tracked per commander, not aggregated.** The 21-damage loss rule
is per-commander, so the model must know *which* commander is the threat. Each commander
card token therefore carries the full per-commander × per-player damage matrix in
`[59..66]` (one bounded slot per player, up to 8), with `[58]` as the focal player's
column. The player-level aggregates in `[49..51]` are kept as supplementary summary
signals.

> **Note:** `FEATURE_DIM` is part of the model's input projection, so changing it
> (as adding the commander-damage matrix did) invalidates existing checkpoints —
> retrain to pick up a new feature layout.

### Model Architecture

**Base Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden dimension | 512 | Balance between capacity and inference speed |
| Feed-forward dimension | 2048 | 4x hidden dimension (standard) |
| Attention heads | 8 | Standard for 512-dim models |
| Encoder layers | 6 | Shallow for fast inference |
| Maximum sequence length | 4096 | Large boards with many cards |
| Dropout | 0.1 | Regularization |
| Activation | GeLU | Standard transformer activation |
| Layer norm | Pre-norm | Better training stability |

**Embedding Strategy:**

```python
class MTGTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len):
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        out = self.token_embedding(x) + self.position_embedding(pos)
        return self.encoder(out)
```

### Policy Head

The policy head converts transformer outputs into a probability distribution over valid actions. Different decision types require different output heads.

**Architecture:**

```
Transformer Output (CLS token)
    → Linear(512 → 512) → GeLU → LayerNorm
    → Decision-Type Router
        → Boolean Head: Linear(512 → 2) → Softmax
        → Card Head: Linear(512 → num_cards) → Softmax
        → Cards Head: Iterative single-card selection
        → Color Head: Linear(512 → 6) → Softmax
        → Number Head: Linear(512 → num_options) → Softmax
        → Player Head: Linear(512 → num_players) → Softmax
        → Order Head: Iterative ordering via attention
```

**Multi-Head Design:**

```python
class PolicyHead(nn.Module):
    def __init__(self, d_model, max_cards, max_players, max_options):
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        # Heads are selected based on decision type
        self.head_boolean = nn.Linear(d_model, 2)
        self.head_card = nn.Linear(d_model, max_cards)
        self.head_color = nn.Linear(d_model, 6)
        self.head_number = nn.Linear(d_model, max_options)
        self.head_player = nn.Linear(d_model, max_players)
```

### Value Head

The value head estimates the expected game outcome from the current state, used for training via reinforcement learning.

```python
class ValueHead(nn.Module):
    def __init__(self, d_model):
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        return torch.tanh(self.network(x))  # Output in [-1, 1]
```

### Decision Decoder

Converts model outputs back to the [`AiDecisionResponse`](../forge/forge-external-ai/src/main/java/forge/externalai/AiDecisionResponse.java) format.

```python
class DecisionDecoder:
    def decode(self, decision_type, model_output, decision_context, board_state) -> AiDecisionResponse:
        if decision_type == "boolean":
            return self._decode_boolean(model_output)
        elif decision_type == "card":
            return self._decode_card(model_output, decision_context, board_state)
        elif decision_type == "cards":
            return self._decode_cards(model_output, decision_context, board_state)
        # ... other decision types
```

## Training Pipeline

### Training Approach

**Primary: Self-Play with Reinforcement Learning**

The AI learns through self-play games, using a combination of:
1. **Policy Gradient** — Improve policy based on game outcomes
2. **Value Target** — Reduce TD error between predicted and actual outcomes
3. **Entropy Regularization** — Encourage exploration

**Optional: Supervised Pre-Training**

If human game data is available (Forge game logs), pre-train on human decisions before self-play fine-tuning.

**Player-type filtering**

Each match replay records every player's type (`human`, `bot`, or `ai`) plus a
composition label (e.g. `human_vs_ai`) under the `match` key. Both the policy and
value extractors accept an optional `player_types` filter (CLI: `--player-types`,
GUI: the Training tab's "Learn From" selector) so you can restrict which players'
moves contribute experiences — e.g. learn only from the external AI's decisions
(`ai`), or only from human/bot play (`human,bot`). When omitted, every recorded
move is used, so legacy replays without player types are never dropped.

### Training Loop

```mermaid
flowchart TB
    subgraph SelfPlay["Self-Play Loop"]
        Init[Initialize Game State]
        Loop[Decision Loop]
        Act[Run Inference]
        Store[Store Experience]
        Next[Next Decision?]
        End[Game Ends]
        Result[Compute Outcome]
    end

    subgraph Training["Training Loop"]
        Sample[Sample Batch from Buffer]
        Loss[Compute Losses]
        PolicyLoss[Policy Loss cross-entropy]
        ValueLoss[Value Loss MSE]
        Update[Update Weights]
        Checkpoint[Save Checkpoint]
    end

    Init --> Loop --> Act --> Store --> Next
    Next --|more decisions| --> Act
    Next --|game over| --> End --> Result --> Buffer
    Buffer --> Sample --> Loss
    Loss --> PolicyLoss
    Loss --> ValueLoss
    PolicyLoss --> Update
    ValueLoss --> Update
    Update --> Checkpoint
```

### Experience Storage

Each experience stores:
- Board state (tokenized)
- Decision context
- Action taken
- Reward (sparse: +1.0 win, -1.0 loss, 0.0 draw at game end)
- Value prediction at each step

### Reward Design

| Event | Reward | Rationale |
|-------|--------|-----------|
| Win game | +1.0 | Primary objective |
| Lose game | -1.0 | Primary objective |
| Draw game | 0.0 | Neutral outcome |
| Win combat | +0.01 | Shaping reward for intermediate progress |
| Lose commander | -0.1 | Commander death is significant setback |
| Opponent wins combat | -0.01 | Shaping reward |

### Replay Buffer

```python
@dataclass
class Experience:
    game_id: str
    step: int
    token_ids: torch.LongTensor       # Tokenized board state
    action_ids: torch.LongTensor      # Taken action
    reward: float                     # Reward at this step
    value_pred: float                 # Model's value prediction
    log_prob: float                   # Log probability of action
    entropy: float                    # Policy entropy
    done: bool                        # Terminal state?

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def add(self, exp: Experience):
        self.buffer.append(exp)

    def sample(self, batch_size=64) -> list[Experience]:
        return random.sample(self.buffer, batch_size)
```

### Loss Function

```python
def compute_loss(policy_log_prob, value_pred, reward, old_log_prob, old_entropy, done):
    # Policy loss (PPO-style clipped surrogate)
    ratio = torch.exp(policy_log_prob - old_log_prob)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
    policy_loss = -torch.min(surr1, surr2)

    # Value loss
    with torch.no_grad():
        target = reward + gamma * value_next * (1 - done)
    value_loss = F.mse_loss(value_pred, target)

    # Entropy bonus
    entropy_loss = -entropy_coeff * old_entropy

    return policy_loss + 0.5 * value_loss + entropy_loss
```

## Configuration

### Server Configuration (`config/default.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8090
  workers: 1
  timeout_ms: 5000

model:
  vocab_size: 30820
  d_model: 512
  nhead: 8
  num_layers: 6
  dim_feedforward: 2048
  max_seq_len: 4096
  dropout: 0.1

inference:
  temperature: 0.7
  top_p: 0.9
  checkpoint_path: "checkpoints/best_model.pt"
  device: "auto"  # auto-detect CUDA/MPS/CPU

logging:
  level: "INFO"
  log_requests: true
  log_decisions: true
```

### Training Configuration (`config/training.yaml`)

```yaml
training:
  learning_rate: 0.0003
  batch_size: 64
  epochs: 1000
  buffer_size: 100000
  update_frequency: 1000
  gamma: 0.99
  entropy_coeff: 0.01
  clip_epsilon: 0.2

self_play:
  games_per_batch: 100
  num_players: 4
  max_games: 100000

checkpoint:
  save_frequency: 1000
  keep_last: 10
  save_dir: "checkpoints/"
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.100.0 | REST API framework |
| uvicorn | >=0.23.0 | ASGI server |
| pydantic | >=2.0.0 | Data validation |
| torch | >=2.0.0 | Deep learning framework |
| numpy | >=1.24.0 | Numerical computing |
| pyyaml | >=6.0 | Config loading |
| loguru | >=0.7.0 | Structured logging |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0.0 | Testing framework |
| pytest-asyncio | >=0.21.0 | Async test support |
| httpx | >=0.24.0 | Async HTTP client for tests |
| black | >=23.0.0 | Code formatting |
| ruff | >=0.1.0 | Linting |
| mypy | >=1.0.0 | Type checking |
