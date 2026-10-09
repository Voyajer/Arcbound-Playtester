# ⚠️Warning⚠️

This project was created partially as an experiment in seeing how capable 'vibe coding' is with a local model (Qwen-3.8q4xl) so consider it slop. It does work but no consideration beyond core functionality was considered.

# Arcbound MTG AI

Transformer-based AI server for Magic: The Gathering (Forge).

## Overview

This project implements a transformer encoder-based AI that learns to play Magic: The Gathering using [card-forge](https://github.com/magefree/mage) as the rules engine. The AI server communicates with Forge via HTTP REST API.

## Quick Start

The easiest way to use Arcbound is the bundled GUI, which manages the server,
models, training, and live game monitoring for you.

```bash
# One-shot: creates a virtual environment, installs, and launches the GUI
./launch-gui.sh

# ...or manually:
pip install -e ".[dev]"
arcbound-gui
```

The GUI starts the server on port 8080 (the default Forge expects) and loads your
last-used model automatically. You can also run the pieces directly:

```bash
# Run just the server (no GUI)
arcbound-server

# Train a model from recorded replays
arcbound-train --replays replays --model MyModel --mode both

# Run the test suite
pytest tests/ -v
```

## Connecting Forge to the AI

The AI plugs into Forge through the `forge-external-ai` module, which replaces
Forge's built-in computer player with a controller that asks the Arcbound server
over HTTP for every decision.

By default the Forge side looks for the server at `http://localhost:8080`, which
matches the server's default port, so a local setup works with no configuration.
To point Forge at a different server or tune its behavior, set environment
variables (or the bundled `external-ai.json` resource):

- `EXTERNAL_AI_URL` — the server's base URL (default `http://localhost:8080`).
- `EXTERNAL_AI_TIMEOUT_MS` — how long to wait for a response (default 30000).
- `EXTERNAL_AI_RETRY_COUNT` — retries on transient failures (default 3).
- `EXTERNAL_AI_FALLBACK` — what to do if the server is unreachable: `PASS`,
  `RANDOM`, or `CANCEL` (default `PASS`).
- `EXTERNAL_AI_STRICT` — when `true`, the game pauses instead of falling back if
  the server cannot be reached (default `false`).

## Where things are stored

Everything lives inside the project directory so the setup is self-contained:

- `replays/` — recorded matches (one JSON file per match). This is the
  training data. By default only the match file is written; it contains every
  game, every move, and every AI decision, so it is the single source of truth
  for training. Set `logging.write_replay_files: true` in
  `config/default.yaml` if you also want a separate per-game replay file.
- `models/<name>/` — a trained model: the checkpoint (`model.pt`), its
  architecture (`config.json`), the card vocabulary (`vocab.json`), and the
  keyword vocabulary (`kw_vocab.json`).
- `logs/` — server logs, **one file per launch** (`server-<timestamp>.log`).
  Each time the server starts it gets its own log file, so runs never mix
  together. The file holds both the server's structured logs and its console
  output.
- `config.json` — app settings, including which model was last used (so the
  server reloads it on startup).
- `config/default.yaml` — server settings (host, port, model architecture,
  logging).

## Project Structure

- `src/arcbound/` - Main package
  - `models/` - Pydantic models for API schemas
  - `routes/` - FastAPI route handlers
  - `decision/` - Decision logic (fallback + transformer)
  - `encoder/` - Transformer encoder implementation
  - `logging/` - Match and game replay logging
  - `training/` - Training pipeline
  - `analysis/` - Post-game move analysis
  - `gui/` - Tkinter GUI (server, models, training, game monitoring)
  - `utils/` - Utility functions

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## What the Model Sees

Before the model makes a decision, the current board state is converted into a
fixed-size feature vector (122 numbers per token) that the transformer reads.
This is the model's "eyes" — everything it can reason about comes from these
features plus the learned card and keyword identities.

- **Per card** — power/toughness, converted mana cost, damage, counters, colors,
  loyalty, which zone it is in, and its combat state (tapped, attacking,
  blocking, blocked). It also gets the card's type (creature, land, enchantment,
  instant, sorcery, artifact), whether it is a token or legendary, and a
  multi-hot of 15 decision-relevant keywords (Flying, Trample, Haste, ...).
- **Per player** — life, library size, hand size, battlefield size, poison, the
  mana pool, commander damage and tax, graveyard and exile sizes, how many lands
  and creatures are on the battlefield, total battlefield power and toughness,
  how many lands are untapped, the hand's mana curve (how many cards at CMC 0–1,
  2–3, and 4+), and how many spells were cast this turn.
- **Per board** — the turn number, the current phase (beginning, main 1,
  combat, main 2, end, cleanup), how many objects are on the stack, combat
  aggregates (number of attackers and blockers, how many attackers are
  unblocked, total attacking power, and whether that damage is lethal), the
  average CMC on the battlefield, and the focal player's relative position
  (life difference, tempo difference, and card advantage versus the opponent).
- **Per decision** — the decision type, how many options there are, the
  selection constraints, and the length of the prompt.

### Hidden information is masked per perspective

The board state Forge sends is the *full* board — both players' hands and both
decklists. But the model never reads it raw: at tokenization time the features
are built from a single **perspective** (one player's point of view), and the
hidden zones are masked to match what that player can actually see:

- **Only the perspective player's hand is emitted.** The opponent's hand is
  dropped entirely — the model still sees the opponent's *hand size* (via the
  player features) but not which cards are in it.
- **Only the perspective player's decklist is emitted.** The opponent's deck is
  never tokenized.
- **Explicitly revealed cards are the exception.** Each card carries a
  `revealed_to` list; if a card in the opponent's hand was revealed to the
  perspective player (by a card effect), it is included.
- **The "focal" features follow the perspective.** `is_focal`, commander
  damage, and the relative position features (life, tempo, and card-advantage
  differences) are all computed from the perspective player, so the same board
  tokenized from two perspectives produces two genuinely different inputs.

In normal play the perspective is the AI itself (the board's focal player), so
the AI only ever sees its own hand and deck — no information leakage. The same
masking is what makes the per-player evaluation graphs possible (see below):
the server can re-tokenize the same full board from the *human's* perspective
to ask "how does the human think it is doing?"

Numeric features are scaled to a 0–1 range with a cap tuned to real games
(life tops out at 50, hand at 10, battlefield at 32, CMC at 12, power and
toughness at 20, counters at 10, deck copies at 8, and the mana pool at 16), so
the common range of values uses most of the scale instead of being compressed
into a tiny slice of it.

**Changing the feature layout invalidates existing models.** The model's input
layer is sized to this feature vector, so when the layout changes (as it did
when these features were added), older checkpoints no longer line up and must be
retrained.

## What the AI Can Decide

Whenever the game needs the AI to make a choice, it asks and the AI picks an option.
Here is the full list of situations the AI knows how to handle, grouped by when they
come up in a game.

### At the start of the game
- **Keep or mulligan** — decide whether to keep your opening hand or shuffle it back and redraw.
- **Tuck cards while mulliganing** — choose which cards to put on the bottom of your deck.
- **Choose your starting hand** — when a card lets you pick which hand to start with.
- **Sideboard between games** — choose which sideboard cards to swap into your main deck.
- **Activate abilities from your opening hand** — for cards that can be activated before the game really starts.

### Playing cards
- **Which spell or ability to play** — pick the next card or ability to cast.
- **Choose targets** — pick the creatures, players, or other things a spell targets.
- **Order multiple plays** — decide the order to play several cards or abilities.
- **Announce a value** — pick the number for an "X" in a cost or effect.
- **Convoke / improvise** — choose which creatures or artifacts to tap to help pay.
- **Splice** — choose which cards to splice onto a spell.
- **Pay with cards** — choose cards to delve, sacrifice, or otherwise spend as part of a cost.
- **Optional costs** — decide whether to pay extra for kicker, overload, and similar options.
- **Choose mana** — pick which mana to use, and which colors to pay with.
- **Help pay** — decide whether to help pay for a teammate's spell, and who pays.

### Combat
- **Declare attackers** — choose which creatures attack.
- **Declare blockers** — choose which creatures block.
- **Order attackers and blockers** — set the order for damage assignment.
- **Assign damage** — decide how much damage each attacker deals to each blocker or player.
- **Divide a shield** — split a shield among the things it protects.

### Hand and graveyard
- **Discard** — choose which cards to discard.
- **Sacrifice** — choose which permanents to sacrifice.
- **Reveal from hand** — choose which cards to reveal.
- **Choose a pile** — pick which group of cards to use.

### Library and zones
- **Search and fetch** — choose which card to find and put into your hand or onto the battlefield.
- **Order cards moving to a zone** — decide the order cards go on top of, or into the bottom of, a zone.
- **Scry and surveil** — choose which cards to put back on top and which to the bottom.
- **Choose a card name or face** — pick a specific card, or which side of a card to use.

### Effects and abilities
- **Choose an entity or card** — pick a player, creature, or other object an effect applies to.
- **Specify a target** — for effects that let you redirect or choose a target.
- **Replacement effects** — choose which replacement effect to apply.
- **Static abilities** — choose which static ability to use.
- **Modes** — pick which mode(s) of a multi-mode effect to use.
- **Triggers** — decide whether to trigger an optional ability.
- **Confirm actions** — answer yes/no prompts for payments, bids, and other confirmations.
- **Call a flip** — decide the result of a coin-flip-like effect.
- **Vote** — cast a vote in multiplayer effects.

### Types, colors, and keywords
- **Choose a type** — pick a card type, subtype, or other category.
- **Choose a color** — pick one or more colors.
- **Choose a keyword** — pick a keyword to give to a creature.
- **Choose a protection type** — pick what a protection effect protects against.
- **Choose a counter type** — pick which type of counter to add.
- **Choose a sector, sprocket, or contraption** — for the more unusual mechanics.

### Dice and numbers
- **Dice rolls** — choose which dice to reroll, ignore, modify, or swap.
- **Choose a number** — pick a number within a range.
- **Yes or no** — answer simple true/false questions.

## How the AI Evaluates the Game: Per-Player Self-Assessment

The model's value head estimates "how good is this position?" — but for whom?
With hidden information, the answer depends on whose hand you can see. So the
server computes the value **from each player's own perspective** and stores
each player's result in its own series.

- **After every action** — the AI's decisions *and* the opponent's moves — the
  server re-tokenizes the full board once per player (each with its own hidden
  zones masked) and records a value point for every player.
- **No zero-sum negation.** MTG is not a perfect-information game, so the
  opponent's self-assessment is a *different information set*, not "the
  negative" of the AI's. Each player's series is its own view of the game.
- **The live Game tab shows a tab per player.** One tab is the AI's own
  self-assessment; another is the human's — i.e. *how the AI thinks the human
  is doing*, computed with the human's hand visible and the AI's hand masked.
  With auto-switch on, the notebook jumps to whichever AI most recently acted.
- **The Analysis tab shows the same per-player graphs for a replay**, so you
  can compare how each player's position looked over the course of the game.

## How the AI Learns: Match Logging and Training

The AI gets better by watching games and learning from them. Here is how that works.

### Every match is recorded as one file

When a match is played (a single game or a best-of-3), the whole thing is saved to a
single replay file. That file contains:

- **Every move** — each action any player takes, along with a full snapshot of the board
  at that moment. This is the same level of detail as a normal human-vs-AI game, so the
  data is useful for training.
- **Every AI decision** — for each choice the AI faced, the file records the options it
  was given, the option it picked, and the model's confidence and value estimate. These
  decision records are what policy learning trains on; the board snapshots are what value
  learning trains on.
- **Who won each game** — in a best-of-3, each game has its own winner and its own reward.
- **Who won the match** — the match-level winner, the final score (for example 2–1), and a
  match-level reward.
- **Who each player is** — each player's type (`human`, `bot`, or `ai`) and a match
  composition label (for example `human_vs_ai`). This lets training choose which players
  to learn from, and lets the analysis tool label each player's moves.

This means the AI can learn from both the big picture (did I win the match?) and the
details (how did each individual game go?). A single-game match works the same way: the
player who wins the only game also wins the match.

### Two ways the AI learns from a recorded match

1. **Learning to pick moves (policy learning).** The AI looks at each decision it faced
   and learns which option was a good one to pick, based on how the game turned out.

2. **Learning to read the board (value learning).** The AI learns to look at a board
   position and estimate how good it is — that is, how likely the player about to move is
   to win from there. This gives the AI a sense of "how good is this position?" that it
   can use when choosing between options.

You can train either one, or both, at the same time.

### Recognizing individual cards (card identity)

The AI does not just see "a 3-mana creature" or "a 0-mana land" — it learns to tell
specific cards apart. When you train, the AI builds a list of every card name it has
seen in your recorded games (from the decks and from every board snapshot) and gives
each card its own little "identity" that it learns over time. This is what lets it
distinguish, say, a Lightning Bolt from a Swords to Plowlands, even though both are
cheap spells that look similar on the surface.

A few practical notes:

- **It only knows the cards it has seen.** The list of card identities is built from
  your replay files, so the AI recognizes the cards that actually appear in the games
  you record. A card it has never seen is treated as an unknown and gets no special
  identity of its own.
- **The list is saved with the model.** When training finishes, the card list is saved
  right next to the model file. When the server starts up, it loads that list so it can
  recognize the same cards during play.
- **Older models keep working.** Models trained before this feature still load and play
  normally — they just do not have per-card identities yet. To get them, train (or
  retrain) a model now and it will pick up the card identities automatically.
- **Keywords are learned too.** Alongside card names, the AI builds a list of the
  keywords it has seen (Flying, Trample, Haste, ...) and gives each its own learned
  identity, so it can reason about what a keyword does rather than only reading the
  card's stats.

### Seeing its own deck (decklist awareness)

At the start of a game, the AI is shown its full deck — every card in the main deck
and the sideboard, not just the cards it has already drawn. Each distinct card is
listed once, together with how many copies of it are in the deck (for example, four
copies of Mountain and two of Lightning Bolt). This lets the AI plan around cards it
may draw later, the same way a human player knows what is in their deck.

A few practical notes:

- **Cards already in play are not repeated.** If a card from the deck is already in
  hand, on the battlefield, or in another visible zone, it is not listed a second
  time — the AI already sees it where it is.
- **The copy count is capped at eight.** Decks rarely have more than four copies of a
  card, so the count is scaled to top out at eight; anything more is treated the same.
- **Older models keep working.** Models trained before this feature still load and
  play normally — they just do not use the deck information yet. To get the benefit,
  train (or retrain) a model now.

### New cards are announced in the terminal

When the server sees a card for the first time during a game, it prints the card's
full details to the terminal — name, mana cost, colors, types, power and toughness,
loyalty, keywords, and the rules text. Each card is announced only once, so you can
watch the server meet new cards as the AI plays through unfamiliar decks.

### Sideboarding

Between games of a multi-game match, the AI makes its own sideboard decisions — it decides
which sideboard cards to put into its main deck, and for each card it adds it removes one
from the main deck, so the main deck always stays the same size. Every sideboard change is
also recorded in the match file, so the AI can learn from its sideboard choices too.

## Analyzing a replay

The GUI's Analysis tab scores every move in a recorded match — the AI's decisions
*and* the human player's moves — and rates each one qualitatively (Best / Good /
Inaccuracy / Mistake / Blunder).

- **AI moves** use the confidence the model recorded when it made the decision.
- **Human moves** (which have no recorded confidence) are scored by loading a model
  and asking it how good the position was, or how likely it was to make the move the
  human actually made. Load a model in the tab to enable this.
- The tab shows a per-player summary (so you can compare the human against the AI),
  the match composition and each player's type, and a **per-player
  evaluation-over-time graph** — one tab per player, each showing that player's
  own self-assessment (its hidden info visible, the opponent's masked), the same
  way the live Game tab does.
- You can export the full report to a text file.

## Models and inference

When the server starts, it automatically loads your last-used model (the one
selected in the GUI, recorded in `config.json`; if there is exactly one model it
uses that). With a model loaded, the AI makes real transformer-based decisions.

### Fallbacks are a last resort, and they are announced

The fallback is **not** a normal way of playing — it only kicks in when the
model genuinely cannot be used (no model is loaded, or model inference throws).
When that happens the AI:

- **Announces it in the terminal in red**, stating *why* the fallback was used
  (e.g. "FALLBACK (model unavailable): no model is loaded — create and train a
  model in the GUI"). If you see red fallback messages, something is wrong and
  you need to know about it.
- **Mixes random actions with pass-throughs**, using the same `epsilon` value
  as the model path. With probability `epsilon` it takes a random action (a
  random option / subset / ordering); otherwise it takes the conservative
  "do nothing" default (the safe option for yes/no, no cards when allowed,
  first option otherwise). So even on the fallback the AI can play cards and
  lands rather than always passing.

### Epsilon exploration is announced in yellow

When the model is loaded and `epsilon` triggers (the AI deliberately picks a
random option instead of its top choice, to keep the training data diverse),
the terminal shows a **yellow** message: "EPSILON exploration (p=0.15): picked
a random option". This lets you confirm exploration is actually happening.

If no model is loaded — for example, on a fresh install before you have trained
anything — the server still runs, but every decision uses the fallback
described above (and you will see the red announcements). The game plays fine;
the AI just does not play well yet. Create a model in the GUI's Model tab and
train it to switch to real inference.

## Training the model

Training reads the replay files in `replays/` and updates a model. You can train
from the GUI's Training tab or from the command line:

```bash
arcbound-train --replays replays --model MyModel --mode both
```

Useful options:

- `--mode` — what to train: `policy` (choosing options), `value` (reading the
board), or `both` (default). Note the GUI's Training tab currently runs policy
training only; use the command line for value or both.
- `--epochs`, `--batch-size`, `--lr` — standard training knobs.
- `--gamma` — discount factor for the per-game value returns.
- `--match-weight` — how much of the policy reward comes from the match outcome
  versus the individual game (0 = game only, 1 = match only, default 0.5).
- `--player-types` — comma-separated player roles to learn from (for example `ai`, or
  `human,bot`). Defaults to all players. Roles are `human`, `bot`, and `ai`. This is
  useful when you want the AI to learn only from its own decisions, or to exclude them.

Training rebuilds the card vocabulary from your replays and saves it next to the
model, so the model learns per-card identities as described above.

## Limitations

A few things to keep in mind about how the AI plays:

- **It learns by example, not by the rules.** The AI is trained on games it has
  watched, so it makes its best guess based on patterns it has seen. It does not
  "know" the rules the way a human does, so on a situation it has not seen much of,
  it may make a suboptimal or even odd choice.

- **It only sees what is on the table — except its own deck.** The AI can see the
  battlefield, your hand, and other public information, and it knows its own full deck
  from the start of the game. But it cannot see your opponent's hidden cards (their
  hand, the top of their deck, and so on), so it has to guess when hidden information matters.
  This is enforced in the tokenizer, not just by convention: Forge sends the full
  board (both hands and both decks), and the server masks the opponent's hidden zones
  before the model ever sees the features, so the AI's decisions and its own
  self-assessment can never leak the opponent's hand.

- **In Commander it tracks commander damage per commander.** The AI knows exactly how
  much damage each specific commander has dealt to each player — not just a total — so
  it can tell that one commander is about to deal the 21 damage that loses the game,
  even if another commander has dealt none.

- **Some choices fall back to a default.** If the model cannot be used (no model
  loaded, or inference fails), the AI falls back to a last-resort heuristic that
  mixes random actions with the conservative default (the safe option for yes/no
  questions, no cards when allowed, usually the first option otherwise), using
  `epsilon` to set the ratio. Every fallback is announced in the terminal in red
  with the reason, so you know immediately that something is wrong. The game will
  still continue, but the choice may not be the best one.

- **It is built mainly for one-on-one play.** Most decisions are tuned for a standard
  two-player game. Multiplayer situations (teams, voting, helping a teammate) are
  supported but are less common and less tested.

- **Unusual mechanics are supported but rare.** Things like contraptions, sectors,
  sprockets, and dice are handled, but they come up rarely in normal games, so the AI
  has less experience with them.

- **It can get stuck in a loop.** If the AI keeps trying the same move that the rules
  do not allow, the game engine will notice and stop it after a while (you may see a
  "looped too much" message). This is a safety net, not a strategy.

- **It is only as good as its training.** The more (and the better) the games the AI is
  trained on, the more it will play like a real player. A freshly started AI will play
  much more randomly than a well-trained one.
