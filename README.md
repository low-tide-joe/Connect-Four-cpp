# Delta Connect

A Connect Four RL agent trained entirely through self-play, inspired by AlphaGo/AlphaZero. The agent learns with no hand-crafted heuristics — only win/loss/draw outcomes.

## How It Works

Training follows the AlphaZero approach and runs in a loop:

```
Self-play → Collect data → Train network → Repeat
```

### 1. Self-Play

The agent plays games against itself. At each move, it runs **MCTS (Monte Carlo Tree Search)** — simulating hundreds of possible continuations from the current position, guided by a neural network. The result is a policy vector showing how often each column was visited during search. The agent samples a move from that distribution and records `(board state, MCTS policy, which player moved)`.

### 2. Labeling

Once the game ends, every recorded position is labeled with the final outcome from that player's perspective: `+1` for win, `-1` for loss, `0` for draw.

### 3. Training the Network

The network has two output heads:

- **Policy head** — learns to predict what MCTS *would* suggest for a given position. Over time this lets the network shortcut expensive search.
- **Value head** — learns to predict the game outcome from a given position. This lets MCTS be steered toward promising positions early, before reaching a terminal state.

The only training signal is the game outcome. No reward shaping, no intermediate bonuses.

### 4. Exploration

Dirichlet noise is added to the MCTS root policy during self-play so the agent doesn't always play the same opening. This ensures diverse training data and prevents the agent from getting stuck in narrow strategies.

### The Network

A small dual-head CNN (~116K parameters):

```
Input: (2, 6, 7) — two binary planes (current player's pieces, opponent's pieces)
  └─ 4× Conv2d(64, 3×3) + BatchNorm + ReLU
        ├─ Policy head → log-softmax over 7 columns
        └─ Value head  → tanh scalar in [-1, 1]
```

## Setup

```bash
uv sync
```

## Training

```bash
uv run python -m alpha_zero.train
```

Default config (in `alpha_zero/config.py`): 50 iterations × 50 self-play games × 100 MCTS simulations per move. Checkpoints are saved to `checkpoints/` after each iteration. Evaluation against a random agent runs every 5 iterations.

To adjust the config, edit `alpha_zero/config.py` or pass a custom `AlphaZeroConfig` to `run_alpha_zero()` in `train.py`.

**Expected training time:** Several hours on CPU. The agent should reach >90% win rate vs a random player within ~20 iterations.

## Play Against the Agent

```bash
uv run python -m alpha_zero.play_human checkpoints/checkpoint_0050.pth
```

You can play against any saved checkpoint. The agent will display its MCTS policy and value estimate at each move so you can see what it's thinking.

To play against an intermediate checkpoint while training is still running:

```bash
# In one terminal:
uv run python -m alpha_zero.train

# In another terminal:
uv run python -m alpha_zero.play_human checkpoints/checkpoint_0010.pth
```

## Project Structure

```
alpha_zero/
    config.py       — all hyperparameters
    game.py         — Connect Four engine (numpy)
    mcts.py         — MCTS with PUCT selection
    network.py      — dual-head CNN (PyTorch)
    utils.py        — board encoding, replay buffer
    self_play.py    — self-play game generation
    train.py        — training orchestrator
    evaluate.py     — win rate evaluation vs random agent
    play_human.py   — interactive terminal play
```
