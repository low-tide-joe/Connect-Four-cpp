# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Clone pybind11 submodule (if not already present)
git submodule update --init

# Build from project root
mkdir -p build && cd build
cmake ..
make game                  # build the static game library
make ConnectFourBitboard   # build the Python extension module

# Package and install as Python wheel
python setup.py bdist_wheel
pip install dist/ConnectFourBitboard-*.whl
```

## Architecture

The project is a Connect Four game engine written in C++, exposed to Python via pybind11, and used as the environment for ML-based agents.

### C++ Game Engine (`src/`)

- `src/game.hpp` / `src/game.cpp` — Core `ConnectFourBitboard` class using two 64-bit integers (one per player) to represent the 6×7 board. Implements win detection, move validation, and `getAdjacentPositions()` for strategic analysis.
- `src/binder.cpp` — pybind11 bindings exposing the C++ class to Python as the `ConnectFourBitboard` module.

Key API:
- `makeMove(col)` — place piece in column 0–6
- `getAvailableActions()` — returns list of valid columns
- `getAdjacentPositions(player)` — returns bitboard of positions adjacent to a player's pieces
- `getPlayerBoardState(player)` — raw 64-bit bitboard
- `gameState`: 0=in progress, 1=win, 2=draw
- `currentPlayer`: 0 or 1

### AI Implementations

- `minimax.cpp` — Standalone minimax with alpha-beta pruning (depth 12). Not integrated into the Python build; compile and run separately.
- `python_files/Deep-Q-Net/` — DQN agent using PyTorch:
  - `DQN.py` — CNN-based `QNetwork` (2-channel 6×7 input → 7 Q-values), reward shaping via `compute_rewards()`, and `train()` loop with experience replay and a target network.
  - `Train.py` — Runs 100 training episodes; saves model to `model_states/DQN_Model.pth`.
  - `Test.py` — Evaluates trained model against random player or human.
- `python_files/MCTS/mcts.py` — Monte Carlo Tree Search skeleton; incomplete/not functional.

### Build System

CMake builds two targets:
1. `game` — static library from `src/game.cpp`
2. `ConnectFourBitboard` — pybind11 Python extension linking against `game`

`setup.py` wraps CMake for `pip`-installable wheel packaging.
