import random
import numpy as np
from typing import Tuple

from .game import ConnectFour, ROWS, COLS

# A single training example: (encoded_state, mcts_policy, game_result)
# encoded_state: np.ndarray of shape (2, ROWS, COLS)
# mcts_policy:   np.ndarray of shape (COLS,)
# game_result:   float in {-1, 0, 1}
TrainingExample = Tuple[np.ndarray, np.ndarray, float]


def encode_board(game_state: ConnectFour) -> np.ndarray:
    """
    Encode a game state into a 2-channel (2, ROWS, COLS) float32 array.
    Uses the canonical board so the network always sees:
        plane 0: current player's pieces (1.0 where they have a piece)
        plane 1: opponent's pieces (1.0 where opponent has a piece)
    """
    canonical = game_state.get_canonical_board()  # (ROWS, COLS), current=+1
    plane0 = (canonical == 1).astype(np.float32)
    plane1 = (canonical == -1).astype(np.float32)
    return np.stack([plane0, plane1], axis=0)


def mirror_example(example: TrainingExample) -> TrainingExample:
    """
    Apply left-right mirror augmentation to a training example.
    Connect Four is horizontally symmetric, so flipping the board and
    reversing the policy vector is an equally valid training example.
    """
    state, policy, result = example
    mirrored_state = state[:, :, ::-1].copy()  # flip columns in both planes
    mirrored_policy = policy[::-1].copy()       # reverse column ordering
    return mirrored_state, mirrored_policy, result


class ReplayBuffer:
    """
    Fixed-capacity replay buffer storing training examples.
    When full, the oldest examples are dropped (FIFO).
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer: list[TrainingExample] = []

    def add(self, examples: list[TrainingExample]):
        """Add a list of examples, dropping oldest entries if over capacity."""
        self._buffer.extend(examples)
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random mini-batch (with replacement if needed)."""
        n = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, n)

    def __len__(self) -> int:
        return len(self._buffer)
