import math
import numpy as np
from typing import Callable, Optional
from .game import ConnectFour, COLS


class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, game_state: ConnectFour, parent: Optional["MCTSNode"] = None, action: Optional[int] = None, prior: float = 0.0):
        self.game_state = game_state
        self.parent = parent
        self.action = action       # move that led to this node
        self.prior = prior         # P(s, a) — prior probability from neural network

        self.children: dict[int, "MCTSNode"] = {}
        self.visit_count = 0       # N(s, a)
        self.value_sum = 0.0       # W(s, a) — sum of backed-up values

    def value(self) -> float:
        """Q(s, a) — mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> "MCTSNode":
        """
        Select the child with the highest PUCT score:
            PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        """
        best_score = -float("inf")
        best_child = None
        sqrt_parent_visits = math.sqrt(self.visit_count)

        for child in self.children.values():
            # child.value() is from child's current_player's perspective (the opponent).
            # Negate it so we're maximizing from the parent's (our) perspective.
            puct = -child.value() + c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            if puct > best_score:
                best_score = puct
                best_child = child

        return best_child

    def expand(self, policy: np.ndarray):
        """
        Create child nodes for each valid action.
        `policy` is a probability distribution over all COLS actions.
        Invalid (full-column) actions have prior 0.
        """
        valid_moves = self.game_state.get_valid_moves()
        for col in range(COLS):
            if valid_moves[col]:
                next_state = self.game_state.make_move(col)
                self.children[col] = MCTSNode(
                    game_state=next_state,
                    parent=self,
                    action=col,
                    prior=float(policy[col]),
                )

    def backpropagate(self, value: float):
        """
        Walk up the tree from this node to the root, updating visit counts
        and value sums. The value is negated at each level because the
        parent node belongs to the opposing player.
        """
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """
    Monte Carlo Tree Search guided by an evaluate_fn.

    evaluate_fn(game_state) -> (policy, value)
        - policy: np.ndarray of shape (COLS,), probability distribution over actions
        - value: float in [-1, 1], predicted outcome for the current player

    For the random-rollout phase (before adding the neural network), use
    `random_evaluate` below as the evaluate_fn.
    """

    def __init__(self, num_simulations: int, c_puct: float, evaluate_fn: Callable):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.evaluate_fn = evaluate_fn

    def search(self, game_state: ConnectFour, add_dirichlet_noise: bool = False,
               dirichlet_alpha: float = 0.5, dirichlet_weight: float = 0.25) -> np.ndarray:
        """
        Run MCTS from `game_state` and return a policy vector
        (normalized visit counts over all COLS actions).

        `add_dirichlet_noise=True` during self-play to encourage exploration.
        """
        root = MCTSNode(game_state)

        # Evaluate and expand root before simulations
        policy, _ = self.evaluate_fn(game_state)
        policy = self._mask_and_normalize(policy, game_state.get_valid_moves())

        if add_dirichlet_noise:
            valid_indices = np.where(game_state.get_valid_moves())[0]
            noise = np.zeros(COLS)
            noise[valid_indices] = np.random.dirichlet([dirichlet_alpha] * len(valid_indices))
            policy = (1 - dirichlet_weight) * policy + dirichlet_weight * noise
            policy = self._mask_and_normalize(policy, game_state.get_valid_moves())

        root.expand(policy)

        for _ in range(self.num_simulations):
            node = root

            # 1. Select: descend the tree using PUCT until an unexpanded or terminal node
            while node.is_expanded() and not node.game_state.is_terminal():
                node = node.select_child(self.c_puct)

            # 2. Evaluate: if not terminal, expand and get a value estimate
            if node.game_state.is_terminal():
                # Terminal node: value from the perspective of node.game_state.current_player
                # (the player whose turn it would be next — i.e., the one who did NOT make
                # the last move). backpropagate() will negate as it ascends to the parent.
                result = node.game_state.get_result(node.game_state.current_player)
                value = result if result is not None else 0.0
            else:
                policy, value = self.evaluate_fn(node.game_state)
                policy = self._mask_and_normalize(policy, node.game_state.get_valid_moves())
                node.expand(policy)

            # 3. Backpropagate: the value is from the current node's player's perspective;
            #    backpropagate negates at each level as we ascend to the parent.
            node.backpropagate(value)

        # Build policy from visit counts
        visit_counts = np.zeros(COLS)
        for col, child in root.children.items():
            visit_counts[col] = child.visit_count

        total = visit_counts.sum()
        if total == 0:
            # Fallback: uniform over valid moves
            valid = game_state.get_valid_moves().astype(float)
            return valid / valid.sum()
        return visit_counts / total

    @staticmethod
    def _mask_and_normalize(policy: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:
        """Zero out invalid actions and renormalize the policy."""
        policy = policy * valid_moves.astype(float)
        total = policy.sum()
        if total == 0:
            # All priors were 0 for valid moves; fall back to uniform
            policy = valid_moves.astype(float)
            total = policy.sum()
        return policy / total


def apply_temperature(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convert visit counts to a probability distribution using temperature.
    - temperature=1.0: proportional to visit counts (exploration)
    - temperature→0: concentrates on the most-visited action (exploitation)
    """
    if temperature == 0 or temperature < 1e-8:
        # Argmax: put all probability on the most-visited action
        best = np.argmax(visit_counts)
        dist = np.zeros(len(visit_counts))
        dist[best] = 1.0
        return dist
    counts = visit_counts ** (1.0 / temperature)
    total = counts.sum()
    return counts / total if total > 0 else counts


# ------------------------------------------------------------------
# Random rollout evaluate_fn (used before the neural network is ready)
# ------------------------------------------------------------------

def random_evaluate(game_state: ConnectFour):
    """
    Evaluate a position by playing out random moves to the end of the game.
    Returns a uniform prior and the rollout value from the current player's perspective.
    """
    valid = game_state.get_valid_moves()
    policy = valid.astype(float) / valid.sum()

    # Play out the game with random moves
    state = game_state
    while not state.is_terminal():
        valid_cols = np.where(state.get_valid_moves())[0]
        col = np.random.choice(valid_cols)
        state = state.make_move(col)

    # Result from the perspective of game_state.current_player
    result = state.get_result(game_state.current_player)
    return policy, float(result) if result is not None else 0.0


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import random

    def play_game(mcts: MCTS, verbose=False) -> int:
        """Play one game and return the winner (1, -1, or 0 for draw)."""
        game = ConnectFour()
        while not game.is_terminal():
            policy = mcts.search(game)
            action = int(np.argmax(policy))
            game = game.make_move(action)
            if verbose:
                print(game)
                print()
        return game._winner

    # Play 20 games MCTS vs random and check MCTS wins most
    mcts = MCTS(num_simulations=50, c_puct=1.5, evaluate_fn=random_evaluate)

    wins, losses, draws = 0, 0, 0
    n_games = 20
    for i in range(n_games):
        game = ConnectFour()
        mcts_player = 1 if i % 2 == 0 else -1
        while not game.is_terminal():
            if game.current_player == mcts_player:
                policy = mcts.search(game)
                col = int(np.argmax(policy))
            else:
                valid = np.where(game.get_valid_moves())[0]
                col = int(np.random.choice(valid))
            game = game.make_move(col)

        result = game._winner
        if result == mcts_player:
            wins += 1
        elif result == -mcts_player:
            losses += 1
        else:
            draws += 1

    print(f"MCTS (random rollout, 50 sims) vs Random over {n_games} games:")
    print(f"  Wins: {wins}, Losses: {losses}, Draws: {draws}")
    win_rate = wins / n_games
    print(f"  Win rate: {win_rate:.1%}")
    assert win_rate >= 0.5, f"MCTS should win >50% vs random, got {win_rate:.1%}"
    print("MCTS test: PASS")
