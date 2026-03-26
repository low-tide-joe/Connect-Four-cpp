import numpy as np
from tqdm import tqdm

from .config import AlphaZeroConfig
from .game import ConnectFour
from .mcts import MCTS, apply_temperature
from .network import ConnectFourNet
from .utils import TrainingExample, encode_board, mirror_example


def make_evaluate_fn(network: ConnectFourNet, device: str = "cpu"):
    """Wrap network.predict so it matches the MCTS evaluate_fn signature."""
    def evaluate_fn(game_state: ConnectFour):
        return network.predict(game_state, device=device)
    return evaluate_fn


def play_game(network: ConnectFourNet, config: AlphaZeroConfig, device: str = "cpu") -> list[TrainingExample]:
    """
    Play a full self-play game using MCTS guided by `network`.
    Returns a list of training examples: (encoded_state, mcts_policy, result).

    The result for each position is the final game outcome from the perspective
    of the player whose turn it was at that position:
        +1 if that player won, -1 if they lost, 0 if draw.
    """
    evaluate_fn = make_evaluate_fn(network, device)
    mcts = MCTS(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        evaluate_fn=evaluate_fn,
    )

    game = ConnectFour()
    # Store (encoded_state, mcts_policy, player_at_that_step) for each move
    trajectory: list[tuple] = []
    move_number = 0

    while not game.is_terminal():
        use_temperature = move_number < config.temperature_threshold
        temperature = 1.0 if use_temperature else 0.0

        visit_counts = mcts.search(
            game,
            add_dirichlet_noise=True,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_weight=config.dirichlet_weight,
        )
        policy = apply_temperature(visit_counts, temperature)

        trajectory.append((encode_board(game), policy, game.current_player))

        # Sample action from the policy
        col = int(np.random.choice(len(policy), p=policy))
        game = game.make_move(col)
        move_number += 1

    # Assign outcomes — convert to each position's player's perspective
    final_winner = game._winner  # 1, -1, or 0
    examples: list[TrainingExample] = []
    for state, policy, player in trajectory:
        if final_winner == 0:
            result = 0.0
        else:
            result = 1.0 if final_winner == player else -1.0
        examples.append((state, policy, result))

    return examples


def generate_self_play_data(network: ConnectFourNet, config: AlphaZeroConfig, device: str = "cpu") -> list[TrainingExample]:
    """
    Run multiple self-play games and collect all training examples.
    Applies mirror augmentation to double the dataset.
    """
    all_examples: list[TrainingExample] = []

    with tqdm(total=config.games_per_iteration, desc="  Self-play", unit="game", leave=False) as pbar:
        for _ in range(config.games_per_iteration):
            examples = play_game(network, config, device)
            mirrored = [mirror_example(e) for e in examples]
            all_examples.extend(examples)
            all_examples.extend(mirrored)
            pbar.update(1)
            pbar.set_postfix(examples=len(all_examples))

    return all_examples
