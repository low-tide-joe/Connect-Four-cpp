import numpy as np

from .config import AlphaZeroConfig
from .game import ConnectFour
from .mcts import MCTS, apply_temperature
from .network import ConnectFourNet
from .self_play import make_evaluate_fn


def evaluate_vs_random(network: ConnectFourNet, config: AlphaZeroConfig,
                       device: str = "cpu", n_games: int = 40) -> float:
    """
    Evaluate the trained agent against a random player.
    The agent plays as both player 1 and player 2 (half the games each)
    to avoid first-mover bias.

    Returns the agent's win rate (wins / n_games). Draws are not counted as wins.
    """
    evaluate_fn = make_evaluate_fn(network, device)
    mcts = MCTS(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        evaluate_fn=evaluate_fn,
    )

    wins = 0
    for i in range(n_games):
        agent_player = 1 if i % 2 == 0 else -1
        game = ConnectFour()

        while not game.is_terminal():
            if game.current_player == agent_player:
                visit_counts = mcts.search(game, add_dirichlet_noise=False)
                col = int(np.argmax(visit_counts))
            else:
                valid_cols = np.where(game.get_valid_moves())[0]
                col = int(np.random.choice(valid_cols))
            game = game.make_move(col)

        if game._winner == agent_player:
            wins += 1

    return wins / n_games
