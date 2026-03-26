"""
Interactive terminal game: human vs trained AlphaZero agent.

Usage:
    uv run python -m alpha_zero.play_human checkpoints/checkpoint_0050.pth
"""

import sys
import numpy as np
import torch

from .config import AlphaZeroConfig
from .game import ConnectFour, COLS
from .mcts import MCTS, apply_temperature
from .network import ConnectFourNet
from .self_play import make_evaluate_fn


def load_network(checkpoint_path: str, config: AlphaZeroConfig, device: str) -> ConnectFourNet:
    network = ConnectFourNet(num_filters=config.num_filters, num_conv_layers=config.num_conv_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (iteration {checkpoint.get('iteration', '?')})")
    return network


def play_vs_human(checkpoint_path: str, config: AlphaZeroConfig = None, device: str = "cpu"):
    if config is None:
        config = AlphaZeroConfig()

    network = load_network(checkpoint_path, config, device)
    evaluate_fn = make_evaluate_fn(network, device)
    mcts = MCTS(num_simulations=config.num_simulations, c_puct=config.c_puct, evaluate_fn=evaluate_fn)

    # Human picks their side
    while True:
        choice = input("Play as X (first) or O (second)? [X/O]: ").strip().upper()
        if choice in ("X", "O"):
            break
    human_player = 1 if choice == "X" else -1
    agent_player = -human_player
    print(f"\nYou are {'X' if human_player == 1 else 'O'}. Agent is {'X' if agent_player == 1 else 'O'}.")
    print("Columns are numbered 0–6 (left to right).\n")

    game = ConnectFour()

    while not game.is_terminal():
        print(game)
        print()

        if game.current_player == human_player:
            # Human move
            while True:
                try:
                    col = int(input(f"Your move (0-{COLS - 1}): "))
                    if 0 <= col < COLS and game.get_valid_moves()[col]:
                        break
                    print(f"Column {col} is not valid. Try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            # Agent move
            print("Agent is thinking...")
            visit_counts = mcts.search(game, add_dirichlet_noise=False)
            policy, value = network.predict(game, device)

            # Display what the agent sees
            print(f"  Agent value estimate: {value:+.3f} (positive = agent is winning)")
            print(f"  MCTS policy: " + "  ".join(f"col{c}:{visit_counts[c]:.2f}" for c in range(COLS)))
            col = int(np.argmax(visit_counts))
            print(f"  Agent plays column {col}\n")

        game = game.make_move(col)

    print(game)
    print()
    if game._winner == human_player:
        print("You win!")
    elif game._winner == agent_player:
        print("Agent wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m alpha_zero.play_human <checkpoint_path>")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    play_vs_human(sys.argv[1], device=device)
