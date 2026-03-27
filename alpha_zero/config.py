from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    # MCTS
    num_simulations: int = 200      # simulations per move during self-play
    c_puct: float = 1.5             # exploration constant in PUCT formula
    temperature_threshold: int = 10 # moves before temperature drops to near-0

    # Self-play & training
    num_iterations: int = 100       # outer loops of (self-play -> train)
    games_per_iteration: int = 100  # self-play games per iteration
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 10            # training epochs per iteration
    replay_buffer_size: int = 50000

    # Network
    num_filters: int = 64
    num_conv_layers: int = 4

    # Evaluation
    eval_games: int = 40            # games vs random agent per eval
    eval_interval: int = 5          # evaluate every N iterations

    # Dirichlet noise (added at MCTS root for exploration during self-play)
    dirichlet_alpha: float = 1.0    # ~10/num_actions per AlphaZero paper (~7 valid moves)
    dirichlet_weight: float = 0.25

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
