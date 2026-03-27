import os
import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .config import AlphaZeroConfig
from .network import ConnectFourNet
from .self_play import generate_self_play_data
from .utils import ReplayBuffer


def train_network(network: ConnectFourNet, replay_buffer: ReplayBuffer,
                  optimizer: optim.Optimizer, config: AlphaZeroConfig, device: str = "cpu"):
    """
    Train the network on mini-batches sampled from the replay buffer.

    Loss = policy_loss + value_loss
        policy_loss: cross-entropy between NN log-policy and MCTS policy target
            = -sum(pi_target * log_pi_pred)  (standard CE for soft targets)
        value_loss:  mean squared error between NN value and game outcome
    """
    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    all_data = list(replay_buffer._buffer)
    n_batches_per_epoch = max(1, len(all_data) // config.batch_size)
    with tqdm(total=config.num_epochs * n_batches_per_epoch, desc="  Training", unit="batch", leave=False) as pbar:
        for epoch in range(config.num_epochs):
            random.shuffle(all_data)  # re-shuffle each epoch for different batch compositions
            for i in range(0, len(all_data), config.batch_size):
                batch = all_data[i: i + config.batch_size]
                if not batch:
                    continue

                states = torch.tensor(
                    np.array([e[0] for e in batch]), dtype=torch.float32
                ).to(device)
                target_policies = torch.tensor(
                    np.array([e[1] for e in batch]), dtype=torch.float32
                ).to(device)
                target_values = torch.tensor(
                    np.array([e[2] for e in batch]), dtype=torch.float32
                ).unsqueeze(1).to(device)

                log_policy, value = network(states)

                # Policy loss: cross-entropy with soft targets
                policy_loss = -(target_policies * log_policy).sum(dim=1).mean()
                # Value loss: MSE
                value_loss = ((value - target_values) ** 2).mean()
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_batches += 1
                pbar.update(1)
                pbar.set_postfix(p_loss=f"{policy_loss.item():.3f}", v_loss=f"{value_loss.item():.3f}")

    if n_batches > 0:
        return total_policy_loss / n_batches, total_value_loss / n_batches
    return 0.0, 0.0


def run_alpha_zero(config: AlphaZeroConfig = None, device: str = "cpu"):
    """
    Main AlphaZero training loop:
      1. Generate self-play games using current network
      2. Add data to replay buffer
      3. Train network on replay buffer
      4. Periodically evaluate and save checkpoint
      5. Repeat
    """
    if config is None:
        config = AlphaZeroConfig()

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    network = ConnectFourNet(
        num_filters=config.num_filters,
        num_conv_layers=config.num_conv_layers,
    ).to(device)

    optimizer = optim.Adam(
        network.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    print(f"Starting AlphaZero training on {device}")
    print(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    print(f"Config: {config.num_iterations} iterations, "
          f"{config.games_per_iteration} games/iter, "
          f"{config.num_simulations} MCTS sims/move\n")

    iter_bar = tqdm(range(1, config.num_iterations + 1), desc="Iterations", unit="iter")
    for iteration in iter_bar:
        iter_bar.set_description(f"Iteration {iteration}/{config.num_iterations}")

        # 1. Self-play
        new_examples = generate_self_play_data(network, config, device)
        replay_buffer.add(new_examples)

        # 2. Train
        policy_loss, value_loss = train_network(network, replay_buffer, optimizer, config, device)

        # 3. Evaluate periodically
        win_rate = None
        if iteration % config.eval_interval == 0:
            from .evaluate import evaluate_vs_random
            win_rate = evaluate_vs_random(network, config, device, n_games=config.eval_games)

        # Update outer bar with key stats
        postfix = dict(buf=len(replay_buffer), p_loss=f"{policy_loss:.3f}", v_loss=f"{value_loss:.3f}")
        if win_rate is not None:
            postfix["win%"] = f"{win_rate:.0%}"
        iter_bar.set_postfix(**postfix)

        # 4. Save checkpoint
        checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{iteration:04d}.pth")
        torch.save({
            "iteration": iteration,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    print("Training complete.")
    return network


if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AlphaZeroConfig()  # uses defaults from config.py
    network = run_alpha_zero(config, device=device)
