import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ConnectFour, ROWS, COLS


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConnectFourNet(nn.Module):
    """
    Dual-head neural network for Connect Four.

    Input:  (batch, 2, ROWS, COLS) — two binary planes
              plane 0: current player's pieces (1 where they have a piece)
              plane 1: opponent's pieces

    Outputs:
        log_policy: (batch, COLS) — log probabilities over actions (LogSoftmax)
        value:      (batch, 1)    — estimated game outcome in [-1, 1] (Tanh)

    The network always reasons from the current player's perspective because
    we pass the canonical board (current player's pieces = +1).
    """

    def __init__(self, num_filters: int = 64, num_conv_layers: int = 4):
        super().__init__()

        # Shared trunk
        layers = [ConvBlock(2, num_filters)]
        for _ in range(num_conv_layers - 1):
            layers.append(ConvBlock(num_filters, num_filters))
        self.trunk = nn.Sequential(*layers)

        # Policy head
        self.policy_conv = ConvBlock(num_filters, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)

        # Value head
        self.value_conv = ConvBlock(num_filters, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.trunk(x)

        # Policy head
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_policy, value

    def predict(self, game_state: ConnectFour, device: str = "cpu"):
        """
        Convenience method for MCTS: takes a game state, runs inference,
        and returns (policy, value) as numpy values.

        The policy is masked to valid moves and renormalized.
        """
        board = game_state.get_canonical_board()  # (ROWS, COLS), current player = +1

        # Build 2-channel input
        plane0 = (board == 1).astype(np.float32)   # current player's pieces
        plane1 = (board == -1).astype(np.float32)  # opponent's pieces
        x = np.stack([plane0, plane1], axis=0)     # (2, ROWS, COLS)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, ROWS, COLS)

        self.eval()
        with torch.no_grad():
            log_policy, value = self(x)

        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()  # (COLS,)
        value = value.squeeze().item()                            # scalar

        # Mask and renormalize to valid moves
        valid = game_state.get_valid_moves().astype(float)
        policy = policy * valid
        total = policy.sum()
        if total > 0:
            policy /= total
        else:
            policy = valid / valid.sum()

        return policy, value


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    net = ConnectFourNet(num_filters=64, num_conv_layers=4)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Network parameters: {total_params:,}")

    # Test output shapes with a batch of 4 random boards
    x = torch.randn(4, 2, ROWS, COLS)
    log_policy, value = net(x)
    assert log_policy.shape == (4, COLS), f"Expected (4, {COLS}), got {log_policy.shape}"
    assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"

    # Verify log_softmax: exp should sum to 1
    policy_sum = torch.exp(log_policy).sum(dim=1)
    assert torch.allclose(policy_sum, torch.ones(4), atol=1e-5), "Policy should sum to 1"
    print("Output shapes: PASS")

    # Test value range
    assert (value >= -1).all() and (value <= 1).all(), "Value should be in [-1, 1]"
    print("Value range: PASS")

    # Test predict() on a real game state
    from .game import ConnectFour
    game = ConnectFour()
    policy, v = net.predict(game)
    assert policy.shape == (COLS,), f"Expected ({COLS},), got {policy.shape}"
    assert abs(policy.sum() - 1.0) < 1e-5, "Predict policy should sum to 1"
    assert -1 <= v <= 1, "Predict value out of range"
    print(f"predict() test: PASS (policy sum={policy.sum():.4f}, value={v:.4f})")

    print(f"\nAll network tests passed. ({total_params:,} parameters)")
