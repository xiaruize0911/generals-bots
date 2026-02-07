"""
Neural network architecture for Generals.io PPO agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PolicyValueNetwork(nn.Module):
    """
    Convolutional policy-value network for 4x4 Generals grid.

    Architecture:
    - 4 convolutional layers for feature extraction
    - Separate policy head (outputs action logits)
    - Separate value head (outputs state value)
    """

    def __init__(self, grid_size=4, channels=(32, 32, 32, 16)):
        """
        Initialize the network.

        Args:
            grid_size: Size of the game grid
            channels: Number of channels in each conv layer
        """
        super().__init__()

        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(9, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1)

        # Policy head: 9 channels = 4 dirs + 4 half-moves + 1 pass
        self.policy_conv = nn.Conv2d(channels[3], 9, kernel_size=1)

        # Value head
        self.value_conv = nn.Conv2d(channels[3], 4, kernel_size=1)
        self.value_linear1 = nn.Linear(grid_size * grid_size * 4, 64)
        self.value_linear2 = nn.Linear(64, 1)

    def forward(self, obs, mask=None, action=None):
        """
        Forward pass through the network.

        Args:
            obs: Observation tensor [batch, 9, H, W]
            mask: Valid action mask [batch, H, W, 4] or None
            action: If provided, evaluate this action [batch, 5]. Otherwise sample.

        Returns:
            (action, value, log_prob, entropy)
        """
        batch_size = obs.shape[0]
        grid_size = obs.shape[-1]

        # Feature extraction backbone
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Value head
        v = F.relu(self.value_conv(x))
        v_flat = v.view(batch_size, -1)
        value_hidden = F.relu(self.value_linear1(v_flat))
        value = self.value_linear2(value_hidden).squeeze(-1)

        # Policy head
        logits = self.policy_conv(x)  # [batch, 9, H, W]

        # Apply action mask if provided
        if mask is not None:
            mask_t = mask.permute(0, 3, 1, 2)  # [batch, 4, H, W]
            mask_penalty = (1 - mask_t) * -1e9
            # Add pass action (always valid)
            pass_mask = torch.zeros(batch_size, 1, grid_size, grid_size, device=obs.device)
            combined_mask = torch.cat([
                mask_penalty,  # 4 directions
                mask_penalty,  # 4 half-move directions
                pass_mask  # pass action
            ], dim=1)
            logits = (logits + combined_mask).view(batch_size, -1)
        else:
            logits = logits.view(batch_size, -1)

        grid_cells = grid_size * grid_size

        if action is None:
            # Sample action
            dist = Categorical(logits=logits)
            idx = dist.sample()
            direction = idx // grid_cells
            position = idx % grid_cells
            row = position // grid_size
            col = position % grid_size
            is_pass = (direction == 8).long()
            is_half = ((direction >= 4) & (direction < 8)).long()
            actual_dir = torch.where(is_pass > 0, torch.zeros_like(direction),
                                   torch.where(is_half > 0, direction - 4, direction))
            action = torch.stack([is_pass, row, col, actual_dir, is_half], dim=-1)
            logprob = dist.log_prob(idx)
            entropy = dist.entropy()
        else:
            # Compute index from provided action
            is_pass, row, col, direction, is_half = action.unbind(-1)
            encoded_dir = torch.where(is_pass > 0, torch.full_like(direction, 8),
                                    torch.where(is_half > 0, direction + 4, direction))
            idx = encoded_dir * grid_cells + row * grid_size + col

            dist = Categorical(logits=logits)
            logprob = dist.log_prob(idx)
            entropy = dist.entropy()

        return action, value, logprob, entropy


def obs_to_array(obs):
    """
    Convert Observation namedtuple to network input array.

    Args:
        obs: Observation from environment

    Returns:
        Tensor [9, H, W] with stacked observation channels
    """
    return torch.stack([
        torch.tensor(obs.armies, dtype=torch.float32),
        torch.tensor(obs.generals, dtype=torch.float32),
        torch.tensor(obs.cities, dtype=torch.float32),
        torch.tensor(obs.mountains, dtype=torch.float32),
        torch.tensor(obs.neutral_cells, dtype=torch.float32),
        torch.tensor(obs.owned_cells, dtype=torch.float32),
        torch.tensor(obs.opponent_cells, dtype=torch.float32),
        torch.tensor(obs.fog_cells, dtype=torch.float32),
        torch.tensor(obs.structures_in_fog, dtype=torch.float32)
    ], dim=0)

