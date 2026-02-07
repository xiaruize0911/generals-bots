"""
PyTorch PPO using raw game API for maximum performance.

This version bypasses the GeneralsEnv wrapper for high FPS.
"""

import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Restrict PyTorch threads
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Fix imports for package context
from generals.core.action import compute_valid_move_mask
from generals.core import game
from generals.core.rewards import composite_reward_fn
from generals.agents.ppo.network import PolicyValueNetwork, obs_to_array


def random_action(obs):
    """Random valid action."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for d in range(4):
                if mask[i, j, d]:
                    valid.append((i, j, d))
    if not valid:
        return torch.tensor([1, 0, 0, 0, 0], dtype=torch.int32)  # pass
    idx = torch.randint(0, len(valid), (1,)).item()
    row, col, direction = valid[idx]
    is_half = torch.randint(0, 2, (1,)).item()
    return torch.tensor([0, row, col, direction, is_half], dtype=torch.int32)


def rollout_step(states, network, device):
    """Vectorized rollout step for all environments."""
    num_envs = len(states)

    # Observations (BEFORE step for reward calculation)
    obs_p0_prior = [game.get_observation(states[i], 0) for i in range(num_envs)]
    obs_p1_prior = [game.get_observation(states[i], 1) for i in range(num_envs)]

    # Actions from network

    obs_arr = torch.stack([obs_to_array(obs) for obs in obs_p0_prior]).to(device)
    masks = torch.stack([
        torch.tensor(compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains), dtype=torch.float32)
        for obs in obs_p0_prior
    ]).to(device)

    # Use eval mode and no_grad for rollout
    network.eval()
    with torch.no_grad():
        # Use autocast for mixed precision if available
        if device.type in ("cuda", "mps"):
            with torch.autocast(device_type=device.type):
                actions_p0, values, logprobs, entropies = network(obs_arr, masks)
        else:
            actions_p0, values, logprobs, entropies = network(obs_arr, masks)
    network.train()

    # Random actions for p1 (move to device to match network outputs)
    actions_p1 = torch.stack([random_action(obs) for obs in obs_p1_prior]).to(device)

    # Step game (convert to CPU numpy arrays for the engine)
    actions = torch.stack([actions_p0, actions_p1], dim=1)
    new_states = []
    infos = []
    for i in range(num_envs):
        act_np = np.array(actions[i].cpu().numpy())
        ns, info = game.step(states[i], act_np)
        new_states.append(ns)
        infos.append(info)
    states = new_states

    # Get new observations (AFTER step)
    obs_p0_new = [game.get_observation(s, 0) for s in states]

    # Compute rewards using composite reward function (returns numpy scalars)
    rewards_list = [
        float(composite_reward_fn(obs_p0_prior[i], actions_p0[i].cpu().numpy(), obs_p0_new[i]))
        for i in range(num_envs)
    ]
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

    # Terminated/truncated - convert array-like fields to Python scalars first
    terminated_list = [bool(np.array(info.is_done).item()) for info in infos]
    # truncated: check state.time >= 500 and not terminated
    truncated_list = [ (int(np.array(s.time).item()) >= 500) and (not t) for s, t in zip(states, terminated_list) ]
    terminated = torch.tensor(terminated_list, dtype=torch.bool, device=device)
    truncated = torch.tensor(truncated_list, dtype=torch.bool, device=device)
    dones = terminated | truncated

    # Auto-reset if done with random but different general locations
    for i in range(num_envs):
        if dones[i]:
            grid = torch.zeros((4, 4), dtype=torch.int32)
            # Sample two different random positions
            positions = torch.randperm(16)[:2]
            pos_a = (positions[0] // 4, positions[0] % 4)
            pos_b = (positions[1] // 4, positions[1] % 4)
            grid[pos_a] = 1
            grid[pos_b] = 2
            states[i] = game.create_initial_state(np.array(grid.cpu().numpy()))

    return states, (obs_arr, masks, actions_p0, logprobs, values, rewards, dones, infos)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute advantages using GAE."""
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(num_steps)):
        next_value = values[t + 1] if t < num_steps - 1 else torch.zeros(num_envs, device=rewards.device)
        next_nonterminal = (~dones[t + 1] if t < num_steps - 1 else ~dones[t]).float()
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        advantages[t] = delta + gamma * lam * next_nonterminal * last_adv
        last_adv = advantages[t]

    return advantages


def ppo_loss(network, obs, mask, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for batch."""
    _, value, logprob, entropy = network(obs, mask, action)

    ratio = torch.exp(logprob - old_logprob)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -torch.min(ratio * advantage, clipped)

    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy

    return policy_loss + value_loss + entropy_loss


def train_step(network, optimizer, batch):
    """Single training step."""
    obs, masks, actions, old_logprobs, advantages, returns = batch

    # Flatten batch
    bs = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(bs, *obs.shape[2:])
    masks_flat = masks.reshape(bs, *masks.shape[2:])
    actions_flat = actions.reshape(bs, -1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)


    optimizer.zero_grad(set_to_none=True)
    # Use autocast for mixed precision if available
    device = next(network.parameters()).device
    if device.type in ("cuda", "mps"):
        with torch.autocast(device_type=device.type):
            losses = ppo_loss(network, obs_flat, masks_flat, actions_flat, old_logprobs_flat, advantages_flat, returns_flat)
            loss = losses.mean()
        loss.backward()
    else:
        losses = ppo_loss(network, obs_flat, masks_flat, actions_flat, old_logprobs_flat, advantages_flat, returns_flat)
        loss = losses.mean()
        loss.backward()
    optimizer.step()

    return loss.item()


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    num_steps = 256
    num_iterations = 200
    lr = 3e-4

    device = torch.device("mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"PyTorch PPO (Raw Game API - High Performance)")
    print(f"Environments:  {num_envs}")
    print(f"Device:        {device}")
    print(f"Grid:          4x4 with composite rewards")
    print()

    # Initialize

    network = PolicyValueNetwork(grid_size=4).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    # Enable cudnn benchmark for better performance if using CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    params = sum(p.numel() for p in network.parameters())
    print(f"Parameters: {params:,}")

    # Initialize states directly (no env wrapper)
    grid = torch.zeros((4, 4), dtype=torch.int32)
    grid[0, 0] = 1
    grid[3, 3] = 2
    grids = [grid] * num_envs
    states = [game.create_initial_state(np.array(g.cpu().numpy())) for g in grids]

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/ppo_training")

    print("\nWarming up...")
    for _ in range(3):
        states, _ = rollout_step(states, network, device)

    print("Training...\n")

    for iteration in range(num_iterations):
        t0 = time.time()

        # Collect rollout
        rollout_data = []
        for _ in range(num_steps):
            states, data = rollout_step(states, network, device)
            rollout_data.append(data)

        # Stack data
        obs = torch.stack([d[0] for d in rollout_data])
        masks = torch.stack([d[1] for d in rollout_data])
        actions = torch.stack([d[2] for d in rollout_data])
        logprobs = torch.stack([d[3] for d in rollout_data])
        values = torch.stack([d[4] for d in rollout_data])
        rewards = torch.stack([d[5] for d in rollout_data])
        dones = torch.stack([d[6] for d in rollout_data])

        # Compute advantages
        advantages = compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        # Train
        batch = (obs, masks, actions, logprobs, advantages, returns)
        loss = train_step(network, optimizer, batch)

        elapsed = time.time() - t0

        # Compute stats each iteration for logging
        avg_reward = rewards.mean().item()
        num_episodes = int(dones.sum().item())
        # compute wins safely from infos
        last_infos = rollout_data[-1][7]
        wins = sum(1 for d, info in zip(dones[-1].tolist(), last_infos) if d and getattr(info, 'winner', None) == 0)
        win_rate = wins / max(num_episodes, 1) * 100
        sps = (num_envs * num_steps) / max(elapsed, 1e-9)

        # Log to TensorBoard every iteration and flush
        try:
            writer.add_scalar("Loss/train", loss, iteration)
            writer.add_scalar("Reward/train", avg_reward, iteration)
            writer.add_scalar("Win Rate/train", win_rate, iteration)
            writer.add_scalar("SPS/train", sps, iteration)
            writer.flush()
        except Exception:
            # don't crash training if logging fails
            pass

        if iteration % 10 == 0:
            print(f"Iter {iteration:4d} | Loss: {loss:.4f} | "
                  f"Reward: {avg_reward:+.4f} | Episodes: {int(num_episodes):3d} | "
                  f"Wins: {wins:2d}/{int(num_episodes)} ({win_rate:.0f}%) | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s")

    print("\nTraining complete!")

    # Save model
    model_path = "pytorch_ppo_model.pth"
    torch.save(network.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Close TensorBoard writer
    try:
        writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

