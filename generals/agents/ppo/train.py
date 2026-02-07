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
from numba import njit, prange

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


@njit(cache=True)
def jit_random_action(armies, owned_cells, mountains):
    """Jit-optimized random valid action selection."""
    mask = compute_valid_move_mask(armies, owned_cells, mountains)
    
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            for d in range(4):
                if mask[r, c, d]:
                    count += 1
    
    if count == 0:
        return np.array([1, 0, 0, 0, 0], dtype=np.int32)
    
    idx = np.random.randint(0, count)
    curr = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            for d in range(4):
                if mask[r, c, d]:
                    if curr == idx:
                        is_half = np.random.randint(0, 2)
                        return np.array([0, r, c, d, is_half], dtype=np.int32)
                    curr += 1
    return np.array([1, 0, 0, 0, 0], dtype=np.int32)


@njit(parallel=True)
def jit_get_obs_batch(armies, generals, cities, mountains, ownership, ownership_neutral, player_idx):
    num_envs = armies.shape[0]
    H, W = armies.shape[1], armies.shape[2]
    obs_batch = np.empty((num_envs, 9, H, W), dtype=np.float32)
    masks_batch = np.empty((num_envs, H, W, 4), dtype=np.float32)
    
    for i in prange(num_envs):
        # Visibility
        own = ownership[i, player_idx]
        visible = game.jit_get_visibility(own)
        invisible = ~visible
        opp_idx = 1 - player_idx
        
        # Channels
        obs_batch[i, 0] = (armies[i] * visible).astype(np.float32)
        obs_batch[i, 1] = (generals[i] * visible).astype(np.float32)
        obs_batch[i, 2] = (cities[i] * visible).astype(np.float32)
        obs_batch[i, 3] = (mountains[i] * visible).astype(np.float32)
        obs_batch[i, 4] = (ownership_neutral[i] * visible).astype(np.float32)
        obs_batch[i, 5] = (ownership[i, player_idx] * visible).astype(np.float32)
        obs_batch[i, 6] = (ownership[i, opp_idx] * visible).astype(np.float32)
        obs_batch[i, 7] = (invisible & ~(mountains[i] | cities[i])).astype(np.float32)
        obs_batch[i, 8] = (invisible & (mountains[i] | cities[i])).astype(np.float32)
        
        # Mask
        m = compute_valid_move_mask(armies[i], ownership[i, player_idx], mountains[i])
        masks_batch[i] = m.astype(np.float32)
        
    return obs_batch, masks_batch


@njit(parallel=True)
def jit_get_random_actions_batch(num_envs, armies, ownership, mountains):
    actions = np.empty((num_envs, 5), dtype=np.int32)
    for i in prange(num_envs):
        actions[i] = jit_random_action(armies[i], ownership[i, 1], mountains[i])
    return actions


def rollout_step(states, network, device):
    """Vectorized rollout step for all environments using Numba parallelization."""
    num_envs = len(states)
    grid_size = states[0].armies.shape[0]

    # Stack state arrays for JIT batch processing
    armies = np.stack([s.armies for s in states])
    ownership = np.stack([s.ownership for s in states])
    ownership_neutral = np.stack([s.ownership_neutral for s in states])
    generals = np.stack([s.generals for s in states])
    cities = np.stack([s.cities for s in states])
    mountains = np.stack([s.mountains for s in states])
    passable = np.stack([s.passable for s in states])
    times = np.array([s.time for s in states])
    winners = np.array([s.winner for s in states])

    # Observations and masks (BEFORE step)
    obs_p0_arr, masks_p0 = jit_get_obs_batch(armies, generals, cities, mountains, ownership, ownership_neutral, 0)
    
    # We also need prior observations for reward calculation (old-style for now to avoid breaking composite_reward_fn)
    # But we can optimize this later.
    obs_p0_prior = [game.get_observation(states[i], 0) for i in range(num_envs)]
    obs_p1_prior = [game.get_observation(states[i], 1) for i in range(num_envs)]

    obs_arr_t = torch.from_numpy(obs_p0_arr).to(device)
    masks_t = torch.from_numpy(masks_p0).to(device)

    # Use eval mode and no_grad for rollout
    network.eval()
    with torch.no_grad():
        if device.type in ("cuda", "mps"):
            with torch.autocast(device_type=device.type):
                actions_p0, values, logprobs, entropies = network(obs_arr_t, masks_t)
        else:
            actions_p0, values, logprobs, entropies = network(obs_arr_t, masks_t)
    network.train()

    # Optimized random actions for p1 (parallelized)
    actions_p1 = jit_get_random_actions_batch(num_envs, armies, ownership, mountains)
    
    # Step game (Highly Optimized with Numba Parallel JIT)
    act0_np = actions_p0.cpu().numpy().astype(np.int32)
    combined_actions = np.zeros((num_envs, 2, 5), dtype=np.int32)
    for i in range(num_envs):
        combined_actions[i, 0] = act0_np[i]
        combined_actions[i, 1] = actions_p1[i]
    
    states, infos = game.batch_step(states, combined_actions)
    
    obs_p0_new = [game.get_observation(s, 0) for s in states]

    # Compute rewards
    rewards_list = []
    for i in range(num_envs):
        r = float(composite_reward_fn(obs_p0_prior[i], act0_np[i], obs_p0_new[i]))
        # Add a significant win bonus/loss penalty
        if infos[i].is_done:
            if infos[i].winner == 0:
                r += 1.0
            elif infos[i].winner == 1:
                r -= 1.0
        rewards_list.append(r)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

    # Terminal flags
    terminated = torch.tensor([bool(info.is_done) for info in infos], dtype=torch.bool, device=device)
    truncated = torch.tensor([(s.time >= 500) and (not t) for s, t in zip(states, terminated)], dtype=torch.bool, device=device)
    dones = terminated | truncated

    # Auto-reset
    if dones.any():
        for i in range(num_envs):
            if dones[i]:
                # Sample random positions for reset
                pos = np.random.choice(grid_size * grid_size, size=2, replace=False)
                grid = np.zeros((grid_size, grid_size), dtype=np.int32)
                grid[pos[0] // grid_size, pos[0] % grid_size] = 1
                grid[pos[1] // grid_size, pos[1] % grid_size] = 2
                states[i] = game.create_initial_state(grid)

    return states, (obs_arr_t, masks_t, actions_p0, logprobs, values, rewards, dones, infos)


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
        
        # Aggregate wins across all rollout steps
        wins = 0
        for step_data in rollout_data:
            step_dones = step_data[6]
            step_infos = step_data[7]
            for i in range(num_envs):
                if step_dones[i].item() and step_infos[i].winner == 0:
                    wins += 1
        
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

