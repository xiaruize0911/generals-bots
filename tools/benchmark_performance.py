"""
Performance benchmark for vectorized environments.

Measures throughput (steps/second) with different configurations.
Uses loops for compatibility.

Usage:
    python benchmark_performance.py [num_envs] [num_steps] [iterations]
    
Examples:
    python benchmark_performance.py 256 100 10
    python benchmark_performance.py 1024 500 5
    python benchmark_performance.py 4096 100 3
"""
import sys
import time
import torch

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent
import numpy as np

# Parse arguments
NUM_ENVS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NUM_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
GRID_DIMS = (10, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("GENERALS.IO PERFORMANCE BENCHMARK")
print("=" * 70)
print(f"Configuration:")
print(f"  Num environments: {NUM_ENVS:,}")
print(f"  Steps per iter:   {NUM_STEPS:,}")
print(f"  Iterations:       {ITERATIONS}")
print(f"  Grid size:        {GRID_DIMS[0]}x{GRID_DIMS[1]}")
print(f"  Device:           {device}")
print(f"  Total steps:      {NUM_ENVS * NUM_STEPS * ITERATIONS:,}")
print("=" * 70)

# Create environment and agents
env = GeneralsEnv(grid_dims=GRID_DIMS, truncation=500)
agent_0 = RandomAgent()
agent_1 = RandomAgent()


def make_step_fn(env, agent_0, agent_1, num_envs):
    """Create a simple (non-vectorized) step function factory.

    This implementation runs the environments sequentially in a Python loop
    which keeps the benchmark dependency-free.
    """

    def step_fn(states, rng):
        new_states = []
        done_count = 0
        for i, s in enumerate(states):
            # Observations for both players
            obs_p0 = get_observation(s, 0)
            obs_p1 = get_observation(s, 1)

            # Actions
            key_p0 = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
            key_p1 = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
            a0 = agent_0.act(obs_p0, key_p0)
            a1 = agent_1.act(obs_p1, key_p1)
            actions = np.stack([a0, a1], axis=0)

            # Step the environment
            timestep, ns = env.step(s, actions, None)
            new_states.append(ns)

            if timestep.terminated or timestep.truncated:
                done_count += 1

        return np.stack(new_states), done_count

    return step_fn


def make_rollout_fn(env, agent_0, agent_1, num_envs, num_steps):
    """Create a simple rollout function that runs num_steps sequentially."""

    step_fn = make_step_fn(env, agent_0, agent_1, num_envs)

    def rollout(states, rng):
        s = list(states)
        total_done = 0
        for _ in range(num_steps):
            s, done_count = step_fn(s, rng)
            total_done += int(done_count)
        return s, rng, total_done

    return rollout


# Create rollout function
rollout_fn = make_rollout_fn(env, agent_0, agent_1, NUM_ENVS, NUM_STEPS)

# Initialize environments
rng = np.random.default_rng(42)
states = [env.reset(rng) for _ in range(NUM_ENVS)]

print("\nWarming up...\n")
states, _ = rollout_fn(states, rng)
print("Warmup complete!\n")

# Benchmark
print("Running benchmark...")
print("-" * 70)

iteration_times = []
all_stats = []

for iteration in range(ITERATIONS):
    iter_start = time.time()
    states, _, episode_count = rollout_fn(states, rng)
    iter_elapsed = time.time() - iter_start
    
    iteration_times.append(iter_elapsed)
    sps = (NUM_ENVS * NUM_STEPS) / iter_elapsed
    all_stats.append({
        'sps': sps,
        'episodes': int(episode_count),
        'time': iter_elapsed
    })
    
    print(f"Iteration {iteration + 1}/{ITERATIONS}: "
          f"{sps:>10,.0f} steps/s | "
          f"{iter_elapsed:>6.3f}s | "
          f"{int(episode_count):>4} episodes")

print("-" * 70)

# Summary statistics
avg_sps = sum(s['sps'] for s in all_stats) / len(all_stats)
max_sps = max(s['sps'] for s in all_stats)
min_sps = min(s['sps'] for s in all_stats)
total_time = sum(iteration_times)
total_steps = NUM_ENVS * NUM_STEPS * ITERATIONS
total_episodes = sum(s['episodes'] for s in all_stats)

print("\nBENCHMARK RESULTS")
print("=" * 70)
print(f"Total time:          {total_time:.2f}s")
print(f"Total steps:         {total_steps:,}")
print(f"Total episodes:      {total_episodes:,}")
print(f"")
print(f"Average throughput:  {avg_sps:,.0f} steps/s")
print(f"Peak throughput:     {max_sps:,.0f} steps/s")
print(f"Min throughput:      {min_sps:,.0f} steps/s")
print(f"")
print(f"Per-env throughput:  {avg_sps / NUM_ENVS:.1f} steps/s")
print(f"Steps/episode:       {total_steps / max(total_episodes, 1):.1f}")
print("=" * 70)
