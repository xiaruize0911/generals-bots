"""Simple performance benchmark using sequential env stepping."""
import time
import numpy as np
from generals import GeneralsEnv


def benchmark(num_envs: int = 64, num_steps: int = 2000):
    print(f"Creating {num_envs} independent environments...")
    envs = [GeneralsEnv(grid_dims=(4, 4), truncation=500, pool_size=100) for _ in range(num_envs)]
    states = [env.reset(seed=i) for i, env in enumerate(envs)]

    pass_action = np.array([1, 0, 0, 0, 0], dtype=np.int32)

    print(f"Running benchmark: {num_envs} envs x {num_steps:,} steps")
    print("=" * 70)
    start_time = time.time()

    for step in range(num_steps):
        for i, env in enumerate(envs):
            actions = np.stack([pass_action, pass_action])
            _, states[i] = env.step(states[i], actions)

    elapsed = time.time() - start_time
    total_steps = num_steps * num_envs
    fps = total_steps / elapsed

    print("=" * 70)
    print(f"Total time:      {elapsed:.2f}s")
    print(f"Total steps:     {total_steps:,}")
    print(f"Average FPS:     {fps:,.0f} steps/second")


if __name__ == "__main__":
    import sys
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    benchmark(num_envs=num_envs, num_steps=num_steps)
