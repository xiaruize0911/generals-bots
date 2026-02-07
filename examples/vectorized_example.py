"""Simple batched example using NumPy RNG to create multiple independent envs."""
import numpy as np

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent


ENV_COUNT = 8
env = GeneralsEnv(grid_dims=(10, 10), truncation=200)

# Create independent starting states
states = [env.init_state(seed=i) for i in range(ENV_COUNT)]

agents = [RandomAgent(), ExpanderAgent()]

for t in range(100):
    actions = []
    for s in states:
        obs0 = get_observation(s, 0)
        obs1 = get_observation(s, 1)
        a0 = agents[0].act(obs0, np.random.default_rng())
        a1 = agents[1].act(obs1, np.random.default_rng())
        actions.append(np.stack([a0, a1]))

    # Step all envs sequentially
    new_states = []
    for s, a in zip(states, actions):
        _, ns = env.step(s, a)
        new_states.append(ns)
    states = new_states
