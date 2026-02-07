"""
NumPy-based Generals.io environment.

This environment uses the NumPy game primitives and a numpy.random.Generator
for randomness. It pre-generates a pool of states for cheap auto-resets.
"""
from typing import NamedTuple

import numpy as np

from generals.core import game
from generals.core.game import GameInfo, GameState, create_initial_state
from generals.core.game import step as game_step
from generals.core.grid import generate_grid
from generals.core.observation import Observation


class TimeStep(NamedTuple):
    observation: Observation
    reward: np.ndarray
    terminated: bool
    truncated: bool
    info: GameInfo


class GeneralsEnv:
    def __init__(
        self,
        grid_dims: tuple[int, int] = (4, 4),
        truncation: int = 500,
        mountain_density: float = 0.15,
        num_cities_range: tuple[int, int] = (0, 2),
        min_generals_distance: int = 3,
        max_generals_distance: int | None = None,
        pool_size: int = 1000,
    ):
        self.grid_dims = grid_dims
        self.truncation = truncation
        self.mountain_density = mountain_density
        self.num_cities_range = num_cities_range
        self.min_generals_distance = min_generals_distance
        self.max_generals_distance = max_generals_distance
        self.pool_size = pool_size
        self._pool: list[GameState] | None = None

    def _make_single_state(self, rng: np.random.Generator) -> GameState:
        grid = generate_grid(
            rng,
            grid_dims=self.grid_dims,
            pad_to=max(self.grid_dims),
            mountain_density=self.mountain_density,
            num_cities_range=self.num_cities_range,
            min_generals_distance=self.min_generals_distance,
            max_generals_distance=self.max_generals_distance,
            castle_val_range=(40, 51),
        )
        return create_initial_state(grid.astype(np.int32))

    def reset(self, seed: int | None = None) -> GameState:
        rng = np.random.default_rng(seed)
        pool = [self._make_single_state(np.random.default_rng(rng.integers(0, 2 ** 31 - 1))) for _ in range(self.pool_size)]
        self._pool = pool
        return self._make_single_state(np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))

    def init_state(self, seed: int | None = None) -> GameState:
        rng = np.random.default_rng(seed)
        return self._make_single_state(rng)

    def step(self, state: GameState, actions: np.ndarray) -> tuple[TimeStep, GameState]:
        assert self._pool is not None, "Call env.reset(seed) before env.step() to generate the state pool."

        new_state, info = game_step(state, actions)

        reward_p0 = 1.0 if info.winner == 0 else (-1.0 if info.winner == 1 else 0.0)
        rewards = np.array([reward_p0, -reward_p0], dtype=float)

        terminated = bool(info.is_done)
        truncated = (int(new_state.time) >= self.truncation) and (not terminated)
        should_reset = terminated or truncated

        pool_idx = int(new_state.pool_idx) if hasattr(new_state, 'pool_idx') else 0
        if should_reset:
            reset_state = self._pool[pool_idx % self.pool_size]
            new_pool_idx = pool_idx + 1
            # update pool_idx on reset_state and new_state if field exists
            try:
                reset_state = reset_state._replace(pool_idx=np.int32(new_pool_idx))
                new_state = new_state._replace(pool_idx=np.int32(new_pool_idx))
            except Exception:
                pass
        final_state = new_state

        obs_p0 = game.get_observation(final_state, 0)
        obs_p1 = game.get_observation(final_state, 1)
        observation = np.stack([obs_p0, obs_p1], axis=0)

        timestep = TimeStep(
            observation=observation,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        return timestep, final_state
