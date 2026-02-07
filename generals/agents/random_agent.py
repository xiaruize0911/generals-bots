import numpy as np

from typing import Tuple

from generals.core.action import compute_valid_move_mask
from generals.core.observation import Observation

from .agent import Agent


class RandomAgent(Agent):
    """Agent that selects random valid actions using numpy RNG."""

    def __init__(self, id: str = "Random", split_prob: float = 0.25, idle_prob: float = 0.05):
        super().__init__(id)
        self.idle_prob = idle_prob
        self.split_prob = split_prob

    def act(self, observation: Observation, rng: np.random.Generator) -> np.ndarray:
        mask = compute_valid_move_mask(observation.armies, observation.owned_cells, observation.mountains)
        valid = np.argwhere(mask)
        num_valid = int(np.sum(np.all(valid >= 0, axis=-1))) if valid.size else 0

        # Pass if no valid moves or randomly with idle_prob
        should_pass = (num_valid == 0) or (rng.random() < self.idle_prob)

        # Select random valid move
        if num_valid == 0:
            move = np.array([-1, -1, -1], dtype=int)
        else:
            idx = rng.integers(0, max(1, num_valid))
            move = valid[min(idx, num_valid - 1)]

        # Random split decision
        split = int(rng.random() < self.split_prob)

        return np.array([int(should_pass), int(move[0]), int(move[1]), int(move[2]), int(split)], dtype=np.int32)

    def reset(self):
        pass
