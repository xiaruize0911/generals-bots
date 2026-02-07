"""
Action utilities for the NumPy-based game core.

Actions are 5-element integer arrays: [pass, row, col, direction, split]
 - pass: 1 to skip turn, 0 to move
 - row, col: Source cell coordinates
 - direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
 - split: 1 to move half army, 0 to move all-but-one
"""
from __future__ import annotations


import numpy as np
from typing import Tuple
from numba import njit

# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)


def create_action(
    to_pass: bool = False, row: int = 0, col: int = 0, direction: int = 0, to_split: bool = False
) -> np.ndarray:
    """Create action array [pass, row, col, direction, split]."""
    return np.array([int(to_pass), int(row), int(col), int(direction), int(to_split)], dtype=np.int32)



@njit(cache=True, fastmath=True)
def compute_valid_move_mask(
    armies, owned_cells, mountains
) -> np.ndarray:
    H, W = armies.shape
    valid_mask = np.zeros((H, W, 4), dtype=np.bool_)
    for i in range(H):
        for j in range(W):
            if owned_cells[i, j] and armies[i, j] > 1:
                for d in range(4):
                    di, dj = DIRECTIONS[d]
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if not mountains[ni, nj]:
                            valid_mask[i, j, d] = True
    return valid_mask



@njit(cache=True, fastmath=True)
def compute_valid_move_mask_obs(observation) -> np.ndarray:
    return compute_valid_move_mask(observation.armies, observation.owned_cells, observation.mountains)


def sample_valid_action(rng: np.random.Generator, observation, allow_pass: bool = True) -> np.ndarray:
    """Sample a random valid action from observation using numpy RNG."""
    valid_mask = compute_valid_move_mask_obs(observation)
    H, W = observation.armies.shape

    positions = np.argwhere(valid_mask)
    num_valid = positions.shape[0]

    should_pass = False
    if allow_pass and rng.random() < 0.1:
        should_pass = True
    if num_valid == 0:
        should_pass = True

    if should_pass:
        return create_action(True, 0, 0, 0, False)

    idx = int(rng.integers(0, num_valid))
    move = positions[idx]
    split = int(rng.integers(0, 2))

    return np.array([0, int(move[0]), int(move[1]), int(move[2]), split], dtype=np.int32)
