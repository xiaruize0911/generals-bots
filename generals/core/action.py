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

# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)


def create_action(
    to_pass: bool = False, row: int = 0, col: int = 0, direction: int = 0, to_split: bool = False
) -> np.ndarray:
    """Create action array [pass, row, col, direction, split]."""
    return np.array([int(to_pass), int(row), int(col), int(direction), int(to_split)], dtype=np.int32)


def compute_valid_move_mask(
    armies: np.ndarray, owned_cells: np.ndarray, mountains: np.ndarray
) -> np.ndarray:
    """Compute valid move mask (H, W, 4) using NumPy.

    Returns a boolean array where True indicates a valid move from (i, j) in
    the given direction.
    """
    H, W = armies.shape

    can_move_from = owned_cells & (armies > 1)
    passable = ~mountains

    i_idx = np.arange(H)[:, None]
    j_idx = np.arange(W)[None, :]

    dest_i = i_idx[:, :, None] + DIRECTIONS[None, None, :, 0]
    dest_j = j_idx[:, :, None] + DIRECTIONS[None, None, :, 1]

    in_bounds = (dest_i >= 0) & (dest_i < H) & (dest_j >= 0) & (dest_j < W)

    safe_dest_i = np.clip(dest_i, 0, H - 1)
    safe_dest_j = np.clip(dest_j, 0, W - 1)

    dest_passable = passable[safe_dest_i, safe_dest_j]

    valid_mask = can_move_from[:, :, None] & in_bounds & dest_passable
    return valid_mask


def compute_valid_move_mask_obs(observation) -> np.ndarray:
    """Compute valid move mask from an Observation object (numpy arrays)."""
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
