"""Expander agent logic using NumPy and numpy RNG (internal module)."""
import numpy as np

from generals.core.action import compute_valid_move_mask_obs, DIRECTIONS


def expander_action(rng: np.random.Generator, observation) -> np.ndarray:
    """
    Expander agent that prioritizes border expansion with strongest cells.

    Strategy:
    1. Find all cells on the border (owned cells adjacent to non-owned)
    2. Score moves by: army_count * can_capture * (opponent_bonus)
    3. Prefer moving strongest border cells to capture new territory
    """
    valid_mask = compute_valid_move_mask_obs(observation)
    H, W = observation.armies.shape

    valid_positions = np.argwhere(valid_mask)
    num_valid = valid_positions.shape[0]
    if num_valid == 0:
        return np.array([1, -1, -1, -1, 0], dtype=np.int32)

    def score_move(move):
        orig_row, orig_col, direction = int(move[0]), int(move[1]), int(move[2])
        di, dj = DIRECTIONS[direction]
        dest_row = np.clip(orig_row + di, 0, H - 1)
        dest_col = np.clip(orig_col + dj, 0, W - 1)

        orig_armies = observation.armies[orig_row, orig_col]
        dest_armies = observation.armies[dest_row, dest_col]
        is_opponent = bool(observation.opponent_cells[dest_row, dest_col])
        is_neutral = bool(observation.neutral_cells[dest_row, dest_col])
        is_owned = bool(observation.owned_cells[dest_row, dest_col])

        can_capture = orig_armies > dest_armies + 1
        is_expansion = (not is_owned) and (is_opponent or is_neutral)

        score = float(orig_armies)
        if is_expansion and can_capture:
            score *= 10.0
            if is_opponent:
                score *= 2.0
        if not (can_capture):
            score = 0.0
        return score

    scores = np.array([score_move(m) for m in valid_positions], dtype=float)
    probs = scores.copy()
    if probs.sum() <= 1e-8:
        # no strong expansion captures, fallback to uniform over valid moves
        probs = np.ones_like(probs)
    probs = probs / (probs.sum() + 1e-8)

    idx = rng.choice(len(valid_positions), p=probs)
    selected_move = valid_positions[idx]

    return np.array([0, int(selected_move[0]), int(selected_move[1]), int(selected_move[2]), 0], dtype=np.int32)

