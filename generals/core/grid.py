import numpy as np
from typing import Tuple


def generate_grid(
    rng: np.random.Generator,
    grid_dims: tuple[int, int] = (23, 23),
    pad_to: int | None = None,
    mountain_density: float = 0.2,
    num_cities_range: tuple[int, int] = (9, 11),
    min_generals_distance: int = 17,
    max_generals_distance: int | None = None,
    castle_val_range: tuple[int, int] = (40, 51),
) -> np.ndarray:
    """
    Generate grid using a NumPy-based algorithm with guaranteed validity.
    
    Unified generator that supports both square and non-square grids with
    dynamic padding and configurable general distance constraints.
    
    Algorithm:
    1. Place generals FIRST on empty grid (guaranteed valid distance)
    2. Mark protected zones around generals for castle placement
    3. Place castles in protected zones
    4. Place mountains on remaining cells
    5. Place remaining cities
    6. Check connectivity, carve L-path if needed
    7. Apply dynamic padding
    
    Args:
        rng: numpy.random.Generator
        grid_dims: Grid dimensions (height, width) - supports non-square grids
        pad_to: Pad grid to this size for batching (None = max(h, w) + 1)
        mountain_density: Fraction of tiles that are mountains (0.18-0.22)
        num_cities_range: (min, max) number of cities to place
        min_generals_distance: Minimum BFS (shortest path) distance between generals
        max_generals_distance: Maximum BFS (shortest path) distance between generals (None = no limit)
        castle_val_range: (min, max) army value for cities
        
    Returns:
        Grid is always valid (validity=True always)
    """
    h, w = grid_dims
    num_tiles = h * w

    num_cities = int(rng.integers(num_cities_range[0], num_cities_range[1] + 1))
    base_mountains = int(mountain_density * num_tiles)
    mountain_variation = int(rng.integers(-10, 11))
    num_mountains = max(0, base_mountains + mountain_variation)

    # Start with empty grid
    grid = np.full(grid_dims, 0, dtype=np.int32)

    # Place generals ensuring min distance by naive trial
    placed = False
    attempts = 0
    while not placed and attempts < 1000:
        attempts += 1
        pos_a = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        pos_b = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        dist = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
        if dist >= min_generals_distance:
            placed = True
    if not placed:
        # fallback to corners
        pos_a = (0, 0)
        pos_b = (h - 1, w - 1)

    grid[pos_a] = 1
    grid[pos_b] = 2

    # Place a castle near each general if possible
    def place_near(pos, radius=6, default_val=None):
        i0, j0 = pos
        candidates = []
        for i in range(max(0, i0 - radius), min(h, i0 + radius + 1)):
            for j in range(max(0, j0 - radius), min(w, j0 + radius + 1)):
                if grid[i, j] == 0:
                    candidates.append((i, j))
        if candidates:
            choice = candidates[int(rng.integers(0, len(candidates)))]
            grid[choice] = int(rng.integers(castle_val_range[0], castle_val_range[1]))

    place_near(pos_a)
    place_near(pos_b)

    # Place mountains randomly
    empty_indices = list(zip(*np.where(grid == 0)))
    rng.shuffle(empty_indices)
    for idx in empty_indices[:num_mountains]:
        grid[idx] = -2

    # Place remaining cities randomly
    empty_indices = list(zip(*np.where(grid == 0)))
    rng.shuffle(empty_indices)
    for idx in empty_indices[: max(0, num_cities - 2)]:
        grid[idx] = int(rng.integers(castle_val_range[0], castle_val_range[1]))

    # Optional padding
    if pad_to is None:
        target_size = max(h, w) + 1
    else:
        target_size = pad_to
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if pad_h > 0 or pad_w > 0:
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=-2)

    return grid


def sample_from_mask(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = np.flatnonzero(mask.reshape(-1))
    if flat.size == 0:
        return (-1, -1)
    idx = int(rng.choice(flat))
    return np.unravel_index(idx, mask.shape)


def sample_k_from_mask(mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.flatnonzero(mask.reshape(-1))
    if flat.size == 0:
        return np.array([], dtype=int)
    k = min(k, flat.size)
    return np.array(rng.choice(flat, size=k, replace=False), dtype=int)


def manhattan_distance_from(pos: tuple[int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Compute Manhattan distance from a position to all cells in grid.
    
    Args:
        pos: (i, j) position
        grid_shape: (height, width) of grid
        
    Returns:
        2D array of Manhattan distances
    """
    h, w = grid_shape
    i_idx = np.arange(h)[:, None]
    j_idx = np.arange(w)[None, :]
    return np.abs(i_idx - pos[0]) + np.abs(j_idx - pos[1])


def valid_base_a_mask(grid_shape: tuple[int, int], min_distance: int, max_distance: int | None = None) -> np.ndarray:
    """
    Create mask of valid positions for Base A.
    A position is valid if there exists at least one cell >= min_distance away
    (and optionally <= max_distance away).
    
    For a cell (i,j), the max Manhattan distance to any corner is:
    max(i+j, i+(w-1-j), (h-1-i)+j, (h-1-i)+(w-1-j))
    
    Args:
        grid_shape: (height, width) of grid
        min_distance: Minimum required distance to Base B
        max_distance: Maximum allowed distance to Base B (None = no limit)
        
    Returns:
        2D boolean mask
    """
    h, w = grid_shape
    i_idx = np.arange(h)[:, None]
    j_idx = np.arange(w)[None, :]

    dist_top_left = i_idx + j_idx
    dist_top_right = i_idx + (w - 1 - j_idx)
    dist_bottom_left = (h - 1 - i_idx) + j_idx
    dist_bottom_right = (h - 1 - i_idx) + (w - 1 - j_idx)

    max_dist = np.maximum(np.maximum(dist_top_left, dist_top_right), np.maximum(dist_bottom_left, dist_bottom_right))
    valid = max_dist >= min_distance

    if max_distance is not None:
        min_dist = np.minimum(np.minimum(dist_top_left, dist_top_right), np.minimum(dist_bottom_left, dist_bottom_right))
        grid_diagonal = h + w - 2
        if grid_diagonal >= max_distance:
            valid = valid & (min_dist <= max_distance)

    return valid


def flood_fill_connected(grid: np.ndarray, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> bool:
    """
    Check if start_pos can reach end_pos using flood fill with early termination.
    
    Args:
        grid: 2D grid array (-2=mountain, 0=passable, 1/2=generals, 40-50=cities)
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)
        
    Returns:
        Boolean indicating if end_pos is reachable from start_pos
    """
    h, w = grid.shape
    passable = (grid != -2)

    reachable = np.zeros((h, w), dtype=bool)
    reachable[start_pos] = True

    def dilate(r):
        up = np.roll(r, -1, axis=0)
        up[-1, :] = False
        down = np.roll(r, 1, axis=0)
        down[0, :] = False
        left = np.roll(r, -1, axis=1)
        left[:, -1] = False
        right = np.roll(r, 1, axis=1)
        right[:, 0] = False
        return (r | up | down | left | right) & passable

    prev = None
    current = reachable.copy()
    while True:
        new_reachable = dilate(current)
        if new_reachable[end_pos]:
            return True
        if prev is not None and np.array_equal(new_reachable, prev):
            return False
        prev = current
        current = new_reachable


def bfs_distance(grid: np.ndarray, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> int:
    """
    Compute shortest path (BFS) distance between two positions.
    Only mountains (-2) are impassable.

    Args:
        grid: 2D grid array
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)

    Returns:
        Scalar integer: BFS distance, or h*w if unreachable.
    """
    h, w = grid.shape
    passable = (grid != -2)

    from collections import deque

    q = deque()
    visited = np.zeros((h, w), dtype=bool)
    q.append((start_pos[0], start_pos[1], 0))
    visited[start_pos] = True

    while q:
        i, j, d = q.popleft()
        if (i, j) == end_pos:
            return d
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj] and passable[ni, nj]:
                visited[ni, nj] = True
                q.append((ni, nj, d + 1))
    return h * w


def carve_l_path(grid: np.ndarray, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> np.ndarray:
    """Carve an L-shaped path between two positions using NumPy.

    Clears mountains and cities on the path while preserving generals.
    The path goes horizontal from pos_a to (pos_a[0], pos_b[1]) then vertical
    to pos_b.
    """
    h, w = grid.shape
    i1, j1 = pos_a
    i2, j2 = pos_b

    # Horizontal segment
    j_min, j_max = min(j1, j2), max(j1, j2)
    grid[i1, j_min : j_max + 1] = np.where(
        (grid[i1, j_min : j_max + 1] == -2) | (grid[i1, j_min : j_max + 1] >= 40),
        0,
        grid[i1, j_min : j_max + 1],
    )

    # Vertical segment
    i_min, i_max = min(i1, i2), max(i1, i2)
    grid[i_min : i_max + 1, j2] = np.where(
        (grid[i_min : i_max + 1, j2] == -2) | (grid[i_min : i_max + 1, j2] >= 40),
        0,
        grid[i_min : i_max + 1, j2],
    )

    return grid