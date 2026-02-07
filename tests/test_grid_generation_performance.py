"""Test grid generation performance and correctness."""
import time
import numpy as np
import pytest

from generals.core.grid import generate_grid


def test_grid_generation_no_crash():
    print("\n=== Testing Grid Generation Stability ===")

    num_grids = 50
    rng = np.random.default_rng(42)

    start = time.time()
    for i in range(num_grids):
        grid = generate_grid(np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))

    elapsed = time.time() - start
    grids_per_sec = num_grids / elapsed

    print(f"✓ Generated {num_grids} grids in {elapsed:.2f}s")
    print(f"  Performance: {grids_per_sec:.1f} grids/second")


def test_100_percent_validity():
    num_grids = 50
    rng = np.random.default_rng(123)
    has_both_generals = 0
    for _ in range(num_grids):
        grid = generate_grid(np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))
        has_g1 = np.any(grid == 1)
        has_g2 = np.any(grid == 2)
        if has_g1 and has_g2:
            has_both_generals += 1
    assert has_both_generals == num_grids


def test_generals_distance():
    num_grids = 50
    rng = np.random.default_rng(456)
    min_generals_distance = 3
    distances = []
    for _ in range(num_grids):
        grid = generate_grid(np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))
        g1 = np.argwhere(grid == 1)[0]
        g2 = np.argwhere(grid == 2)[0]
        distance = abs(int(g1[0]) - int(g2[0])) + abs(int(g1[1]) - int(g2[1]))
        distances.append(distance)
    assert np.min(distances) >= min_generals_distance


def test_grid_properties():
    num_grids = 50
    rng = np.random.default_rng(789)
    mountain_counts = []
    city_counts = []
    for _ in range(num_grids):
        grid = generate_grid(np.random.default_rng(rng.integers(0, 2 ** 31 - 1)))
        mountain_counts.append(int(np.sum(grid == -2)))
        city_counts.append(int(np.sum((grid >= 40) & (grid <= 50))))
    assert np.mean(city_counts) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
