"""Tests for NumPy-based game implementation."""
import numpy as np
import pytest

from generals.core import game


def create_test_grid(size=4):
    grid = np.zeros((size, size), dtype=np.int32)
    grid[0, 0] = 1
    grid[size - 1, size - 1] = 2
    return grid


def test_create_initial_state():
    grid = create_test_grid(4)
    state = game.create_initial_state(grid)

    assert hasattr(state, "armies")
    assert hasattr(state, "ownership")
    assert hasattr(state, "generals")

    assert state.armies[0, 0] == 1
    assert state.armies[3, 3] == 1
    assert state.ownership[0, 0, 0] == True
    assert state.ownership[1, 3, 3] == True
    assert int(state.time) == 0
    assert int(state.winner) == -1


def test_step_pass_action():
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)

    actions = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.int32)
    new_state, info = game.step(state, actions)

    assert np.array_equal(new_state.armies, state.armies)
    assert int(new_state.time) == 1
    assert int(new_state.winner) == -1


def test_step_move_to_neutral():
    grid = create_test_grid(3)
    state = game.create_initial_state(grid)

    armies = state.armies.copy()
    armies[0, 0] = 5
    state = state._replace(armies=armies)

    actions = np.array([[0, 0, 0, 3, 0], [1, 0, 0, 0, 0]], dtype=np.int32)
    new_state, info = game.step(state, actions)

    assert new_state.armies[0, 0] == 1
    assert new_state.armies[0, 1] == 4
    assert new_state.ownership[0, 0, 1] == True


def test_step_move_to_own_cell():
    grid = create_test_grid(3)
    state = game.create_initial_state(grid)

    armies = state.armies.copy()
    armies[0, 0] = 5
    armies[0, 1] = 3
    ownership = state.ownership.copy()
    ownership[0, 0, 1] = True
    ownership_neutral = state.ownership_neutral.copy()
    ownership_neutral[0, 1] = False

    state = state._replace(armies=armies, ownership=ownership, ownership_neutral=ownership_neutral)

    actions = np.array([[0, 0, 0, 3, 0], [1, 0, 0, 0, 0]], dtype=np.int32)
    new_state, info = game.step(state, actions)

    assert new_state.armies[0, 0] == 1
    assert new_state.armies[0, 1] == 7


def test_get_observation():
    grid = create_test_grid(4)
    state = game.create_initial_state(grid)

    obs = game.get_observation(state, 0)
    assert hasattr(obs, "armies")
    assert hasattr(obs, "owned_cells")
    assert hasattr(obs, "fog_cells")
    assert hasattr(obs, "timestep")
    assert obs.armies[0, 0] == 1
    assert obs.armies[3, 3] == 0


def test_global_update():
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)
    armies = state.armies.copy()
    armies[0, 0] = 5
    state = state._replace(armies=armies, time=np.int32(2))
    state = game.global_update(state)

    assert state.armies[0, 0] == 6


def test_batch_step():
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)

    # Simple batching via list of states
    batched = [state, state]
    actions = [np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.int32) for _ in batched]

    new_states, infos = game.batch_step(batched, actions)

    assert len(new_states) == 2
    assert hasattr(new_states[0], "armies")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
