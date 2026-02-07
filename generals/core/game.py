"""
NumPy game logic for Generals.io.

This module contains the core game mechanics implemented imperatively
with NumPy so that the project no longer depends on external libraries. The functions
mirror the previous behavior but use standard NumPy arrays and python
control flow.
"""
from typing import Tuple, NamedTuple, Protocol, Any

import numpy as np

from generals.core.observation import Observation


class Game(Protocol):
    """Protocol for game objects used by the GUI."""
    agents: list[str]
    channels: Any
    grid_dims: tuple[int, int]
    general_positions: dict[str, Any]
    time: int

    def get_infos(self) -> dict[str, dict[str, Any]]:
        """Return player stats."""
        ...


class GameState(NamedTuple):
    armies: np.ndarray
    ownership: np.ndarray
    ownership_neutral: np.ndarray
    generals: np.ndarray
    cities: np.ndarray
    mountains: np.ndarray
    passable: np.ndarray
    general_positions: np.ndarray
    time: np.int32
    winner: np.int32
    pool_idx: np.int32


class GameInfo(NamedTuple):
    army: np.ndarray
    land: np.ndarray
    is_done: bool
    winner: int
    time: int


# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)


def create_initial_state(grid: np.ndarray) -> GameState:
    H, W = grid.shape

    is_general_0 = grid == 1
    is_general_1 = grid == 2
    generals = is_general_0 | is_general_1

    mountains = grid == -2
    passable = grid != -2
    cities = (grid >= 40) & (grid <= 50)

    ownership = np.stack([is_general_0, is_general_1])
    ownership_neutral = passable & ~is_general_0 & ~is_general_1

    armies = np.where(is_general_0 | is_general_1, 1, 0).astype(np.int32)
    armies = np.where(cities, grid, armies)

    general_pos_0 = np.argwhere(is_general_0)
    general_pos_1 = np.argwhere(is_general_1)
    gp0 = general_pos_0[0] if general_pos_0.size else np.array([-1, -1])
    gp1 = general_pos_1[0] if general_pos_1.size else np.array([-1, -1])
    general_positions = np.stack([gp0, gp1])

    return GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        cities=cities,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        time=np.int32(0),
        winner=np.int32(-1),
        pool_idx=np.int32(0),
    )


def get_visibility(ownership: np.ndarray) -> np.ndarray:
    H, W = ownership.shape
    ownership_float = ownership.astype(np.float32)
    padded = np.pad(ownership_float, 1, mode="constant", constant_values=0)

    stacked = np.stack(
        [
            padded[0:H, 0:W],
            padded[0:H, 1 : W + 1],
            padded[0:H, 2 : W + 2],
            padded[1 : H + 1, 0:W],
            padded[1 : H + 1, 1 : W + 1],
            padded[1 : H + 1, 2 : W + 2],
            padded[2 : H + 2, 0:W],
            padded[2 : H + 2, 1 : W + 1],
            padded[2 : H + 2, 2 : W + 2],
        ],
        axis=0,
    )

    return np.max(stacked, axis=0) > 0


def execute_action(state: GameState, player_idx: int, action: np.ndarray) -> GameState:
    pass_turn, si, sj, direction, split_army = action
    if int(pass_turn) == 1:
        return state
    return _execute_move(state, int(player_idx), int(si), int(sj), int(direction), int(split_army))


def _execute_move(state: GameState, player_idx: int, si: int, sj: int, direction: int, split_army: int) -> GameState:
    H, W = state.armies.shape

    in_bounds = (si >= 0) and (si < H) and (sj >= 0) and (sj < W)

    di = si + int(DIRECTIONS[direction, 0])
    dj = sj + int(DIRECTIONS[direction, 1])
    dest_in_bounds = (di >= 0) and (di < H) and (dj >= 0) and (dj < W)

    owns_source = bool(state.ownership[player_idx, si, sj])
    source_army = int(state.armies[si, sj])

    if split_army == 1:
        army_to_move = source_army // 2
    else:
        army_to_move = source_army - 1
    army_to_move = max(0, min(army_to_move, source_army - 1))

    valid_move = in_bounds and dest_in_bounds and owns_source and (army_to_move > 0) and bool(state.passable[di, dj])

    if valid_move:
        return _apply_move(state, player_idx, si, sj, di, dj, army_to_move)
    return state


def _apply_move(state: GameState, player_idx: int, si: int, sj: int, di: int, dj: int, army_to_move: int) -> GameState:
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()

    target_owner_0 = bool(ownership[0, di, dj])
    target_owner_1 = bool(ownership[1, di, dj])
    target_neutral = bool(ownership_neutral[di, dj])

    moving_to_own = (player_idx == 0 and target_owner_0) or (player_idx == 1 and target_owner_1)

    if moving_to_own:
        armies[di, dj] = armies[di, dj] + army_to_move
        armies[si, sj] = armies[si, sj] - army_to_move
    else:
        target_army = armies[di, dj]
        attacker_wins = army_to_move > target_army
        remaining_army = abs(int(target_army) - army_to_move)

        armies[di, dj] = remaining_army
        armies[si, sj] = armies[si, sj] - army_to_move

        if attacker_wins:
            ownership[player_idx, di, dj] = True
            # remove other player's ownership if present
            ownership[1 - player_idx, di, dj] = False
            if target_neutral:
                ownership_neutral[di, dj] = False

    is_general = bool(state.generals[di, dj])
    general_captured = (not moving_to_own) and (armies[di, dj] >= 0) and is_general and (army_to_move > 0)

    winner = state.winner
    if general_captured:
        winner = np.int32(player_idx)

    return state._replace(armies=armies, ownership=ownership, ownership_neutral=ownership_neutral, winner=winner)


def global_update(state: GameState) -> GameState:
    time = int(state.time)
    armies = state.armies.copy()

    if time % 50 == 0:
        armies = armies + state.ownership[0].astype(np.int32) + state.ownership[1].astype(np.int32)

    if (time % 2 == 0) and (time > 0):
        structure_mask = (state.generals | state.cities).astype(np.int32)
        armies = armies + structure_mask * state.ownership[0].astype(np.int32) + structure_mask * state.ownership[1].astype(np.int32)

    return state._replace(armies=armies)


def _determine_move_order(state: GameState, actions: np.ndarray) -> int:
    pass_0, row_0, col_0, dir_0, _ = actions[0]
    pass_1, row_1, col_1, dir_1, _ = actions[1]

    only_p0_passes = (pass_0 == 1) and (pass_1 != 1)

    si_0, sj_0 = int(row_0), int(col_0)
    di_0, dj_0 = si_0 + int(DIRECTIONS[int(dir_0), 0]), sj_0 + int(DIRECTIONS[int(dir_0), 1])

    si_1, sj_1 = int(row_1), int(col_1)
    di_1, dj_1 = si_1 + int(DIRECTIONS[int(dir_1), 0]), sj_1 + int(DIRECTIONS[int(dir_1), 1])

    p0_chasing = (di_0 == si_1) and (dj_0 == sj_1)
    p1_chasing = (di_1 == si_0) and (dj_1 == sj_0)

    p0_reinforcing = bool(state.ownership[0, di_0, dj_0]) if 0 <= di_0 < state.armies.shape[0] and 0 <= dj_0 < state.armies.shape[1] else False
    p1_reinforcing = bool(state.ownership[1, di_1, dj_1]) if 0 <= di_1 < state.armies.shape[0] and 0 <= dj_1 < state.armies.shape[1] else False

    army_0 = int(state.armies[si_0, sj_0]) if 0 <= si_0 < state.armies.shape[0] and 0 <= sj_0 < state.armies.shape[1] else 0
    army_1 = int(state.armies[si_1, sj_1]) if 0 <= si_1 < state.armies.shape[0] and 0 <= sj_1 < state.armies.shape[1] else 0

    p1_wins_by_chase = p1_chasing and (not p0_chasing)
    tie_on_chase = p0_chasing == p1_chasing
    p1_wins_by_reinforce = tie_on_chase and p1_reinforcing and (not p0_reinforcing)
    tie_on_reinforce = p0_reinforcing == p1_reinforcing
    p1_wins_by_army = tie_on_chase and tie_on_reinforce and (army_1 > army_0)

    p1_goes_first = p1_wins_by_chase or p1_wins_by_reinforce or p1_wins_by_army or only_p0_passes

    return 1 if p1_goes_first else 0


def step(state: GameState, actions: np.ndarray) -> tuple[GameState, GameInfo]:
    done_before = int(state.winner) >= 0

    first_player = _determine_move_order(state, actions)
    second_player = 1 - first_player

    state = execute_action(state, first_player, actions[first_player])
    state = execute_action(state, second_player, actions[second_player])

    if not done_before:
        state = state._replace(time=np.int32(int(state.time) + 1))

    if int(state.winner) >= 0:
        state = _transfer_loser_cells_to_winner(state)
    else:
        state = global_update(state)

    return state, get_info(state)


def _transfer_loser_cells_to_winner(state: GameState) -> GameState:
    winner_idx = int(state.winner)
    loser_idx = 1 - winner_idx

    ownership = state.ownership.copy()
    ownership[winner_idx] = ownership[winner_idx] | ownership[loser_idx]
    ownership[loser_idx] = np.zeros_like(ownership[loser_idx], dtype=bool)
    new_ownership_neutral = state.ownership_neutral & ~ownership[loser_idx]

    return state._replace(ownership=ownership, ownership_neutral=new_ownership_neutral)


def get_info(state: GameState) -> GameInfo:
    armies = state.armies
    ownership = state.ownership

    army = np.array([int(np.sum(armies * ownership[0])), int(np.sum(armies * ownership[1]))], dtype=np.int32)
    land = np.array([int(np.sum(ownership[0])), int(np.sum(ownership[1]))], dtype=np.int32)

    return GameInfo(
        army=army,
        land=land,
        is_done=int(state.winner) >= 0,
        winner=int(state.winner),
        time=int(state.time),
    )


def get_observation(state: GameState, player_idx: int) -> Observation:
    visible = get_visibility(state.ownership[player_idx])
    invisible = ~visible
    opponent_idx = 1 - player_idx

    info = get_info(state)

    return Observation(
        armies=state.armies * visible,
        generals=state.generals * visible,
        cities=state.cities * visible,
        mountains=state.mountains * visible,
        neutral_cells=state.ownership_neutral * visible,
        owned_cells=state.ownership[player_idx] * visible,
        opponent_cells=state.ownership[opponent_idx] * visible,
        fog_cells=invisible & ~(state.mountains | state.cities),
        structures_in_fog=invisible & (state.mountains | state.cities),
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=info.land[opponent_idx],
        opponent_army_count=info.army[opponent_idx],
        timestep=np.int32(state.time),
    )


def batch_step(states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, list[GameInfo]]:
    """Vectorized step for multiple environments (simple Python loop)."""
    new_states = []
    infos = []
    for s, a in zip(states, actions):
        ns, info = step(s, a)
        new_states.append(ns)
        infos.append(info)
    return np.array(new_states, dtype=object), infos
