"""
NumPy game logic for Generals.io.

This module contains the core game mechanics implemented imperatively
with NumPy so that the project no longer depends on external libraries. The functions
mirror the previous behavior but use standard NumPy arrays and python
control flow.
"""
from typing import Tuple, NamedTuple, Protocol, Any

import numpy as np
from numba import njit, prange

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


@njit(cache=True)
def jit_get_visibility(ownership):
    H, W = ownership.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.bool_)
    padded[1 : H + 1, 1 : W + 1] = ownership

    visible = np.zeros((H, W), dtype=np.bool_)
    for i in range(H):
        for j in range(W):
            # Check 3x3 neighborhood
            for di in range(3):
                for dj in range(3):
                    if padded[i + di, j + dj]:
                        visible[i, j] = True
                        break
                if visible[i, j]:
                    break
    return visible


def get_visibility(ownership: np.ndarray) -> np.ndarray:
    return jit_get_visibility(ownership)


@njit(cache=True)
def jit_apply_move(player_idx, si, sj, di, dj, army_to_move, armies, ownership, ownership_neutral, generals, winner):
    target_owner_0 = ownership[0, di, dj]
    target_owner_1 = ownership[1, di, dj]
    target_neutral = ownership_neutral[di, dj]

    moving_to_own = (player_idx == 0 and target_owner_0) or (player_idx == 1 and target_owner_1)

    if moving_to_own:
        armies[di, dj] += army_to_move
        armies[si, sj] -= army_to_move
    else:
        target_army = armies[di, dj]
        attacker_wins = army_to_move > target_army
        remaining_army = abs(target_army - army_to_move)

        armies[di, dj] = remaining_army
        armies[si, sj] -= army_to_move

        if attacker_wins:
            ownership[player_idx, di, dj] = True
            ownership[1 - player_idx, di, dj] = False
            if target_neutral:
                ownership_neutral[di, dj] = False
            
            # Check for general capture only if the attacker actually takes the cell
            if generals[di, dj]:
                winner = player_idx

    return winner


@njit(cache=True)
def jit_execute_move(player_idx, si, sj, direction, split_army, armies, ownership, ownership_neutral, generals, passable, winner):
    H, W = armies.shape
    in_bounds = (si >= 0) and (si < H) and (sj >= 0) and (sj < W)

    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) and (di < H) and (dj >= 0) and (dj < W)

    owns_source = ownership[player_idx, si, sj]
    source_army = armies[si, sj]

    if split_army == 1:
        army_to_move = source_army // 2
    else:
        army_to_move = source_army - 1
    army_to_move = max(0, min(army_to_move, source_army - 1))

    valid_move = in_bounds and dest_in_bounds and owns_source and (army_to_move > 0) and passable[di, dj]

    if valid_move:
        winner = jit_apply_move(player_idx, si, sj, di, dj, army_to_move, armies, ownership, ownership_neutral, generals, winner)
    return winner


def execute_action(state: GameState, player_idx: int, action: np.ndarray) -> GameState:
    pass_turn, si, sj, direction, split_army = action
    if int(pass_turn) == 1:
        return state

    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    winner = int(state.winner)

    winner = jit_execute_move(
        int(player_idx), int(si), int(sj), int(direction), int(split_army),
        armies, ownership, ownership_neutral, state.generals, state.passable, winner
    )

    return state._replace(armies=armies, ownership=ownership, ownership_neutral=ownership_neutral, winner=np.int32(winner))


# Placeholder to remove old internal functions that were replaced by jit
# _execute_move and _apply_move are no longer needed


@njit(cache=True)
def jit_global_update(time, armies, ownership_0, ownership_1, generals, cities):
    if time % 50 == 0:
        armies += ownership_0.astype(np.int32) + ownership_1.astype(np.int32)

    if (time % 2 == 0) and (time > 0):
        structure_mask = (generals | cities).astype(np.int32)
        armies += structure_mask * ownership_0.astype(np.int32) + structure_mask * ownership_1.astype(np.int32)
    return armies


def global_update(state: GameState) -> GameState:
    armies = jit_global_update(
        int(state.time),
        state.armies.copy(),
        state.ownership[0],
        state.ownership[1],
        state.generals,
        state.cities
    )
    return state._replace(armies=armies)


@njit(cache=True)
def jit_determine_move_order(armies, ownership, actions):
    pass_0, row_0, col_0, dir_0, _ = actions[0]
    pass_1, row_1, col_1, dir_1, _ = actions[1]

    only_p0_passes = (pass_0 == 1) and (pass_1 != 1)

    si_0, sj_0 = int(row_0), int(col_0)
    di_0, dj_0 = si_0 + int(DIRECTIONS[int(dir_0), 0]), sj_0 + int(DIRECTIONS[int(dir_0), 1])

    si_1, sj_1 = int(row_1), int(col_1)
    di_1, dj_1 = si_1 + int(DIRECTIONS[int(dir_1), 0]), sj_1 + int(DIRECTIONS[int(dir_1), 1])

    H, W = armies.shape

    p0_chasing = (di_0 == si_1) and (dj_0 == sj_1)
    p1_chasing = (di_1 == si_0) and (dj_1 == sj_0)

    p0_reinforcing = ownership[0, di_0, dj_0] if 0 <= di_0 < H and 0 <= dj_0 < W else False
    p1_reinforcing = ownership[1, di_1, dj_1] if 0 <= di_1 < H and 0 <= dj_1 < W else False

    army_0 = armies[si_0, sj_0] if 0 <= si_0 < H and 0 <= sj_0 < W else 0
    army_1 = armies[si_1, sj_1] if 0 <= si_1 < H and 0 <= sj_1 < W else 0

    p1_wins_by_chase = p1_chasing and (not p0_chasing)
    tie_on_chase = p0_chasing == p1_chasing
    p1_wins_by_reinforce = tie_on_chase and p1_reinforcing and (not p0_reinforcing)
    tie_on_reinforce = p0_reinforcing == p1_reinforcing
    p1_wins_by_army = tie_on_chase and tie_on_reinforce and (army_1 > army_0)

    p1_goes_first = p1_wins_by_chase or p1_wins_by_reinforce or p1_wins_by_army or only_p0_passes

    return 1 if p1_goes_first else 0


def _determine_move_order(state: GameState, actions: np.ndarray) -> int:
    return jit_determine_move_order(state.armies, state.ownership, actions)


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


@njit(cache=True)
def jit_transfer_loser_cells_to_winner(winner_idx, ownership, ownership_neutral):
    loser_idx = 1 - winner_idx
    ownership[winner_idx] |= ownership[loser_idx]
    ownership[loser_idx][:] = False
    new_ownership_neutral = ownership_neutral & ~ownership[winner_idx]
    return ownership, new_ownership_neutral


def _transfer_loser_cells_to_winner(state: GameState) -> GameState:
    winner_idx = int(state.winner)
    ownership, new_ownership_neutral = jit_transfer_loser_cells_to_winner(
        winner_idx, state.ownership.copy(), state.ownership_neutral.copy()
    )
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


@njit(parallel=True)
def jit_batch_step(
    armies_batch,
    ownership_batch,
    ownership_neutral_batch,
    generals_batch,
    cities_batch,
    mountains_batch,
    passable_batch,
    time_batch,
    winner_batch,
    actions_batch
):
    num_envs = armies_batch.shape[0]
    for i in prange(num_envs):
        armies = armies_batch[i]
        ownership = ownership_batch[i]
        ownership_neutral = ownership_neutral_batch[i]
        generals = generals_batch[i]
        cities = cities_batch[i]
        passable = passable_batch[i]
        actions = actions_batch[i]
        
        done_before = winner_batch[i] >= 0
        
        first_player = jit_determine_move_order(armies, ownership, actions)
        second_player = 1 - first_player
        
        # First player
        act_f = actions[first_player]
        if act_f[0] == 0:
            winner_batch[i] = jit_execute_move(
                first_player, int(act_f[1]), int(act_f[2]), int(act_f[3]), int(act_f[4]),
                armies, ownership, ownership_neutral, generals, passable, int(winner_batch[i])
            )
            
        # Second player
        act_s = actions[second_player]
        if act_s[0] == 0:
            winner_batch[i] = jit_execute_move(
                second_player, int(act_s[1]), int(act_s[2]), int(act_s[3]), int(act_s[4]),
                armies, ownership, ownership_neutral, generals, passable, int(winner_batch[i])
            )
            
        if not done_before:
            time_batch[i] += 1
            
        if winner_batch[i] >= 0:
            # Transfer loser cells to winner
            w_idx = int(winner_batch[i])
            l_idx = 1 - w_idx
            ownership[w_idx] |= ownership[l_idx]
            ownership[l_idx][:] = False
            ownership_neutral_batch[i] &= ~ownership[w_idx]
        else:
            # Global update
            t = int(time_batch[i])
            if t % 50 == 0:
                armies += ownership[0].astype(np.int32) + ownership[1].astype(np.int32)
            if (t % 2 == 0) and (t > 0):
                mask = (generals | cities).astype(np.int32)
                armies += mask * ownership[0].astype(np.int32) + mask * ownership[1].astype(np.int32)


def batch_step(states: list[GameState], actions: np.ndarray) -> Tuple[list[GameState], list[GameInfo]]:
    num_envs = len(states)
    armies = np.stack([s.armies for s in states])
    ownership = np.stack([s.ownership for s in states])
    ownership_neutral = np.stack([s.ownership_neutral for s in states])
    generals = np.stack([s.generals for s in states])
    cities = np.stack([s.cities for s in states])
    mountains = np.stack([s.mountains for s in states])
    passable = np.stack([s.passable for s in states])
    time = np.array([s.time for s in states], dtype=np.int32)
    winner = np.array([s.winner for s in states], dtype=np.int32)
    
    jit_batch_step(
        armies, ownership, ownership_neutral, generals, cities, mountains, passable, time, winner, actions
    )
    
    new_states = []
    infos = []
    for i in range(num_envs):
        ns = GameState(
            armies=armies[i],
            ownership=ownership[i],
            ownership_neutral=ownership_neutral[i],
            generals=generals[i],
            cities=cities[i],
            mountains=mountains[i],
            passable=passable[i],
            general_positions=states[i].general_positions, # not updated here for simplicity
            time=np.int32(time[i]),
            winner=np.int32(winner[i]),
            pool_idx=states[i].pool_idx
        )
        new_states.append(ns)
        infos.append(get_info(ns))
        
    return new_states, infos
