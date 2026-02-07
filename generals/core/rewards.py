import numpy as np

from generals.core.observation import Observation


def _to_np(x):
    # Convert torch tensors or numpy arrays to numpy ndarray
    try:
        return np.array(x)
    except Exception:
        return x


def compute_num_cities_owned(observation: Observation) -> np.ndarray:
    """Count number of cities owned by the agent."""
    owned_cities_mask = _to_np(observation.cities) & _to_np(observation.owned_cells)
    num_cities_owned = np.sum(owned_cities_mask)
    return np.array(num_cities_owned, dtype=np.float32)


def compute_num_generals_owned(observation: Observation) -> np.ndarray:
    """Count number of generals owned by the agent."""
    owned_generals_mask = _to_np(observation.generals) & _to_np(observation.owned_cells)
    num_generals_owned = np.sum(owned_generals_mask)
    return np.array(num_generals_owned, dtype=np.float32)


def calculate_army_size(castles: np.ndarray, ownership: np.ndarray) -> np.ndarray:
    """Calculate total army size in castles (cities/generals) owned by the player."""
    castles_arr = _to_np(castles)
    ownership_arr = _to_np(ownership)
    return np.array(np.sum(castles_arr * ownership_arr), dtype=np.float32)


def city_reward_fn(
    prior_obs: Observation,
    prior_action: np.ndarray,
    obs: Observation,
    shaping_weight: float = 0.3,
) -> np.ndarray:
    """
    Reward function that shapes the reward based on the number of cities owned.
    """
    original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

    # If game is done, don't shape the reward
    game_done = (_to_np(obs.owned_army_count) == 0) | (_to_np(obs.opponent_army_count) == 0)

    city_now = calculate_army_size(obs.cities, obs.owned_cells)
    city_prev = calculate_army_size(prior_obs.cities, prior_obs.owned_cells)
    city_change = city_now - city_prev

    shaped_reward = original_reward + shaping_weight * city_change

    return np.where(game_done, original_reward, shaped_reward)


def ratio_reward_fn(
    prior_obs: Observation,
    prior_action: np.ndarray,
    obs: Observation,
    clip_value: float = 1.5,
    shaping_weight: float = 0.5,
) -> np.ndarray:
    """
    Reward function that shapes based on army ratio between player and opponent.
    """
    original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

    # If game is done, don't shape the reward
    game_done = (_to_np(obs.owned_army_count) == 0) | (_to_np(obs.opponent_army_count) == 0)

    def calculate_ratio_reward(my_army, opponent_army):
        my = _to_np(my_army).astype(np.float32)
        opp = _to_np(opponent_army).astype(np.float32)
        ratio = my / np.maximum(opp, 1.0)
        ratio = np.log(ratio) / np.log(clip_value)
        return np.clip(ratio, -1.0, 1.0)

    prev_ratio_reward = calculate_ratio_reward(prior_obs.owned_army_count, prior_obs.opponent_army_count)
    current_ratio_reward = calculate_ratio_reward(obs.owned_army_count, obs.opponent_army_count)
    ratio_reward = current_ratio_reward - prev_ratio_reward

    shaped_reward = original_reward + shaping_weight * ratio_reward

    return np.where(game_done, original_reward, shaped_reward)


def win_lose_reward_fn(prior_obs: Observation, prior_action: np.ndarray, obs: Observation) -> np.ndarray:
    """
    Simple reward function based on generals owned with small bonus for splitting.
    """
    original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

    # Encourage splitting a bit; handle scalar/prior_action safely
    pa = _to_np(prior_action)
    split_bonus = 0.0
    if isinstance(pa, np.ndarray) and pa.size > 4:
        try:
            if pa[4] == 1:
                split_bonus = 0.0015
        except Exception:
            split_bonus = 0.0

    return original_reward + split_bonus


def composite_reward_fn(
    prior_obs: Observation,
    prior_action: np.ndarray,
    obs: Observation,
    city_weight: float = 0.4,
    ratio_weight: float = 0.3,
    maximum_army_ratio: float = 1.6,
    maximum_land_ratio: float = 1.3,
) -> np.ndarray:
    """
    Composite reward function combining multiple reward signals.
    """
    original_reward = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

    # If game is done, don't shape the reward (except split bonus)
    game_done = (_to_np(obs.owned_army_count) == 0) | (_to_np(obs.opponent_army_count) == 0)

    def calculate_ratio_reward(mine, opponents, max_ratio):
        m = _to_np(mine).astype(np.float32)
        o = _to_np(opponents).astype(np.float32)
        ratio = m / np.maximum(o, 1.0)
        ratio = np.log(ratio) / np.log(max_ratio)
        return np.clip(ratio, -1.0, 1.0)

    previous_army_ratio = calculate_ratio_reward(
        prior_obs.owned_army_count, prior_obs.opponent_army_count, maximum_army_ratio
    )
    current_army_ratio = calculate_ratio_reward(
        obs.owned_army_count, obs.opponent_army_count, maximum_army_ratio
    )
    army_reward = current_army_ratio - previous_army_ratio

    previous_land_ratio = calculate_ratio_reward(
        prior_obs.owned_land_count, prior_obs.opponent_land_count, maximum_land_ratio
    )
    current_land_ratio = calculate_ratio_reward(
        obs.owned_land_count, obs.opponent_land_count, maximum_land_ratio
    )
    land_reward = current_land_ratio - previous_land_ratio

    city_reward = compute_num_cities_owned(obs) - compute_num_cities_owned(prior_obs)

    shaped_reward = (
        original_reward + ratio_weight * army_reward + city_weight * city_reward + ratio_weight * land_reward
    )

    return np.where(game_done, original_reward, shaped_reward)
