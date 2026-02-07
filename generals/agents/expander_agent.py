import numpy as np

from generals.core.observation import Observation

from .agent import Agent
from ._expander_logic import expander_action


class ExpanderAgent(Agent):
    """Agent that aggressively expands territory by capturing new cells (NumPy RNG)."""

    def __init__(self, id: str = "Expander"):
        super().__init__(id)

    def act(self, observation: Observation, rng: np.random.Generator) -> np.ndarray:
        return expander_action(rng, observation)

    def reset(self):
        pass
