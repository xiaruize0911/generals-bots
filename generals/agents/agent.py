from abc import ABC, abstractmethod

import numpy as np

from generals.core.observation import Observation


class Agent(ABC):
    """Base class for agents using NumPy/standard RNGs."""

    def __init__(self, id: str = "NPC"):
        self.id = id

    @abstractmethod
    def act(self, observation: Observation, rng: np.random.Generator) -> np.ndarray:
        """
        Select an action given an observation.

        Args:
            observation: Current game observation
            rng: numpy.random.Generator for stochastic decisions

        Returns:
            Action array [pass, row, col, direction, split]
        """
        raise NotImplementedError

    def reset(self):
        """Reset agent state between episodes."""
        pass

    def __str__(self):
        return self.id
