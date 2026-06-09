from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """
    Abstract interface for all standalone TO baseline agents.
    Every algorithm file in algorithms/baselines/ MUST inherit this.
    """

    @abstractmethod
    def train(self, env, num_episodes: int, **kwargs):
        """Run training loop. env is already wrapped with the correct Gym Wrapper."""
        pass

    @abstractmethod
    def predict(self, observation, deterministic: bool = True):
        """
        Return a single action for the given observation.
        deterministic=True must produce greedy (no-exploration) actions.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model weights to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model weights from disk."""
        pass
