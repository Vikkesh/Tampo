import gymnasium as gym
import numpy as np


class FlatVectorWrapper(gym.Wrapper):
    """
    Wraps TaskOffloadingEnv to provide a flat 1D observation vector.
    Used for D3QN, SAC, MAPPO and other standard RL baselines.

    The server feature dimension is read directly from the environment's
    get_server_features() method, not from sampling the observation space
    (which returns a flat Box, not a Dict).
    """

    SERVER_FEATURE_DIM = 20   # matches TaskOffloadingEnv.get_server_features() hard cap

    def __init__(self, env, max_tasks: int = 30, task_feature_dim: int = None):
        super().__init__(env)
        self.max_tasks = max_tasks
        # Default to the env's own width rather than a literal, so adding a node feature
        # cannot silently truncate or mis-shape the flattened observation.
        self.task_feature_dim = (
            task_feature_dim if task_feature_dim is not None
            else getattr(env, 'task_feature_dim', 9)
        )

        flat_dim = (self.max_tasks * self.task_feature_dim) + self.SERVER_FEATURE_DIM

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def _build_flat_obs(self) -> np.ndarray:
        """Pull real graph features and server state from the environment."""
        task_features = self.env.get_task_feature_matrix()   # [N, task_feature_dim]
        server_features = self.env.get_server_features()     # [20]

        N = task_features.shape[0]
        flat_tasks = task_features.flatten()

        if N < self.max_tasks:
            pad_size = (self.max_tasks - N) * self.task_feature_dim
            flat_tasks = np.pad(flat_tasks, (0, pad_size), 'constant')
        elif N > self.max_tasks:
            flat_tasks = flat_tasks[:self.max_tasks * self.task_feature_dim]

        return np.concatenate([flat_tasks, server_features]).astype(np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._build_flat_obs()

    def step(self, action):
        obs_raw, reward, done, info = self.env.step(action)
        return self._build_flat_obs(), reward, done, info
