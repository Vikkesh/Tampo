import gym
import numpy as np
from collections import deque


class SequenceWrapper(gym.Wrapper):
    """
    Wraps TaskOffloadingEnv to provide tasks as a topologically-sorted 2-D sequence.
    Used for TPTO, MTD3 and other sequence-to-sequence TO models.

    Output observation shape: [max_tasks, task_feature_dim]
    Tasks are ordered so every parent appears before its children (Kahn's algorithm).
    If the graph has a cycle (should not happen with valid DAGs), falls back to
    numerical node order.
    """

    def __init__(self, env, max_tasks: int = 30, task_feature_dim: int = 6):
        super().__init__(env)
        self.max_tasks = max_tasks
        self.task_feature_dim = task_feature_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_tasks, self.task_feature_dim),
            dtype=np.float32
        )

    # ── topological sort ────────────────────────────────────────────────────
    @staticmethod
    def _topological_sort(adj_matrix: np.ndarray) -> list:
        """Kahn's BFS topological sort.  adj_matrix[i,j] == 1 means edge i→j."""
        N = adj_matrix.shape[0]
        in_degree = adj_matrix.sum(axis=0).astype(int)   # column sums
        queue = deque(i for i in range(N) if in_degree[i] == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in range(N):
                if adj_matrix[node, child] > 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(order) != N:
            # Cycle detected or disconnected component — safe fallback
            return list(range(N))
        return order

    # ── observation builder ─────────────────────────────────────────────────
    def _build_sequence_obs(self) -> np.ndarray:
        """Read the current task graph from the env and return a sorted sequence."""
        task_features = self.env.get_task_feature_matrix()  # [N, 6]
        adj_matrix = self.env.get_adjacency_matrix()

        N = task_features.shape[0]

        if adj_matrix is not None and adj_matrix.shape == (N, N):
            order = self._topological_sort(np.asarray(adj_matrix, dtype=np.float32))
            task_features = task_features[order]

        # Pad or truncate to max_tasks
        padded = np.zeros((self.max_tasks, self.task_feature_dim), dtype=np.float32)
        actual_len = min(N, self.max_tasks)
        padded[:actual_len] = task_features[:actual_len]
        return padded

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._build_sequence_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._build_sequence_obs(), reward, done, info
