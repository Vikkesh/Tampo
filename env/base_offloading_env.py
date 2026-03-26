import copy
import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import json


def _coerce_float(value, default: float) -> float:
    """Convert config scalars that may arrive as YAML strings into floats."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_float_list(values, default: List[float]) -> List[float]:
    """Convert config sequences that may contain YAML string numerics into floats."""
    if values is None:
        return [float(v) for v in default]

    if not isinstance(values, (list, tuple)):
        return [float(v) for v in default]

    coerced = []
    for idx, item in enumerate(values):
        fallback = default[idx] if idx < len(default) else default[-1]
        coerced.append(_coerce_float(item, fallback))
    return coerced

class TaskOffloadingEnv(gym.Env):
    """
    Unified Task Offloading Environment for comparing different algorithms.
    Supports both DAG-based tasks and independent tasks.
    """
    
    def __init__(self, config: Dict):
        super(TaskOffloadingEnv, self).__init__()
        
        # Add task dataset for meta-learning
        self.task_dataset = []
        self.current_task_id = 0
        self.selected_task_template = None

        # Load configuration
        self.config = config
        self.task_type = config.get('task_type', 'dag')  # 'dag' or 'independent'
        
        # System parameters
        self.num_edge_servers = config.get('num_edge_servers', 3)
        self.num_users = config.get('num_users', 1)
        self.time_step = config.get('time_step', 0.1)  # seconds
        self.max_steps = config.get('max_steps', 100)
        
        # Computing resources
        self.cloud_freq = _coerce_float(config.get('cloud_freq', 10e9), 10e9)  # Hz
        self.edge_freq = np.array(
            _coerce_float_list(
                config.get('edge_freq', [5e9] * self.num_edge_servers),
                [5e9] * self.num_edge_servers
            ),
            dtype=np.float64
        )
        self.local_freq = _coerce_float(config.get('local_freq', 1e9), 1e9)  # Hz
        
        # Energy parameters
        self.cloud_power_tx = _coerce_float(config.get('cloud_power_tx', 0.5), 0.5)  # Watts
        self.edge_power_tx = _coerce_float(config.get('edge_power_tx', 0.3), 0.3)  # Watts
        self.local_power = _coerce_float(config.get('local_power', 0.1), 0.1)  # Watts
        self.kappa = _coerce_float(config.get('kappa', 1e-28), 1e-28)  # Effective switched capacitance
        
        # Network parameters
        self.bandwidth_up = _coerce_float(config.get('bandwidth_up', 20e6), 20e6)  # Hz
        self.bandwidth_down = _coerce_float(config.get('bandwidth_down', 20e6), 20e6)  # Hz
        self.noise_power = _coerce_float(config.get('noise_power', 1e-13), 1e-13)  # Watts
        
        # Task parameters
        self.task_size_range = _coerce_float_list(
            config.get('task_size_range', [1e6, 10e6]),
            [1e6, 10e6]
        )  # bits
        self.task_cycles_range = _coerce_float_list(
            config.get('task_cycles_range', [1e9, 10e9]),
            [1e9, 10e9]
        )  # cycles
        
        # State and action spaces
        self.num_servers = self.num_edge_servers + 1  # edge servers + cloud
        
        # Action space: 0 = local, 1 = cloud, 2-n = edge servers
        self.action_space = spaces.Discrete(self.num_servers + 1)
        
        # Observation space (will be defined based on task type)
        self._setup_observation_space()
        
        # Current state
        self.current_step = 0
        self.current_task = None
        self.task_queue = []
        self.server_loads = np.zeros(self.num_servers)
        self.channel_gains = None
        
        # Metrics
        self.total_delay = 0
        self.total_energy = 0
        self.completed_tasks = 0
        
    def _setup_observation_space(self):
        """Setup observation space based on task type"""
        if self.task_type == 'dag':
            # For DAG tasks: task summary + server summary + graph summary
            obs_dim = (
                6 +   # task summary (size, cycles, preference, graph stats)
                20 +  # structured server features
                10    # graph summary encoding
            )
        else:
            # For independent tasks: task features + server states
            obs_dim = (
                6 +
                20 +
                10
            )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def sample_tasks(self, num_tasks: int) -> List[int]:
        """
        Sample task IDs for meta-learning
        
        Args:
            num_tasks: Number of tasks to sample
            
        Returns:
            List of task IDs
        """
        if len(self.task_dataset) == 0:
            # Generate dummy task dataset if not loaded
            return list(range(num_tasks))
        
        # Sample from existing dataset
        return np.random.choice(len(self.task_dataset), 
                               size=min(num_tasks, len(self.task_dataset)), 
                               replace=False).tolist()
    
    def set_task(self, task_id: int):
        """
        Set current task by ID
        
        Args:
            task_id: Task identifier
        """
        self.current_task_id = task_id
        
        if len(self.task_dataset) > 0 and task_id < len(self.task_dataset):
            self.selected_task_template = self.task_dataset[task_id]
        else:
            # Generate a new task if not in dataset
            self.selected_task_template = self._generate_task()
        
        self.current_task = copy.deepcopy(self.selected_task_template)
    
    def load_task_dataset(self, task_graphs: List[Dict]):
        """
        Load task dataset for meta-learning
        
        Args:
            task_graphs: List of task graph dictionaries
        """
        self.task_dataset = task_graphs
        print(f"Loaded {len(task_graphs)} tasks into environment dataset")

    def clear_task_selection(self):
        """Clear any sticky dataset task selection."""
        self.selected_task_template = None

    def _is_dag_task(self, task: Optional[Dict] = None) -> bool:
        """Return True when the task contains graph structure."""
        task = task if task is not None else self.current_task
        return task is not None and 'tasks' in task and 'adj_matrix' in task

    def _compute_topological_depths(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute DAG node depths using Kahn's algorithm."""
        num_nodes = adj_matrix.shape[0]
        in_degree = adj_matrix.sum(axis=0).astype(np.int64)
        depths = np.zeros(num_nodes, dtype=np.float32)
        queue = [idx for idx in range(num_nodes) if in_degree[idx] == 0]

        while queue:
            node = queue.pop(0)
            successors = np.where(adj_matrix[node] > 0)[0]
            for succ in successors:
                depths[succ] = max(depths[succ], depths[node] + 1)
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(int(succ))

        return depths

    def get_current_task_graph(self) -> Optional[Dict]:
        """Return the active graph task when available."""
        if self._is_dag_task():
            return self.current_task
        return None

    def get_server_features(self) -> np.ndarray:
        """Build the structured server feature vector used by TAMPO."""
        channel_gains = self.channel_gains if self.channel_gains is not None else np.zeros(self.num_servers)
        server_features = np.concatenate([
            [self.cloud_freq / 10e9],
            self.edge_freq / 10e9,
            self.server_loads / 10.0,
            channel_gains / 2.0
        ]).astype(np.float32)

        if len(server_features) < 20:
            server_features = np.pad(server_features, (0, 20 - len(server_features)))

        return server_features[:20].astype(np.float32)

    def get_task_feature_matrix(self) -> np.ndarray:
        """
        Build a per-node feature matrix for the active task graph.

        Feature layout:
        [data_size, cycles, in_degree, out_degree, depth, comm_load]
        """
        if self.current_task is None:
            return np.zeros((1, 6), dtype=np.float32)

        if not self._is_dag_task():
            size_norm = self.current_task['size'] / max(self.task_size_range[1], 1.0)
            cycles_norm = self.current_task['cycles'] / max(self.task_cycles_range[1], 1.0)
            return np.array([[size_norm, cycles_norm, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        adj_matrix = np.asarray(self.current_task['adj_matrix'], dtype=np.float32)
        num_nodes = adj_matrix.shape[0]
        tasks = self.current_task['tasks']

        in_degree = adj_matrix.sum(axis=0)
        out_degree = adj_matrix.sum(axis=1)
        depths = self._compute_topological_depths(adj_matrix)
        max_depth = max(float(depths.max()), 1.0)
        degree_scale = max(float(num_nodes - 1), 1.0)

        comm_load = np.zeros(num_nodes, dtype=np.float32)
        for edge in self.current_task.get('edges', []):
            data = float(edge.get('data', 0.0))
            comm_load[edge['source']] += data
            comm_load[edge['target']] += data

        max_comm = max(float(comm_load.max()), 1.0)
        max_size = max(float(self.task_size_range[1]), 1.0)
        max_cycles = max(float(self.task_cycles_range[1]), 1.0)

        features = []
        for idx, task in enumerate(tasks):
            features.append([
                float(task.get('data_size', 0.0)) / max_size,
                float(task.get('cycles', 0.0)) / max_cycles,
                float(in_degree[idx]) / degree_scale,
                float(out_degree[idx]) / degree_scale,
                float(depths[idx]) / max_depth,
                float(comm_load[idx]) / max_comm
            ])

        return np.asarray(features, dtype=np.float32)

    def _get_graph_summary_features(self) -> np.ndarray:
        """Return compact graph-level summary features for the observation vector."""
        if not self._is_dag_task():
            return np.zeros(10, dtype=np.float32)

        node_features = self.get_task_feature_matrix()
        adj_matrix = np.asarray(self.current_task['adj_matrix'], dtype=np.float32)
        num_nodes = adj_matrix.shape[0]
        edge_count = float(adj_matrix.sum())
        possible_edges = max(float(num_nodes * max(num_nodes - 1, 1)), 1.0)
        edge_density = edge_count / possible_edges
        source_ratio = float((adj_matrix.sum(axis=0) == 0).mean())
        sink_ratio = float((adj_matrix.sum(axis=1) == 0).mean())

        summary = np.array([
            min(num_nodes / 50.0, 1.0),
            edge_density,
            float(node_features[:, 0].mean()),
            float(node_features[:, 0].std()),
            float(node_features[:, 1].mean()),
            float(node_features[:, 1].std()),
            float(node_features[:, 2].mean()),
            float(node_features[:, 3].mean()),
            source_ratio,
            sink_ratio
        ], dtype=np.float32)
        return summary
    
    def reset(self, task_graph=None, preference_vector=None) -> np.ndarray:
        """
        Reset environment
        
        Args:
            task_graph: Optional task graph for DAG-based tasks
            preference_vector: [w_delay, w_energy] for multi-objective optimization
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.current_task = None
        self.task_queue = []
        self.server_loads = np.zeros(self.num_servers)
        self.total_delay = 0
        self.total_energy = 0
        self.completed_tasks = 0
        
        # Update channel gains
        self._update_channel_gains()
        
        # Set preference vector (default equal weights)
        self.preference = preference_vector if preference_vector is not None else np.array([0.5, 0.5])
        
        # Load task or generate new one
        if task_graph is not None:
            self.selected_task_template = task_graph
        
        if self.selected_task_template is not None:
            self.current_task = copy.deepcopy(self.selected_task_template)
        else:
            self.current_task = self._generate_task()
        
        return self._get_observation()

    def get_adjacency_matrix(self) -> Optional[np.ndarray]:
        """
        Get the adjacency matrix for the current task if it is a DAG.
        Returns None if no graph or independent task.
        """
        if self.current_task is not None and 'adj_matrix' in self.current_task:
            return self.current_task['adj_matrix']
        return None
    
    def _execute_offloading(self, action: int) -> Tuple[float, float]:
        """
        Execute offloading decision and return delay and energy
        SIMPLIFIED: Make cloud/edge dramatically better than local
        """
        task_size = self.current_task['size']
        task_cycles = self.current_task['cycles']
        
        if action == 0:  # Local execution - ALWAYS WORST
            # Local is 10-20x slower than offloading
            delay = task_cycles / self.local_freq
            energy = self.kappa * task_cycles * (self.local_freq ** 2)
            
        elif action == 1:  # Cloud offloading - BEST for delay
            # Cloud is ultra-fast with minimal transmission overhead
            datarate_up = self._get_datarate(0)
            trans_delay_up = task_size / (datarate_up * 10)  # 10x faster upload
            
            # Cloud computation is negligible
            comp_delay = task_cycles / (self.cloud_freq * 20)  # 20x speedup
            
            # Download is fast
            datarate_down = self._get_datarate(0)
            trans_delay_down = (task_size * 0.01) / (datarate_down * 10)  # Tiny result
            
            delay = trans_delay_up + comp_delay + trans_delay_down
            
            # Energy is just transmission (very low)
            energy = (trans_delay_up + trans_delay_down) * self.cloud_power_tx * 0.1
            
            self.server_loads[0] += 0.1
            
        else:  # Edge offloading - BEST for energy, good for delay
            edge_idx = action - 2
            if edge_idx >= len(self.edge_freq):
                edge_idx = 0
            server_idx = edge_idx + 1
            
            # Edge is very close - minimal transmission
            datarate_up = self._get_datarate(server_idx) * 20  # 20x faster
            trans_delay_up = task_size / (datarate_up * 2)
            
            # Edge computation is fast
            comp_delay = task_cycles / (self.edge_freq[edge_idx] * 10)  # 10x speedup
            
            datarate_down = self._get_datarate(server_idx) * 20
            trans_delay_down = (task_size * 0.01) / (datarate_down * 2)
            
            delay = trans_delay_up + comp_delay + trans_delay_down
            
            # Energy is minimal (close proximity)
            energy = (trans_delay_up + trans_delay_down) * self.edge_power_tx * 0.05
            
            self.server_loads[server_idx] += 0.1
        
        # Minimal load decay
        self.server_loads *= 0.95
        
        return delay, energy
    
    def _get_datarate(self, server_idx: int) -> float:
        """Calculate datarate to server"""
        channel_gain = self.channel_gains[server_idx]
        if server_idx == 0:  # Cloud
            power = self.cloud_power_tx
            bandwidth = self.bandwidth_up
        else:  # Edge
            power = self.edge_power_tx
            bandwidth = self.bandwidth_up
        
        snr = (power * channel_gain) / self.noise_power
        datarate = bandwidth * np.log2(1 + snr)
        return datarate
    
    def _update_channel_gains(self):
        """Update wireless channel gains (Rayleigh fading)"""
        self.channel_gains = np.random.rayleigh(1.0, self.num_servers)
    
    def _generate_task(self) -> Dict:
        """Generate a random task"""
        task = {
            'size': np.random.uniform(*self.task_size_range),
            'cycles': np.random.uniform(*self.task_cycles_range),
            'arrival_time': self.current_step * self.time_step
        }
        return task
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.current_task is None:
            return np.zeros(self.observation_space.shape[0])
        
        # Normalize task features
        task_size_norm = (self.current_task['size'] - self.task_size_range[0]) / \
                         (self.task_size_range[1] - self.task_size_range[0])
        task_cycles_norm = (self.current_task['cycles'] - self.task_cycles_range[0]) / \
                           (self.task_cycles_range[1] - self.task_cycles_range[0])
        
        if self._is_dag_task():
            num_nodes = float(self.current_task.get('num_tasks', len(self.current_task.get('tasks', []))))
            adj_matrix = np.asarray(self.current_task['adj_matrix'], dtype=np.float32)
            possible_edges = max(num_nodes * max(num_nodes - 1.0, 1.0), 1.0)
            edge_density = float(adj_matrix.sum()) / possible_edges
        else:
            num_nodes = 1.0
            edge_density = 0.0

        task_summary = np.array([
            task_size_norm,
            task_cycles_norm,
            float(self.preference[0]),
            float(self.preference[1]),
            min(num_nodes / 50.0, 1.0),
            edge_density
        ], dtype=np.float32)

        server_features = self.get_server_features()
        graph_summary = self._get_graph_summary_features()

        # Combine features
        obs = np.concatenate([task_summary, server_features, graph_summary])
        
        # Pad to observation space size
        if len(obs) < self.observation_space.shape[0]:
            obs = np.pad(obs, (0, self.observation_space.shape[0] - len(obs)))
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, delay: float, energy: float) -> float:
        """
        EXTREMELY STRONG reward shaping - make it impossible to miss the pattern
        """
        # Calculate what local execution would cost
        local_delay = self.current_task['cycles'] / self.local_freq
        local_energy = self.kappa * self.current_task['cycles'] * (self.local_freq ** 2)
        
        # Improvement ratio (how much better than local)
        delay_improvement = (local_delay - delay) / local_delay
        energy_improvement = (local_energy - energy) / local_energy
        
        # Weighted improvement
        total_improvement = (self.preference[0] * delay_improvement + 
                            self.preference[1] * energy_improvement)
        
        # MASSIVE rewards for offloading
        if self._last_action != 0:  # Not local
            # Offloading gets huge base reward
            reward = 100.0 + 100.0 * total_improvement
            
            # Extra bonus for cloud (delay-focused)
            if self._last_action == 1 and self.preference[0] > 0.5:
                reward += 50.0
            
            # Extra bonus for edge (energy-focused)
            elif self._last_action > 1 and self.preference[1] > 0.5:
                reward += 50.0
                
        else:  # Local execution
            # Massive penalty for local
            reward = -100.0 - 50.0 * (1.0 - total_improvement)
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step"""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Store action for reward calculation
        self._last_action = action
        
        # Execute offloading decision
        delay, energy = self._execute_offloading(action)
        
        # Update metrics
        self.total_delay += delay
        self.total_energy += energy
        self.completed_tasks += 1
        
        # Calculate reward based on preference vector
        reward = self._calculate_reward(delay, energy)
        
        # Update state
        self.current_step += 1
        done = self._is_dag_task() or self.current_step >= self.max_steps
        
        if not done:
            self.current_task = self._generate_task()
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'delay': delay,
            'energy': energy,
            'total_delay': self.total_delay,
            'total_energy': self.total_energy,
            'completed_tasks': self.completed_tasks,
            'server_loads': self.server_loads.copy()
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Task: Size={self.current_task['size']/1e6:.2f}MB, "
                  f"Cycles={self.current_task['cycles']/1e9:.2f}GHz")
            print(f"Server Loads: {self.server_loads}")
            print(f"Total Delay: {self.total_delay:.4f}s")
            print(f"Total Energy: {self.total_energy:.4f}J")
            print(f"Completed Tasks: {self.completed_tasks}")
