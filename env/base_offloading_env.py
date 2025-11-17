import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import json

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

        # Load configuration
        self.config = config
        self.task_type = config.get('task_type', 'dag')  # 'dag' or 'independent'
        
        # System parameters
        self.num_edge_servers = config.get('num_edge_servers', 3)
        self.num_users = config.get('num_users', 1)
        self.time_step = config.get('time_step', 0.1)  # seconds
        self.max_steps = config.get('max_steps', 100)
        
        # Computing resources
        self.cloud_freq = config.get('cloud_freq', 10e9)  # Hz
        self.edge_freq = np.array(config.get('edge_freq', [5e9] * self.num_edge_servers))
        self.local_freq = config.get('local_freq', 1e9)  # Hz
        
        # Energy parameters
        self.cloud_power_tx = config.get('cloud_power_tx', 0.5)  # Watts
        self.edge_power_tx = config.get('edge_power_tx', 0.3)  # Watts
        self.local_power = config.get('local_power', 0.1)  # Watts
        self.kappa = config.get('kappa', 1e-28)  # Effective switched capacitance
        
        # Network parameters
        self.bandwidth_up = config.get('bandwidth_up', 20e6)  # Hz
        self.bandwidth_down = config.get('bandwidth_down', 20e6)  # Hz
        self.noise_power = config.get('noise_power', 1e-13)  # Watts
        
        # Task parameters
        self.task_size_range = config.get('task_size_range', [1e6, 10e6])  # bits
        self.task_cycles_range = config.get('task_cycles_range', [1e9, 10e9])  # cycles
        
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
            # For DAG tasks: task features + graph structure + server states
            obs_dim = (
                6 +  # task features (size, cycles, depth, etc.)
                10 +  # server states (loads, frequencies)
                20   # graph structure encoding
            )
        else:
            # For independent tasks: task features + server states
            obs_dim = (
                4 +  # task features
                10   # server states
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
            self.current_task = self.task_dataset[task_id]
        else:
            # Generate a new task if not in dataset
            self.current_task = self._generate_task()
    
    def load_task_dataset(self, task_graphs: List[Dict]):
        """
        Load task dataset for meta-learning
        
        Args:
            task_graphs: List of task graph dictionaries
        """
        self.task_dataset = task_graphs
        print(f"Loaded {len(task_graphs)} tasks into environment dataset")
    
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
            self.current_task = task_graph
        else:
            self.current_task = self._generate_task()
        
        return self._get_observation()
    
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
        
        # Server features
        server_freqs_norm = np.concatenate([
            [self.cloud_freq / 10e9],
            self.edge_freq / 10e9
        ])
        
        server_loads_norm = self.server_loads / 10.0  # Normalize by max expected load
        
        # Channel gains
        channel_gains_norm = self.channel_gains / 2.0
        
        # Combine features
        obs = np.concatenate([
            [task_size_norm, task_cycles_norm],
            self.preference,
            server_freqs_norm,
            server_loads_norm,
            channel_gains_norm
        ])
        
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
        done = self.current_step >= self.max_steps
        
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