import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch

class CommonEvaluator:
    """
    Unified evaluation framework for all algorithms.
    Ensures consistent and fair comparison across different approaches.
    
    Features:
    - Fixed evaluation episodes
    - Consistent preference vectors
    - Same task distribution
    - Standardized metrics calculation
    - Deterministic action selection
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize common evaluator
        
        Args:
            env: Task offloading environment
            config: Evaluation configuration
        """
        self.env = env
        self.config = config
        
        # Standard evaluation preferences
        self.eval_preferences = [
            np.array([0.8, 0.2]),  # Delay-focused
            np.array([0.5, 0.5]),  # Balanced
            np.array([0.2, 0.8])   # Energy-focused
        ]
        
        # Evaluation parameters
        self.num_episodes = config.get('eval_episodes', 20)
        self.max_steps_per_episode = config.get('max_steps', 100)
        
        # Task seeds for reproducibility
        self.eval_seeds = list(range(1000, 1000 + self.num_episodes))
        
        print(f"✓ Common Evaluator Initialized")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Max steps per episode: {self.max_steps_per_episode}")
        print(f"  Preference vectors: {len(self.eval_preferences)}")
    
    def evaluate_heuristic(self, algorithm, dag_parser, num_tasks: int = None) -> Dict:
        """
        Evaluate heuristic algorithms (HEFT, PSO, GA)
        
        Args:
            algorithm: Algorithm instance (HEFTScheduler, PSOScheduler, GAScheduler)
            dag_parser: DAG parser for loading task graphs
            num_tasks: Number of tasks to test (uses eval_episodes if None)
            
        Returns:
            Dictionary with metrics
        """
        if num_tasks is None:
            num_tasks = self.num_episodes
        
        print(f"\n📊 Evaluating {algorithm.__class__.__name__}...")
        
        # Load consistent task set
        np.random.seed(42)  # Fixed seed for reproducibility
        dags = dag_parser.load_dataset(num_graphs=num_tasks)
        
        if len(dags) == 0:
            print("⚠️  Warning: No DAG graphs loaded.")
            return None
        
        delays = []
        energies = []
        
        # Evaluate each task with different preferences
        episodes_per_pref = num_tasks // len(self.eval_preferences)
        
        for i, dag in enumerate(dags):
            # Select preference based on episode index
            pref_idx = i % len(self.eval_preferences)
            preference = self.eval_preferences[pref_idx]
            
            # For heuristic algorithms, use their optimize/schedule methods
            if hasattr(algorithm, 'schedule'):  # HEFT
                schedule, delay, energy = algorithm.schedule(dag)
            elif hasattr(algorithm, 'optimize'):  # PSO, GA
                schedule, delay, energy = algorithm.optimize(dag, preference)
            else:
                raise ValueError(f"Unknown algorithm type: {algorithm.__class__.__name__}")
            
            delays.append(delay)
            energies.append(energy)
        
        # Calculate metrics
        result = self._calculate_metrics(delays, energies)
        
        print(f"  ✓ Avg Delay: {result['avg_delay']:.4f}s")
        print(f"  ✓ Avg Energy: {result['avg_energy']:.4f}J")
        
        return result
    
    def evaluate_rl_agent(self, agent, agent_type: str) -> Dict:
        """
        Evaluate RL-based algorithms (PPO, GMORL, TAM-PO)
        
        Args:
            agent: Agent instance (PPOAgent, GMORLAgent, TAMPOFramework)
            agent_type: Type of agent ('ppo', 'gmorl', 'tampo')
            
        Returns:
            Dictionary with metrics
        """
        print(f"\n📊 Evaluating {agent_type.upper()}...")
        
        delays = []
        energies = []
        
        # Evaluate across all preference vectors
        episodes_per_pref = self.num_episodes // len(self.eval_preferences)
        
        for pref_idx, preference in enumerate(self.eval_preferences):
            for episode in range(episodes_per_pref):
                # Use fixed seed for this episode
                episode_idx = pref_idx * episodes_per_pref + episode
                np.random.seed(self.eval_seeds[episode_idx])
                
                # Reset environment with preference
                state = self.env.reset(preference_vector=preference)
                
                episode_delay = 0
                episode_energy = 0
                done = False
                steps = 0
                
                while not done and steps < self.max_steps_per_episode:
                    # Select action (deterministic for evaluation)
                    action = self._get_action(agent, agent_type, state, preference)
                    
                    # Step environment
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Accumulate metrics
                    episode_delay += info.get('delay', 0)
                    episode_energy += info.get('energy', 0)
                    
                    state = next_state
                    steps += 1
                
                delays.append(episode_delay)
                energies.append(episode_energy)
        
        # Calculate metrics
        result = self._calculate_metrics(delays, energies)
        
        print(f"  ✓ Avg Delay: {result['avg_delay']:.4f}s")
        print(f"  ✓ Avg Energy: {result['avg_energy']:.4f}J")
        
        return result
    
    def _get_action(self, agent, agent_type: str, state: np.ndarray, preference: np.ndarray) -> int:
        """
        Get action from agent (handles different agent types)
        
        Args:
            agent: Agent instance
            agent_type: Type of agent
            state: Current state
            preference: Preference vector
            
        Returns:
            Action (int)
        """
        if agent_type == 'ppo':
            # PPO: select_action(state, deterministic=True)
            action, _, _ = agent.select_action(state, deterministic=True)
            return action
            
        elif agent_type == 'gmorl':
            # GMORL: select_action(state, preference, deterministic=True)
            action, _, _ = agent.select_action(state, preference, deterministic=True)
            return action
            
        elif agent_type == 'tampo':
            # TAM-PO: use fully deterministic argmax
            task_features = self._extract_task_features(state)
            server_features = self._extract_server_features(state)
            
            with torch.no_grad():
                device = agent.device
                task_tensor = torch.FloatTensor(task_features).to(device)
                server_tensor = torch.FloatTensor(server_features).to(device)
                pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(device)
                
                logits = agent.meta_learner.meta_policy(
                    task_tensor, server_tensor, pref_tensor, num_tasks=1
                )
                
                # Fully deterministic: argmax without any randomness
                probs = torch.softmax(logits[:, 0, :], dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            
            return action
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _extract_task_features(self, state: np.ndarray) -> np.ndarray:
        """Extract task features from state"""
        task_feat = state[:6].reshape(1, 1, -1)
        return task_feat
    
    def _extract_server_features(self, state: np.ndarray) -> np.ndarray:
        """Extract server features from state"""
        server_feat = state[6:26].reshape(1, -1)
        return server_feat
    
    def _calculate_metrics(self, delays: List[float], energies: List[float]) -> Dict:
        """
        Calculate standardized metrics
        
        Args:
            delays: List of delay values
            energies: List of energy values
            
        Returns:
            Dictionary with metrics
        """
        delays = np.array(delays)
        energies = np.array(energies)
        
        # Remove outliers (>3 std dev)
        delay_mean = np.mean(delays)
        delay_std = np.std(delays)
        energy_mean = np.mean(energies)
        energy_std = np.std(energies)
        
        # Filter outliers
        delay_mask = np.abs(delays - delay_mean) <= 3 * delay_std
        energy_mask = np.abs(energies - energy_mean) <= 3 * energy_std
        valid_mask = delay_mask & energy_mask
        
        if np.sum(valid_mask) < len(delays) * 0.8:
            # If too many outliers, don't filter
            valid_delays = delays
            valid_energies = energies
        else:
            valid_delays = delays[valid_mask]
            valid_energies = energies[valid_mask]
        
        metrics = {
            'avg_delay': np.mean(valid_delays),
            'std_delay': np.std(valid_delays),
            'min_delay': np.min(valid_delays),
            'max_delay': np.max(valid_delays),
            'median_delay': np.median(valid_delays),
            
            'avg_energy': np.mean(valid_energies),
            'std_energy': np.std(valid_energies),
            'min_energy': np.min(valid_energies),
            'max_energy': np.max(valid_energies),
            'median_energy': np.median(valid_energies),
            
            'num_episodes': len(valid_delays),
            'num_outliers_removed': len(delays) - len(valid_delays)
        }
        
        return metrics
    
    def compare_algorithms(self, results: Dict[str, Dict]) -> None:
        """
        Print detailed comparison table
        
        Args:
            results: Dictionary mapping algorithm names to their metrics
        """
        print("\n" + "="*100)
        print(" "*35 + "DETAILED COMPARISON")
        print("="*100)
        
        # Header
        print(f"\n{'Algorithm':<15} {'Avg Delay':<12} {'Std Delay':<12} {'Avg Energy':<12} {'Std Energy':<12} {'Episodes':<10}")
        print("-" * 100)
        
        # Rows
        for alg_name, metrics in results.items():
            if metrics is None:
                continue
            print(f"{alg_name:<15} "
                  f"{metrics['avg_delay']:<12.4f} "
                  f"{metrics['std_delay']:<12.4f} "
                  f"{metrics['avg_energy']:<12.4f} "
                  f"{metrics['std_energy']:<12.4f} "
                  f"{metrics['num_episodes']:<10}")
        
        print("-" * 100)
        
        # Find winners
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) > 0:
            best_delay = min(valid_results.items(), key=lambda x: x[1]['avg_delay'])
            best_energy = min(valid_results.items(), key=lambda x: x[1]['avg_energy'])
            
            print(f"\n🏆 Best Delay:  {best_delay[0]:<15} ({best_delay[1]['avg_delay']:.4f}s)")
            print(f"🏆 Best Energy: {best_energy[0]:<15} ({best_energy[1]['avg_energy']:.4f}J)")
            
            # Best balanced (using equal weights)
            best_balanced = min(valid_results.items(), 
                              key=lambda x: 0.5 * x[1]['avg_delay'] / 10.0 + 0.5 * x[1]['avg_energy'])
            print(f"🏆 Best Balanced: {best_balanced[0]:<15}")
        
        print("\n" + "="*100 + "\n")