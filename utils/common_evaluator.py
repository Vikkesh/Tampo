import numpy as np
from typing import Dict, List, Tuple, Optional, Any

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
    
    def evaluate_rl_agent(self, agent, agent_type: str,
                           test_dags: list = None) -> Dict:
        """
        Evaluate RL-based algorithms (PPO, GMORL, TAM-PO)

        Args:
            agent:      Agent instance (PPOAgent, GMORLAgent, TAMPOFramework)
            agent_type: Type of agent ('ppo', 'gmorl', 'tampo')
            test_dags:  Explicit list of DAG dicts to evaluate on.  Every
                        algorithm in the same benchmark run receives this SAME
                        list in the SAME order, guaranteeing a fair comparison.
                        If None, falls back to the env's loaded task_dataset.

        Returns:
            Dictionary with metrics
        """
        print(f"\n📊 Evaluating {agent_type.upper()}...")

        delays   = []
        energies = []

        # Determine the DAG pool to evaluate on
        if test_dags is not None:
            dags_to_eval = test_dags
        elif len(getattr(self.env, 'task_dataset', [])) > 0:
            dags_to_eval = self.env.task_dataset
        else:
            print("  ⚠ No test DAGs available — cannot evaluate.")
            return None

        # Per-episode action traces, so the reported metrics can be traced back to the
        # placements that produced them.  A policy that emits one action for every node
        # is degenerate; without this you cannot see that from makespan/energy alone.
        action_traces = []

        # Evaluate every (dag, preference) combination.
        # Each algorithm sees IDENTICAL dags in IDENTICAL order.
        episode_idx = 0
        for dag_idx, dag in enumerate(dags_to_eval):
            for preference in self.eval_preferences:
                np.random.seed(1000 + episode_idx)   # reproducible noise per episode
                episode_idx += 1

                # Hard-reset with the explicit DAG — bypasses set_task() entirely
                state = self.env.reset(task_graph=dag, preference_vector=preference)

                done  = False
                steps = 0
                actions = []

                while not done and steps < self.max_steps_per_episode:
                    action = self._get_action(agent, agent_type, state, preference)
                    next_state, reward, done, info = self.env.step(action)
                    actions.append(int(action))
                    state  = next_state
                    steps += 1

                # Read final accumulated metrics from the environment
                delays.append(self.env.total_delay)
                energies.append(self.env.total_energy)
                action_traces.append({
                    'dag_index': dag_idx,
                    'num_tasks': len(actions),
                    'w_delay': float(preference[0]),
                    'w_energy': float(preference[1]),
                    'actions': actions,
                    'makespan': float(self.env.total_delay),
                    'energy': float(self.env.total_energy),
                })

        # Calculate metrics
        result = self._calculate_metrics(delays, energies)
        result['action_traces'] = action_traces
        result['action_summary'] = self._summarize_actions(action_traces)

        print(f"  ✓ Episodes      : {len(delays)} ({len(dags_to_eval)} DAGs × {len(self.eval_preferences)} preferences)")
        print(f"  ✓ Avg Makespan  : {result['avg_makespan']:.4f}s")
        print(f"  ✓ Avg Energy    : {result['avg_energy']:.6f}J")
        self._print_action_summary(result['action_summary'])

        return result

    ACTION_NAMES = ['local', 'cloud', 'edge0', 'edge1', 'edge2']

    def _action_label(self, action: int) -> str:
        if action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"srv{action}"

    def _summarize_actions(self, traces: List[Dict]) -> Dict:
        """
        Aggregate action usage overall and per preference vector.

        `per_episode_entropy` is the mean normalised entropy of the action distribution
        WITHIN an episode.  0.0 means the agent placed every node of a DAG on the same
        server — the degenerate policy.  This is the number that reveals whether the
        agent is scheduling or just picking one server per graph.
        """
        n_actions = self.env.action_space.n
        overall = np.zeros(n_actions, dtype=np.int64)
        per_pref = {}
        episode_entropies = []

        for tr in traces:
            counts = np.bincount(tr['actions'], minlength=n_actions).astype(np.int64)
            overall += counts

            key = (tr['w_delay'], tr['w_energy'])
            per_pref.setdefault(key, np.zeros(n_actions, dtype=np.int64))
            per_pref[key] += counts

            total = counts.sum()
            if total > 0 and n_actions > 1:
                p = counts[counts > 0] / total
                episode_entropies.append(float(-(p * np.log(p)).sum() / np.log(n_actions)))

        def _frac(c):
            t = c.sum()
            return (c / t).tolist() if t else [0.0] * n_actions

        return {
            'overall_fractions': _frac(overall),
            'per_preference_fractions': {f"{k[0]:.1f}/{k[1]:.1f}": _frac(v) for k, v in per_pref.items()},
            'mean_per_episode_entropy': float(np.mean(episode_entropies)) if episode_entropies else 0.0,
            'degenerate_episodes': int(sum(1 for e in episode_entropies if e < 1e-9)),
            'total_episodes': len(traces),
        }

    def _print_action_summary(self, summary: Dict) -> None:
        n = len(summary['overall_fractions'])
        names = " ".join(
            f"{self._action_label(a)}={summary['overall_fractions'][a]*100:4.1f}%" for a in range(n)
        )
        print(f"  ✓ Actions       : {names}")
        print(f"  ✓ Within-episode entropy: {summary['mean_per_episode_entropy']:.3f} "
              f"(0 = one server for the whole DAG, 1 = uniform)")
        if summary['degenerate_episodes']:
            print(f"  ⚠ {summary['degenerate_episodes']}/{summary['total_episodes']} episodes "
                  f"placed EVERY node on a single server — policy is degenerate.")
        print("    Action mix per preference (delay/energy):")
        for pref, fracs in summary['per_preference_fractions'].items():
            mix = " ".join(f"{self._action_label(a)}={fracs[a]*100:4.1f}%" for a in range(n))
            print(f"      {pref:>8s} → {mix}")

    
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
            return agent.select_action(state, preference, deterministic=True)
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

        NOTE: No outlier filtering is applied.  Removing different numbers of
        episodes per algorithm makes the averages non-comparable — an "outlier"
        for LSTM is a real evaluation episode that GCN also ran on.  All episodes
        are included so every algorithm is scored on identical data.
        """
        delays   = np.array(delays)
        energies = np.array(energies)

        metrics = {
            'avg_makespan':    np.mean(delays),
            'std_makespan':    np.std(delays),
            'min_makespan':    np.min(delays),
            'max_makespan':    np.max(delays),
            'median_makespan': np.median(delays),

            'avg_energy':    np.mean(energies),
            'std_energy':    np.std(energies),
            'min_energy':    np.min(energies),
            'max_energy':    np.max(energies),
            'median_energy': np.median(energies),

            'num_episodes':        len(delays),
            'num_outliers_removed': 0,   # no longer filtered
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
        print(f"\n{'Algorithm':<15} {'Avg Makespan':<12} {'Std Makespan':<12} {'Avg Energy':<12} {'Std Energy':<12} {'Episodes':<10}")
        print("-" * 100)
        
        # Rows
        for alg_name, metrics in results.items():
            if metrics is None:
                continue
            print(f"{alg_name:<15} "
                  f"{metrics['avg_makespan']:<12.4f} "
                  f"{metrics['std_makespan']:<12.4f} "
                  f"{metrics['avg_energy']:<12.4f} "
                  f"{metrics['std_energy']:<12.4f} "
                  f"{metrics['num_episodes']:<10}")
        
        print("-" * 100)
        
        # Find winners
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) > 0:
            best_delay = min(valid_results.items(), key=lambda x: x[1]['avg_makespan'])
            best_energy = min(valid_results.items(), key=lambda x: x[1]['avg_energy'])
            
            print(f"\n🏆 Best Makespan:  {best_delay[0]:<15} ({best_delay[1]['avg_makespan']:.4f}s)")
            print(f"🏆 Best Energy:    {best_energy[0]:<15} ({best_energy[1]['avg_energy']:.4f}J)")
            
            # Best balanced (using equal weights)
            best_balanced = min(valid_results.items(), 
                              key=lambda x: 0.5 * x[1]['avg_makespan'] / 10.0 + 0.5 * x[1]['avg_energy'])
            print(f"🏆 Best Balanced: {best_balanced[0]:<15}")
        
        print("\n" + "="*100 + "\n")
