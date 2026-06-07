import numpy as np
import yaml
import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from env.base_offloading_env import TaskOffloadingEnv
from utils.dag_parser import DAGParser
from utils.metrics import calculate_hypervolume, normalize_objectives
from utils.common_evaluator import CommonEvaluator

from algorithms.heuristic.heft import HEFTScheduler
from algorithms.heuristic.pso import PSOScheduler
from algorithms.heuristic.ga import GAScheduler
from algorithms.rl.ppo_baseline import PPOAgent
from algorithms.rl.gmorl import GMORLAgent
from algorithms.rl.tampo import TAMPOFramework

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config):
    """Setup the task offloading environment"""
    env_config = {
        **config.get('system', {}),
        **config.get('computing', {}),
        **config.get('energy', {}),
        **config.get('network', {}),
        **config.get('tasks', {})
    }
    env = TaskOffloadingEnv(env_config)
    return env

def get_user_input():
    """Get user input for which algorithms to run"""
    print("\n" + "="*70)
    print(" "*20 + "ALGORITHM SELECTION")
    print("="*70 + "\n")
    
    algorithms = {}
    
    # Heuristic algorithms
    print("HEURISTIC ALGORITHMS:")
    print("-" * 40)
    
    heft = input("Run HEFT? (yes/no): ").strip().lower()
    algorithms['HEFT'] = heft in ['yes', 'y']
    
    pso = input("Run PSO? (yes/no): ").strip().lower()
    algorithms['PSO'] = pso in ['yes', 'y']
    
    ga = input("Run GA? (yes/no): ").strip().lower()
    algorithms['GA'] = ga in ['yes', 'y']
    
    if any([algorithms['HEFT'], algorithms['PSO'], algorithms['GA']]):
        heuristic_tasks = input("Number of test tasks for heuristics (default 10): ").strip()
        algorithms['heuristic_tasks'] = int(heuristic_tasks) if heuristic_tasks else 10
    else:
        algorithms['heuristic_tasks'] = 10
    
    # RL algorithms
    print("\n\nREINFORCEMENT LEARNING ALGORITHMS:")
    print("-" * 40)
    
    ppo_run = input("Run PPO? (yes/no): ").strip().lower()
    algorithms['PPO'] = ppo_run in ['yes', 'y']
    
    gmorl_run = input("Run GMORL? (yes/no): ").strip().lower()
    algorithms['GMORL'] = gmorl_run in ['yes', 'y']
    
    tampo_lstm_run = input("Run TAMPO-LSTM? (yes/no): ").strip().lower()
    algorithms['TAMPO_LSTM'] = tampo_lstm_run in ['yes', 'y']
    
    tampo_gcn_run = input("Run TAMPO-GCN? (yes/no): ").strip().lower()
    algorithms['TAMPO_GCN'] = tampo_gcn_run in ['yes', 'y']

    tampo_gat_run = input("Run TAMPO-GAT? (yes/no): ").strip().lower()
    algorithms['TAMPO_GAT'] = tampo_gat_run in ['yes', 'y']

    # Common training/evaluation parameters for RL
    if any([algorithms['PPO'], algorithms['GMORL'], algorithms['TAMPO_LSTM'], algorithms['TAMPO_GCN'], algorithms['TAMPO_GAT']]):
        print("\n\nRL TRAINING & EVALUATION PARAMETERS:")
        print("-" * 40)
        
        # Training episodes/iterations
        if algorithms['PPO']:
            ppo_episodes = input("Number of training episodes for PPO (default 100): ").strip()
            algorithms['ppo_episodes'] = int(ppo_episodes) if ppo_episodes else 100
        else:
            algorithms['ppo_episodes'] = 100
        
        if algorithms['GMORL']:
            gmorl_episodes = input("Number of training episodes for GMORL (default 100): ").strip()
            algorithms['gmorl_episodes'] = int(gmorl_episodes) if gmorl_episodes else 100
        else:
            algorithms['gmorl_episodes'] = 100
        
        if algorithms['TAMPO_LSTM'] or algorithms['TAMPO_GCN'] or algorithms['TAMPO_GAT']:
            tampo_iterations = input("Number of meta-iterations for TAMPO (default 100): ").strip()
            algorithms['tampo_iterations'] = int(tampo_iterations) if tampo_iterations else 100
        else:
            algorithms['tampo_iterations'] = 100
        
        # Common evaluation episodes
        eval_episodes = input("Number of evaluation episodes for all RL algorithms (default 20): ").strip()
        algorithms['eval_episodes'] = int(eval_episodes) if eval_episodes else 20
    else:
        algorithms['ppo_episodes'] = 100
        algorithms['gmorl_episodes'] = 100
        algorithms['tampo_iterations'] = 100
        algorithms['eval_episodes'] = 20
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "CONFIGURATION SUMMARY")
    print("="*70)
    
    selected = []
    if algorithms['HEFT']:
        selected.append(f"HEFT")
    if algorithms['PSO']:
        selected.append(f"PSO")
    if algorithms['GA']:
        selected.append(f"GA")
    if algorithms['PPO']:
        selected.append(f"PPO ({algorithms['ppo_episodes']} training episodes)")
    if algorithms['GMORL']:
        selected.append(f"GMORL ({algorithms['gmorl_episodes']} training episodes)")
    if algorithms['TAMPO_LSTM']:
        selected.append(f"TAMPO_LSTM ({algorithms['tampo_iterations']} meta-iterations)")
    if algorithms['TAMPO_GCN']:
        selected.append(f"TAMPO_GCN ({algorithms['tampo_iterations']} meta-iterations)")
    if algorithms['TAMPO_GAT']:
        selected.append(f"TAMPO_GAT ({algorithms['tampo_iterations']} meta-iterations)")

    if len(selected) == 0:
        print("\n⚠️  No algorithms selected. Exiting...")
        return None
    
    print("\nSelected Algorithms:")
    for i, alg in enumerate(selected, 1):
        print(f"  {i}. {alg}")
    
    if any([algorithms['HEFT'], algorithms['PSO'], algorithms['GA']]):
        print(f"\nHeuristic test tasks: {algorithms['heuristic_tasks']}")
    
    if any([algorithms['PPO'], algorithms['GMORL'], algorithms['TAMPO_LSTM'], algorithms['TAMPO_GCN'], algorithms['TAMPO_GAT']]):
        print(f"RL evaluation episodes: {algorithms['eval_episodes']} (common for all RL algorithms)")
    
    print("\n" + "="*70)
    
    confirm = input("\nProceed with this configuration? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Configuration cancelled.")
        return None
    
    return algorithms

def test_heft(env, dag_parser, evaluator, num_tasks=10):
    """Test HEFT algorithm using common evaluator"""
    print("\n[1/6] Testing HEFT...")
    heft = HEFTScheduler(env)
    result = evaluator.evaluate_heuristic(heft, dag_parser, num_tasks)
    if result:
        print("✓ HEFT completed")
    return result

def test_pso(env, dag_parser, config, evaluator, num_tasks=10):
    """Test PSO algorithm using common evaluator"""
    print("\n[2/6] Testing PSO...")
    pso = PSOScheduler(env, config['algorithms']['pso'])
    result = evaluator.evaluate_heuristic(pso, dag_parser, num_tasks)
    if result:
        print("✓ PSO completed")
    return result

def test_ga(env, dag_parser, config, evaluator, num_tasks=10):
    """Test GA algorithm using common evaluator"""
    print("\n[3/6] Testing GA...")
    ga = GAScheduler(env, config['algorithms']['ga'])
    result = evaluator.evaluate_heuristic(ga, dag_parser, num_tasks)
    if result:
        print("✓ GA completed")
    return result

def test_ppo(env, config, evaluator, train_episodes=100):
    """Test PPO algorithm using common evaluator"""
    
    # Define checkpoint path
    checkpoint_path = "models/ppo_checkpoint.pth"
    
    # Always use checkpoint if it exists - no user prompt
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found existing PPO checkpoint - resuming training")
    else:
        print(f"\n🆕 No existing checkpoint found - starting fresh training")
    
    print(f"\n[4/6] Training PPO ({train_episodes} episodes)...")
    
    # Pass checkpoint path to agent initialization
    ppo = PPOAgent(env, config['training'], model_path=checkpoint_path)
    ppo.train(num_episodes=train_episodes, checkpoint_path=checkpoint_path)
    
    print(f"\nEvaluating PPO...")
    result = evaluator.evaluate_rl_agent(ppo, agent_type='ppo')
    if result:
        print("✓ PPO completed")
    return result

def test_gmorl(env, config, evaluator, train_episodes=100):
    """Test GMORL algorithm using common evaluator"""
    
    # Define checkpoint path
    checkpoint_path = "models/gmorl_checkpoint.pth"
    
    # Always use checkpoint if it exists - no user prompt
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found existing GMORL checkpoint - resuming training")
    else:
        print(f"\n🆕 No existing checkpoint found - starting fresh training")
    
    print(f"\n[5/6] Training GMORL ({train_episodes} episodes)...")
    
    # Pass checkpoint path to agent initialization
    gmorl = GMORLAgent(env, config['training'], model_path=checkpoint_path)
    gmorl.train(num_episodes=train_episodes, checkpoint_path=checkpoint_path)
    
    print(f"\nEvaluating GMORL...")
    result = evaluator.evaluate_rl_agent(gmorl, agent_type='gmorl')
    if result:
        print("✓ GMORL completed")
    return result

def test_tampo(env, dag_parser, config, evaluator, train_iterations=100, encoder_type='lstm'):
    """Test TAM-PO algorithm using common evaluator"""
    
    # Combine training config with TAMPO specific config
    tampo_config = {**config['training'], **config['algorithms'].get('tampo', {})}
    tampo_config['encoder_type'] = encoder_type
    
    # Define checkpoint path dynamically based on encoder type
    checkpoint_path = f"models/tampo_{encoder_type}_checkpoint.pth"
    
    # Always use checkpoint if it exists - no user prompt
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found existing TAM-PO ({encoder_type.upper()}) checkpoint - resuming training")
    else:
        print(f"\n🆕 No existing checkpoint found - starting fresh training")
    
    # Load task dataset
    print("\n📚 Loading task dataset...")
    task_graphs = dag_parser.load_dataset(num_graphs=50)
    
    if len(task_graphs) == 0:
        print("⚠️  Warning: No task graphs loaded.")
        return None
    
    # Convert DAG format
    tasks_for_env = []
    for dag in task_graphs:
        task = {
            'num_tasks': dag['num_tasks'],
            'tasks': dag['tasks'],
            'edges': dag['edges'],
            'adj_matrix': dag['adj_matrix'],
            'size': sum(t['data_size'] for t in dag['tasks']),
            'cycles': sum(t['cycles'] for t in dag['tasks'])
        }
        tasks_for_env.append(task)
    
    # Load tasks into environment
    env.load_task_dataset(tasks_for_env)
    print(f"✓ Loaded {len(tasks_for_env)} tasks")
    
    # Create TAM-PO framework - checkpoint loaded automatically in __init__
    print(f"\n[6/6] Training TAM-PO ({train_iterations} meta-iterations)...")
    tampo_framework = TAMPOFramework(env, tampo_config, model_path=checkpoint_path)
    
    # Train
    tampo_framework.train(
        num_iterations=train_iterations,
        meta_batch_size=min(10, len(tasks_for_env)),
        checkpoint_path=checkpoint_path
    )
    
    # Evaluate using common evaluator
    print(f"\nEvaluating TAM-PO...")
    result = evaluator.evaluate_rl_agent(tampo_framework, agent_type='tampo')
    if result:
        print("✓ TAM-PO completed")
    
    if hasattr(env, 'clear_task_selection'):
        env.clear_task_selection()
    
    return result

def save_results(results, output_dir):
    """Save results to JSON and generate plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) == 0:
        print("\n⚠️  No valid results to save")
        return
    
    # Save JSON
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(valid_results, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots
    plot_comparison(valid_results, output_dir)

def plot_comparison(results, output_dir):
    """Generate comparison plots"""
    algorithms = list(results.keys())
    delays = [results[alg]['avg_delay'] for alg in algorithms]
    energies = [results[alg]['avg_energy'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Delay comparison
    color_map = plt.get_cmap('tab10')
    colors = [color_map(i % color_map.N) for i in range(len(algorithms))]
    ax1.bar(algorithms, delays, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Average Delay (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Delay Comparison', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (alg, delay) in enumerate(zip(algorithms, delays)):
        ax1.text(i, delay, f'{delay:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Energy comparison
    ax2.bar(algorithms, energies, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Energy (J)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (alg, energy) in enumerate(zip(algorithms, energies)):
        ax2.text(i, energy, f'{energy:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_file}")
    plt.close()
    
    # Pareto front plot
    plt.figure(figsize=(10, 8))
    
    for i, alg in enumerate(algorithms):
        plt.scatter(
            results[alg]['avg_delay'], 
            results[alg]['avg_energy'],
            label=alg, 
            s=300, 
            alpha=0.7, 
            color=colors[i % len(colors)],
            edgecolors='black',
            linewidths=2
        )
        
        plt.annotate(
            alg,
            (results[alg]['avg_delay'], results[alg]['avg_energy']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
        )
    
    plt.xlabel('Average Delay (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Energy (J)', fontsize=14, fontweight='bold')
    plt.title('Pareto Front: Delay vs Energy Trade-off', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.4, linestyle='--')
    
    if len(delays) > 0 and len(energies) > 0:
        plt.axvline(x=min(delays), color='green', linestyle='--', alpha=0.5, label='Best Delay')
        plt.axhline(y=min(energies), color='blue', linestyle='--', alpha=0.5, label='Best Energy')
    
    pareto_file = os.path.join(output_dir, 'pareto_front.png')
    plt.savefig(pareto_file, dpi=300, bbox_inches='tight')
    print(f"Pareto front plot saved to {pareto_file}")
    plt.close()

def main():
    # Print banner
    print("\n" + "="*70)
    print(" "*10 + "TASK OFFLOADING ALGORITHM COMPARISON FRAMEWORK")
    print(" "*20 + "Interactive Mode")
    print("="*70 + "\n")
    
    # Get user input
    user_choices = get_user_input()
    
    if user_choices is None:
        return
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_config('configs/default_config.yaml')
    print("✓ Configuration loaded")
    
    # Setup environment
    print("\n🏗️  Setting up environment...")
    env = setup_environment(config)
    print(f"✓ Environment created with {env.num_servers} servers")
    
    # Setup DAG parser
    dag_parser = DAGParser(data_folder="data/meta_offloading_20/offload_random20_1")
    
    # Create common evaluator
    print("\n🔧 Initializing Common Evaluator...")
    evaluator = CommonEvaluator(env, {
        'eval_episodes': user_choices['eval_episodes'],
        'max_steps': config['system']['max_steps']
    })
    
    # Run selected algorithms
    results = {}
    
    print("\n" + "="*70)
    print("🚀 Starting Algorithm Execution")
    print("="*70)
    
    # Heuristic algorithms
    if user_choices['HEFT']:
        result = test_heft(env, dag_parser, evaluator, user_choices['heuristic_tasks'])
        if result:
            results['HEFT'] = result
    
    if user_choices['PSO']:
        result = test_pso(env, dag_parser, config, evaluator, user_choices['heuristic_tasks'])
        if result:
            results['PSO'] = result
    
    if user_choices['GA']:
        result = test_ga(env, dag_parser, config, evaluator, user_choices['heuristic_tasks'])
        if result:
            results['GA'] = result
    
    # RL algorithms
    if user_choices['PPO']:
        result = test_ppo(env, config, evaluator, user_choices['ppo_episodes'])
        if result:
            results['PPO'] = result
    
    if user_choices['GMORL']:
        result = test_gmorl(env, config, evaluator, user_choices['gmorl_episodes'])
        if result:
            results['GMORL'] = result
    
    if user_choices['TAMPO_LSTM']:
        result = test_tampo(
            env, dag_parser, config, evaluator,
            user_choices['tampo_iterations'],
            encoder_type='lstm'
        )
        if result:
            results['TAMPO_LSTM'] = result
    
    if user_choices['TAMPO_GCN']:
        result = test_tampo(
            env, dag_parser, config, evaluator,
            user_choices['tampo_iterations'],
            encoder_type='gcn'
        )
        if result:
            results['TAMPO_GCN'] = result

    if user_choices['TAMPO_GAT']:
        result = test_tampo(
            env, dag_parser, config, evaluator,
            user_choices['tampo_iterations'],
            encoder_type='gat'
        )
        if result:
            results['TAMPO_GAT'] = result

    # Display detailed comparison using common evaluator
    if len(results) > 0:
        evaluator.compare_algorithms(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', timestamp)
    save_results(results, output_dir)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
