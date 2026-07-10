import os
import json
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── allow running from the repo root ──────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.base_offloading_env import TaskOffloadingEnv
from algorithms.rl.tampo import TAMPOFramework
from utils.common_evaluator import CommonEvaluator   # was incorrectly imported as Evaluator
from utils.seeding import set_seed


def _load_env_config():
    with open('configs/default_config.yaml', 'r') as f:
        full_config = yaml.safe_load(f)
    config = {}
    for section in ('system', 'computing', 'energy', 'network', 'tasks'):
        config.update(full_config['environment'][section])
    return config, full_config


def run_benchmark(algorithms, checkpoint_dir, dataset_path, output_dir, use_best=False, seed=None):
    print("=" * 60)
    print("  TO Algorithm Benchmarking Suite")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    env_config, full_config = _load_env_config()

    exp_cfg = full_config.get('experiment', {})
    seed = exp_cfg.get('seed', 42) if seed is None else seed
    set_seed(seed, deterministic_torch=exp_cfg.get('deterministic_torch', False))
    print(f"  Seed: {seed}")

    env = TaskOffloadingEnv(env_config)

    # Load Golden Dataset
    with open(dataset_path, 'r') as f:
        test_dataset = json.load(f)
    print(f"\nLoaded {len(test_dataset)} DAGs from: {dataset_path}")
    env.load_task_dataset(test_dataset)

    evaluator = CommonEvaluator(env, full_config.get('evaluation', {}))
    results = {}

    for algo in algorithms:
        print(f"\n{'─'*50}")
        print(f"  Evaluating: {algo}")
        print(f"{'─'*50}")

        if algo.startswith("TAMPO"):
            # Detect encoder type from algorithm name
            if 'GCN' in algo.upper():
                encoder_type = 'gcn'
            elif 'GAT' in algo.upper():
                encoder_type = 'gat'
            else:
                encoder_type = 'lstm'

            tampo_config = full_config['algorithms']['tampo'].copy()
            tampo_config['encoder_type'] = encoder_type

            agent = TAMPOFramework(env, tampo_config)

            # --use_best loads _best.pth (lowest meta-loss checkpoint) without
            # touching the regular checkpoint so training can still be resumed.
            ckpt_name = (f"tampo_{encoder_type}_checkpoint_best.pth"
                         if use_best
                         else f"tampo_{encoder_type}_checkpoint.pth")
            checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

            # Fallback: if _best.pth requested but doesn't exist, try regular
            if use_best and not os.path.exists(checkpoint_path):
                fallback = os.path.join(checkpoint_dir,
                                        f"tampo_{encoder_type}_checkpoint.pth")
                print(f"  _best.pth not found — falling back to {fallback}")
                checkpoint_path = fallback

            if os.path.exists(checkpoint_path):
                agent.load(checkpoint_path)
                print(f"  Checkpoint loaded: {checkpoint_path}")
            else:
                print(f"  ⚠ No checkpoint found at {checkpoint_path} — using untrained weights.")

            algo_results = evaluator.evaluate_rl_agent(
                agent,
                agent_type='tampo',
                test_dags=test_dataset   # explicit list — same dags, same order for all algos
            )
            results[algo] = algo_results

        else:
            # Placeholder: real baselines will be hooked in here when implemented.
            # Each baseline must inherit BaseAgent (algorithms/baselines/base_agent.py)
            # and be wrapped with FlatVectorWrapper or SequenceWrapper accordingly.
            print(f"  [Placeholder] {algo} not yet implemented — skipping.")

    if not results:
        print("\nNo results collected. Train at least one algorithm first.")
        return

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    _save_csv(results, run_output_dir)
    _save_action_csvs(results, run_output_dir, evaluator)
    _plot_results(results, run_output_dir)
    _print_summary_table(results, evaluator)


def _save_csv(results: dict, output_dir: str):
    import csv
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = ['algorithm', 'avg_makespan', 'std_makespan', 'min_makespan', 'max_makespan',
                  'avg_energy', 'std_energy', 'min_energy', 'max_energy', 'num_episodes',
                  'within_episode_entropy', 'degenerate_episodes']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algo, metrics in results.items():
            if metrics is None:
                continue
            row = {'algorithm': algo}
            row.update({k: metrics.get(k, '') for k in fieldnames[1:-2]})
            summary = metrics.get('action_summary', {})
            row['within_episode_entropy'] = summary.get('mean_per_episode_entropy', '')
            row['degenerate_episodes'] = summary.get('degenerate_episodes', '')
            writer.writerow(row)
    print(f"\nResults CSV → {csv_path}")


def _save_action_csvs(results: dict, output_dir: str, evaluator):
    """
    Write the per-episode action traces and the aggregate action mix.

    `action_traces.csv` is the ground truth for "what did the agent actually do?".
    Each row is one (dag, preference) episode with the full placement sequence, so a
    degenerate policy ("all 20 nodes → cloud") is visible at a glance rather than being
    hidden behind an averaged makespan.
    """
    import csv

    traces_path = os.path.join(output_dir, "action_traces.csv")
    with open(traces_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'dag_index', 'w_delay', 'w_energy', 'num_tasks',
                         'makespan', 'energy', 'action_sequence'])
        for algo, metrics in results.items():
            if not metrics or 'action_traces' not in metrics:
                continue
            for tr in metrics['action_traces']:
                writer.writerow([
                    algo, tr['dag_index'], tr['w_delay'], tr['w_energy'], tr['num_tasks'],
                    f"{tr['makespan']:.6f}", f"{tr['energy']:.6f}",
                    " ".join(evaluator._action_label(a) for a in tr['actions']),
                ])
    print(f"Action traces CSV → {traces_path}")

    dist_path = os.path.join(output_dir, "action_distribution.csv")
    with open(dist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'preference', 'action', 'fraction'])
        for algo, metrics in results.items():
            if not metrics or 'action_summary' not in metrics:
                continue
            summary = metrics['action_summary']
            for a, frac in enumerate(summary['overall_fractions']):
                writer.writerow([algo, 'all', evaluator._action_label(a), f"{frac:.6f}"])
            for pref, fracs in summary['per_preference_fractions'].items():
                for a, frac in enumerate(fracs):
                    writer.writerow([algo, pref, evaluator._action_label(a), f"{frac:.6f}"])
    print(f"Action distribution CSV → {dist_path}")


def _plot_results(results: dict, output_dir: str):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return

    algos = list(valid.keys())
    avg_delays = [valid[a]['avg_makespan'] for a in algos]
    avg_energies = [valid[a]['avg_energy'] for a in algos]
    x = np.arange(len(algos))
    width = 0.35

    # ── Bar Chart ────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(max(8, len(algos) * 2), 6))
    bars1 = ax1.bar(x - width / 2, avg_delays, width, label='Avg Makespan (s)', color='steelblue')
    ax1.set_ylabel('Average Makespan (s)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=30, ha='right')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, avg_energies, width, label='Avg Energy (J)', color='coral')
    ax2.set_ylabel('Average Energy (J)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    plt.title('TO Algorithm Comparison: Delay & Energy')
    fig.tight_layout()
    bar_path = os.path.join(output_dir, 'comparison_bar.png')
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Bar chart → {bar_path}")

    # ── Pareto Front Scatter ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    for i, algo in enumerate(algos):
        ax.scatter(avg_delays[i], avg_energies[i], s=180, color=colors[i],
                   label=algo, zorder=3)
        ax.annotate(algo, (avg_delays[i], avg_energies[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel('Average Makespan (s)')
    ax.set_ylabel('Average Energy (J)')
    ax.set_title('Performance Trade-off: Pareto Front')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    pareto_path = os.path.join(output_dir, 'pareto_front.png')
    plt.savefig(pareto_path, dpi=150)
    plt.close()
    print(f"Pareto plot → {pareto_path}")


def _print_summary_table(results: dict, evaluator):
    evaluator.compare_algorithms(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all trained TO algorithms.")
    parser.add_argument("--algorithms", nargs='+',
                        default=["TAMPO_GCN", "TAMPO_GAT", "TAMPO_LSTM"],
                        help="List of algorithm keys to benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default="models/",
                        help="Directory containing model checkpoint files")
    parser.add_argument("--dataset_path", type=str, default="data/test_dags.json",
                        help="Path to the immutable Golden Test Dataset JSON")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to write CSV and plots")
    parser.add_argument("--use_best", action="store_true",
                        help="Load *_best.pth checkpoints (lowest meta-loss) instead of "
                             "the regular checkpoint. Falls back to the regular checkpoint "
                             "if no _best.pth exists. Does NOT overwrite any files.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override experiment.seed from the config.")
    args = parser.parse_args()
    run_benchmark(args.algorithms, args.checkpoint_dir, args.dataset_path,
                  args.output_dir, use_best=args.use_best, seed=args.seed)
