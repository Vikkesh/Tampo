"""
training_setup.py — Canonical training graph pool loader.

Import this instead of hand-writing DAGParser calls in notebooks.
Keeps the training/testing graph split consistent across all experiments.

TRAINING SIZES : ALL 9 sizes — 10, 15, 20, 25, 30, 35, 40, 45, 50 node graphs
                 20 graphs per size  →  180 total training graphs

ZERO-SHOT TEST : test_dags.json contains 500 brand-new random DAGs.
                 Same 9 sizes, but completely different graph topologies.
                 The agent is tested on graph structures it has never encountered,
                 proving it learned the physics — not specific graph patterns.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dag_parser import DAGParser

# ── Configuration ─────────────────────────────────────────────────────────────
# All 9 available node sizes are used for training.
# Zero-shot generalisation is tested via UNSEEN GRAPH TOPOLOGIES (different
# random DAG instances) in test_dags.json — not unseen sizes.
TRAIN_SIZES     = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # all available sizes
GRAPHS_PER_SIZE = 20                                       # 20 × 9 = 180 total
DATA_ROOT       = 'data/meta_offloading_n'


def load_training_graphs(
    sizes: list = None,
    graphs_per_size: int = None,
    data_root: str = None,
    shuffle: bool = True,
    verbose: bool = True,
) -> list:
    """
    Load and return a mixed pool of training graphs.

    Args:
        sizes           : List of node counts to include. Defaults to TRAIN_SIZES.
        graphs_per_size : Number of graphs to sample per size. Defaults to GRAPHS_PER_SIZE.
        data_root       : Root folder containing offload_random{N} subdirectories.
        shuffle         : Shuffle the pool so sizes are interleaved in meta-batches.
        verbose         : Print loading summary.

    Returns:
        List of raw DAG dicts from DAGParser.load_dataset().
    """
    sizes           = sizes           or TRAIN_SIZES
    graphs_per_size = graphs_per_size or GRAPHS_PER_SIZE
    data_root       = data_root       or DATA_ROOT

    if verbose:
        print("=" * 56)
        print("  Loading training graph pool")
        print(f"  Sizes         : {sizes}")
        print(f"  Graphs / size : {graphs_per_size}")
        print(f"  Expected total: {len(sizes) * graphs_per_size}")
        print(f"  Zero-shot test: unseen topologies in test_dags.json")
        print("=" * 56)

    all_graphs = []
    for size in sizes:
        folder = os.path.join(data_root, f'offload_random{size}')
        try:
            parser = DAGParser(data_folder=folder)
            graphs = parser.load_dataset(num_graphs=graphs_per_size)
            all_graphs.extend(graphs)
            if verbose:
                print(f"  ✓  offload_random{size:2d}  →  {len(graphs)} graphs loaded")
        except Exception as exc:
            if verbose:
                print(f"  ✗  offload_random{size:2d}  →  FAILED: {exc}")

    if shuffle:
        random.shuffle(all_graphs)

    if verbose:
        print(f"\n  Total training pool : {len(all_graphs)} graphs  (shuffled={shuffle})")
        print("  Zero-shot test uses brand-new random topologies in test_dags.json.\n")

    return all_graphs


def build_env_task_list(graphs: list) -> list:
    """
    Convert raw DAGParser output into the dict format expected by
    TaskOffloadingEnv.load_task_dataset().
    """
    return [
        {
            'num_tasks': dag['num_tasks'],
            'tasks':     dag['tasks'],
            'edges':     dag['edges'],
            'adj_matrix':dag['adj_matrix'],
            'size':      sum(t['data_size'] for t in dag['tasks']),
            'cycles':    sum(t['cycles']    for t in dag['tasks']),
        }
        for dag in graphs
    ]


if __name__ == '__main__':
    # Quick sanity check — run from the repo root
    graphs = load_training_graphs(verbose=True)
    print(f"Sample graph sizes: {[g['num_tasks'] for g in graphs[:10]]}")
