"""
Aggregate multi-seed benchmark results into a publication table.

A single seed makes a run repeatable, not representative. This script collects the
per-seed benchmark CSVs produced by `benchmark.py`, reports mean +/- std across seeds
for every (algorithm, metric), and runs a paired Wilcoxon signed-rank test on each
encoder pair -- paired because every encoder is trained and evaluated under the same
set of seeds.

Expected layout (one benchmark run per seed):

    <results_root>/seed_0/run_<timestamp>/benchmark_results.csv
    <results_root>/seed_1/run_<timestamp>/benchmark_results.csv
    ...

If a seed directory holds several run_* dirs, the most recent one is used.

Usage:
    python utils/aggregate_seeds.py --results_root results --seeds 0 1 2 3 4
"""

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

# Metrics aggregated across seeds. Per-seed std/min/max columns are summaries of the
# episodes *within* one run, so averaging them across seeds is not meaningful.
METRICS = ['avg_makespan', 'avg_energy', 'within_episode_entropy', 'degenerate_episodes']


def _latest_run_csv(seed_dir: str):
    """Return the benchmark CSV from the newest run_* directory under seed_dir."""
    runs = sorted(glob.glob(os.path.join(seed_dir, 'run_*')))
    if not runs:
        direct = os.path.join(seed_dir, 'benchmark_results.csv')
        return direct if os.path.exists(direct) else None
    for run in reversed(runs):
        candidate = os.path.join(run, 'benchmark_results.csv')
        if os.path.exists(candidate):
            return candidate
    return None


def collect(results_root: str, seeds):
    """Build {algorithm: {metric: {seed: value}}} from the per-seed benchmark CSVs."""
    data = defaultdict(lambda: defaultdict(dict))
    missing = []

    for seed in seeds:
        seed_dir = os.path.join(results_root, f'seed_{seed}')
        csv_path = _latest_run_csv(seed_dir)
        if csv_path is None:
            missing.append(seed)
            continue

        print(f'  seed {seed}: {csv_path}')
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                algo = row['algorithm']
                for metric in METRICS:
                    raw = row.get(metric, '')
                    if raw == '' or raw is None:
                        continue
                    try:
                        data[algo][metric][seed] = float(raw)
                    except ValueError:
                        pass

    if missing:
        print(f'\n  WARNING: no benchmark CSV for seed(s) {missing}. '
              f'Those seeds are excluded — the table will be under-powered.')
    return data


def summarise(data, seeds, out_dir: str):
    """Print and write mean +/- std per (algorithm, metric) across seeds."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary_across_seeds.csv')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'metric', 'mean', 'std', 'n_seeds', 'per_seed'])

        for algo in sorted(data):
            print(f'\n{algo}')
            for metric in METRICS:
                by_seed = data[algo][metric]
                values = [by_seed[s] for s in seeds if s in by_seed]
                if not values:
                    continue
                mean = statistics.fmean(values)
                # Sample std is undefined for n=1; report 0 rather than crashing.
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                print(f'  {metric:24s} {mean:12.4f} +/- {std:<10.4f} (n={len(values)})')
                writer.writerow([algo, metric, f'{mean:.6f}', f'{std:.6f}', len(values),
                                 ';'.join(f'{v:.6f}' for v in values)])

    print(f'\nSummary CSV -> {out_path}')
    return out_path


def compare(data, seeds, out_dir: str):
    """Paired Wilcoxon signed-rank test between every pair of algorithms."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print('\nscipy not available — skipping significance tests. '
              'The mean +/- std table above is still valid.')
        return None

    algos = sorted(data)
    if len(algos) < 2:
        return None

    out_path = os.path.join(out_dir, 'significance_across_seeds.csv')
    rows = []

    print('\nPaired Wilcoxon signed-rank tests (across seeds)')
    for metric in ['avg_makespan', 'avg_energy']:
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                a, b = algos[i], algos[j]
                # Pair only on seeds where BOTH algorithms have a result.
                shared = [s for s in seeds
                          if s in data[a][metric] and s in data[b][metric]]
                if len(shared) < 2:
                    continue
                xs = [data[a][metric][s] for s in shared]
                ys = [data[b][metric][s] for s in shared]
                if all(x == y for x, y in zip(xs, ys)):
                    print(f'  {metric:14s} {a} vs {b}: identical values — skipped')
                    continue
                try:
                    stat, p = wilcoxon(xs, ys)
                except ValueError as e:
                    print(f'  {metric:14s} {a} vs {b}: test failed ({e})')
                    continue
                verdict = 'significant' if p < 0.05 else 'NOT significant'
                # With n paired seeds there are only 2^n sign assignments, so the smallest
                # two-sided p attainable is 2/2^n. At n=5 that is 0.0625 -- p<0.05 is
                # unreachable no matter how large the effect. Say so rather than let the
                # reader mistake an impossible test for a null result.
                if len(shared) < 6:
                    note = f'  (CANNOT reach p<0.05 with {len(shared)} seeds; floor is ' \
                           f'{2 / 2 ** len(shared):.4f} — use >=6)'
                else:
                    note = ''
                print(f'  {metric:14s} {a} vs {b}: p={p:.4f}  {verdict}'
                      f'  (n={len(shared)}){note}')
                rows.append([metric, a, b, len(shared), f'{stat:.4f}', f'{p:.6f}', verdict])

    if rows:
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'algorithm_a', 'algorithm_b', 'n_seeds',
                             'statistic', 'p_value', 'verdict'])
            writer.writerows(rows)
        print(f'\nSignificance CSV -> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multi-seed benchmark results into a publication table.')
    parser.add_argument('--results_root', type=str, default='results',
                        help='Directory holding seed_<n>/ subdirectories.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Seeds to aggregate.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write the aggregate CSVs '
                             '(default: <results_root>/aggregate).')
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_root, 'aggregate')

    print(f'Collecting from {args.results_root} for seeds {args.seeds}')
    data = collect(args.results_root, args.seeds)
    if not data:
        raise SystemExit(
            f'No benchmark results found under {args.results_root}/seed_*/. '
            f'Run benchmark.py with --output_dir {args.results_root}/seed_<n> first.')

    summarise(data, args.seeds, out_dir)
    compare(data, args.seeds, out_dir)

    print('\nReport a difference only if it exceeds the across-seed spread.')


if __name__ == '__main__':
    main()
