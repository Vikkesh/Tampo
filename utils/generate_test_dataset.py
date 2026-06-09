import os
import sys
import json
import argparse
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.base_offloading_env import TaskOffloadingEnv


def _load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'default_config.yaml')
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    config = {}
    for section in ('system', 'computing', 'energy', 'network', 'tasks'):
        config.update(full_config['environment'][section])
    return config


def generate_golden_dataset(num_dags: int, output_file: str):
    print(f"Generating Golden Dataset of {num_dags} DAGs...")

    config = _load_config()
    env = TaskOffloadingEnv(config)

    dataset = []
    np_seed = 0

    while len(dataset) < num_dags:
        import numpy as np
        np.random.seed(np_seed)
        np_seed += 1

        # reset() internally calls _generate_task() which may produce a DAG when
        # task_type == 'dag'.  We collect whatever task was generated.
        env.reset()
        task = env.current_task

        if task is None:
            continue

        # Serialise numpy arrays to plain Python so json.dump works.
        def _serialise(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialise(i) for i in obj]
            return obj

        dataset.append(_serialise(task))

        if len(dataset) % 50 == 0:
            print(f"  Generated {len(dataset)}/{num_dags} DAGs...")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the immutable Golden Test Dataset.")
    parser.add_argument("--num_dags", type=int, default=500,
                        help="Number of DAG workflows to generate (default: 500)")
    parser.add_argument("--output", type=str, default="data/test_dags.json",
                        help="Output JSON path (default: data/test_dags.json)")
    args = parser.parse_args()
    generate_golden_dataset(args.num_dags, args.output)
