import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dag_parser import DAGParser

def generate_golden_dataset(num_dags: int, output_file: str):
    print(f"Generating Golden Dataset of {num_dags} DAGs from meta_offloading_n...")

    # Load from meta_offloading_n — variable-size graphs for better generalisation testing
    folders = [
        'data/meta_offloading_n/offload_random10',
        'data/meta_offloading_n/offload_random20',
        'data/meta_offloading_n/offload_random30',
        'data/meta_offloading_n/offload_random40',
        'data/meta_offloading_n/offload_random50',
    ]
    
    # Sample equally from each size
    per_folder = max(1, num_dags // len(folders))
    dataset = []
    
    for folder in folders:
        try:
            parser = DAGParser(folder)
            graphs = parser.load_dataset(num_graphs=per_folder)
            dataset.extend(graphs)
            print(f"Loaded {len(graphs)} graphs from {folder}")
        except Exception as e:
            print(f"Warning: Failed to load from {folder}: {e}")

    # Serialise numpy arrays to plain Python so json.dump works
    def _serialise(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(i) for i in obj]
        return obj

    serialised_dataset = [_serialise(task) for task in dataset[:num_dags]]

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(serialised_dataset, f, indent=2)

    print(f"\nDataset of {len(serialised_dataset)} DAGs saved → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the immutable Golden Test Dataset.")
    parser.add_argument("--num_dags", type=int, default=500,
                        help="Number of DAG workflows to generate (default: 500)")
    parser.add_argument("--output", type=str, default="data/test_dags.json",
                        help="Output JSON path (default: data/test_dags.json)")
    args = parser.parse_args()
    generate_golden_dataset(args.num_dags, args.output)
