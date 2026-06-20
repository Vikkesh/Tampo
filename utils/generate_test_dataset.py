import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dag_parser import DAGParser

def generate_golden_dataset(num_dags: int, output_file: str):
    print(f"Generating Golden Dataset of {num_dags} DAGs from meta_offloading_n...")

    # All 9 available graph sizes — sample equally across the full complexity spectrum.
    # The agent is only trained on sizes 10–30; sizes 35–50 are NEVER seen during
    # training, making them a true zero-shot generalisation test.
    folders = [
        'data/meta_offloading_n/offload_random10',
        'data/meta_offloading_n/offload_random15',
        'data/meta_offloading_n/offload_random20',
        'data/meta_offloading_n/offload_random25',
        'data/meta_offloading_n/offload_random30',
        'data/meta_offloading_n/offload_random35',
        'data/meta_offloading_n/offload_random40',
        'data/meta_offloading_n/offload_random45',
        'data/meta_offloading_n/offload_random50',
    ]

    # Sample equally from each size bucket
    per_folder = max(1, num_dags // len(folders))
    dataset = []

    for folder in folders:
        try:
            parser = DAGParser(folder)
            # Offset by 20 to ensure 100% segregation. 
            # The training pool uses the first 20 graphs (files 0-19) from each folder.
            # The test pool uses the remaining graphs starting from file 20.
            graphs = parser.load_dataset(num_graphs=per_folder, offset=20)
            dataset.extend(graphs)
            size_label = folder.split('offload_random')[-1]
            print(f"  Loaded {len(graphs):3d} graphs from offload_random{size_label}")
        except Exception as e:
            print(f"  Warning: Failed to load from {folder}: {e}")

    print(f"\nTotal graphs collected before trim: {len(dataset)}")

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

    print(f"Dataset of {len(serialised_dataset)} DAGs saved → {output_file}")
    print(f"\nSize breakdown: ~{per_folder} graphs per size × {len(folders)} sizes")
    print(f"  Train-seen sizes  (10–30):  {per_folder * 5} graphs")
    print(f"  Zero-shot sizes   (35–50):  {per_folder * 4} graphs  ← agent never trained on these")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the immutable Golden Test Dataset.")
    parser.add_argument("--num_dags", type=int, default=500,
                        help="Number of DAG workflows to generate (default: 500)")
    parser.add_argument("--output", type=str, default="data/test_dags.json",
                        help="Output JSON path (default: data/test_dags.json)")
    args = parser.parse_args()
    generate_golden_dataset(args.num_dags, args.output)
