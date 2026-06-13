# Task Offloading Algorithm Benchmarking — Project README

## What This Project Does

This repository implements a **fair, unified benchmarking framework** for comparing multiple Deep Reinforcement Learning (DRL) algorithms on the Task Offloading (TO) problem.

All algorithms share the same physics engine (`TaskOffloadingEnv`), the same immutable test dataset (`data/test_dags.json`), and the same evaluation metrics. No algorithm calculates its own latency or energy — all physics are handled by the environment.

### ⚙️ How It Works: Physics Engine & Reward System

The heart of this benchmarking framework is the `TaskOffloadingEnv`, which simulates the physical constraints of Edge/Cloud networks while enforcing a strict, multi-objective evaluation schema. 

#### 1. True DAG-Aware Physics Engine
Most standard offloading environments treat tasks as independent or evaluate them sequentially without topological consideration. This framework implements a true Directed Acyclic Graph (DAG) physics engine to ensure fair comparison between architectures:

*   **Topological Stepping (Kahn's Algorithm):** Instead of processing nodes sequentially by ID, the environment utilizes Kahn's algorithm to resolve task dependencies. The agent interacts with the environment precisely in the order of graph execution depth. 
*   **Dependency Blockers:** A task cannot begin computation until **all** of its parent tasks have finished their execution *and* successfully transmitted their output data across the network to the current server.
*   **Dynamic Server Queues:** The engine maintains an internal `server_available` timeline for every device in the network. If an agent dumps multiple tasks onto the Cloud server simultaneously, the tasks will physically queue up, heavily delaying the start time of the later tasks. 
*   **Global Makespan Calculation:** The absolute standard metric for DAG scheduling. The environment calculates the *Makespan*—the total time required from the start of the first node to the completion of the final node. No algorithm calculates its own delay; the environment acts as the absolute ground truth.

#### 2. The Three-Component Penalty Reward System
Because standard DRL agents tend to collapse into "lazy" local minimum policies (e.g., exclusively offloading to the Cloud), the reward system applies distinct pressures to force the agent to balance trade-offs. The total reward combines:

1.  🚀 **Computation Improvement (The Carrot):** The environment calculates how much faster a task runs on the selected server compared to the baseline of running it locally on the mobile device. 
2.  🚦 **Server Congestion Penalty (The Stick):** If an agent repeatedly chooses the same high-powered server (e.g., the Cloud), it is aggressively penalized based on the current length of that server's queue. This forces the agent to dynamically distribute load across Edge and Local devices.
3.  📡 **Communication Penalty (The Stick):** If two tightly coupled tasks (a parent and its child) are scheduled on different machines, the agent receives a strict penalty corresponding to the data transmission overhead. This encourages algorithms to group dependent tasks together logically.

#### 3. User Preference Alignment (MORL)
As demonstrated in the TAMPO framework, the reward system dynamically adapts to user priorities via a **Preference Vector** `[w_delay, w_energy]`.
*   During training, the environment randomizes this vector, forcing the Meta-RL agent to learn a generalized policy.
*   During benchmarking, the algorithms are fed specific profiles (e.g., `[0.8, 0.2]` for Performance Mode, `[0.2, 0.8]` for Battery Saver Mode) to evaluate their zero-shot adaptability.

---

## Project Structure

```
tampo/
├── algorithms/
│   ├── rl/
│   │   └── tampo.py              ← TAMPO meta-RL + Bi-GCN/LSTM encoder
│   └── baselines/
│       ├── base_agent.py         ← Abstract interface all baselines must implement
│       └── [d3qn.py, ...]        ← One file per baseline (added progressively)
├── env/
│   ├── base_offloading_env.py    ← Core TaskOffloadingEnv (ground truth physics)
│   └── wrappers/
│       ├── flat_vector_wrapper.py   ← For D3QN, SAC, MAPPO (1D flat obs)
│       └── sequence_wrapper.py     ← For TPTO, MTD3 (topo-sorted sequence obs)
├── utils/
│   ├── common_evaluator.py          ← Unified metrics evaluation
│   ├── generate_test_dataset.py     ← Generates the Golden Test Dataset
│   └── dag_parser.py
├── configs/
│   └── default_config.yaml       ← All system/algorithm hyperparameters
├── data/
│   └── test_dags.json            ← Immutable 500-DAG evaluation set (generated once)
├── models/                       ← Trained checkpoints (created after training)
├── results/                      ← CSV + plots output from benchmark.py
├── dev_logs/                     ← Full implementation history
├── benchmark.py                  ← Master evaluation script
├── main.py                       ← Interactive training runner
├── requirements.txt              ← All dependencies
├── Papers referred.md            ← All cited references
└── Colab_Test_Run.ipynb          ← Google Colab notebook for testing
```

---

## Implemented Algorithms

| Algorithm | Type | Encoder | Observation Wrapper |
|---|---|---|---|
| TAMPO-GCN | TAMPO meta-RL | Bi-GCNConv(6→16)→(16→1)→FNN | None (raw graph) |
| TAMPO-GAT | TAMPO meta-RL | Bi-GATv2Conv(6→16,heads=4)→(16→1)→FNN | None (raw graph) |
| TAMPO-LSTM | TAMPO meta-RL | BiLSTM → Attention | None (raw graph) |

*More baselines will be added to this table as they are implemented.*

---

## How to Run Algorithms

This section covers every method available to train and evaluate TO algorithms.

### Method 1 — Interactive CLI (Local, Recommended for Development)

The simplest way to run any algorithm locally. `main.py` will prompt you for which algorithms to run and how many training iterations to use.

```bash
python main.py
```

Example session:
```
Run TAMPO-LSTM? [y/n]: n
Run TAMPO-GCN?  [y/n]: y
Number of TAMPO meta-learning iterations: 10
```

**When to use this:** During development when you want to quickly test a single algorithm or compare two algorithms one after the other.

---

### Method 2 — Config File (Local, Recommended for Reproducibility)

Edit `configs/default_config.yaml` to set the algorithm and all hyperparameters before running. This is the most reproducible approach as the exact config can be committed to Git.

```yaml
algorithms:
  tampo:
    encoder_type: 'gcn'   # Change to 'lstm' for the LSTM variant
    hidden_dims: [128, 128]
    num_meta_iterations: 100
    meta_batch_size: 10
```

Then run without any prompts:
```bash
python main.py --config configs/default_config.yaml
```

**When to use this:** When running long overnight training jobs, or when you want the exact run to be reproducible from a committed config file.

---

### Method 3 — Google Colab Notebook (Recommended for GPU Training)

Use `Colab_Test_Run.ipynb` for full training and benchmarking on Colab's free GPUs. Run cells top to bottom.

```
Step 1 → Clone repo
Step 2 → Install dependencies from requirements.txt
Step 3 → Generate the Golden Dataset (run once only)
Step 4 → Train an algorithm
Step 5 → Run benchmarks
Step 6 → Download results
```

See the **Colab Workflow** section below for the exact commands.

**When to use this:** When running long training jobs that would be slow on a local machine, or when sharing the experiment with collaborators.

---

### Method 4 — benchmark.py Direct Evaluation (No Training)

If you already have a trained checkpoint saved in `models/`, you can skip training and go straight to evaluation. This runs all selected algorithms against the Golden Dataset and outputs a CSV and plots.

```bash
python benchmark.py \
  --algorithms TAMPO_GCN TAMPO_LSTM \
  --checkpoint_dir models/ \
  --dataset_path data/test_dags.json \
  --output_dir results/
```

| Argument | Default | Description |
|---|---|---|
| `--algorithms` | `TAMPO_GCN TAMPO_LSTM` | Space-separated list of algorithms to evaluate |
| `--checkpoint_dir` | `models/` | Folder containing `.pth` checkpoint files |
| `--dataset_path` | `data/test_dags.json` | The immutable Golden Test Dataset |
| `--output_dir` | `results/` | Where to write the CSV and plots |

**When to use this:** After training is complete and you just want to regenerate comparison charts.

---

### Method 5 — Programmatic API (Advanced, For Custom Scripts)

Each algorithm can be instantiated and trained directly in Python without going through `main.py`. This is useful for hyperparameter sweeps or embedding the training loop inside a larger experiment script.

```python
import yaml
from env.base_offloading_env import TaskOffloadingEnv
from algorithms.rl.tampo import TAMPOFramework

with open('configs/default_config.yaml') as f:
    full_config = yaml.safe_load(f)

cfg = {}
for sec in ('system', 'computing', 'energy', 'network', 'tasks'):
    cfg.update(full_config['environment'][sec])

env = TaskOffloadingEnv(cfg)
tampo_cfg = full_config['algorithms']['tampo']
tampo_cfg['encoder_type'] = 'gcn'   # or 'lstm'

agent = TAMPOFramework(env, tampo_cfg)
agent.train(num_meta_iterations=100)
agent.save('models/tampo_gcn_checkpoint.pth')
```

**When to use this:** Hyperparameter sweeps, ablation studies, or integrating training into a larger pipeline.

---


## How to Configure and Switch Algorithms

### 1. Configuring TAMPO (GCN vs LSTM)

Algorithm configurations live in `configs/default_config.yaml`. Here is the exact structure for TAMPO and the **recommended values**:

```yaml
algorithms:
  tampo:
    enabled: true                  # Set to true to run this algorithm
    hidden_dims: [256, 256]        # Recommended: [256, 256] for complex DAGs, [128, 128] for simpler ones
    num_attention_heads: 8         # Recommended: 8 (used only when encoder_type is 'lstm')
    encoder_type: 'gcn'            # Options: 'gcn' (Bi-GCN), 'gat' (Bi-GATv2), or 'lstm' (BiLSTM)
    num_meta_iterations: 100       # Recommended: 100+ for actual training, 1 for smoke tests
    meta_batch_size: 10            # Recommended: 10-20 to ensure stable meta-gradients

    # GAT-specific parameters (only active when encoder_type: 'gat')
    num_gat_heads: 4               # Attention heads for GATv2Conv layer 1 (intermediate dim = gat_hidden_dim)
    gat_hidden_dim: 16             # Must be divisible by num_gat_heads; keeps intermediate dim equal to GCN
    gat_add_self_loops: true       # Match GCNConv default for fair comparison
```

### 2. How to Compare Multiple Algorithms

To compare two or more algorithms on the exact same dataset, you simply pass their names as a space-separated list to the benchmarking script:

```bash
# Compare TAMPO-GCN and TAMPO-LSTM
python benchmark.py --algorithms TAMPO_GCN TAMPO_LSTM

# Future: Compare all implemented algorithms at once
python benchmark.py --algorithms TAMPO_GCN TAMPO_LSTM D3QN TPTO
```
The script will load the saved model checkpoints for each algorithm, run them against the dataset, and plot them side-by-side in `results/comparison_bar.png` and `results/pareto_front.png`.

---

## Understanding the Benchmarking Lifecycle

To ensure a fair comparison, the framework strictly separates the **Data**, the **Models**, and the **Results** into distinct directories.

### 1. The Dataset (`data/`)
*   **Generate Once, Use Forever:** You must generate the dataset **ONLY ONCE** before you begin your experiments by running `python utils/generate_test_dataset.py`.
*   **Why?** This creates an immutable JSON file (`data/test_dags.json`) containing 500 fixed workflow graphs. Every algorithm you train will be evaluated against this exact same file. If you regenerate this file, your new algorithms will be tested on different graphs than your old algorithms, completely destroying the fairness of the benchmark.

### 2. The Checkpoints (`models/`)
*   **What happens when you run an algorithm?** When you train an algorithm (e.g., via `main.py`), the script initializes the neural network weights and improves them over time using reinforcement learning.
*   **How are they saved?** At the end of training, the framework automatically saves the neural network's final "brain" as a `.pth` file inside the `models/` directory (e.g., `models/tampo_gcn_checkpoint.pth`).
*   **What do they mean?** These checkpoints mean you do not have to retrain the algorithm every time you want to evaluate it. The `benchmark.py` script automatically looks in the `models/` folder, loads the pre-trained `.pth` file, and uses it to make offloading decisions on the dataset.

### 3. The Results (`results/`)
*   When you run `benchmark.py`, it evaluates the checkpoints from `models/` against the dataset from `data/`.
*   It automatically creates the `results/` directory and populates it with a CSV of the raw metrics, as well as bar charts and Pareto front scatter plots visually comparing the algorithms you selected.

---

## Colab Workflow (Recommended for GPU Training)

### Step 1 — Upload or Clone
```python
# In Colab
!git clone https://github.com/your-repo/tampo.git
%cd tampo
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Generate the Golden Dataset (Run Once Only)
```bash
python utils/generate_test_dataset.py --num_dags 500
```
> ⚠️ **Never regenerate** `data/test_dags.json` between algorithm runs. It is the shared immutable test set. Regenerating it would make all previous benchmark results incomparable.

### Step 4 — Train an Algorithm
```bash
python main.py
```

### Step 5 — Run Benchmarks
```bash
python benchmark.py --algorithms TAMPO_GCN TAMPO_LSTM --checkpoint_dir models/ --output_dir results/
```

Results will be in `results/benchmark_results.csv`, `results/comparison_bar.png`, and `results/pareto_front.png`.

### Step 6 — Download Results
```python
# In Colab
from google.colab import files
files.download('results/comparison_bar.png')
files.download('results/pareto_front.png')
files.download('results/benchmark_results.csv')
```

---

## Adding a New Baseline

1. Get the reference paper (see `.agents/agents.md` Rule 1 — must cite, no scratch implementations).
2. Create `algorithms/baselines/<algorithm_name>.py`.
3. Inherit from `BaseAgent` in `algorithms/baselines/base_agent.py`.
4. Wrap the env with `FlatVectorWrapper` (for MLP/standard RL) or `SequenceWrapper` (for sequence models). Graph-based algorithms use no wrapper.
5. Update `Papers referred.md` with the full IEEE-format citation.
6. Add the algorithm key to `benchmark.py` and hook it into the evaluation loop.
7. Update the **Implemented Algorithms** table in this README.

---

## Papers

See [Papers referred.md](Papers%20referred.md) for all full citations.