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
2.  🚦 **Server Congestion Penalty (The Stick):** If an agent repeatedly chooses the same high-powered server (e.g., the Cloud), it is penalized by how long the task actually sat in that server's queue. This forces the agent to dynamically distribute load across Edge and Local devices.
3.  📡 **Communication Penalty (The Stick):** If two tightly coupled tasks (a parent and its child) are scheduled on different machines, the agent is penalized by the transmission time actually incurred. This encourages algorithms to group dependent tasks together logically.

Both penalties are expressed as a **relative overhead**, `cost / (cost + local_delay)` ∈ [0, 1) — dimensionless, smooth, and never fully saturated, so the term keeps a usable gradient at any congestion level and needs no retuning when task sizes or clock speeds change.

`step()` exposes the two objective components separately as `info['r_delay']` and `info['r_energy']`, each clipped to [-1, 1]. **Multi-objective learners must discount these**, not re-derive improvements from raw delay/energy — doing the latter silently discards both penalties. See `dev_logs/reward_signal_and_determinism_overhaul.md`.

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

---

## Graph Encoder & Observability Overhaul (2026-07-10)

> 📖 **Reading the output:** see **[docs/READING_RESULTS.md](docs/READING_RESULTS.md)** — written in two layers, plain-English first, technical second.

Measured with dropout disabled, all three encoders emitted **identical logits at every node** of a DAG (`max|logit - logit_step0| = 0.00e+00`). Each one picked a single server and sent the whole graph there. Every GCN/GAT/LSTM comparison produced before this fix is void. Full analysis in `dev_logs/graph_encoder_and_observability_overhaul.md`.

| Area | What Changed | Why It Matters |
|---|---|---|
| `base_offloading_env.py` | Node features `6 → 9`: `is_current`, `is_scheduled`, `assigned_server` | The policy had no way to know **which node** it was scheduling. (The critic did — it reads the flat obs.) |
| `base_offloading_env.py` | `get_server_features()` reports real queue times | `server_loads` was zeroed in `reset()` and **never written again**; `_execute_offloading` advances `server_available`, a different array. The agent was blind to congestion. |
| `tampo.py` | GNN emits per-node embeddings; no more `.mean()` to a scalar | The whole graph was crushed to **2 floats**, then broadcast to every node slot — so decoder attention ran over N identical keys. |
| `tampo.py` | Decoder indexes the current node's embedding (pointer-style) | Makes the policy a function of the node being placed, not just of the graph. |
| `tampo.py` | `_select_action` forces `eval()` | `.eval()` was only called inside the MAML inner loop, so "deterministic" benchmark actions were argmaxes over **dropout-corrupted** logits. |
| `tampo.py`, `common_evaluator.py`, `benchmark.py` | Action histograms, entropy, per-episode traces | A degenerate all-cloud policy posts a respectable makespan. None of this was visible before. |

### Was the GCN still faithful to GDRL?

**More faithful than before.** `GDRL/Feature.py` ends with `gnn2(...).squeeze(2)` → a `[35]`
per-node vector which it concatenates whole. **GDRL never pools.** The `.mean()` was TAMPO's own
addition. Restoring per-node outputs moves the code *toward* the reference.

Remaining deviations — all pre-existing, now documented in `Papers referred.md`:
- The **bidirectional stream** is TAMPO's (`grep -c "bwd" GDRL/Feature.py` → `0`).
- A **mean readout** replaces GDRL's fixed-size concatenation, which assumes 35 nodes; our DAGs are 10–50.
- A **sequential pointer decoder** (Vinyals et al., 2015) replaces GDRL's single-shot TRPO actor head, because server queue state evolves as nodes are placed.

You can still cite Cai et al. (2025) for the graph feature extractor. Do not describe it as a reproduction of GDRL — suggested phrasing is in `Papers referred.md`.

> ⚠ **Existing checkpoints are incompatible.** The node feature width and GNN output shape both changed. Delete `models/*.pth` and retrain; `TAMPOFramework.load()` raises an explanatory error rather than a shape mismatch.

### What this does and does not prove

It establishes that the policy is now **capable** of a per-node, congestion-aware, preference-conditioned schedule, and that the GCN-vs-GAT comparison is controlled (identical skeleton; the conv operator is the only variable). It does **not** yet show that any encoder learns to use that capability — watch `within_episode_entropy` climb above zero before comparing encoders.

### Training budget — the old runs were barely trained

`num_meta_iterations`, `meta_batch_size`, `num_episodes` and `num_attention_heads` in `default_config.yaml` are **inert**: no Python reads them. Iteration count reaches `train()` from the caller — `Colab_Test_Run.ipynb`'s `NUM_ITERATIONS` (the source of truth) or `main.py`'s `train_iterations`. The YAML keys are now annotated as such.

The real value was `NUM_ITERATIONS = 75`. Measured, at `meta_learning_rate = 5.0e-5` Adam moves each weight by ~`9.3e-6` per meta-iteration against a mean init magnitude of `0.029`:

| iterations | mean \|Δparam\| as % of init scale |
|---|---|
| **75** (old) | **2.4%** — essentially untrained |
| 300 | 9.7% |
| **1000** (new target) | **32.4%** |

**Do not raise the learning rate to compensate.** A measured sweep collapsed the policy (action entropy → `0.00`) at both `1.5e-4` and `3.0e-4`, while the loss stayed finite and trended *down* in every case. Only the new action-entropy diagnostic catches it.

### Wall-clock cost is the real constraint

Measured per-iteration cost, before vs after this overhaul (CPU, `meta_batch=4`, 10/20/30-node graphs):

| encoder | before | after | change |
|---|---|---|---|
| GCN | 3.52 s | 3.28 s | 1.08× faster |
| GAT | 8.30 s | **4.12 s** | **2.02× faster** (batched conv replaced a per-graph Python loop) |
| LSTM | 14.09 s | 14.11 s | unchanged |

The overhaul does not make iterations slower. But **LSTM costs ~4× GCN** — CuDNN cannot be used for MAML's second-order gradients — and it dominates any run that trains all three encoders in one loop. A free-tier Colab T4 session (~4 h kill) fits roughly 75 iterations of all three at `meta_batch_size=15`, which is far from converged.

Two changes make this workable:

- **`train(time_budget_s=...)`** stops at an iteration boundary, saves a checkpoint, and prints resume instructions — instead of being killed mid-iteration and losing everything since the last 10-iteration autosave.
- **`episodes_per_task`** is now a live config key (was hardcoded `5`).

`meta_batch_size` does **not** buy you learning: Adam's step size is ≈ `lr` regardless of how many tasks the gradient averages over. Shrinking it trades gradient quality for more optimiser steps per hour — which is exactly the trade you want on a time-limited session. The notebook now trains **one encoder per session** (`ENCODERS = ['gcn']`) at `META_BATCH_SIZE=6`, `EPISODES_PER_TASK=3`, `TIME_BUDGET_HOURS=3.5`.

Don't chase a fixed iteration number. The stopping rule is behavioural: train until `within_episode_entropy` rises off `0.000` and `avg_makespan` plateaus across two consecutive checkpoints.

### Exact cross-session resume

Because the three encoders need more than one free-Colab session, training is split across sessions — and the split is made **invisible to the result**. Each checkpoint stores the weights, the Adam state, the iteration counter, and the **full RNG stream** (python + numpy + torch). On resume, `train()` restores that stream rather than re-seeding, so a run split across any number of sessions is **bit-identical** to one continuous run. Verified: `3+3` and `2+2+2` iteration splits each reproduce a continuous 6-iteration run exactly.

Two determinism fixes made this work: seeding now happens in `__init__` **before** weight initialisation (it was in `train()`, after — so initial weights weren't reproducible), and the RNG state is checkpointed (it wasn't). `save()` now delegates to `_save_checkpoint` so it can't silently drop the RNG state; writes are atomic.

Every encoder is seeded identically at construction, so all three face the **identical** sequence of graphs, preferences and channel conditions — only their weights and actions differ. That is what makes the comparison controlled.

**Operator guide:** `docs/RUNNING_THE_EXPERIMENT.md` walks through three cases — multiple free Colab sessions (Drive-backed, auto-resuming), a single long VM session, and a multi-seed publication run. The notebook training cell is an auto-advancing driver: set it once, re-run each session, and it pours the budget into `gcn → gat → lstm` until each reaches target, resuming exactly.

---

## Reward-Signal & Determinism Overhaul (2026-07-10)

Two from-scratch Colab runs on identical code reported `avg_energy` of **0.59 J** and **6863 J** for the same algorithm. Root-caused and fixed. Full analysis in `dev_logs/reward_signal_and_determinism_overhaul.md`.

| Area | What Changed | Why It Matters |
|---|---|---|
| `default_config.yaml` | `kappa: 1e-23` → `1e-27` | Real DAG nodes take `cycles` from the `.gv` `expect_size` (~2.8e7), not `task_cycles_range` (~1e9). At 1e-23 a local node cost **283 J** vs **0.04 J** for cloud — `total_energy` just counted local picks, and `e_imp` saturated at 0.9998 (cloud) vs 0.9999 (edge), giving no gradient to tell servers apart. |
| `base_offloading_env.py` | Congestion penalty now measures the **real** queue wait | It was recomputed *after* `_execute_offloading` had overwritten `server_available`, so it reduced to `min(comp_delay/5, 1)` — it never measured queueing at all. |
| `base_offloading_env.py` | Comm penalty uses the actual Shannon-rate transfer time | It divided raw bytes by raw `bandwidth_up` (42× below the real datarate) and pinned at exactly 1.0 whenever any parent was cross-server. |
| `base_offloading_env.py` | `step()` returns `info['r_delay']`, `info['r_energy']` | The reward carrying both penalties was computed, stored on the experience dict, and **never read by any loss function**. |
| `tampo.py` | `mo_return` discounts the env's reward components | The agent was optimising queue-free comp-delay while being scored on queue-driven makespan. |
| `tampo.py` | PPO clipped surrogate in the MAML inner loop | `inner_steps: 5` reused one on-policy batch with uncorrected vanilla policy gradient. |
| `utils/seeding.py` (new) | `set_seed()` wired into `main.py`, `benchmark.py`, notebook | Training was **completely unseeded**. Evaluation was seeded; training was not. |
| `main.py` | `setup_environment()` reads `config['environment']` | Every section lookup returned `{}`, so the env ignored `default_config.yaml` entirely and used hardcoded defaults. |

### Reproducibility

A fixed seed makes one run **repeatable**; it does not make it **representative**. A single-seed three-way table cannot distinguish a real architectural difference from luck.

Multi-seed means **retraining** under each seed — re-benchmarking one set of weights under different evaluation seeds measures nothing about training variance. Each seed needs its own checkpoint dir and its own results dir:

```bash
for s in 0 1 2 3 4 5 6 7; do
  # train all three encoders with seed=$s into models_seed_$s/  (see docs)
  python benchmark.py --seed 42 --checkpoint_dir models_seed_$s --output_dir results/seed_$s
done
python utils/aggregate_seeds.py --results_root results --seeds 0 1 2 3 4 5 6 7
```

`utils.seeding.SEEDS` has **8** seeds, not 5, for a concrete reason: a two-sided Wilcoxon on 5 paired seeds floors at p = 0.0625, so it can never reach p < 0.05 however decisively one encoder wins. Use ≥ 6 seeds to claim significance, or report mean ± std and say the study is descriptive.

Full walkthrough, compute budget and aggregation: **`docs/RUNNING_THE_EXPERIMENT.md` (Case 3)**.

### Known limitation

The task's own upload time is still **not** charged to the timeline (`finish_time = start_time + comp_delay`); `trans_time` feeds only the energy formula. For the median node, cloud upload takes 0.083 s while local execution takes 0.028 s, so offloading should often be *slower*. Fixing this changes the physics engine and invalidates all previously reported numbers — see §9 of the dev log.

---

## Convergence Overhaul (2026-07-06)

A professional root-cause analysis identified **15 distinct reasons** the TAMPO agent was failing to converge. All 15 have been fixed. See `dev_logs/convergence_fixes_overhaul.md` for full details.

> ⚠️ **RC#15 in the table below is superseded.** Its `kappa: 1e-28 → 1e-23` change was justified with arithmetic that was wrong by nine orders of magnitude and assumed `cycles ≈ 1e9`. It has been reverted to `1e-27`. See the 2026-07-10 overhaul above.

### Critical Fixes Applied

| RC | File | What Changed | Why It Matters |
|---|---|---|---|
| #1+#4 | `tampo.py` | `mo_return` now stores discounted per-objective improvement over local baseline | Old code stored raw costs (inverted sign) with no discounting — agent was penalised for good actions and could not plan |
| #2 | `tampo.py` | GCN/GAT `encoded_tasks` broadcast from context (was all-zeros) | Decoder attention was attending over zeros — no graph info in decisions |
| #3 | `base_offloading_env.py` | Reward scale 5.0 → 1.0, clip ±5.0 → ±1.0 | Old scale caused value network targets of ~200 from iteration 1 |
| #5 | `tampo.py` | Value network now receives server_features (20-dim) in addition to flat obs | Was comparing apples/oranges — policy saw graph, value saw summary |
| #13 | `tampo.py` | Decoder uses `context_projection(context)` to initialise h_t | Old code discarded the second half of the bidirectional context |
| #15 | `default_config.yaml` | `kappa: 1e-28` → `1e-23` | Local energy was 1e-10 J vs transmission 0.025 J — 8 OOM gap made energy objective unlearnable |

### Stabilisation Fixes

| RC | What Changed |
|---|---|
| #6 | `meta_learning_rate` 3e-4→5e-5; `inner_lr` 0.01→0.005; `inner_steps` 3→5 |
| #7 | `meta_policy.eval()` during inner-loop adaptation; `train()` restored after |
| #8 | Zero-mean unit-variance advantage normalisation before policy gradient |
| #10 | `random.shuffle(all_experiences)` before 80/20 train/test split |
| #11 | Categorical(probs).sample() during training; no more biased epsilon-greedy |
| #12 | Removed `sys.stdout = StringIO()` redirect that silenced gradient diagnostics |
| #9 | HyperVolume reference point updated to [2.0, 2.0] for improvement scale |
| #14 | Value network `hidden_dim` doubled for sufficient model capacity |