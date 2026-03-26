# Tampo Project Documentation

## 1. Project Overview & Workflow

The **Task Offloading Algorithm Comparison Framework (Tampo)** is a comprehensive framework that models and evaluates various scheduling and task offloading algorithms in Edge/Cloud computing environments.

### Complete Workflow (From Start to Finish)
1. **Environment Setup:** Loads system configurations (from `configs/default_config.yaml`) defining server frequencies, transmission power, task parameters, and constraints.
2. **Data Parsing:** The `utils/dag_parser.py` parses Directed Acyclic Graph (`.gv`/Graphviz) files representing dependent tasks, converting them into computational graphs with specific data sizes and required CPU cycles.
3. **Environment Initialization:** The `TaskOffloadingEnv` initializes the simulation, providing standardized `gym`-based interfaces (`state`, `step`, `reward`) for task offloading policies.
4. **Algorithm Selection & Execution:** Triggered via `main.py`, the user selects which algorithms (Heuristics or RL) to run. 
5. **Execution Loop:**
   - **For Heuristics:** Schedules task nodes according to pre-defined logic (like earliest completion).
   - **For RL:** Uses training loops where the agent iteratively takes an action, receives a reward (negative latency/energy cost), and updates its network weights.
6. **Evaluation & Metrics:** Logs the output and calculates metrics like Hypervolume (for multi-objective trade-offs between Energy, Cost, and Time) using `utils/metrics.py`.

---

## 2. Algorithms Used

The project categorizes its algorithms into two main types: **Heuristics** and **Reinforcement Learning (RL)**.

### Heuristic Algorithms
1. **HEFT (Heterogeneous Earliest Finish Time):** A widely adopted scheduling algorithm that prioritizes tasks in a DAG based on their downward rank (critical path) and assigns them to the processor that minimizes the earliest finish time.
2. **PSO (Particle Swarm Optimization):** An evolutionary meta-heuristic where "particles" (candidate schedules) move around in a search space, adjusting their positions based on their own best experience and the global best experience to find optimal offloading decisions.
3. **GA (Genetic Algorithm):** An evolutionary approach that relies on bio-inspired operators such as mutation, crossover, and selection to iteratively evolve high-quality scheduling solutions.

### Reinforcement Learning Algorithms
1. **PPO (Proximal Policy Optimization):** A state-of-the-art policy gradient RL method. The baseline PPO uses standard Multi-Layer Perceptrons (MLPs) to predict actions but is robust, stable, and simple to tune.
2. **GMORL (Generalized Multi-Objective RL):** Focuses on balancing competing objectives (e.g., energy vs. latency). It incorporates a `HistogramEncoder` to discretize and understand continuous workload distributions effectively.
3. **TAMPO (The specialized Meta-RL approach):** Uses a `DAGEncoder` network. The current implementation now consumes true DAG node-feature matrices plus adjacency, can run with an **LSTM encoder**, a **GCN encoder**, or an internal **both/fused** mode, and performs its MAML-style inner loop through a functional parameter path rather than temporary live-weight swapping.

---

## 3. System Architecture: GCN Integration (Native PyTorch)

To optimally handle Directed Acyclic Graph (DAG) structures, the TAMPO framework now routes parsed DAGs through a true graph-input pipeline and supports a **Graph Convolutional Network (GCN)** alongside the original **BiLSTM** encoder.

### 3.1 Design Choice: Native PyTorch vs PyTorch Geometric
The decision was made to implement the GCN using **Native PyTorch Matrix Multiplications** (`torch.bmm`) rather than relying on external libraries like `torch_geometric`. 
* **Why?** PyTorch Geometric requires highly specific, compiled CUDA wheels (`torch-scatter`, `torch-sparse`). In dynamic environments like Google Colab or external compute clusters, this frequently leads to dependency conflicts and environment breaks. 
* **Result:** The native approach keeps the repo lightweight while still implementing the standard normalized message-passing step directly in PyTorch.

### 3.2 The Mathematical Logic
The GCN operates using a symmetric normalized adjacency mechanism. In `tampo/algorithms/rl/tampo.py` -> `DAGEncoder._apply_gcn()`, the layer logic calculates:

1. **Self-Loops:** $\tilde{A} = A + I$ (Ensures a node considers its own features during the message passing phase).
2. **Degree Matrix:** Computes the diagonal degree matrix ($\tilde{D}$) based on the edges in $\tilde{A}$.
3. **Normalization:** Calculates the symmetrically normalized Laplacian: $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.
4. **Forward Pass:** For each layer $l$, the hidden states are updated via the matrix operation: $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$.

### 3.3 The Data Plumbing Pipeline (How the Adjacency Matrix flows)
For the maintenance team taking over, here is exactly how the graph topology now routes through the codebase. If you adapt the graph parsing or the environment in the future, **ensure this unbroken pipeline is maintained**:

1. **`tampo/utils/dag_parser.py` (`parse_gv_file`)**: 
   - When a `.gv` graph file is loaded, it extracts edges and builds a dense zero-initialized numpy matrix: `adj_matrix = np.zeros((num_tasks, num_tasks))`.
   - It sorts tasks by node id and maps edges through `id_to_index`, ensuring the task-feature rows and adjacency rows refer to the same node ordering.
   - It attaches this structural map directly to the returned DAG dictionary.
   
2. **`tampo/env/base_offloading_env.py` (`TaskOffloadingEnv`)**:
   - The active DAG is preserved across `reset()` after `set_task(task_id)` is called.
   - The environment exposes `get_adjacency_matrix()`, `get_task_feature_matrix()`, and `get_server_features()` so TAMPO can consume real graph inputs.
   - DAG-backed episodes terminate after one graph-level decision instead of silently swapping in random scalar tasks mid-rollout.
   
3. **`tampo/algorithms/rl/tampo.py` (Data Collection & Loss Computation)**:
   - Inside `_collect_task_experiences()`, the agent now collects true graph node features, server features, and adjacency from the active DAG task.
   - `_pad_graph_batch(...)` pads graphs of different sizes and produces a `node_mask` so the encoder can batch them safely.
   - `DAGEncoder` supports `lstm`, `gcn`, and `both`, and the policy now makes a graph-level offloading decision from the encoded DAG context.
   - `TAMPOFramework.select_action(...)` is graph-aware and is reused during evaluation.
   - The MAML inner loop now uses `torch.func.functional_call`, which keeps the meta-gradient chain intact across the adaptation step and makes the TAMPO meta-update closer to standard functional MAML practice.

### 3.4 Runtime Toggling (GCN vs LSTM)
The framework is configured to dynamically switch between the sequence approximation (BiLSTM) and the topological approximation (GCN) via runtime parameter injection in `tampo/main.py`.

In `configs/default_config.yaml`, locate the `tampo` block:
```yaml
  tampo:
    enabled: true
    hidden_dims: [256, 256]
    num_attention_heads: 8
    encoder_type: 'gcn'  # options: 'lstm', 'gcn', 'both'
```

### 3.5 Operational Gotchas & Future-Proofing
* **Graph Sizes & Batching:** The current TAMPO path now pads variable-size graphs into dense batches and carries a `node_mask`, which is safer than the earlier fixed-size-only stack.
* **Fallbacks:** If independent non-DAG tasks are passed, TAMPO falls back to a single-node identity graph so the policy path stays valid.
* **PyTorch Requirement for TAMPO Meta-Learning:** The corrected MAML path relies on `torch.func.functional_call`, so TAMPO's meta-learning code expects a modern PyTorch version with `torch.func` support.

## 4. Testing & Exploration Guide (How to Run)

The entry point for this framework is completely interactive. It is designed to let you selectively toggle which algorithms to train, evaluate, and compare against each other in a single run.

### 4.1 Starting the Framework
If you are running inside a Jupyter Notebook or Google Colab, you must use the magic command so standard input (prompts) works correctly:
```python
%run tampo/main.py
```
If you are running locally in a standard Linux/Unix terminal, execute:
```bash
cd tampo
python3 main.py
```

### 4.2 Interactive Options & When to Use Them
Upon execution, the script will prompt you with a series of `yes/no` questions. As the developer maintaining this system, here is a detailed breakdown of what each option does and exactly when you should select it.

#### Group A: Heuristic Baselines (HEFT, PSO, GA)
*   **Run HEFT? (Heterogeneous Earliest Finish Time)**
    *   *Goal:* Calculates the mathematical "critical path" of the graph based on computational weights and assigns nodes to servers that finish earliest.
    *   *When to choose:* **Always.** Use this as your bedrock baseline. It executes instantly. If your RL model (TAMPO/PPO) cannot beat HEFT's latency, your RL model is under-trained or has a structural bug.
*   **Run PSO? (Particle Swarm) / Run GA? (Genetic Algorithm)**
    *   *Goal:* Uses heavily mathematical evolutionary/swarm mechanics to find the best schedule. 
    *   *When to choose:* Select these only when generating final benchmarking charts for a paper/presentation. 
    *   *Warning:* Unlike RL, which is slow to train but fast to execute, PSO and GA require hundreds of iterations *per test graph*. Skip these if you are just doing a quick debug run, as they will bottleneck your testing pipeline.

#### Group B: Reinforcement Learning Models
*   **Run PPO? (Proximal Policy Optimization)**
    *   *Goal:* Tests a standard, flat Multi-Layer Perceptron (MLP) reinforcement learning approach. 
    *   *When to choose:* Use this as your "naive" ML baseline. Select this when you need to definitively prove that advanced structural graph-encoding (GCN/LSTM) is vastly superior to standard vector-based RL.
*   **Run GMORL? (Generalized Multi-Objective RL)**
    *   *Goal:* Optimizes heavily for the Pareto front (Delay vs. Energy) using a Histogram Encoder.
    *   *When to choose:* Select this when your current sprint/focus is specifically on tuning multi-objective trade-offs (e.g., punishing power consumption) rather than pure offloading latency.
*   **Run TAMPO-LSTM? / Run TAMPO-GCN?**
    *   *Goal:* These are the two exposed TAMPO benchmark variants. They share the same graph input pipeline but use different encoders internally.
    *   *When to choose:* Select one when validating a single encoder family, or select both when you want a direct side-by-side benchmark from a single interactive run.

### 4.3 Training & Evaluation Parameters
After selecting the RL algorithms, you will be prompted for duration parameters:

1.  **Training Episodes / Meta-iterations (e.g., `tampo_iterations`):**
    *   *Goal:* Determines how many loops the RL agent spends learning and applying gradient updates.
    *   *Situational Advice:* 
        *   **Smoke Test/Debug:** Enter `3` or `5`. Use this immediately after rewriting PyTorch modules (like the GCN layer) purely to verify that forward and backward passes compile without tensor dimension mismatch errors.
        *   **Production Run:** Enter `100+` (e.g., `200`) when you are ready to collect real, converged system metrics inside Colab.
2.  **Evaluation Episodes / Heuristic Test Tasks:**
    *   *Goal:* The number of unseen test graphs the models must solve *after* training is locked.
    *   *Situational Advice:* `5` to `10` is enough for a rapid pipeline test. Use `50+` (the full parsed dataset size) when publishing final charts to ensure statistical significance across randomized task bounds.

### 4.4 Outputs & Interpretations
Once the selected evaluation finishes, the framework automatically hands the results dictionary to `utils/metrics.py`. 

The framework generates a timestamped directory in `tampo/results/YYYYMMDD_HHMMSS/` containing:
1.  **`results.json`**: Raw numerical data. Look here to inspect the exact `avg_delay`, `avg_energy`, and `hypervolume` score. This is highly useful for automated unit tests or CI/CD parsing.
2.  **`plots/`**: Automatically generated visual images.
    *   *Bar Charts:* Use these to quickly compare absolute Latency/Cost against HEFT.
    *   *Pareto Front Scatters:* Crucial for multi-objective analysis. A model is considered Pareto-superior if its scatter points push closer to the bottom-left vertex (lower delay AND lower energy) than the alternatives.
