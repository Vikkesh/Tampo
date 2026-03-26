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

## 3. System Architecture: GCN Integration (Paper-Standard PyG)

To optimally handle Directed Acyclic Graph (DAG) structures, the TAMPO framework now routes parsed DAGs through a true graph-input pipeline and supports a **Graph Convolutional Network (GCN)** alongside the original **BiLSTM** encoder.

### 3.1 Design Choice: PyTorch Geometric (`torch_geometric`)
The active GCN implementation now uses **PyTorch Geometric (PyG)**, which is the more common research-paper-standard library choice for graph neural networks.
* **Why?** PyG provides tested graph layers, `edge_index` graph formatting, graph batching, and node-to-dense packing utilities that align much better with the standard GCN literature and open-source baselines.
* **Result:** TAMPO now uses the same style of GCN stack most readers expect when they see a graph-learning paper: parsed graph -> `Data`/`Batch` -> `GCNConv` layers -> graph pooling -> decoder.

### 3.2 The Mathematical Logic
The GCN still follows the Kipf-Welling update, but the normalization and message passing are now handled by `torch_geometric.nn.GCNConv` in `tampo/algorithms/rl/tampo.py` -> `DAGEncoder._apply_gcn()`.

1. **Graph Input:** The parser provides a DAG adjacency matrix, which TAMPO converts into PyG's `edge_index` representation.
2. **Undirected Standardization:** The DAG edges are passed through `to_undirected(...)` so the GCN branch uses the common undirected neighborhood aggregation expected by standard `GCNConv`.
3. **Self-Loops and Normalization:** `GCNConv` applies the standard self-loop insertion and symmetric degree normalization internally.
4. **Forward Pass:** For each layer $l$, the hidden states follow the standard GCN rule: $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$.
5. **Dense Readout Bridge:** After graph convolution, `to_dense_batch(...)` reconstructs a padded `[batch, num_nodes, hidden]` tensor so the attention/pooling/decoder path can stay compatible with the rest of TAMPO.

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
   - `_pad_graph_batch(...)` still pads graphs for dense components like the LSTM path and attention masking.
   - `_build_pyg_batch(...)` converts each DAG into a PyG `Data` object and merges them into a batched `Batch`, which is the active GCN input path.
   - `DAGEncoder` supports `lstm`, `gcn`, and `both`, where the GCN branch now uses stacked `GCNConv` layers instead of manual dense adjacency multiplication.
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
* **Graph Sizes & Batching:** The current TAMPO path uses both padded dense tensors and PyG graph batches. Dense padding is used for the LSTM/attention bridge, while the actual GCN layers consume PyG's sparse-style graph representation.
* **Fallbacks:** If independent non-DAG tasks are passed, TAMPO falls back to a single-node identity graph so the policy path stays valid.
* **Dependency Requirement:** The GCN path now expects `torch-geometric` in the runtime environment. Since this repo may be edited locally but trained in Colab, install the graph dependencies in Colab rather than manually on the local editing machine.
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
Upon execution, `main.py` presents an interactive run menu. This is the current operator-facing control surface for the project, so when new algorithms are added, this section should be kept in sync with `get_user_input()`.

### 4.2A Running "Both" TAMPO Encodings
There are now **two different meanings** of "both" for TAMPO, and it is important not to mix them up:

1. **Run both benchmark variants side-by-side in one interactive session**
   * This is the normal benchmarking workflow.
   * Start the program with `%run tampo/main.py` in Colab or `python3 main.py` inside `tampo/`.
   * When prompted, answer:
     * `Run TAMPO-LSTM?` -> `yes`
     * `Run TAMPO-GCN?` -> `yes`
   * Then enter one shared value for `Number of meta-iterations for TAMPO`.
   * Result: the framework trains and evaluates **two separate TAMPO runs**, one with `encoder_type='lstm'` and one with `encoder_type='gcn'`.
   * These are saved independently and reported independently as `TAMPO_LSTM` and `TAMPO_GCN`.

2. **Run the internal fused encoder mode**
   * This is the experimental hybrid mode inside one TAMPO model.
   * It is **not exposed in the interactive prompt menu**.
   * To use it, set `encoder_type: 'both'` in the TAMPO config file and run TAMPO through code/config rather than the side-by-side prompt flow.
   * Result: one TAMPO model uses both the LSTM stream and GCN stream together inside the same `DAGEncoder`.

### 4.2B Important Variables, Paths, and Control Points
If you want to quickly change how TAMPO runs, these are the most important places:

* **Interactive yes/no prompts:** [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L39)
  * `algorithms['TAMPO_LSTM']` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L76)
  * `algorithms['TAMPO_GCN']` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L79)
  * `algorithms['tampo_iterations']` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L100)
  * `algorithms['eval_episodes']` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L107)

* **Where the chosen TAMPO variant is actually launched:** [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L459)
  * `TAMPO_LSTM` calls `test_tampo(..., encoder_type='lstm')` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L459)
  * `TAMPO_GCN` calls `test_tampo(..., encoder_type='gcn')` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L468)

* **Where TAMPO turns that choice into a real encoder config:** [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L234)
  * `tampo_config['encoder_type'] = encoder_type` in [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L239)

* **Default TAMPO config file:** [default_config.yaml](/home/vikkesh/Tampo-clone/tampo/configs/default_config.yaml#L95)
  * `config['algorithms']['tampo']['encoder_type']` is documented in [default_config.yaml](/home/vikkesh/Tampo-clone/tampo/configs/default_config.yaml#L99)
  * valid values are `'lstm'`, `'gcn'`, and `'both'`

* **Checkpoint files per exposed encoder run:**
  * `models/tampo_lstm_checkpoint.pth` created from [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L242)
  * `models/tampo_gcn_checkpoint.pth` created from [main.py](/home/vikkesh/Tampo-clone/tampo/main.py#L242)

* **Where the encoder implementation lives:** [tampo.py](/home/vikkesh/Tampo-clone/tampo/algorithms/rl/tampo.py#L64)
  * `DAGEncoder`
  * `encoder_type`
  * `_build_pyg_batch(...)`
  * `_apply_gcn(...)`
  * `_apply_lstm(...)`

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
*   **Run TAMPO-LSTM?**
    *   *Goal:* Runs the TAMPO meta-RL pipeline with the sequence-oriented BiLSTM encoder over topologically ordered node features.
    *   *When to choose:* Use this when you want the non-GNN structural baseline inside TAMPO, or when you want to compare sequence modeling versus graph convolution using the same downstream decoder and MAML loop.
*   **Run TAMPO-GCN?**
    *   *Goal:* Runs the TAMPO meta-RL pipeline with the paper-standard PyG `GCNConv` encoder over the parsed DAG graph.
    *   *When to choose:* Use this when your experiment specifically targets graph neural scheduling quality, or when you want the architecture most aligned with standard GCN literature.
*   **Internal `encoder_type: both` option**
    *   *Goal:* Fuses the LSTM stream and the GCN stream inside the same `DAGEncoder`.
    *   *When to choose:* This is currently a config-only option for internal experiments. It is not exposed in the interactive prompt menu, which keeps the main benchmarking workflow easier to compare and report.

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
