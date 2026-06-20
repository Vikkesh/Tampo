# TAMPO Core Logic & Architecture Concepts Explained

*This document captures the detailed conceptual logic behind the TAMPO meta-RL framework, providing a comprehensive explanation of its multi-objective evaluation, episodic execution, physics trade-offs, network architecture (encoders), threshold-triggered communications, and hypervolume usage.*

---

## 1. Multi-Objective Evaluation & Preference Vectors

TAMPO is a Multi-Objective Meta-RL (MORL) framework designed to handle two conflicting goals simultaneously: minimizing **Latency (Makespan)** vs. minimizing **Battery usage (Energy)**. A **preference vector** `[w_delay, w_energy]` tells the agent what the user cares about right now, e.g., `[0.8, 0.2]` means 80% priority on speed, 20% on battery.

### How the Preference Vector Is Used Inside the Network

**A. Value Network (`MultiObjectiveValueNetwork`):**
The value network's job is to predict how "good" the current situation is. But the exact same physical situation (e.g., "Cloud server is far away") is good for a speed-focused user and bad for a battery-focused user. The preference vector is concatenated directly onto the state vector before the neural network processes it:

```
Input to Value Network = [Task Features + Server State] + [w_delay, w_energy]
```

This allows a single network to predict two different values for the same physical state depending on the user's goal.

**B. Computing the Weighted Advantage:**
During training, after each step, the agent computes a separate advantage for delay and energy. The preference vector then scales these before summing:

```python
# Simplified code from tampo.py
advantages = mo_returns - values.detach()            # [advantage_delay, advantage_energy]
weighted_advantages = (advantages * preferences).sum(dim=-1)  # scalar: (adv_d * 0.8) + (adv_e * 0.2)
policy_loss = -(log_probs * weighted_advantages).mean()
```

**Concrete example:** Agent sends a heavy file to the Cloud. Result: `delay_advantage = +10` (fast!), `energy_advantage = -5` (battery drain!).
- **Battery Saver `[0.2, 0.8]`:** `(10 × 0.2) + (-5 × 0.8) = 2 - 4 = -2` → Agent is penalized. Learns to avoid Cloud.
- **Performance Mode `[0.8, 0.2]`:** `(10 × 0.8) + (-5 × 0.2) = 8 - 1 = +7` → Agent is rewarded. Learns Cloud is correct.

### Why This Does NOT Confuse the Model

A common concern is whether training a single model on contradictory objectives (reward the Cloud in one episode, penalize it in the next) would prevent the model from ever converging.

**The key insight:** Without the preference vector in the input, the model *would* be confused — the environment would appear to randomly reward and punish the exact same action, causing massive reward variance and preventing convergence.

By making the preference vector part of the input observation, the model never sees two "identical" states with different rewards. The two scenarios are actually distinct inputs:
- `[Heavy Task, Cloud Far, pref=0.2, 0.8]` → bad action → negative gradient
- `[Heavy Task, Cloud Far, pref=0.8, 0.2]` → good action → positive gradient

The model learns a **conditional policy** (an "If-Then" map) rather than a fixed rule:
> *"IF state is [Heavy Task] AND preference is [0.8, 0.2], THEN send to Cloud."*
> *"IF state is [Heavy Task] AND preference is [0.2, 0.8], THEN send to Edge."*

This is a well-established technique called **Conditioned Reinforcement Learning**, used extensively in modern MORL research.

### Training Phase vs. Testing Phase Preferences

**During Training (`_sample_preference` in `tampo.py`):**
The preference vector is sampled randomly from a **continuous uniform distribution** on every single episode:
```python
w_delay = np.random.uniform(0.2, 0.8)
w_energy = 1.0 - w_delay
```
This means the agent is exposed to every preference from `[0.2, 0.8]` all the way to `[0.8, 0.2]` (e.g., `[0.34, 0.66]`, `[0.71, 0.29]`, etc.). It learns a smooth, generalized Pareto curve — not just 3 modes.

**During Evaluation (`CommonEvaluator` in `utils/common_evaluator.py`):**
The evaluator uses only 3 fixed, hardcoded preference vectors for clean, comparable metrics:
```python
self.eval_preferences = [
    np.array([0.8, 0.2]),  # Performance Mode
    np.array([0.5, 0.5]),  # Balanced Mode
    np.array([0.2, 0.8])   # Battery Saver Mode
]
```
The 500 test graphs cycle round-robin through these three profiles (Graph 1 → `[0.8, 0.2]`, Graph 2 → `[0.5, 0.5]`, Graph 3 → `[0.2, 0.8]`, Graph 4 → `[0.8, 0.2]`, ...).

Because the model was trained on the full continuous spectrum, it handles these 3 fixed test values trivially — they are just 3 points it has already seen and mastered during training.

**Testing the Convergence:**
When `benchmark.py` evaluates the algorithm against the golden dataset (`test_dags.json`), it evaluates the exact same graphs across three distinct user profiles:
1.  **Performance Mode** `[w_delay: 0.8, w_energy: 0.2]`
2.  **Balanced Mode** `[w_delay: 0.5, w_energy: 0.5]`
3.  **Battery Saver Mode** `[w_delay: 0.2, w_energy: 0.8]`

Because TAMPO produces multiple distinct data points that form a Pareto front across these profiles, it proves the meta-agent can generalize its behavior on the fly based on the preference vector without retraining.

---

## 2. Episodic Execution & DAG Processing

In TAMPO's reinforcement learning loop, **1 Episode = Processing exactly 1 complete DAG workflow.**

**The Flow of an Episode:**
1.  The environment loads a single Directed Acyclic Graph (DAG) (e.g., 20 interdependent tasks) and assigns a random preference vector.
2.  The DAG features and adjacency matrices are extracted. Depending on the encoder (LSTM, GCN, or GAT), the node representations are formed.
3.  The agent operates sequentially over the tasks in **topological order**. The RL decoder assigns a server for the first task.
4.  The environment calculates the start and finish time for that task based on network bandwidth and server availability (queue state).
5.  The environment updates the `server_available` timeline and moves to the next task in the sequence.
6.  Once the final task in the DAG is processed, the episode completes. The environment computes the total *Makespan* and *Total Energy* consumed.
7.  The multi-objective reward is returned to the agent, and the step history is used for gradient updates.

---

## 3. The Edge vs. Cloud Physics Trade-off

The framework simulates a physics engine with a mobile device communicating with Edge servers and a Cloud server.

**The Trade-offs:**
*   **The Edge:**
    *   **Pros:** Physically closer with a `1.5x` bandwidth multiplier, resulting in fast uploads and lower transmission power (`0.3W`).
    *   **Cons:** Computationally slower (`5.0 GHz`).
*   **The Cloud:**
    *   **Pros:** Massive CPU capacity (`10.0 GHz` — twice as fast as the Edge).
    *   **Cons:** Distant with no bandwidth multiplier, leading to high network latency and higher transmission power (`0.5W`).

The agent must learn that "Data-Heavy / Computation-Light" tasks should stay at the Edge, while "Data-Light / Computation-Heavy" tasks should be sent to the Cloud to offset the transmission delay.

---

## 4. Encoder Architecture Switching (LSTM vs. GCN/GAT)

The original TAMPO paper relies on an **LSTM** to encode the graph structure sequentially. However, the codebase supports dynamically switching the encoder architecture (`encoder_type` config parameter) between `lstm`, `gcn`, and `gat`.

*   **LSTM Encoder (Original TAMPO):**
    When `encoder_type='lstm'`, the `DAGEncoder` uses a bidirectional LSTM over the topologically sorted node features. This processes nodes sequentially, passing hidden states forward and backward to capture dependencies.
    *Impact:* Standard LSTM operations use fast CuDNN kernels by default. However, because TAMPO uses MAML meta-learning (which requires second-order gradients), CuDNN is dynamically disabled (`torch.backends.cudnn.flags(enabled=False)`) during meta-updates since CuDNN RNNs do not support double backward passes.

*   **GCN / GAT Encoder (Parallel):**
    When `encoder_type='gcn'` or `gat'`, the framework relies on Graph Neural Networks (GNNs) using `torch_geometric`. This processes the *entire* graph structure simultaneously in a parallel pass by building PyG batches.
    *Impact:* By passing messages directly across the adjacency matrix edges, GCNs and GATs can identify critical paths and bottlenecks more explicitly than sequential LSTMs. After the parallel encoding, the RL decoder still assigns tasks sequentially to accommodate dynamic server queue states and communication penalties.

---

## 5. Hypervolume Tracking and Threshold-Triggered Updates

The framework uses an intelligent, dynamic threshold system to determine when lower-layer agents should communicate their gradients back to the higher-layer meta-learner.

**Hypervolume Calculation:**
Instead of constantly sending updates, each `LowerLayerAgent` maintains a `performance_buffer` storing the normalized delay and energy outcomes of recent episodes. A `HypervolumeCalculator` computes the 2D hypervolume of these outcomes against a fixed reference point (e.g., `[10.0, 1.0]`). The hypervolume measures the quality of the Pareto front generated by the agent.

**Threshold Trigger (`hv_threshold`):**
1. The agent maintains a moving average of the hypervolume (`hv_history`) over a specific window.
2. After each episode, if the moving average falls below the defined `hv_threshold` (e.g., `0.5`), it indicates the agent is struggling to find optimal trade-offs.
3. This triggers the agent to set `self.update_needed = True` and package its current policy gradients (`policy_grads`) and performance history into an `update_package`.
4. The `HigherLayerMetaLearner` periodically calls `collect_updates()`. If packages are ready, it aggregates the gradients from all struggling agents and performs a refinement step on the global `meta_policy`.

---

## 6. MAML Inner and Outer Loops

The learning process uses Model-Agnostic Meta-Learning (MAML):
- **Inner Loop (Fast Adaptation):** The `LowerLayerAgent` computes task-specific functional gradients (`create_graph=True`) and applies standard SGD steps to locally adapt its policy to a specific graph/preference context.
- **Outer Loop (Meta-Update):** The `HigherLayerMetaLearner` computes the loss of the *adapted* policy on a separate test set of experiences and backpropagates through the entire adaptation graph to update the core `meta_policy` parameters, training the agent to be "fast at adapting".

---

## 7. Dataset Separation & Train/Test Size Split

A strict separation between training and testing data is enforced on two levels:

### Level 1: File-Level Separation (No Data Leakage)
The `test_dags.json` golden dataset is generated once and **never loaded during training**. The training loop exclusively uses `data/meta_offloading_n/offload_random{N}` folders via `DAGParser`. The benchmark loads `test_dags.json`. These two file sets are completely disjoint.

### Level 2: Graph Complexity Separation (Zero-Shot Gap)
Beyond file separation, there is a deliberate gap in the **graph sizes** seen during training vs. testing. This is the most important element for proving the agent has genuinely learned the physics — not just memorized specific workflows.

| Phase | Graph Sizes | Graphs per Size | Total Graphs | Source |
|-------|------------|----------------|-------------|--------|
| **Training** | 10, 15, 20, 25, 30 nodes | 30 each | **150 graphs** | `meta_offloading_n/offload_random{10,15,20,25,30}` |
| **Testing** | 10, 15, 20, 25, 30, 35, 40, 45, 50 nodes | ~55 each | **500 graphs** | `meta_offloading_n/offload_random{10..50}` |

**Key insight:** Sizes 35, 40, 45, and 50 nodes (~222 of the 500 test graphs) are **never seen during training**. A model that merely memorized 20-node graph patterns will collapse on 50-node graphs. A model that learned the fundamental physics of queue management, dependency resolution, and preference-conditioned offloading will handle them gracefully.

### How to Load the Training Pool (Colab Notebooks)

Instead of hand-coding folder paths, use the canonical helper:

```python
# In Colab, after cloning the repo:
from utils.training_setup import load_training_graphs, build_env_task_list

task_graphs = load_training_graphs()          # 150 graphs, sizes 10–30, shuffled
tasks_for_env = build_env_task_list(task_graphs)
env.load_task_dataset(tasks_for_env)
```

To customise (e.g., if Colab GPU time is limited):
```python
# Smaller pool for a quick test run
task_graphs = load_training_graphs(sizes=[10, 20, 30], graphs_per_size=20)
```

### Why 150 Training Graphs Is Enough

Because TAMPO is a **Meta-RL** framework (MAML), it is not learning a fixed policy from a large dataset like supervised learning. Instead, it is learning *how to adapt quickly*. The MAML outer loop needs enough task diversity to learn a good prior — 150 graphs across 5 different sizes provides that diversity. What matters more than graph count is:
1. **Size diversity** ✓ (5 different node counts)
2. **Topology diversity** ✓ (random DAG structure per graph)
3. **Preference diversity** ✓ (random `w_delay` sampled every episode)
4. **Server-load diversity** ✓ (dynamic queue state resets every episode)

