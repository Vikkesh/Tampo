# Physics Engine, Reward System & Offloading Metrics — Master Reference Guide

> **This is the single source of truth.** Every variable, formula, path, and scenario playbook for the scoring and reward system in this codebase is documented here. Do not look anywhere else.

---

## Section 1 — Comprehensive File & Path Map

Every file in the repository that touches reward calculation, metric collection, or offloading physics is listed below with its role.

### Core Physics & Reward Engine

| File | Role |
|---|---|
| `env/base_offloading_env.py` | **Ground truth.** All delay, energy, reward calculations happen here. Every algorithm must go through `env.step(action)`. No algorithm computes its own physics. |
| `configs/default_config.yaml` | Declares every tunable physics and reward parameter. This is the single configuration file. |

### Evaluation & Benchmarking

| File | Role |
|---|---|
| `utils/common_evaluator.py` | Standardized evaluation harness. Runs all algorithms through the same fixed preference vectors and episode seeds. Calls `env.step()` for every agent type. |
| `benchmark.py` | CLI entry point for head-to-head benchmarking. Loads `test_dags.json`, runs `CommonEvaluator`, writes CSV and plots to `results/`. |
| `utils/metrics.py` | Utility functions: `calculate_hypervolume()` (2-D hypervolume for Pareto reporting), `normalize_objectives()` (delay/energy normalization), `MovingAverage` class. |
| `main.py` | Interactive training + evaluation runner. Calls `CommonEvaluator.evaluate_heuristic()` and `evaluate_rl_agent()`. |

### Heuristic Schedulers (Each Computes Their Own Makespan & Energy)

| File | Role |
|---|---|
| `algorithms/heuristic/heft.py` | `HEFTScheduler.schedule()` returns `(schedule, makespan, total_energy)`. Has its own energy formula (see Section 2C). |
| `algorithms/heuristic/pso.py` | `PSOScheduler.optimize()` returns `(schedule, delay, energy)`. |
| `algorithms/heuristic/ga.py` | `GAScheduler.optimize()` returns `(schedule, delay, energy)`. |

> **Note:** Heuristic algorithms compute their own makespan/energy directly (not via `env.step()`) because they solve the full DAG in one shot, not node-by-node. They do NOT receive a reward signal.

### RL Agents (All Physics Via `env.step()`)

| File | Role |
|---|---|
| `algorithms/rl/tampo.py` | TAMPO framework. Uses `env.step()` at every node. Reads `info['makespan']` and `info['total_energy']` at episode end. |
| `algorithms/rl/ppo_baseline.py` | PPO baseline agent. Same env interface. |
| `algorithms/rl/gmorl.py` | GMORL (multi-objective) agent. Same env interface. |

### Environment Wrappers (Observation Shape Only — Do Not Change Rewards)

| File | Role |
|---|---|
| `env/wrappers/flat_vector_wrapper.py` | Flattens the per-node feature matrix to a 1-D vector for D3QN/SAC-style agents. Physics unchanged. |
| `env/wrappers/sequence_wrapper.py` | Returns topologically sorted 2-D task sequence for TPTO/MTD3-style agents. Physics unchanged. |

### Dataset Files

| File | Role |
|---|---|
| `utils/generate_test_dataset.py` | Generates the immutable `data/test_dags.json` golden test set. Uses `offset=20` so it never overlaps with the training pool. |
| `utils/training_setup.py` | Loads the training graph pool (all 9 node sizes, 20 graphs each = 180 total). |
| `utils/dag_parser.py` | Parses `.gv` files into `{num_tasks, tasks, edges, adj_matrix}` dicts. `load_dataset(num_graphs, offset)` controls segregation. |
| `data/test_dags.json` | The frozen golden test dataset. **Never modify after generation.** |
| `data/meta_offloading_n/offload_random{N}/` | Source `.gv` files for all 9 sizes (N=10,15,20,25,30,35,40,45,50). |
| `data/meta_offloading_20/offload_random20_{N}/` | Legacy 20-node folders used only by `main.py`'s interactive mode. |

---

## Section 2 — Reward & Metric Mechanics: Cloud vs. Edge

All physics live in `env/base_offloading_env.py`. The action space is:

| Action Index | Target |
|---|---|
| `0` | Local Device |
| `1` | Cloud Server |
| `2` | Edge Server 0 |
| `3` | Edge Server 1 |
| `4` | Edge Server 2 |

### 2A. DAG Topological Execution Order

Before any step executes, `reset()` computes a topological ordering using Kahn's algorithm:

```
adj = current_task['adj_matrix']              # shape: [N, N]
depths = _compute_topological_depths(adj)      # depth of each node
topo_order = np.argsort(depths).tolist()       # process shallowest first
```

Each call to `step(action)` advances `current_node_idx` by 1. This guarantees every parent node is scheduled **before** any child, making the dependency resolution causal and correct.

---

### 2B. `_execute_offloading()` — The Physics Function

This is called inside `step()` before reward is computed. It returns `(comp_delay, energy)`.

#### Step 1 — Determine Bandwidth to Target Server
```python
# action == 0 → Local (no transmission)
# action == 1 → Cloud: use raw bandwidth_up
bw_up = self._get_datarate(0)

# action >= 2 → Edge: boosted by 1.5x (closer proximity)
bw_up = self._get_datarate(server_idx - 1) * 1.5
```

`_get_datarate(server_idx)`:
```
SNR  = (power * channel_gain) / noise_power
rate = bandwidth_up * log2(1 + SNR)   [Shannon capacity]
```
Where `channel_gain` is sampled fresh every episode from a Rayleigh distribution (σ=1.0).

#### Step 2 — Resolve DAG Dependencies (Critical Path)
```python
data_ready = 0.0
for edge where edge.target == current_node:
    parent_finish = node_finish_times[parent_id]
    parent_server = node_assignments[parent_id]

    if parent_server == action:           # same processor
        comm_cost = 0.0
    else:                                 # cross-server transfer
        comm_cost = edge.data / bw_up
        # Also adds to cross_server_energy
        if parent_server == 1:
            cross_server_energy += comm_cost * cloud_power_tx
        elif parent_server > 1:
            cross_server_energy += comm_cost * edge_power_tx

    data_ready = max(data_ready, parent_finish + comm_cost)
```

#### Step 3 — Compute Start Time (Enforce Queue)
```python
start_time = max(data_ready, server_available[action])
```
This is the key line that enforces **queueing**. If a server is busy, the task must wait.

#### Step 4 — Compute Processing Delay
| Action | Formula |
|---|---|
| Local (`0`) | `cycles / local_freq` |
| Cloud (`1`) | `cycles / cloud_freq` |
| Edge (`>=2`) | `cycles / edge_freq[action - 2]` |

```python
finish_time = start_time + comp_delay
node_finish_times[node_id] = finish_time
node_assignments[node_id] = action
server_available[action] = finish_time   # server is busy until this time
```

#### Step 5 — Compute Energy
| Action | Formula |
|---|---|
| Local (`0`) | `kappa * cycles * (local_freq^2)` |
| Cloud (`1`) | `(trans_time + result_time) * cloud_power_tx + cross_server_energy` |
| Edge (`>=2`) | `(trans_time + result_time) * edge_power_tx + cross_server_energy` |

Where:
- `trans_time = data_size / bw_up`
- `result_time = (data_size * 0.05) / bw_up` — result download assumed to be 5% of input size

**Only `comp_delay` is returned, not `finish_time`.** The returned delay is the pure computation time on the target processor (used for reward baseline comparison). The total makespan is only computed at episode end as `max(node_finish_times)`.

---

### 2C. HEFT Energy Formula (Different from RL Environment)
HEFT computes energy independently (not via `env.step()`):
```python
# Local:
energy = kappa * task.cycles * (local_freq^2)

# Cloud:
trans_time = data_size / bandwidth_up
result_size = data_size * 0.1           # HEFT uses 10% result, env uses 5%
trans_time += result_size / bandwidth_down
energy = trans_time * cloud_power_tx

# Edge:
trans_time = data_size / bandwidth_up
result_size = data_size * 0.1
trans_time += result_size / bandwidth_down
energy = trans_time * edge_power_tx
```

> **⚠️ Inconsistency Note:** HEFT uses `data_size * 0.1` for result size and uses `bandwidth_down` for result return. The RL environment uses `data_size * 0.05` and reuses `bw_up` for result. This is a known inconsistency. It does not affect fairness because HEFT is evaluated via `evaluator.evaluate_heuristic()` which calls `algorithm.schedule(dag)` directly, while RL agents go through `env.step()`. The two paths are consistent within themselves.

---

### 2D. The Dense Reward Function — `_calculate_reward()`

Called at every single node step. Returned to the RL agent as `step()` output.

**Mathematical Formula:**
```
reward = clip( 5.0 * (w_delay * combined_delay_metric + w_energy * e_imp), -5.0, 5.0 )

where:
  combined_delay_metric = comp_imp - (w_cong * congestion_penalty) - (w_comm * comm_penalty)
  comp_imp  = (local_delay - action_delay) / local_delay
  e_imp     = (local_energy - action_energy) / local_energy
  local_delay  = cycles / local_freq
  local_energy = kappa * cycles * (local_freq^2)
```

**Component 1 — Baseline Improvement (vs Local Execution):**
- Positive when the chosen server is faster/cheaper than running locally.
- Negative when sending overhead makes it worse than local.
- Both delay and energy are each normalized to [-1, 1] before being weighted.

**Component 2 — Congestion Penalty:**
```python
wait_time = server_available[action] - node_finish_times[node_id] + comp_delay
wait_time = max(0.0, wait_time)
congestion_penalty = min(wait_time / 5.0, 1.0)   # capped at 1.0 when wait > 5s
```
- Penalizes stacking tasks onto a server that is already backlogged.
- The denominator `5.0` is a hardcoded normalization constant — congestion is considered maximum at 5 seconds of queue wait.

**Component 3 — Communication Penalty:**
```python
comm_penalty = 0.0
for edge where edge.target == current_node:
    if node_assignments[parent] != action:         # cross-server dependency
        comm_penalty += edge.data / bandwidth_up
comm_penalty = min(comm_penalty, 1.0)              # capped at 1.0
```
- Penalizes placing a child task on a different physical server than its parent.
- Directly incentivizes the GCN/GAT encoder to leverage graph structure.

**Independent Task Fallback (when `task_type != 'dag'`):**
```python
d_imp = (local_delay - delay) / max(local_delay, 1e-9)
e_imp = (local_energy - energy) / max(local_energy, 1e-9)
reward = clip(5.0 * (preference[0] * d_imp + preference[1] * e_imp), -5.0, 5.0)
```
No congestion or communication penalties apply. Independent tasks are not the focus of this codebase.

---

### 2E. Final Benchmark Metrics (What is Actually Reported)

After a full episode (all DAG nodes processed), `step()` computes:
```python
self.total_delay = max(self.node_finish_times)   # true critical-path makespan
```

`CommonEvaluator.evaluate_rl_agent()` reads:
```python
episode_delay = info.get('makespan', info.get('total_delay', 0))
episode_energy = info.get('total_energy', 0)
```

`_calculate_metrics()` then computes over all episodes:
- `avg_makespan`, `std_makespan`, `min_makespan`, `max_makespan`, `median_makespan`
- `avg_energy`, `std_energy`, `min_energy`, `max_energy`, `median_energy`
- Outlier removal: episodes where `|value - mean| > 3 * std` are dropped (unless >20% would be removed, in which case all are kept)

**Hypervolume** (`utils/metrics.py`):
```python
# For 2-D Pareto front (delay, energy):
sorted by delay ascending
hv = sum of rectangular areas from each solution to the reference point
```
Used internally by TAMPO's meta-learning threshold check.

---

## Section 3 — Detailed Variable Dictionary

### 3A. `configs/default_config.yaml` — All Tunable Parameters

#### `environment.system`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `num_edge_servers` | int | `3` | Number of edge servers. Changes `action_space.n` to `num_edge_servers + 2`. Adding servers gives the agent more parallelism. |
| `num_users` | int | `1` | Unused in current physics. Placeholder for multi-user extension. |
| `time_step` | float | `0.1` | Simulation time granularity in seconds. Only used for independent task arrival time (`_generate_task()`). |
| `max_steps` | int | `100` | Hard episode step limit. For DAG tasks, the real limit is `num_tasks` (nodes in the DAG). `max_steps` only applies as a safety cap in `CommonEvaluator`. |

#### `environment.computing`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `cloud_freq` | float | `10e9` Hz | Cloud CPU speed. Higher = faster cloud compute = smaller `comp_delay` = easier to beat local. If always-cloud is occurring, lower this. |
| `edge_freq` | list[float] | `[5e9, 5e9, 5e9]` Hz | Per-edge-server CPU speeds. Can be made heterogeneous (e.g., `[8e9, 5e9, 3e9]`) to create server specialization. |
| `local_freq` | float | `1e9` Hz | Local device CPU speed. This is the baseline denominator in the reward formula. Making this higher makes it harder to beat local, increasing reward pressure. |
| `cloud_cycles_per_bit` | float | `1000` | Legacy field. Not used in `_execute_offloading()`. Present in config but ignored. |
| `edge_cycles_per_bit` | float | `1000` | Same — legacy, unused. |
| `local_cycles_per_bit` | float | `1000` | Same — legacy, unused. |

#### `environment.energy`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `cloud_power_tx` | float | `0.5` W | Transmission power draw when sending to cloud. Used in both `_execute_offloading()` energy and `_get_datarate()` SNR. Higher = more energy cost + higher Shannon capacity. |
| `edge_power_tx` | float | `0.3` W | Transmission power draw when sending to edge. Lower than cloud = edge is cheaper to use energetically. |
| `local_power` | float | `0.1` W | Declared in config but **NOT used** in `_execute_offloading()`. Local energy uses the `kappa` formula instead. |
| `kappa` | float | `1e-28` | Effective switched capacitance for local CPU. Local energy = `kappa * cycles * freq^2`. Extremely small by design (CPU energy is physically tiny). If agents never learn energy trade-offs, increase this. |

#### `environment.network`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `bandwidth_up` | float | `20e6` Hz | Uplink bandwidth (user → server). Used in Shannon capacity formula and in HEFT's direct division. This is the primary transmission bottleneck. Lower = slower upload = bigger latency penalty for offloading. |
| `bandwidth_down` | float | `20e6` Hz | Downlink bandwidth (server → user). Used **only** in HEFT (result download). RL env uses `bw_up * 0.05` as a proxy. |
| `noise_power` | float | `1e-13` W | Thermal noise floor. Used in `SNR = (power * gain) / noise_power`. Higher = worse channel quality = lower data rate. |
| `rayleigh_var` | float | `1.0` | Rayleigh fading variance. Controls channel gain stochasticity. Used as the `scale` parameter in `np.random.rayleigh()`. |
| `path_loss_a`, `path_loss_b` | float | `35`, `133.6` | Declared in config but **NOT used** in `_update_channel_gains()` (which uses only Rayleigh). These are legacy fields for a more complex channel model. |
| `antenna_gain`, `shadow_fading`, `noise_dBm` | float | various | Legacy, unused in current `_update_channel_gains()`. |

#### `environment.reward`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `congestion_penalty_weight` (`w_cong`) | float | `0.4` | Weight applied to the queueing penalty in the delay component. Higher = agent avoids server pile-ups more aggressively. Lower = agent will linearly queue to the fastest server. |
| `comm_penalty_weight` (`w_comm`) | float | `0.3` | Weight applied to cross-server dependency transfer penalty. Higher = agent groups dependent tasks (respects graph structure). Lower = agent ignores edges entirely. |
| `improvement_baseline` | str | `'local'` | Conceptual label only. The formula always uses local as the baseline. Changing this string has no code effect currently. |
| `dag.num_tasks_range` | list[int] | `[10, 30]` | Only used by `_generate_task()` for synthetic tasks. Irrelevant when loading real DAG files. |
| `dag.dependency_prob` | float | `0.3` | Only used for synthetic DAG generation. Irrelevant with real data. |
| `poisson_lambda` | float | `0.5` | Only used for synthetic task arrival. Irrelevant with real data. |

#### `training`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `batch_size` | int | `32` | PPO/GMORL batch size for gradient updates. |
| `num_episodes` | int | `1000` | Default training episodes for PPO/GMORL in `main.py`. |
| `learning_rate` | float | `3e-4` | Adam optimizer learning rate for all RL policy networks. |
| `gamma` | float | `0.99` | Discount factor for future rewards. Close to 1.0 = values long-horizon makespan improvements. |
| `meta_batch_size` | int | `8` | MAML outer loop: number of task DAGs sampled per meta-update. |
| `num_meta_iterations` | int | `300` | MAML outer loop iterations. |
| `inner_lr` | float | `0.01` | MAML inner-loop gradient step size. Controls fast-adaptation speed. |
| `inner_steps` | int | `3` | Number of gradient steps in the MAML inner loop (per task). |
| `hypervolume_threshold` | float | `0.7` | TAMPO Pareto front threshold. When the moving average Pareto hypervolume exceeds this, higher-level meta-updates are triggered. |
| `moving_average_window` | int | `50` | Window size for TAMPO's `MovingAverage` tracker used in hypervolume checks. |

#### `algorithms.tampo`
| Variable | Type | Default | Effect |
|---|---|---|---|
| `hidden_dims` | list[int] | `[256, 256]` | Hidden layer sizes for all TAMPO network components. |
| `num_attention_heads` | int | `8` | Multi-head attention heads in the decoder. |
| `encoder_type` | str | `'gcn'` | Toggles the graph encoder. Options: `'lstm'`, `'gcn'`, `'gat'`. |
| `meta_learning_rate` | float | `3e-4` | Outer-loop meta-optimizer learning rate. |
| `num_meta_iterations` | int | `100` | TAMPO-specific iteration count (overrides `training.num_meta_iterations`). |
| `meta_batch_size` | int | `10` | TAMPO-specific outer batch size. |
| `num_gat_heads` | int | `4` | Number of attention heads in GAT encoder layer 1. Only active when `encoder_type: 'gat'`. |
| `gat_hidden_dim` | int | `16` | Intermediate dimension in GAT. `actual_dim = num_gat_heads * (gat_hidden_dim // num_gat_heads)`. |
| `gat_add_self_loops` | bool | `true` | Adds self-loop edges to GAT computation. Matches GCNConv default for fair comparison. |

### 3B. Hardcoded Constants in `env/base_offloading_env.py`
These are NOT in the config file. They must be changed directly in source code.

| Constant | Location | Value | Effect |
|---|---|---|---|
| `result_size_ratio` | `_execute_offloading()` line ~455 | `0.05` | Result download = 5% of input data size. Affects both cloud and edge energy in the RL env. |
| `result_size_ratio (HEFT)` | `heft.py` line ~249 | `0.1` | HEFT uses 10% — see inconsistency note in Section 2C. |
| `congestion_normalization` | `_calculate_reward()` line ~576 | `5.0` | Congestion penalty saturates at 5 seconds of wait time. |
| `reward_scale` | `_calculate_reward()` line ~593 | `5.0` | All rewards are multiplied by 5.0 before clipping. |
| `reward_clip` | `_calculate_reward()` line ~593 | `[-5.0, 5.0]` | Hard clip on reward magnitude. |
| `edge_bandwidth_boost` | `_execute_offloading()` line ~407 | `1.5` | Edge server bandwidth is 1.5× base rate. |
| `graph_scale_norm` | `_get_graph_summary_features()` line ~291 | `50.0` | Graph size normalized to [0,1] by dividing by 50. Graphs with >50 nodes will have this feature clipped at 1.0. |
| `rayleigh_scale` | `_update_channel_gains()` line ~477 | `1.0` | Rayleigh fading σ. |

### 3C. `utils/common_evaluator.py` Hardcoded Values
| Constant | Default | Effect |
|---|---|---|
| `eval_preferences` | `[[0.8,0.2], [0.5,0.5], [0.2,0.8]]` | The three fixed preference vectors used in all evaluations. Tests delay-focused, balanced, and energy-focused policies. Cannot be changed via config. Modify the list directly in the file. |
| `num_episodes` | `config.get('eval_episodes', 20)` | Default 20 evaluation episodes. Set via `CommonEvaluator(env, {'eval_episodes': N})`. |
| `eval_seeds` | `range(1000, 1020)` | Fixed random seeds per episode for reproducibility. |
| `outlier_threshold` | `3.0` (std deviations) | Episodes further than 3σ from mean are discarded. If >20% are outliers, no filtering is applied. |
| `balanced_score_weights` | `0.5 / 10.0` (delay), `0.5` (energy) | Used only in `compare_algorithms()` to pick a "best balanced" winner. The `/10.0` de-emphasizes delay scale. |

---

## Section 4 — Scenario Guide: Tweaking for Convergence & Use Cases

### 4A. Convergence Optimization

#### Problem: Agent Always Chooses Cloud (Action 1)
The reward for cloud execution is consistently positive because `comp_imp` is large (cloud is 10× faster than local). The transmission penalty is not strong enough.

**Fix:**
```yaml
# configs/default_config.yaml
environment:
  computing:
    cloud_freq: 5000000000.0   # reduce from 10e9 to 5e9 (equal to edge)
  network:
    bandwidth_up: 5000000.0    # reduce from 20MHz to 5MHz (increase transmission time)
  reward:
    congestion_penalty_weight: 0.6   # increase from 0.4 to 0.6
```

#### Problem: Agent Always Chooses Local (Action 0)
Transmission cost dominates. The improvement ratio `comp_imp` goes negative because transmission overhead exceeds the speed advantage.

**Fix:**
```yaml
environment:
  network:
    bandwidth_up: 50000000.0   # increase from 20MHz to 50MHz
  computing:
    local_freq: 500000000.0    # reduce local from 1GHz to 0.5GHz (harder to be local-optimal)
```

#### Problem: Reward Oscillates / Doesn't Converge
Usually caused by the reward scale being too large relative to policy gradient step size.

**Fix options:**
1. Reduce the hardcoded `reward_scale` from `5.0` to `1.0` in `_calculate_reward()` (`env/base_offloading_env.py` line ~593).
2. Reduce `learning_rate` in config from `3e-4` to `1e-4`.
3. Reduce `inner_lr` (MAML inner loop) from `0.01` to `0.001`.

#### Problem: MAML Not Adapting (Inner Loop Ignored)
The inner adaptation step size is too small for visible effect.

**Fix:**
```yaml
training:
  inner_lr: 0.05      # increase from 0.01
  inner_steps: 5      # increase from 3
```

#### Problem: Agent Ignores Energy Objective
The energy metric is in Joules, which for local execution equals `kappa * cycles * freq^2 = 1e-28 * 1e9 * (1e9)^2 = 1e-10 J`. This is numerically tiny compared to delay (which is in seconds, often 0.01 to 10 seconds). The `e_imp` term gets swamped by `comp_imp`.

**Fix:** Increase `kappa` so local energy is meaningfully large:
```yaml
environment:
  energy:
    kappa: 1.0e-25    # increase from 1e-28 by 3 orders of magnitude
```
This makes local energy ~0.1 J, which is now comparable to transmission energy.

#### Problem: GCN/GAT Encoders Not Outperforming LSTM
Graph structure (edges, dependencies) is not influencing the policy enough. The agent doesn't need to look at the graph to make good decisions because the communication penalty is weak.

**Fix:**
```yaml
environment:
  reward:
    comm_penalty_weight: 0.6   # increase from 0.3 to 0.6
```
Now placing a child task cross-server is much more costly, forcing graph-aware placement.

---

### 4B. Edge-Heavy vs Cloud-Heavy Scenarios

#### To Make Edge the Dominant Strategy:
```yaml
environment:
  computing:
    cloud_freq: 3000000000.0       # slow down cloud (3 GHz)
    edge_freq: [8e9, 8e9, 8e9]     # speed up edge (8 GHz each)
  energy:
    cloud_power_tx: 1.0            # make cloud transmission more expensive
    edge_power_tx: 0.1             # make edge transmission cheap
  network:
    bandwidth_up: 10000000.0       # 10 MHz uplink (slow transmission penalizes cloud more)
```

The 1.5× bandwidth boost for edge (hardcoded in `_execute_offloading()`) combined with faster `edge_freq` and cheaper `edge_power_tx` will make edge uniformly preferred. With 3 edge servers, the agent can also parallelize across them.

#### To Make Cloud the Dominant Strategy:
```yaml
environment:
  computing:
    cloud_freq: 20000000000.0      # very fast cloud (20 GHz)
    edge_freq: [2e9, 2e9, 2e9]     # slow edge (2 GHz)
  energy:
    cloud_power_tx: 0.1            # cheap cloud transmission
    edge_power_tx: 0.5             # expensive edge transmission
  network:
    bandwidth_up: 100000000.0      # high bandwidth (100 MHz → transmission penalty negligible)
```

#### To Create a Balanced Edge/Cloud Tradeoff (Ideal for Multi-Objective Benchmarking):
Use default values but make energy costs meaningful:
```yaml
environment:
  energy:
    kappa: 1.0e-25        # local energy matters
    cloud_power_tx: 0.5   # cloud transmission costly
    edge_power_tx: 0.3    # edge cheaper but not free
  computing:
    cloud_freq: 10e9      # cloud faster (delay)
    edge_freq: [5e9, 5e9, 5e9]  # edge slower (delay)
```
In this setup:
- **Delay preference** (`[0.8, 0.2]`) → agent should prefer cloud (faster compute)
- **Energy preference** (`[0.2, 0.8]`) → agent should prefer edge (cheaper transmission) or local (no transmission)
- **Balanced** (`[0.5, 0.5]`) → agent should spread tasks across edge servers

This is the configuration that makes the Pareto front plot in `benchmark.py` most informative.

---

### 4C. Multi-Objective Balancing (Preference Vector Tuning)

The preference vector `[w_delay, w_energy]` is sampled randomly during training and fixed during evaluation. It directly appears in the reward formula:
```
reward ∝ w_delay * combined_delay_metric + w_energy * e_imp
```

**Training sampling** (in `tampo.py`, `_sample_preference()`):
- Randomly samples `w_delay ∈ [0.2, 0.8]` then sets `w_energy = 1 - w_delay`.
- This covers the full Pareto tradeoff spectrum during training.

**Evaluation fixed vectors** (in `common_evaluator.py`):
```python
self.eval_preferences = [
    np.array([0.8, 0.2]),   # aggressive delay minimization
    np.array([0.5, 0.5]),   # balanced
    np.array([0.2, 0.8])    # aggressive energy minimization
]
```

**To test extreme preferences:**
Modify `eval_preferences` directly in `utils/common_evaluator.py`:
```python
self.eval_preferences = [
    np.array([1.0, 0.0]),   # pure delay (ignores energy completely)
    np.array([0.5, 0.5]),
    np.array([0.0, 1.0])    # pure energy (ignores delay completely)
]
```

**To test finer granularity for a Pareto curve:**
```python
self.eval_preferences = [
    np.array([w, 1-w]) for w in [0.1, 0.3, 0.5, 0.7, 0.9]
]
```
Increase `num_episodes` proportionally so each preference gets enough coverage.

**Effect of Preference on Convergence:**
- Pure `[1.0, 0.0]` training: Agent will maximize speed at all energy costs, collapsing onto a single strategy (always cloud or always fastest edge). Pareto front degenerates to a point.
- Random `[w, 1-w]` training (current setup): Agent learns a conditional policy that adjusts based on the current preference. The Pareto front becomes a curve, enabling the hypervolume metric to be meaningful.
- If the agent fails to differentiate between `[0.8, 0.2]` and `[0.2, 0.8]` in evaluation, it means the preference vector is not being properly routed into the policy network. Check that `env.preference` is being passed to `env.reset(preference_vector=preference)` in the evaluation loop.
