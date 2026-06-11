# Development Log: Physics Engine and Reward System Overhaul

## Overview
This update fundamentally restructures the core simulation engine and the underlying reward mechanisms of the Task Offloading (TO) framework. Previously, the environment lacked true DAG dependency resolution, stepping through task nodes without accounting for topological ordering, which produced a "flat" optimization problem. The reward system compared action latencies against local execution alone, leading all agents into identical policy attractors (e.g., invariably offloading to the cloud) because cross-server constraints and congestion were not enforced.

This overhaul corrects those issues, establishing a fair, physically accurate, and complex benchmarking foundation where graph-aware architectures can truly differentiate themselves.

## Modifications in Detail

### 1. `env/base_offloading_env.py` (Physics & Rewards)
- **Library Migration:** Replaced the unmaintained `gym` imports with `gymnasium` across the environment and wrappers (`flat_vector_wrapper.py`, `sequence_wrapper.py`) to resolve Colab compatibility issues with NumPy 2.x.
- **DAG Topological Stepping:** 
  - Integrated Kahn's algorithm (`_compute_topological_depths`) to generate a valid depth-ordered `topo_order`.
  - Added timeline tracking arrays: `node_finish_times`, `node_assignments`, and `server_available`.
  - `step()` now processes exactly one DAG node (in topological order) per iteration rather than acting holistically.
- **True Makespan Calculation:** `_execute_offloading()` was rewritten to replicate the critical-path latency logic established in `heft.py`.
  - Step 1: Wait for data-ready time from *all* parent nodes.
  - Step 2: Earliest start is max of data-ready and `server_available`.
  - Step 3: Compute processing delay and update global timeline.
  - Cross-server communication latencies and energy penalties are now applied strictly when a parent and child are scheduled on different servers.
- **Three-Component Reward System:** `_calculate_reward()` now generates differentiated, topology-sensitive pressure:
  1. *Computation Improvement:* The base advantage over local processing.
  2. *Congestion Penalty:* Tracks `server_available` wait time. Penalizes stacking multiple nodes onto the same computational bottleneck.
  3. *Communication Penalty:* Penalizes placing closely coupled dependent nodes on different physical machines.

### 2. `configs/default_config.yaml`
- Added the `reward` block to parameterize penalties:
  - `congestion_penalty_weight: 0.4`
  - `comm_penalty_weight: 0.3`
  - `improvement_baseline: 'local'`

### 3. `algorithms/rl/tampo.py`
- Refactored `_collect_task_experiences()` and `evaluate()` loops to dynamically pull the actual node length of the DAG for each episode (instead of the fixed `max_steps=50`).
- Swapped recording metrics to retrieve the newly accurate global `makespan` at the conclusion of the DAG step sequence.

### 4. `utils/generate_test_dataset.py`
- Dropped the buggy reliance on `TaskOffloadingEnv.reset()`.
- Switched data generation to utilize the `DAGParser` explicitly, sampling from variable-length graph folders (`meta_offloading_n/offload_random10` up to `50`). This builds a comprehensive golden dataset to properly evaluate agent generalization over graph scale.

### 5. `benchmark.py` & `utils/common_evaluator.py`
- Reprogrammed plotting and CSV utilities to record and visualize `makespan` instead of average step delays. This aligns with standard scheduling academic metrics—dense per-node rewards for training, global makespan for final benchmarking.

## Status
All updates applied locally. Testing is to be conducted inside the primary Google Colab environment.
