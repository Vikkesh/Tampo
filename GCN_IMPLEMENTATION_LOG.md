# Detailed Change Log & Implementation Logic: GCN Integration

This log reflects the current, corrected TAMPO graph pipeline after the GCN rework. The old version described a mathematically valid GCN kernel, but the end-to-end system was still passing flattened environment state instead of true DAG node features. That gap has now been closed.

## Current GCN Design Standard
The active implementation follows the Kipf-Welling style update:

$$
X^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X^{(l)}W^{(l)}\right)
$$

with the following practical choices:

* `A` is built from the DAG adjacency matrix parsed from the `.gv` file.
* The dense adjacency is symmetrized before propagation so the native GCN behaves like the common undirected normalization used in standard GCN layers.
* Self-loops are added every forward pass.
* Degree normalization is applied with dense `torch.bmm` operations.
* Variable-size graphs are padded inside the batch and accompanied by a node mask.
* A graph-level context vector is produced by masked pooling over encoded node states.

The implementation stays in native PyTorch instead of `torch_geometric`, which keeps the project portable in environments like Colab while still matching the core GCN message-passing math.

---

## Latest End-to-End Graph Pipeline

### 1. `tampo/utils/dag_parser.py`
**What changed now**
* The parser still creates `adj_matrix`, but it now also sorts tasks by node id and maps edges through an explicit `id_to_index` dictionary.

**Why this mattered**
* `pydotplus` does not guarantee node order.
* Without this fix, the task feature list and adjacency rows/columns could point at different nodes.
* This alignment fix is essential for both the LSTM baseline and the GCN path.

### 2. `tampo/env/base_offloading_env.py`
**What changed now**
* Added sticky dataset-task handling so `set_task(task_id)` survives `reset()`.
* Added `clear_task_selection()` to avoid stale task reuse outside TAMPO.
* Added `get_task_feature_matrix()` to expose real per-node graph features.
* Added `get_server_features()` to expose a structured server-state vector.
* Added `get_current_task_graph()` and kept `get_adjacency_matrix()`.
* Updated the observation builder to include graph summary information instead of mostly padded zeros.
* DAG-backed episodes now terminate after a single graph-level decision rather than swapping in a random scalar task mid-rollout.

**New node feature layout**
Each node now uses a 6D feature vector:
1. normalized node `data_size`
2. normalized node `cycles`
3. normalized `in_degree`
4. normalized `out_degree`
5. normalized DAG `depth`
6. normalized communication load

**Why this mattered**
* TAMPO now receives a real `[num_nodes, feature_dim]` matrix instead of a fake `state[:6]` slice.
* The environment no longer destroys the selected DAG during `reset()` or after the first step.

### 3. `tampo/algorithms/rl/tampo.py`
**What changed now**
* `DAGEncoder` now supports `encoder_type in {'lstm', 'gcn', 'both'}`.
* Added dense graph padding and `node_mask` support for batching.
* Replaced the old “1 token pretending to be a graph” path with true graph tensors.
* Replaced the old temporary-weight-swapping MAML shortcut with a proper functional forward pass using `torch.func.functional_call`.
* `GCN` path now:
  * symmetrizes adjacency
  * adds self-loops
  * normalizes by degree
  * applies stacked dense message-passing layers
  * masks padded nodes after every layer
* `LSTM` path now operates over node sequences and supports padded batches via packed sequences.
* The final graph representation is built with masked pooling over encoded node states.
* The policy now decodes a single graph-level offloading decision from the encoded DAG instead of treating the raw flat state as if it were a node sequence.
* Loss computation now pads graph batches properly before forwarding them through the policy.
* The MAML inner loop now keeps the full update chain differentiable, so second-order meta-gradients can flow from test loss back through the adapted parameters to the original meta-parameters.

**Latest encoder structure**
```text
Node feature matrix [B, N, F]
  -> task embedding
  -> one of:
       LSTM branch
       GCN branch
       BOTH branch (LSTM + GCN fused)
  -> multi-head self-attention over encoded nodes
  -> masked graph pooling
  -> server-state encoder
  -> preference-conditioned decoder
  -> action logits
```

### 4. `tampo/main.py`
**What changed now**
* Environment setup now merges `system`, `computing`, `energy`, `network`, and `tasks` config sections instead of passing only `system`.
* `test_tampo(...)` now accepts an explicit `encoder_type` argument.
* DAG tasks loaded into TAMPO now preserve `adj_matrix`.
* TAMPO variants save to isolated checkpoints:
  * `models/tampo_lstm_checkpoint.pth`
  * `models/tampo_gcn_checkpoint.pth`
  * manual config support also exists for `both`
* Plot colors were made dynamic so more algorithms can be shown in a single results figure.

### 5. `tampo/utils/common_evaluator.py`
**What changed now**
* TAMPO evaluation now walks the loaded DAG task dataset instead of silently resetting into scalar random tasks.
* Evaluation calls the framework’s graph-aware `select_action(...)` helper directly.

### 6. `tampo/configs/default_config.yaml`
**What changed now**
* `encoder_type` remains configurable and now documents all supported options:
  * `'lstm'`
  * `'gcn'`
  * `'both'`

---

## Latest MAML Update

The TAMPO graph pipeline now sits on top of a more correct meta-learning implementation.

### What changed
The previous `_forward_with_params(...)` workaround temporarily assigned tensors into the live model before a forward pass and then restored the original weights. That made the code runnable, but it weakened the core MAML logic because PyTorch could not reliably preserve the full "original weights -> inner update -> adapted weights -> test loss" dependency chain.

The current implementation now:

* imports `functional_call` from `torch.func`
* keeps the live `meta_policy` module untouched
* forwards with an explicit `params_dict`
* includes model buffers alongside the parameter dictionary
* preserves differentiability through the inner-loop update path

### Why this matters
This change does not alter the GCN math itself, but it makes the TAMPO meta-learning stack more research-correct:

* the GCN/LSTM encoder can now participate in proper second-order meta-gradients
* fast adaptation behavior is better aligned with real MAML
* the meta-update is no longer based on manual weight swapping

---

## What Was Corrected Conceptually

The previous prototype had the right matrix multiplication inside `_apply_gcn()`, but the actual training path had four structural issues:

1. adjacency could be dropped before the environment used it
2. `reset()` could overwrite the selected DAG
3. the policy saw a single fake token instead of `N` graph nodes
4. batching assumed every graph tensor was already shaped correctly

Those issues are now addressed in the active implementation.

The final remaining conceptual issue from the previous revision, the approximate MAML inner loop, has now also been corrected.

---

## Practical Outcome

The current TAMPO GCN path is now a real graph encoder pipeline:

* parsed DAG -> aligned adjacency
* environment -> graph node feature matrix + adjacency + server features
* TAMPO -> padded graph batch + node mask
* encoder -> LSTM, GCN, or fused both
* MAML inner loop -> functional parameter update path
* decoder -> one graph-level offloading decision

That makes the implementation substantially closer to standard research and production practice than the previous prototype log described.
