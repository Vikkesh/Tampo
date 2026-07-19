# Dev Log: Graph Encoder & Policy Observability Overhaul

**Date:** 2026-07-10
**Scope:** Make the GCN / GAT / LSTM comparison a valid experiment. Before this change all
three encoders were structurally incapable of emitting a per-node policy.
**Files:** `env/base_offloading_env.py`, `algorithms/rl/tampo.py`, `utils/common_evaluator.py`,
`benchmark.py`, `configs/default_config.yaml`, `env/wrappers/*.py`, `Papers referred.md`,
`Colab_Test_Run.ipynb`, `docs/READING_RESULTS.md` (new)

**Prerequisite:** `dev_logs/reward_signal_and_determinism_overhaul.md` (same day) fixed the
*reward signal*. This log fixes the *agent that consumes it*.

---

## 0. The Finding

With dropout disabled, the policy's output logits were measured at every node of a 20-node DAG:

```
GCN   eval (dropout OFF)   max|logit - logit_step0| = 0.00e+00   unique_actions=[1]
GAT   eval (dropout OFF)   max|logit - logit_step0| = 0.00e+00   unique_actions=[0]
LSTM  eval (dropout OFF)   max|logit - logit_step0| = 0.00e+00   unique_actions=[0]
```

**Exactly zero.** All three encoders emitted identical logits at every node. The agent chose
one server and sent the entire DAG there. Any GCN-vs-GAT-vs-LSTM ranking produced before this
fix compared three architectures that all collapse to "pick one server per graph."

Three independent causes, all now fixed.

---

## 1. The Policy Could Not See Which Node It Was Scheduling

`MetaPolicyNetwork.forward` received `task_features` (the whole `[N, 6]` node matrix — identical
at every step), `server_features`, `preference`, and `adjacency`. None of them encode the current
node index. The decoder's attention query was `cat([h_t, pref_encoded])`, and `h_t` was
initialised from the graph context. There was no path by which "I am placing node 7" could reach
the logits.

Meanwhile `MultiObjectiveValueNetwork` receives the flat `state`, whose first six entries *are*
the current node's features. **The critic knew which node it was evaluating; the actor did not.**

### Fix

- `get_task_feature_matrix()` gains three columns (`6 → 9`):
  `is_current`, `is_scheduled`, `assigned_server_norm`.
  The last one lets a child node see where its parents landed, which is the information required
  to avoid the cross-server communication penalty.
- `current_node_idx` is threaded through `_select_action` → `_collect_task_experiences` →
  the experience dict → `_prepare_policy_batch` → `MetaPolicyNetwork.forward` →
  `PreferenceConditionedDecoder.forward` → `_forward_with_params` (for the MAML functional call).
- The decoder indexes the current node's embedding out of `encoded_tasks`
  (`encoded_tasks[arange(B), current_node_idx]`, pointer-style, cf. Vinyals et al. 2015),
  projects it, and uses it in **both** the attention query and the decision head input.
  `lstm_cell` input grows `hidden*3 → hidden*4`; `decision_head` input likewise.
- `TAMPOFramework._current_node_idx()` maps `env.current_node_idx` through `env.topo_order`
  to the row index used by the feature matrix.

---

## 2. `server_loads` Was Dead — The Agent Was Blind to Congestion

```python
# __init__ and reset()
self.server_loads = np.zeros(self.num_servers)
# get_server_features()
self.server_loads / 10.0,
```

`_execute_offloading` advances `self.server_available`. Nothing ever wrote `self.server_loads`
again. The policy therefore read `[0, 0, 0, 0]` for the queue state at every step of every
episode, while `server_available` correctly held e.g. `[0, 0.0470, 0, 0, 0]`.

Combined with the constant `task_features`, **every input to the GCN/GAT policy was constant
within an episode.** The context was constant, the LSTM state was constant, the logits were
constant. Confirmed empirically:

```
server_features   : (21, 20), identical across all steps? True
  server_loads slice (idx 4:8) = [0. 0. 0. 0.]
task_feature_matrix: identical across all steps? True
env.server_available DOES change: [0. 0.04696084 0. 0. 0.]
```

This is why fixing the congestion *penalty* (previous dev log) was necessary but not sufficient:
the reward could punish queueing all it liked, but the agent had no observation with which to
respond.

### Fix

`get_server_features()` now reports the real timeline, two ways:

- `rel_load = server_available / server_available.max()` — which processor is the bottleneck.
- `abs_load = tanh(server_available)` — how loaded the system is overall, bounded as makespan grows.
- plus `progress = current_node_idx / num_nodes`.

Layout: `cloud_freq(1) + edge_freq(3) + rel_load(5) + abs_load(5) + channel_gains(4) + progress(1) = 19`,
zero-padded to the 20-dim contract. `self.server_loads` is now also kept in sync so `render()`
and `info` stop lying.

---

## 3. The GNN Crushed the Graph Into Two Scalars

```python
summary = torch.stack([x_fwd.mean(), x_bwd.mean()])   # [2]
...
encoded_tasks = context.unsqueeze(1).expand(-1, N, -1)
```

`gnn2_fwd` was `GCNConv(16, 1)`, so each stream produced one scalar per node — and then `.mean()`
reduced the entire graph to a single number. Two numbers total (forward + backward), concatenated
with 20 server features.

`encoded_tasks` was then the graph context broadcast to all `N` node slots, so the decoder's
multi-head attention ran over `N` **identical** keys and values. Attention over identical keys is
a no-op: uniform weights, output equal to the shared value.

RC#2 in `convergence_fixes_overhaul.md` claimed to fix "encoded_tasks all-zeros" by introducing
this broadcast. It replaced *all-zeros* with *all-identical*. The tensor became non-zero, which is
what the test asserted, but it carried no more per-node information than zeros did.

### 3a. This was a deviation from GDRL, not fidelity to it

`GDRL/Feature.py`, the cited reference:

```python
con_output = self.gnn2(con_output, self.edge_index).squeeze(2)   # [B, 35] — per node
var_input  = torch.cat((var_state, con_output), dim=1)           # all 35 values concatenated
```

**GDRL never pools.** It keeps one value per node and concatenates all 35. The `.mean()` was
TAMPO's own addition, introduced to cope with variable node counts. Restoring per-node outputs
moves the implementation *closer* to the reference.

`grep -c "bwd\|backward\|reversed" GDRL/Feature.py` → `0`. The bidirectional stream is likewise
TAMPO's, not GDRL's.

### Fix

- `gnn2_fwd` / `gnn2_bwd`: `GCNConv(gnn_hidden_dim, hidden_dim)` (was `→ 1`).
  Same for `gat2_fwd` / `gat2_bwd`.
- Both streams concatenated → `[N, hidden_dim * 2]` per-node embeddings, which matches the
  BiLSTM output width and the decoder's `embed_dim`, keeping the three encoders interchangeable.
- `_apply_gcn` and `_apply_gat` now delegate to a shared `_apply_bidirectional_gnn`, so the conv
  operator is provably the only difference between them — the precondition for attributing any
  measured GCN-vs-GAT gap to attention. It also runs the convs over the whole PyG batch at once
  instead of a per-graph Python loop.
- Graph-level context = masked **mean readout** over node embeddings ⊕ server features → GDRL's
  unchanged two-block FNN head. The readout replaces GDRL's fixed-size concatenation, which
  assumes a constant node count.
- Padding slots are masked to zero; the readout divides by the true node count.

New config key: `algorithms.tampo.gnn_hidden_dim: 16` (GDRL's intermediate width).

---

## 4. "Deterministic" Evaluation Was Not Deterministic

`.eval()` appears exactly twice in `tampo.py`, both inside `inner_loop_update`. The network is
therefore in **train mode with dropout active** during benchmarking, and
`select_action(deterministic=True)` took an argmax over dropout-corrupted logits.

That is where the apparent per-node action variation in previous benchmark runs came from — not
from reasoning. In train mode the same rollout gives `max|logit - logit_step0| ≈ 0.6–0.8` and
actions scattered across servers; in eval mode it gives exactly `0.0`.

Dropout during collection also silently broke the PPO ratio introduced earlier the same day:
`old_log_prob` was recorded under one dropout mask, while the loss recomputes `log_prob` under a
different one, so `exp(log_prob - old_log_prob)` was noise rather than an importance ratio.

### Fix

`_select_action` now forces `policy.eval()` for the forward pass and restores the previous mode in
a `finally` block. Exploration comes from `Categorical(probs).sample()`, not from dropout.

---

## 5. Action Observability

None of this was visible from the reported metrics. A degenerate all-cloud policy posts a
perfectly respectable makespan. Added:

**Training** — `TAMPOFramework._action_counts` accumulates every sampled action per meta-iteration.
`_format_action_distribution` renders the histogram plus a normalised entropy
`H(p) / ln(num_actions)`; printed on the same cadence as the loss line:

```
  [actions] local=18.7% cloud=36.7% edge0=16.7% edge1=22.7% edge2= 5.3% | entropy=0.91 (0=collapsed, 1=uniform) | n=150
```

**Evaluation** — `CommonEvaluator.evaluate_rl_agent` records the full per-node action sequence for
every `(dag, preference)` episode. `_summarize_actions` computes:

- overall action fractions,
- fractions **per preference vector** — the direct test of preference conditioning, which was
  never verified before,
- `mean_per_episode_entropy` — entropy *within* one DAG. An agent that sends DAG A entirely to
  cloud and DAG B entirely to edge0 has a balanced *overall* mix and a within-episode entropy of
  `0.0`. Only the latter detects it.
- `degenerate_episodes` — count of episodes with zero within-episode entropy, printed as a ⚠.

**Files** — `benchmark.py` writes `action_traces.csv` (one row per episode, full placement
sequence) and `action_distribution.csv` (long format, per preference). `benchmark_results.csv`
gains `within_episode_entropy` and `degenerate_episodes`.

---

## 6. Incidental

- `task_feature_dim` is now read from `env.task_feature_dim` (`TaskOffloadingEnv.TASK_FEATURE_DIM = 9`)
  rather than the literal `6` hardcoded in `TAMPOFramework.__init__`, so a feature addition cannot
  silently desync the encoder's input layer. `FlatVectorWrapper` and `SequenceWrapper` default to
  the env's width for the same reason.
- `TAMPOFramework.load()` raises an explanatory `RuntimeError` on pre-overhaul checkpoints instead
  of a raw shape mismatch. **Existing `models/*.pth` are incompatible and must be deleted.**
- `value_loss_coef` / `entropy_coef` were hardcoded `0.5` / `0.01`; now config keys.

---

## 7. Verification

**Observation now varies within an episode:**

```
task_feature_dim = 9
task_feature_matrix changes between steps? True
server_features    changes between steps? True
is_current column moved: step0 node=0 -> step1 node=1
is_scheduled count: 0 -> 1
```

**Logits are now a function of the current node** — probe holding graph, servers and preference
fixed while varying only `current_node_idx`:

```
GCN   logit spread across nodes = 0.2669
GAT   logit spread across nodes = 0.3203
LSTM  logit spread across nodes = 0.3871
```

(was exactly `0.0` for all three). Preference conditioning remains intact (`0.19–0.44` spread
across the three preference vectors).

**Encoder unit test** (notebook cell 1, rewritten) now asserts what actually matters — that real
nodes have *distinct* embeddings and padding is masked:

```
PASS [GCN] context=(3, 256) encoded_tasks=(3, 20, 256) per-node spread=1.9755 padding masked
PASS [GAT] context=(3, 256) encoded_tasks=(3, 20, 256) per-node spread=1.1870 padding masked
```

The old assertion (`encoded_tasks.abs().max() > 1e-9`) passed throughout the broken period.

**All three encoders train** with finite losses and a live action histogram:

```
gcn   Iter 5/6 | Loss: 2.6561 | [actions] local=18.7% cloud=36.7% edge0=16.7% edge1=22.7% edge2=5.3% | entropy=0.91
gat   Iter 5/6 | Loss: 0.3047 | [actions] local=78.7% cloud= 9.3% edge0= 5.3% edge1= 3.3% edge2=3.3% | entropy=0.49
lstm  Iter 5/6 | Loss: 0.6638 | [actions] local=53.3% cloud=30.7% edge0= 0.7% edge1=10.7% edge2=4.7% | entropy=0.69
```

**The benchmark's degeneracy detector fires correctly.** After a deliberately short 4-iteration
run, all three policies are still greedy-degenerate, and the harness says so plainly rather than
reporting a plausible-looking makespan:

```
  ✓ Actions       : local= 0.0% cloud= 0.0% edge0=100.0% edge1= 0.0% edge2= 0.0%
  ✓ Within-episode entropy: 0.000
  ⚠ 9/9 episodes placed EVERY node on a single server — policy is degenerate.
```

---

## 8. What This Does and Does Not Establish

**Established:** the policy is now *capable* of a per-node, congestion-aware,
preference-conditioned schedule. The logit spread across nodes is non-zero for all three
encoders; the observation varies step to step; the GNN preserves per-node structure; the
GCN-vs-GAT comparison is now controlled (shared skeleton, operator is the only variable).

**Not established:** that any encoder *learns* to use this capability, or that GCN/GAT beat LSTM.
The short verification runs above still end in a degenerate greedy policy — which is expected at
4–6 meta-iterations from random init, and is exactly what the new diagnostics exist to reveal.

The honest reading is that every benchmark number produced before this change is void, and the
open question is now testable for the first time. Run the multi-seed protocol in
`docs/READING_RESULTS.md`, watch `within_episode_entropy` rise above zero, and only then compare
encoders.

---

## 9. Training Budget: Why Runs Never Converged

`num_meta_iterations` was suspected of being too small. Measured directly.

### 9a. Four config keys are inert

`grep` for `config.get('<key>')` across the codebase:

| key | in YAML | read by Python |
|---|---|---|
| `num_meta_iterations` | 2 places (300, 100) | **0** |
| `meta_batch_size` | 2 places (8, 10) | **0** |
| `num_episodes` | 1 | **0** |
| `num_attention_heads` | 1 | **0** (decoder hardcodes 8) |

Iteration count and meta-batch size reach `TAMPOFramework.train()` from the caller only:
`Colab_Test_Run.ipynb` (`NUM_ITERATIONS`, `META_BATCH_SIZE`) and `main.py`'s
`train_iterations` argument. **This is by design** — the Colab cell is intended to be the
source of truth. The YAML keys are now explicitly annotated `INERT` so nobody edits them
expecting an effect.

So the real iteration count was the notebook's `NUM_ITERATIONS = 75`, not `100` or `300`.

### 9b. 75 iterations barely perturbs the network

Instrumented the GCN policy: cloned all parameters, trained 10 meta-iterations at the
configured `meta_learning_rate = 5.0e-5`, and measured displacement.

```
mean |Δparam| per meta-iteration = 9.342e-06     (≈ 0.19 × lr; Adam step ≈ lr, damped by grad clipping)
mean |param| at init             = 0.0288
```

Extrapolating:

| iterations | mean \|Δparam\| | as % of init scale |
|---|---|---|
| 75 (old default) | 0.0007 | **2.4%** |
| 100 | 0.0009 | 3.2% |
| 300 | 0.0028 | 9.7% |
| 1000 | 0.0093 | **32.4%** |

At 75 iterations the average weight moves 2.4% of its initial magnitude. That is an
essentially untrained network, and it explains why every short verification run in §7
ended in a degenerate greedy policy.

### 9c. Do NOT compensate by raising the learning rate

The obvious response — raise `meta_learning_rate` instead of iterating longer — was tested
and is wrong. Sweep at 12 iterations, seed 42, GCN and LSTM:

| `meta_learning_rate` | loss finite | loss trend | **action entropy** |
|---|---|---|---|
| **5.0e-5** (current) | yes | ↓ | **0.72 / 0.77** — healthy |
| 1.5e-4 | yes | ↓ | **0.00 / 0.00** — collapsed |
| 3.0e-4 | yes | ↓ | **0.00 / 0.00** — collapsed |

The loss stayed finite and trended *downward* in every case. A loss curve alone would have
endorsed `3.0e-4`. Only the action-entropy diagnostic added in §5 reveals that the policy
has collapsed onto a single server. `3.0e-4` was the pre-RC#6 default.

`meta_learning_rate` stays at `5.0e-5`. The knob is the iteration count.

### 9d. Wall-clock cost IS the constraint

An earlier draft of this log claimed "cost is not the constraint," extrapolating from a
`meta_batch_size=4` CPU probe. That was wrong. Real reported data point: **75 iterations of
all three encoders took ~4 hours on a Colab free-tier T4**, and the session limit is ~4h.

Per-iteration cost measured before vs after this overhaul (CPU, `meta_batch=4`,
graphs of 10/20/30 nodes, 3 timed iterations after a warmup; pre-change tree extracted with
`git archive HEAD`):

| encoder | OLD (HEAD) | NEW | speedup |
|---|---|---|---|
| gcn | 3.52 s | 3.28 s | 1.08× |
| gat | 8.30 s | 4.12 s | **2.02×** |
| lstm | 14.09 s | 14.11 s | 1.00× |

The overhaul does not slow iterations down. GAT gained 2× because `_apply_bidirectional_gnn`
runs the convolutions over the whole PyG batch at once, replacing the per-graph Python loop
(which called the conv once per graph, 32 times per forward pass).

**LSTM is ~4× GCN** and dominates any all-three-encoders loop, because CuDNN must be
disabled for MAML's second-order gradients.

### 9e. Changes made

- `TAMPOFramework.train(..., time_budget_s=None)` — stops at an iteration boundary once the
  wall-clock budget is exhausted, saves a checkpoint, and prints resume instructions.
  Without it a Colab session is killed mid-iteration and loses everything since the last
  10-iteration autosave. Progress lines now print `s/it` so a session can be planned.
- `episodes_per_task` promoted from a hardcoded `5` in `_collect_task_experiences` to a live
  config key. Scales rollout cost ~linearly.
- `Colab_Test_Run.ipynb`: trains **one encoder per session** (`ENCODERS = ['gcn']`), with
  `META_BATCH_SIZE` `15 → 6`, `EPISODES_PER_TASK` `5 → 3`, `TIME_BUDGET_HOURS = 3.5`.

The rationale for shrinking the meta-batch rather than the iteration count: **Adam's step
size is ≈ `lr` regardless of how many tasks the gradient averages over.** `meta_batch_size`
buys gradient quality, not weight displacement. Halving it roughly halves per-iteration cost
and therefore doubles the optimiser steps available in a fixed session, at the price of
noisier gradients. `inner_steps` scales cost almost linearly too and is the next lever.

Verified: `time_budget_s=8` stopped a 100-iteration run after 11 iterations in 8.3 s with a
saved checkpoint; resuming stacked correctly (11 → 13); `episodes_per_task` 2 vs 5 produced
20 vs 50 experiences on a 10-node DAG.

The iteration number is a budget, not a convergence guarantee. The stopping criterion is
behavioural, and is documented in the cell and in `docs/READING_RESULTS.md`:

1. `within_episode_entropy` has risen off `0.000` — while it is zero the agent places every
   node of a DAG on one server and has not learned to schedule at all.
2. `avg_makespan` has plateaued across two consecutive checkpoints.

---

---

## 9f. Exact Cross-Session Resume (the multi-session comparison depends on it)

Free Colab kills a session at ~4h; the three encoders need far more, so training is split
across sessions. For the comparison to be valid, a split run must equal a continuous run.

**Two determinism bugs fixed:**

1. **Weight init was not seeded.** Seeding had been added inside `train()`, but weight
   initialisation happens in `__init__`, *before* `train()` runs. So the initial weights
   depended on whatever global RNG state preceded construction — two "identical" runs
   diverged from iteration 0. Fix: `TAMPOFramework(__init__)` now takes `seed` and calls
   `set_seed` **before** building the networks. torch init consumes the torch RNG (so
   different-sized encoders diverge there, which is fine); it does not touch numpy, so every
   encoder enters training with an identical numpy stream — identical graphs, preferences and
   channel gains.

2. **RNG stream was not checkpointed.** `_save_checkpoint` now stores
   `{python, numpy, torch, torch_cuda}` RNG state. `load()` captures it (without applying —
   applying in `load()` would be undone by any draw before the loop); `train()` restores it
   at the top of the loop when resuming, instead of re-seeding. So iterations N..M of a
   resumed run continue the exact stream.

Also: `save()` used to duplicate `_save_checkpoint` *without* the RNG state, and the notebook
calls `save()` at the end of every session — it would have silently overwritten the good
checkpoint. `save()` now delegates to `_save_checkpoint`. Writes are atomic (temp + rename,
with a direct-write fallback for Google Drive's FUSE mount, which can reject cross-name
rename).

**Verified bit-identical:** loss sequences for `3+3` and `2+2+2` iteration splits each equal
a continuous 6-iteration run exactly. The full notebook driver (session 1: `0→2` on budget;
session 2: `2→4`; session 3: `4→8` done) produced a checkpoint whose loss history matches a
continuous 8-iteration run. Cross-encoder: GCN and LSTM draw an identical graph-ID sequence
under the same seed.

## 9g. Stopping / Saving Mechanism and the Notebook Driver

`train(..., time_budget_s=None)` stops three ways, always leaving a valid checkpoint:
reaching `num_iterations`; exhausting the wall-clock budget (stops at the next iteration
boundary, saves, returns — this is what prevents a mid-iteration kill); or a hard kill, after
which the 10-iteration autosave loses at most <10 iterations. `episodes_per_task` is now a
live config key (was hardcoded `5`).

The notebook training cell is an **auto-advancing, resumable driver**: it constructs each
encoder with `model_path` (auto-resume) and `seed`, skips encoders already at
`TARGET_ITERATIONS`, and pours the remaining session budget into the next unfinished one.
Checkpoints write directly to `CKPT_DIR`, which points at Google Drive (`USE_DRIVE=True`) so
they survive session death, or local disk for a single VM. A new "Persistence & experiment
configuration" cell mounts Drive and fails loudly if `CKPT_DIR` is not writable — so a failed
mount can't masquerade as "starting fresh" and discard a prior session.

Full operator guide: `docs/RUNNING_THE_EXPERIMENT.md` (three cases: multi-session Colab,
single VM, multi-seed publication run).

## 10. Still Open (Deliberately Deferred)

- The task's own upload time is not charged to `finish_time` (§9 of the reward dev log).
- The energy formula double-counts transfers for non-source nodes.
- `_best.pth` selects on meta-loss rather than a scheduling metric.
- `entropy_coef: 0.01` may be too weak to hold entropy up at higher learning rates — a
  joint `(meta_learning_rate, entropy_coef)` sweep was not run.
