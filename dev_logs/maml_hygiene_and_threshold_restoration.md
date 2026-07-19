# MAML Hygiene & Threshold-Mechanism Restoration (2026-07-20)

Codebase audit ahead of the GCN/GAT/LSTM comparison runs. Every encoder and pipeline was
exercised on CPU (venv, small dims); three defects were found and fixed, plus a fourth
uncovered while verifying the first. None of them affected the DAG training/benchmark
pipeline's correctness before the fix — but two affected gradient quality and one
disconnected a mechanism the TAMPO paper (§3.2.3) claims as a core contribution.

## 1. Orphaned `__init__` tail in `LowerLayerAgent` (fixed)

When `_ppo_policy_loss` / `_old_log_probs` were inserted into the class, the tail of
`__init__` — the entire hypervolume-tracking state (`hv_threshold`, `hv_window`,
`hv_calculator`, `performance_buffer`, `hv_history`, `update_needed`, `update_package`) —
was stranded *after the `return`* inside the `_old_log_probs` staticmethod. Dead code;
none of the attributes ever existed. `update_performance()`, `get_update_package()` and
`HigherLayerMetaLearner.collect_updates()` all raised `AttributeError` (confirmed by
direct call). The threshold-adaptive communication mechanism of the paper was therefore
completely non-functional, silently.

**Fix:** block moved back into `__init__`. Verified: all attributes exist; a poor-performing
agent's trigger fires; `collect_updates()` returns exactly its package.

## 2. Inverted hypervolume semantics (found while verifying #1, fixed)

`update_performance` stored **benefit** improvements (higher = better) but
`HypervolumeCalculator` uses a **minimisation** convention (Pareto filter keeps smaller
coordinates; HV grows as points drop below the reference `[2,2]`). Net effect: a uniformly
poor agent (improvement −0.9) scored HV ≈ 8.4 and a uniformly good one ≈ 1.2 — exactly
backwards, so the paper's `HV_avg < τ → request meta-update` trigger could never fire for
the agents that needed help. The code comment even said "flip sign so higher = better";
the flip was never implemented.

**Fix:** improvements are negated into costs before buffering. Measured after fix:
good agent HV = 8.41 (no trigger), poor agent HV = 2.20 < τ → trigger fires,
`collect_updates()` returns only the poor agent's package.

**Threshold recalibrated:** with costs in [−1,1] against reference [2,2], a single-point
front spans ≈ [1.2 (poor), 8.4 (good)], so the old `hypervolume_threshold: 0.7` sat below
the attainable minimum and could never fire even with correct signs. Default moved to 3.0
with the unit derivation documented in `configs/default_config.yaml`.

**Status of the mechanism:** implemented and unit-tested, but **not wired into the
single-device training loop**, which performs a meta-update every iteration — equivalent
to the trigger always firing (τ→∞). That is deliberately the right setting for the encoder
comparison: gating meta-updates would confound the encoder variable with communication
scheduling. If the paper must *demonstrate* the threshold mechanism, run it as a separate
communication-overhead experiment with `num_agents > 1` driving
`update_performance()`/`collect_updates()`.

## 3. Dropout regime mismatch in the outer MAML loss (fixed)

RC#7 disabled dropout for the inner loop, but `inner_loop_update` unconditionally restored
`train()` mode on exit — so `meta_update` computed the outer test-set loss with dropout
ACTIVE, while the `old_log_prob` in the PPO ratio had been recorded in eval mode at
collection time. Measured: the same batch evaluated twice under the outer loss gave
0.338 vs 0.276 — pure dropout noise inside the importance ratio, and inconsistent
second-order gradients through `create_graph=True`.

**Fix:** `meta_update` holds the policy in `eval()` for the entire MAML computation
(inner adaptation + outer loss) and restores `train()` afterwards; `inner_loop_update`
now restores the *caller's* mode instead of forcing `train()`. Every log-prob the PPO
ratio compares — collection, inner steps, outer loss — is now computed under the identical
dropout-free regime. Exploration comes from Categorical sampling, as designed. Verified:
outer loss is bit-deterministic on a fixed batch; value network still receives gradients
(norm 0.82); 1+1 resume still reproduces 2 continuous iterations bit-identically.

Note: this changes RNG consumption relative to pre-fix code, so pre-fix checkpoints
resumed under post-fix code will not reproduce a pre-fix continuous run. All production
training starts from scratch after the 2026-07-10 overhaul anyway.

## 4. Stale 6-wide feature fallbacks vs `TASK_FEATURE_DIM = 9` (fixed)

Three fallback paths still produced 6-wide feature rows against encoders built for 9,
crashing with an opaque `mat1 and mat2 shapes cannot be multiplied (1x6 and 9x16)`:
`get_task_feature_matrix()` with no task set, the independent-task branch, and
`_extract_task_features`'s state-slice fallback in `tampo.py`. The DAG pipeline never hit
them (train() always calls `set_task` first), but `task_type: 'independent'` crashed
immediately.

**Fix:** all three now emit `task_feature_dim`-wide rows (independent task marks itself
`is_current=1.0`). Verified: independent-task env constructs, resets, and a GCN policy
selects an action on it.

## Fairness audit for the encoder comparison (measured, no changes made)

- **Parameter counts** (hidden_dim=128): GCN 1.13M, GAT 1.14M, **LSTM 1.72M** — the LSTM
  encoder has ~2.8× the encoder parameters (0.92M vs 0.33M). Decoders are identical
  (0.72M). GCN/GAT widths are faithful to GDRL's Feature.py (16-dim intermediate). If
  anything this biases *against* the GNNs; capacity-matching would mean deviating further
  from the cited reference and must be reported as such if done.
- **LSTM sequence order:** node-id order violated topology in 0/20 sampled graphs
  (sizes 10–50), so the BiLSTM receives a valid topological sort, as its paper specifies.
  No handicap.
- **Receptive field:** sampled DAGs are shallow (max depth 2 at 20 nodes, 5 at 50), so the
  2-hop bidirectional GNN receptive field covers most of every graph, and the comm-cost
  physics depends only on direct parents (1 hop). The GNN inductive bias is well matched
  to the task; the architecture does not structurally prevent GCN/GAT from using their
  graph awareness.
- All encoders share: identical decoder, identical node features (which already include
  in/out-degree, depth, comm_load, parent placements), identical seeding/task streams,
  identical hyperparameters.

## Verification summary

Encoders: shapes, per-node distinctness (spread 0.94–1.36), padding masking, eval-mode
determinism, edge-direction sensitivity — all three PASS. Environment: pre-reset safety,
topo-order validity, reward bounds, real queue waits, Pareto tension (cloud fastest /
edge cheapest) — PASS. Training: 2 meta-iterations per encoder, finite decreasing losses,
grad norms 6.7–14.1, action entropy 0.85–0.92 — PASS. Persistence: save/load exact,
incompatible-checkpoint guard raises, 1+1 == 2 bit-identical — PASS. Evaluation:
deterministic repeats, action traces/entropy metrics, both wrappers — PASS.
