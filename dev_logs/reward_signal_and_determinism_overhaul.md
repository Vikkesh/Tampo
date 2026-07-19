# Dev Log: Reward-Signal Correctness & Training Determinism Overhaul

**Date:** 2026-07-10
**Scope:** Root-cause and fix for benchmark results diverging by orders of magnitude between two
Colab sessions started from scratch on identical code and config.
**Files:** `env/base_offloading_env.py`, `algorithms/rl/tampo.py`, `configs/default_config.yaml`,
`utils/seeding.py` (new), `main.py`, `benchmark.py`, `Colab_Test_Run.ipynb`

---

## 0. The Symptom

Two from-scratch Colab runs, no checkpoints, same branch, same config:

| run | algorithm | avg_makespan | avg_energy |
|---|---|---|---|
| A | TAMPO_GCN | 1.265 | 424.69 |
| A | TAMPO_GAT | 0.092 | 9.35 |
| A | TAMPO_LSTM | 2.604 | 6863.72 |
| B | TAMPO_GCN | 0.176 | 5.59 |
| B | TAMPO_GAT | 0.178 | 0.99 |
| B | TAMPO_LSTM | 0.177 | 0.59 |

A four-orders-of-magnitude spread in `avg_energy` for the same algorithm.

Note also that run A's episode counts (52/51/53) and run B's (52/52/54) are not multiples of 3,
while `CommonEvaluator` runs exactly `len(dags) x 3` episodes. Both runs predate commit `9f9d94a`
and were still applying per-algorithm outlier filtering, so the three rows in each table were not
even scored on the same episodes. Neither table is a valid comparison.

---

## 1. Root Cause: `kappa` Put the Energy Metric on a Cliff

`local_energy = kappa * cycles * local_freq^2`.

Real DAG nodes take `cycles` from the `.gv` `expect_size` attribute (`utils/dag_parser.py:64`),
whose median across the dataset is **2.83e7** — not the ~1e9 in `task_cycles_range`, which only
applies to synthetically generated independent tasks.

With `kappa: 1.0e-23` (introduced as RC#15 in `convergence_fixes_overhaul.md`):

| action | energy per median node |
|---|---|
| local | **283.45 J** |
| cloud | 0.043 J |
| edge | 0.018 J |

`total_energy` therefore did not measure energy. It counted how many nodes the policy sent to
local, times ~283. One local pick buried an entire episode of offloaded nodes. Run A's LSTM at
6863 J is a policy sending most nodes local; run B's at 0.59 J is a policy sending essentially
none. Two policies differing slightly in behaviour reported metrics differing by 10^4.

RC#15's justification was arithmetically wrong: it claimed `kappa=1e-28` gave local energy of
`1e-10 J`. With `cycles=1e9` it gives `0.1 J`; with the real `cycles=2.8e7` it gives `0.0028 J`.
The parameter was moved five orders of magnitude out of range to fix a problem that did not exist.

A second consequence: `e_imp = (local_E - action_E) / local_E` evaluated to **0.99994** for cloud
and **0.99998** for edge. The energy objective was saturated and carried no information about
*which* server to choose.

**Fix:** `kappa: 1.0e-23` -> `1.0e-27`.

| kappa | local energy | e_imp (cloud) | e_imp (edge) |
|---|---|---|---|
| 1e-23 (old) | 283.4 J | 0.9998 | 0.9999 |
| 1e-26 | 0.283 J | 0.847 | 0.938 |
| **1e-27 (new)** | **0.0283 J** | **-0.53** | **+0.38** |
| 1e-28 | 0.0028 J | -14.3 | -5.2 |

At `1e-27`, local energy sits between edge and cloud. Ordering becomes: **edge cheapest, local
middling, cloud priciest** on energy, while delay runs the other way (cloud fastest, edge middle,
local slowest). That is a genuine Pareto tension — the precondition for the hypervolume metric and
the Pareto front plot to mean anything. `1e-27` is also squarely inside the physically defensible
CMOS range.

---

## 2. Root Cause: The Penalties Never Reached the Learner

`env.step()` returned a reward containing the congestion and communication penalties.
`_collect_task_experiences` stored it as `exp['reward']`. **Nothing ever read it again.**

Both `_compute_loss` and `_compute_loss_with_params` trained exclusively on `mo_return`, which was
rebuilt from `raw_delay` (pure `comp_delay`) and `raw_energy`. No congestion term. No comm term.

The agent was never taught to avoid queueing — while makespan, the reported metric, is almost
entirely queue-driven. It optimised one objective and was scored on another.

**Fix:** `_calculate_reward()` now returns `(reward, r_delay, r_energy)`. The two per-objective
components — each clipped to `[-1, 1]`, with `r_delay` carrying both penalties — are surfaced in
`step()`'s `info` dict. `_collect_task_experiences` discounts *those* backward into `mo_return`,
replacing the local re-derivation from raw physics. The `raw_delay` / `raw_energy` / `node_cycles`
temporary fields are gone.

---

## 3. Root Cause: Both Penalties Were Themselves Broken

Fixing (2) alone would have accomplished nothing, because neither penalty worked.

### 3a. The congestion penalty measured nothing

```python
wait_time = self.server_available[action] - self.node_finish_times[node_id] + delay
```

`_execute_offloading` runs *before* `_calculate_reward` and sets both
`server_available[action]` and `node_finish_times[node_id]` to `finish_time`. So this reduced to
`finish_time - finish_time + delay`, i.e. **`wait_time == delay`, always**. The congestion penalty
was `min(comp_delay / 5.0, 1.0)` ~ 0.002, regardless of server load.

**Fix:** `_execute_offloading` now captures the real wait
(`max(0, server_available[idx] - data_ready)`) into `self._last_queue_wait` *before* advancing the
timeline, and `_calculate_reward` reads it.

### 3b. The communication penalty was saturated

```python
comm_penalty += comm_data / self.bandwidth_up          # 20e6
```

but `_execute_offloading` transmits at the Shannon datarate,
`bandwidth_up * log2(1 + SNR)` ~ **8.44e8** — 42x larger. Dividing raw bytes by the raw bandwidth
gave `2.83e7 / 2.0e7 = 1.4`, clipped to `1.0`. The term pinned at exactly 1.0 whenever *any*
parent lived on another server. Binary, no gradient.

**Fix:** use the transfer times actually incurred (already computed at the Shannon rate) via
`self._last_comm_time`.

### 3c. Both now use a smooth relative overhead

Normalising by `local_delay` with a hard `min(x/scale, 1.0)` was still wrong: these are
`daggen --ccr 0.5` graphs, so a *single* cross-server parent transfer (0.0336 s) already exceeds
`local_delay` (0.0283 s) and pins the clip.

Both penalties now use

```
penalty = cost / (cost + local_delay)        # [0, 1), smooth, never pins
```

a dimensionless relative overhead that keeps a gradient at any congestion or chattiness level, and
needs no retuning when task sizes or clock speeds change.

Measured on a 20-node DAG, round-robin placement: `comm_penalty` mean 0.380, std 0.425, fraction
pinned at 1.0 = **0.00** (was 1.00).

---

## 4. Root Cause: Training Was Never Seeded

No `torch.manual_seed`, `np.random.seed`, or `random.seed` anywhere in `tampo.py`, `main.py`,
`benchmark.py`, or `Colab_Test_Run.ipynb`. Evaluation *was* seeded
(`common_evaluator.py:137`) and is reproducible given a fixed checkpoint. Training was not.

**Fix:** new `utils/seeding.py` exposing `set_seed(seed, deterministic_torch=False)` and a `SEEDS`
tuple for multi-seed reporting. Called from `main.py`, `benchmark.py` (with a `--seed` override),
and both notebook training cells, which additionally re-seed before each encoder so the three-way
comparison isolates the encoder rather than the RNG draw order.

Verified: three iterations x three encoders, same seed reproduces the loss trajectory bit-exactly;
a different seed diverges.

> A fixed seed makes a single run *repeatable*. It does not make it *representative*. A single-seed
> GCN-vs-GAT-vs-LSTM table cannot distinguish a real architectural difference from luck. Report
> mean +/- std over `utils.seeding.SEEDS`.

### 4.1 Update (2026-07-17): `SEEDS` corrected from 5 to 8, and multi-seed tooling added

`SEEDS = (0,1,2,3,4)` was a bad default, and `docs/RUNNING_THE_EXPERIMENT.md` compounded it by
calling `k = 5` "the practical minimum" for a Wilcoxon signed-rank test. That is arithmetically
impossible advice.

A two-sided Wilcoxon on `n` paired samples enumerates `2^n` sign assignments, so the smallest
p-value it can *ever* return is `2 / 2^n`. Measured with scipy against perfectly separated data
(one arm beating the other on every seed by a factor of 10):

| n | 3 | 4 | **5** | **6** | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| best attainable two-sided p | 0.250 | 0.125 | **0.0625** | **0.031** | 0.016 | 0.0078 | 0.002 |

At `k = 5` the test cannot reach `p < 0.05` under any outcome. Five full training runs would
have been spent on a test that was incapable of passing, and the resulting `p = 0.0625` would
most likely have been misread as "no difference" when it actually means "no verdict possible".

**Fix:**
- `utils/seeding.py`: `SEEDS = (0,...,7)`, with the derivation in a comment so the number is not
  silently "rounded back down" later. `k >= 6` to claim significance, `k = 8` for margin.
- `utils/aggregate_seeds.py` (new): aggregates `results/seed_*/run_*/benchmark_results.csv` into
  `mean +/- std` per `(encoder, metric)` and runs paired Wilcoxon per encoder pair. When `n < 6`
  it prints the attainable floor instead of a bare "NOT significant". Stdlib `csv` +
  `statistics`; scipy only for the test, and it degrades to the descriptive table without it.
- `README.md`: its multi-seed snippet looped `benchmark.py --seed $s` over one checkpoint dir —
  that only re-seeds *evaluation* and measures nothing about training variance. Multi-seed
  requires **retraining** per seed into a per-seed checkpoint dir; corrected.
- `docs/RUNNING_THE_EXPERIMENT.md`: Case 3 rewritten with the seed-count table, a compute-budget
  formula anchored to the user's real measured T4 rate, per-seed directory layout, and the
  `DRIVE_ROOT`/`SEED` coupling that voids the study if changed independently.

### 4.2 Update (2026-07-17): golden dataset in a fresh session

`data/test_dags.json` is untracked, so a re-cloned Colab VM has no test set — but the notebook
said "Run **once only**. Never re-run", which read as "do not regenerate" and left no way to get
the file back. Two further problems: the cell generated `--num_dags 20` while the docs claimed
500 (a 20-DAG test set is far too small to publish from), and nothing verified the set was stable
across sessions.

Confirmed by reading the full path that generation is deterministic: `DAGParser.load_dataset`
sorts `os.listdir` and slices at a fixed `offset=20`; `parse_gv_file` is a pure parse of the
committed `.gv` files; `grep` finds no RNG anywhere in `utils/dag_parser.py`. Same commit + same
args ⇒ byte-identical output. (Could not execute it locally — `networkx` is absent and the user's
standing instruction is to install nothing — so this rests on reading, and the notebook now
prints an MD5 so the invariant is checked at runtime rather than asserted here.)

**Fix:** the Section 2 cell now generates 500 DAGs, prints an MD5 + size histogram, asserts the
count, and is documented as "run in every new session". The freeze applies to the *arguments*,
not the file. A Drive copy/restore alternative is documented for anyone who prefers it.

---

## 5. Root Cause: The MAML Inner Loop Was Off-Policy and Uncorrected

`inner_loop_update` takes `inner_steps: 5` gradient steps, resampling minibatches from a single
batch of experiences collected under the meta-policy, using a vanilla policy-gradient loss
`-(log_probs * advantages).mean()`. After the first step the policy has moved; the data is
off-policy and nothing corrected for it.

**Fix:** `_select_action(..., return_log_prob=True)` records the behaviour-policy log-prob at
collection time. Both loss functions now use a PPO clipped surrogate
(`ppo_clip_eps: 0.2`), falling back to vanilla PG when `old_log_prob` is absent so older
checkpoints and buffers keep loading. `value_loss_coef` and `entropy_coef` were hardcoded at
`0.5` / `0.01`; they are now config keys with those defaults.

---

## 6. Incidental Bug: `main.py` Ignored the Config File

`setup_environment()` read `config.get('system')`, `config.get('computing')`, etc. off the
top-level dict, but those sections live under `environment:`. Every lookup returned `{}`, so the
env silently fell back to its hardcoded defaults and ignored `default_config.yaml` entirely —
including `kappa`. `reward` was dropped the same way, leaving the penalty weights at defaults.
`main()` also indexed `config['system']['max_steps']`, which raises `KeyError`.

Fixed both. (`benchmark.py` and the notebook already read `full_config['environment']` correctly,
which is why training worked at all.)

---

## 7. Verification

Fixed-policy rollouts on a 20-node DAG, `preference = [0.5, 0.5]`:

| policy | makespan | energy | r_delay | r_energy | queue_wait | comm |
|---|---|---|---|---|---|---|
| all-local | 0.4696 | 0.4696 | -0.273 | +0.000 | 0.1030 | 0.0000 |
| all-cloud | 0.0470 | 0.9924 | +0.777 | -1.000 | 0.0103 | 0.0000 |
| all-edge0 | 0.0939 | 0.3904 | +0.630 | +0.169 | 0.0206 | 0.0000 |
| spread-edge | 0.1070 | 0.6183 | +0.614 | -0.264 | 0.0060 | 0.0360 |
| round-robin | 2.1663 | 0.8018 | +0.500 | -0.420 | 0.0063 | 0.4572 |

- Cloud is fastest and priciest; edge is cheapest and slower. Real Pareto tension.
- `queue_wait` is nonzero and load-dependent (0.1030 s all-local vs 0.0060 s spread-edge).
- All-local energy is 0.47 J, not 6020 J. The cliff is gone.

Learner checks (`_collect_task_experiences` after `set_task`, 10-node DAG, 3 episodes):

- 30 experiences for 3 x 10 nodes — episode boundaries split correctly.
- `mo_return` within the theoretical `sum(gamma^k) = 9.56` bound, confirming the per-episode reset
  holds and step rewards stay in `[-1, 1]`.
- `_ppo_policy_loss` at `ratio = e^2` matches the hand-computed clipped value exactly.
- No `raw_delay` / `raw_energy` / `node_cycles` keys left on experience dicts.
- All three encoders (gcn, gat, lstm) meta-train without error.

Notebook cell 2.5 was rewritten to assert all of the above (kappa calibration, component clipping,
congestion responds to load, comm penalty unsaturated, cloud/edge Pareto tension) rather than the
old assertions, which checked for `kappa ~1e-23` and would now fail.

---

## 8. Config Changes

```yaml
environment:
  energy:
    kappa: 1.0e-27          # was 1.0e-23

experiment:                  # new section
  seed: 42
  deterministic_torch: false

training:
  ppo_clip_eps: 0.2         # new
  value_loss_coef: 0.5      # new (was hardcoded)
  entropy_coef: 0.01        # new (was hardcoded)
```

---

## 9. Deliberately Not Done

**The task's own upload time is still not charged to the timeline.** In `_execute_offloading`,
`finish_time = start_time + comp_delay`, where `comp_delay = cycles / freq`. The `trans_time`
computed a few lines later feeds only the energy formula. For the median node, cloud upload takes
**0.083 s** while running the whole node locally takes **0.028 s** — offloading should often be
*slower*, and that is the trade-off `--ccr 0.5` was generated to create.

Fixing this would make the delay objective non-degenerate on its own, but it changes the physics
engine, invalidates every previously reported makespan/energy number, and conflicts with
`CLAUDE.md` §2. It also exposes a related inconsistency: the energy formula charges `data_size`
upload for *every* offloaded node *and* separately charges parent->child transfers via
`cross_server_energy`, double-counting for non-source nodes. Delay and energy should be made
consistent about what data actually moves, in one deliberate change, with its own dev log.

Also unaddressed: `num_meta_iterations: 100` with `meta_learning_rate: 5.0e-5` is likely too few
steps to converge even now that the signal is correct.
