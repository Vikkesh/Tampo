# How to Read the Results

This guide explains every number this project prints, at two levels.

- **Part 1 — Plain English.** No maths. Read this if you want to know whether a run went
  well and what the results mean.
- **Part 2 — Technical.** Read this if you are tuning the algorithm, debugging a run, or
  writing up the results.

You can stop after Part 1 and still understand the outcome of an experiment.

---
---

# PART 1 — PLAIN ENGLISH

## What this project is actually doing

Imagine a phone that has a job to do — say, editing a video. The job is broken into many
small steps, and some steps must finish before others can start (you cannot add subtitles
before the video is decoded). For each step, the phone must decide **where to run it**:

| Choice | What it means | Trade-off |
|---|---|---|
| **local** | Run it on the phone itself | No network needed, but the phone is slow |
| **cloud** | Send it to a big distant data centre | Very fast, but costly to send data there |
| **edge0/1/2** | Send it to a nearby small server | Middle ground: fairly fast, cheap to reach |

Every choice costs **time** and **battery**. Sending data to the cloud drains battery.
Running everything on the phone is slow. And if you send everything to the same server it
forms a queue and everything waits.

The AI ("the agent") learns to make these choices. We are testing **three different AI
designs** to see which makes the best choices:

- **TAMPO-LSTM** — reads the job's steps one after another, like reading a sentence.
- **TAMPO-GCN** — looks at the whole job as a network of connected steps, all at once.
- **TAMPO-GAT** — same as GCN, but it also learns *which* connections matter most.

The whole point of GCN and GAT is that they should "understand the shape" of the job
better than reading it as a flat sequence. Whether they actually do is what the benchmark
is meant to answer.

## The two numbers that matter

**Makespan** — total time from starting the first step to finishing the last. Measured in
seconds. **Lower is better.**

**Energy** — total battery used. Measured in joules. **Lower is better.**

These two fight each other. Getting things done faster usually burns more battery. So
there is no single "best" answer — there is a trade-off curve. That is why we test the
agent under three different *priorities*:

| Priority | Written as | Means |
|---|---|---|
| Performance mode | `0.8 / 0.2` | "I care mostly about speed" |
| Balanced | `0.5 / 0.5` | "Both matter equally" |
| Battery saver | `0.2 / 0.8` | "I care mostly about battery" |

**A good agent behaves differently in each mode.** In performance mode it should lean on
the cloud. In battery-saver mode it should lean on the nearby edge servers or stay local.
If it does the exact same thing in all three modes, it has not learned the trade-off — it
has just memorised one habit.

## Reading the training output

While the model is learning, you will see lines like this:

```
  Iter   5/75 | Loss: 2.6561 | Avg(10): 3.6410 | Best: 2.6561
  [actions] local=18.7% cloud=36.7% edge0=16.7% edge1=22.7% edge2= 5.3% | entropy=0.91 (0=collapsed, 1=uniform) | n=150
```

You only need to watch **two things**.

**1. Is `Loss` going down over time?** Loss is "how wrong the model currently is". It
bounces around a lot — that is normal. What matters is that `Avg(10)` (the average of the
last 10 rounds) trends downward. If it climbs steadily, or turns into `nan`, training has
broken.

**2. Is `entropy` staying healthy?** This is the single most useful number in the whole
output.

> `entropy` measures **how varied the agent's choices are**.
> - `1.00` = it spreads work evenly across all five options.
> - `0.00` = it sends *everything* to one place and never varies.

An entropy of `0.00` means the agent has stopped thinking. It found one option, latched
onto it, and now applies it to every single step of every job. This is the most common
failure in this project, and it is **invisible** if you only look at makespan and energy.

You want entropy somewhere in the middle — roughly **0.4 to 0.9**. Early in training it
will be high (the agent is guessing randomly). It should come down as the agent gets
opinionated, but it should not hit zero.

**Rule of thumb:** if entropy drops below `0.1` and stays there, stop the run. Something
is wrong.

## Reading the benchmark output

After training, the benchmark tests the agent on jobs it has never seen:

```
📊 Evaluating TAMPO...
  ✓ Episodes      : 9 (3 DAGs × 3 preferences)
  ✓ Avg Makespan  : 0.0706s
  ✓ Avg Energy    : 0.223824J
  ✓ Actions       : local= 0.0% cloud= 0.0% edge0=100.0% edge1= 0.0% edge2= 0.0%
  ✓ Within-episode entropy: 0.000 (0 = one server for the whole DAG, 1 = uniform)
  ⚠ 9/9 episodes placed EVERY node on a single server — policy is degenerate.
    Action mix per preference (delay/energy):
       0.8/0.2 → local= 0.0% cloud= 0.0% edge0=100.0% edge1= 0.0% edge2= 0.0%
       0.5/0.5 → local= 0.0% cloud= 0.0% edge0=100.0% edge1= 0.0% edge2= 0.0%
       0.2/0.8 → local= 0.0% cloud= 0.0% edge0=100.0% edge1= 0.0% edge2= 0.0%
```

Read it in this order:

**First, look for the ⚠ warning.** If it appears, **the makespan and energy numbers above
it are meaningless as a measure of intelligence.** The agent put every step on one server.
It did not schedule anything. It might still post a decent makespan — a fast server is a
fast server — but it did not learn.

**Second, look at "Action mix per preference".** The three rows should look *different*
from each other. If speed-mode and battery-mode produce identical rows, the agent is
ignoring the user's priority. That defeats the entire purpose of the framework.

**Third, and only then, read the makespan and energy.**

## Healthy vs unhealthy — at a glance

| Signal | Healthy | Unhealthy |
|---|---|---|
| Training `Avg(10)` | drifts down | climbs, or becomes `nan` |
| Training `entropy` | 0.4 – 0.9 | below 0.1 |
| ⚠ degenerate warning | absent | present |
| Action mix across the 3 priorities | rows differ | rows identical |
| `avg_energy` | roughly 0.2 – 1.5 J | in the hundreds or thousands |
| `num_episodes` | exactly `DAGs × 3` | anything else |

That `avg_energy` range is worth remembering. Running a step on the phone costs roughly
**2,000× more energy** than sending it to a nearby server. So if the agent sends even a
handful of steps to the phone, energy jumps enormously. An `avg_energy` in the thousands
almost always means "the agent decided to do everything locally," not "there is an energy
bug."

## Why you must never trust a single run

Training an AI involves a lot of randomness: the starting weights, the order it sees jobs
in, the choices it gambles on early. Run the same code twice and you get different
results.

We now fix this randomness with a **seed** — a number that makes the run repeatable. Same
seed, same result, every time.

But here is the trap. A seed makes a run **repeatable**; it does not make it
**representative**. Suppose GAT scores 0.171 and GCN scores 0.176 with seed 42. Is GAT
better? You cannot say. Seed 42 might simply have handed GAT a lucky start.

So run each design with several different seeds and report the average *and* the spread:

```
TAMPO_GCN   makespan 0.176 ± 0.021
TAMPO_GAT   makespan 0.171 ± 0.019
```

Here GAT's 0.005 lead is much smaller than the ±0.02 wobble. **The two are
indistinguishable.** You cannot claim GAT wins. If GAT had scored 0.120 ± 0.015, the gap
would be far bigger than the noise, and that would be a real, defensible finding.

> **The rule:** if the difference *between* two designs is smaller than the variation
> *within* one design across seeds, there is no difference you can report.

## The files a run produces

| File | What it holds |
|---|---|
| `benchmark_results.csv` | One row per design: average makespan, energy, and the degeneracy check |
| `action_traces.csv` | Every decision the agent made, job by job — the ground truth |
| `action_distribution.csv` | How often each server was chosen, overall and per priority |
| `comparison_bar.png` | Bar charts of average time and energy |
| `pareto_front.png` | The speed-vs-battery trade-off curve |

If a result looks surprising, open `action_traces.csv`. It contains the actual sequence of
choices, like `cloud cloud cloud cloud ...`, and it will usually explain the surprise
immediately.

---
---

# PART 2 — TECHNICAL

## Experimental setup

- **Environment:** `TaskOffloadingEnv`, the single source of truth for all physics. Every
  algorithm reaches it through `env.step(action)`; no algorithm computes its own latency
  or energy.
- **Action space:** `Discrete(5)` — `0=local`, `1=cloud`, `2..4=edge0..2`.
- **Episode:** one complete DAG, decided node-by-node in topological order (Kahn). Episode
  length equals the node count.
- **Evaluation:** `len(test_dags) × 3` episodes — every DAG under each of
  `[0.8,0.2]`, `[0.5,0.5]`, `[0.2,0.8]`. Actions are greedy (`deterministic=True`) and the
  policy runs in `eval()` mode, so dropout is off.
- **Metrics:** `makespan = max(node_finish_times)` (true critical path), and
  `total_energy = Σ per-node energy`. No outlier filtering — every episode counts.

## Training output, line by line

```
    [diag] returns |G_d|=1.631  |G_e|=0.817  policy=-0.0269  value=2.6680  entropy=-1.4373
  [meta] grad_norm_total=19.8811  tasks=2  inner_steps=5
  Iter   1/75 | Loss: 1.8625 | Avg(10): 1.8625 | Best: 1.8625
  [actions] local=18.7% cloud=36.7% edge0=16.7% edge1=22.7% edge2= 5.3% | entropy=0.91 | n=150
```

### `[diag]` — loss decomposition, printed every 50 loss evaluations

**`|G_d|`, `|G_e|`** — mean absolute discounted return per objective. Step rewards are
clipped to `[-1, 1]` and `gamma = 0.99`, so the hard bound is
`Σ_{k<N} gamma^k` ≈ **9.56** for a 10-node DAG, **25.8** for 30 nodes, **39.5** for 50.

- Above that bound → the per-episode return reset in `_collect_task_experiences` has
  broken and returns are bleeding across episode boundaries.
- `|G_e|` pinned near the bound → the energy objective has saturated. This is what
  `kappa = 1e-23` used to cause: `e_imp` was `0.9998` for cloud and `0.9999` for edge, so
  the objective carried no information about which server to pick.

**`policy`** — the PPO clipped surrogate. Near `0` at the first inner step is *correct*:
the importance ratio is 1 and advantages are normalised to zero mean, so
`-min(1·A, clip(1)·A).mean() = -A.mean() ≈ 0`. It becomes non-zero once the adapted
parameters diverge from the behaviour policy.

**`value`** — MSE of the two-headed critic against `mo_return`. Should decline and stay
within about an order of magnitude of `policy`. A `value` term that runs away while
`policy` stays flat means `0.5 * value_loss` is dominating the gradient.

**`entropy`** — printed as its *negative* (it enters the loss as a bonus). `-1.437` means
actual entropy `1.437`. Maximum for 5 actions is `ln(5) = 1.609`. Drift toward `0` is
policy collapse.

### `[meta]` — the outer loop

**`grad_norm_total`** — summed over `meta_policy` parameters, **after** accumulation and
**before** `clip_grad_norm_(max_norm=1.0)`. A value far above 1.0 just means clipping is
active. `WARNING: No gradients detected` means the inner loop produced nothing — usually
an empty `train_experiences`.

**`tasks`** — how many of the sampled task DAGs contributed a valid loss.
**`inner_steps`** — MAML inner-loop gradient steps (config `inner_steps`, default 5).

### `Iter` — progress

`Loss` is the current meta-loss, already divided by the number of valid tasks. `Best`
tracks the minimum and is what `_best.pth` stores.

> ⚠ **`_best.pth` is the lowest-meta-loss checkpoint, not the best-scheduling
> checkpoint.** Meta-loss is a weak proxy for benchmark performance. Compare `--use_best`
> against the plain checkpoint before trusting it.

### `[actions]` — the collapse detector

Aggregated over every action sampled in the iteration's rollouts (`n` = total decisions).
Entropy is normalised: `H(p) / ln(num_actions)`, so `1.0` is uniform and `0.0` is a point
mass.

Because collection uses `Categorical(probs).sample()`, this distribution reflects the
policy's *stochastic* behaviour. It will look healthier than the greedy behaviour the
benchmark measures. **A low training entropy is damning; a high one is not proof of
health.** Confirm with the benchmark's within-episode entropy.

## Benchmark output

### `Within-episode entropy` — the number that matters

Mean over episodes of the normalised entropy of the action distribution *within a single
DAG*.

This is deliberately different from the overall action mix. An agent could send every node
of DAG A to cloud and every node of DAG B to edge0; its *overall* mix would look
beautifully balanced while its *within-episode* entropy is `0.0`. Such an agent is picking
one server per graph, not scheduling.

`degenerate_episodes` counts episodes with exactly zero within-episode entropy.

### `Action mix per preference`

The direct test of preference conditioning. TAMPO's central claim is a policy conditioned
on `[w_delay, w_energy]`. If the three rows are identical, the preference vector is not
reaching the policy, or the agent has learned to ignore it. Either way, the Pareto front
plot is meaningless — the three points will coincide.

Expected direction, given the calibrated physics (`kappa = 1e-27`):

| per median node | delay | energy |
|---|---|---|
| local | 0.0283 s | 0.0283 J |
| cloud | 0.0028 s | 0.0435 J |
| edge | 0.0057 s | 0.0177 J |

So `[0.8, 0.2]` should skew toward **cloud**, and `[0.2, 0.8]` toward **edge**. Local is
dominated on both axes by edge and is only rational under heavy edge congestion.

## Output files

### `benchmark_results.csv`

| Column | Meaning |
|---|---|
| `avg_makespan`, `std_makespan`, `min/max_makespan` | Critical-path time, seconds |
| `avg_energy`, `std_energy`, `min/max_energy` | Total energy, joules |
| `num_episodes` | **Must equal `len(test_dags) × 3`.** Anything else means stale code with outlier filtering still enabled |
| `within_episode_entropy` | Mean per-episode action entropy; `0.0` = degenerate |
| `degenerate_episodes` | Count of single-server episodes |

### `action_traces.csv`

One row per `(dag, preference)` episode, with the full placement sequence as
`cloud cloud edge0 ...`. This is the ground truth. Every claim about agent behaviour
should be checkable here.

### `action_distribution.csv`

Long-format: `algorithm, preference, action, fraction`. `preference = "all"` is the
aggregate; the other rows are per preference vector. Suitable for direct plotting.

## Sanity checks worth internalising

**Energy scale.** With `kappa = 1e-27` and `local_freq = 1e9`, local energy per node is
`kappa · cycles · f²  = 1e-9 · cycles`, and local delay is `cycles / f = 1e-9 · cycles`.
They coincide numerically. So an **all-local** episode reports `makespan ≈ energy`. That
is a coincidence of the calibration, not a bug — but it is a handy fingerprint: if a row
shows makespan and energy nearly equal, the agent went all-local.

**Cycles come from the graph file.** Real DAG nodes take `cycles` from the `.gv`
`expect_size` attribute (median ≈ `2.8e7`), *not* from `task_cycles_range` (≈ `1e9`, which
only applies to synthetic independent tasks). Every energy calculation depends on this.
Getting it wrong is exactly how `kappa` ended up five orders of magnitude too large.

**Episode count parity.** `52`, `51`, `53` are not multiples of 3. If you see counts like
that, the run predates the removal of per-algorithm outlier filtering, and the algorithms
were not scored on the same episodes. The comparison is invalid.

## Training budget and session planning

Measured per-iteration wall clock (CPU, `meta_batch=4`, 10/20/30-node graphs):

| encoder | s/iteration | why |
|---|---|---|
| GCN | 3.28 | batched graph convolution |
| GAT | 4.12 | same skeleton, attention-weighted aggregation |
| **LSTM** | **14.11** | CuDNN disabled — MAML needs double-backward through the RNN |

LSTM is ~4× the cost of GCN and dominates any loop that trains all three encoders together.
A Colab free-tier T4 session is killed at ~4 h, which fits roughly 75 iterations of all
three at `meta_batch_size=15`. That is nowhere near converged: at
`meta_learning_rate = 5.0e-5`, 75 iterations move the average weight only 2.4% of its
initial magnitude.

**Train one encoder per session.** `train(time_budget_s=...)` stops at an iteration boundary,
saves a checkpoint, and prints resume instructions rather than being killed mid-iteration.
The progress line reports `s/it` so you can plan.

Levers on per-iteration cost, in order of usefulness:

| knob | effect on cost | effect on learning |
|---|---|---|
| `META_BATCH_SIZE` (notebook) | ~linear | gradient quality only — **Adam's step is ≈ `lr` regardless of batch size** |
| `EPISODES_PER_TASK` (config) | ~linear on rollouts | per-task adaptation quality |
| `inner_steps` (config) | ~linear on second-order grads | inner-loop adaptation depth |
| `meta_learning_rate` | none | **do not touch** — collapses the policy above `1e-4` |

Because a smaller meta-batch does not shrink Adam's step, halving it roughly doubles the
number of optimiser steps you can afford per session. That is the trade to make.

**Fairness:** before comparing encoders, every one must have the same total iterations,
`META_BATCH_SIZE`, `EPISODES_PER_TASK`, `inner_steps` and seed.

## Multi-seed protocol

```bash
for s in 0 1 2 3 4; do
    python benchmark.py --seed $s --output_dir results/seed_$s
done
```

Then aggregate: report `mean ± std` across seeds for each `(algorithm, metric)` pair. Do
not report a single-seed table.

For a significance claim on `k = 5` seeds, a paired comparison (same seeds across
algorithms — which the notebook guarantees by re-seeding before each encoder) with a
Wilcoxon signed-rank test is the standard non-parametric choice; `k = 5` is the practical
minimum and `k ≥ 10` is preferable.

## Known limitations to disclose in a write-up

**Upload time is not charged to the timeline.** In `_execute_offloading`,
`finish_time = start_time + comp_delay` where `comp_delay = cycles / freq`. The
`trans_time` computed a few lines later feeds only the energy formula. For the median
node, cloud upload takes **0.083 s** against **0.028 s** to execute the node locally — so
offloading should frequently be *slower*, which is the trade-off `daggen --ccr 0.5`
generates. Cross-server parent→child transfers *are* charged; a node's own input upload is
not. This flatters every offloading decision.

**Energy double-counts transfers for non-source nodes.** The energy formula charges
`data_size` upload for every offloaded node *and* separately charges parent→child
transfers via `cross_server_energy`.

**Meta-loss is used for checkpoint selection.** `_best.pth` minimises meta-loss, not
makespan or hypervolume.

**Encoder deviations from the cited papers** are catalogued in `Papers referred.md` and
`dev_logs/graph_encoder_and_observability_overhaul.md`. Summary: the bidirectional stream
is ours, not GDRL's or GAPO's; the mean readout replaces GDRL's fixed-size concatenation;
the sequential pointer decoder replaces GDRL's single-shot actor head.
