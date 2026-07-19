# Running the Experiment — Full Walkthrough

How to train and benchmark TAMPO-GCN, TAMPO-GAT and TAMPO-LSTM for a publication-grade
comparison, in three deployment scenarios. For interpreting the numbers afterwards, see
`docs/READING_RESULTS.md`.

---

## The one thing that makes this rigorous: exact resume

Free Colab kills a session at ~4 hours, and the three encoders together need far more than
that. So training is **split across sessions** — and the split must not change the result.

It doesn't. Each checkpoint stores the model weights, the Adam optimizer state, the
iteration counter, **and the full RNG stream position** (python + numpy + torch). On resume,
`train()` restores that stream instead of re-seeding, so iterations 251–500 of a run resumed
in a new session draw the identical graphs, preferences and channel conditions they would
have drawn had the session never stopped.

Verified: a run split `3+3` and a run split `2+2+2` each reproduce a single continuous
6-iteration run **bit-for-bit** (identical loss sequence). Splitting an encoder across any
number of sessions is therefore invisible to the result.

And because every encoder is seeded identically at construction (before weight init), all
three face the **identical task sequence** — same graphs, same preferences, same channel
noise. Only their weights and action choices differ, which is the whole point of comparing
encoders. (Verified: GCN and LSTM draw an identical graph-ID sequence under the same seed.)

---

## The golden dataset in a new session

`data/test_dags.json` is **not tracked in git**, and a new Colab session starts from a fresh
clone. So the file does not exist until you make it — you **must run the Section 2 cell in
every new session**. This does not contradict "keep the test set frozen":

**Regenerating is safe, because generating it is deterministic.** `utils/generate_test_dataset.py`
is a pure parse of the `.gv` files that *are* committed (`data/meta_offloading_n/offload_random*/`):
it sorts filenames, takes a fixed slice at `offset=20`, and draws no random numbers anywhere in
the path. Same commit + same arguments ⇒ **byte-identical** `test_dags.json`. The cell prints
an MD5 so you can confirm it rather than trust it — if that hash ever differs between sessions,
stop, because your test set moved.

What must stay frozen is the **arguments**, not the file: keep `--num_dags 500`. Changing them
changes which graphs are in the set and silently invalidates comparison against results you
already collected.

The `offset=20` is what keeps this a genuine held-out set: training draws the first 20 files
per size folder, the test set starts at file 20, so the two never overlap. Sizes 35–50 are
never trained on at all, making them a true zero-shot test.

**Optional (belt and braces):** if you would rather not rely on regeneration, copy the file to
Drive once and restore it each session instead:

```python
# after generating it the first time
!cp data/test_dags.json /content/drive/MyDrive/TAMPO_checkpoints/test_dags.json

# in later sessions, in place of the Section 2 cell
!cp /content/drive/MyDrive/TAMPO_checkpoints/test_dags.json data/test_dags.json
```

Both routes are valid. Regenerating is simpler and self-verifying; copying is immune to a
future change in the generator or the `.gv` files.

---

## Stopping and saving — the mechanism

`framework.train(num_iterations, meta_batch_size, checkpoint_path, time_budget_s)` can stop
three ways, and always leaves a valid checkpoint:

| Trigger | What happens |
|---|---|
| **Reached `num_iterations`** | Final checkpoint saved, returns normally. |
| **`time_budget_s` exhausted** | At the next iteration boundary: saves, prints "budget reached", returns. This is what prevents Colab from killing the run mid-iteration with unsaved work. Set it below the session limit (e.g. 3.3 h for a 4 h cap). |
| **Hard kill (disconnect)** | The 10-iteration autosave means at most the last <10 iterations are lost. If `CKPT_DIR` is on Drive, that autosave already persisted. |

Every save is **atomic** (temp file + rename, with a direct-write fallback for Drive's FUSE
mount), so a crash mid-write can never leave a corrupt, unloadable checkpoint.

The notebook's training cell wraps this in an **auto-advancing loop**: it pours the session
budget into the encoders in order, one at a time, each up to `TARGET_ITERATIONS`, skipping
any that are already done. So you set it once and just re-run the cell each session.

---

## Choosing when to stop training

Do **not** fix an iteration count and trust it. After each session run the benchmark cell and
read two numbers (both explained in `docs/READING_RESULTS.md`):

1. **`within_episode_entropy`** — while it is `0.000`, the policy places every node of a DAG
   on one server and has not learned to schedule. Keep training.
2. **`avg_makespan`** — once it has plateaued across two consecutive sessions, the encoder
   has converged.

`TARGET_ITERATIONS = 600` in the notebook is a ceiling, not a promise. Raise it if an encoder
is still improving at 600; stop early if it plateaued at 400.

**Fairness rule:** before you compare encoders, all three must reach the **same total
iterations** with the **same** `META_BATCH_SIZE`, `EPISODES_PER_TASK`, `inner_steps` and
`SEED`.

---

## Per-iteration cost (measured, CPU; a T4 is faster but same ratios)

| encoder | s/iteration | note |
|---|---|---|
| GCN | ~3.3 | batched graph convolution |
| GAT | ~4.1 | attention-weighted, same skeleton |
| **LSTM** | ~14 | CuDNN disabled — MAML needs double-backward through the RNN |

LSTM is ~4× the others and dominates the budget. Levers to fit more optimiser steps into a
session, in order of usefulness: shrink `META_BATCH_SIZE` (does **not** reduce learning —
Adam's step size is ≈ `lr` regardless of batch), shrink `EPISODES_PER_TASK`, shrink
`inner_steps`. Do **not** raise `meta_learning_rate` — it collapses the policy (measured).

---
---

# CASE 1 — Multiple free Colab sessions (with Google Drive)

The default. Checkpoints live on Drive so they survive session kills.

### One-time setup
1. Open `Colab_Test_Run.ipynb` in Colab. Runtime → Change runtime type → **T4 GPU**.
2. Run the setup cells top to bottom:
   - **Clone repo** (cell 0a) and **install dependencies** (cell 0b).
   - **CUDA config + imports** (cell 0c).
3. (Optional) Run **Section 2.5** verification and the **Section 1** encoder unit test —
   both should print all `PASS`.

### Every session
4. Run **Section 2 — Generate the Golden Test Dataset**. Yes, *every* session — see
   [The golden dataset in a new session](#the-golden-dataset-in-a-new-session) below.
   Check the printed MD5 matches the previous session's.
5. Run the **"Persistence & experiment configuration"** cell.
   - Keep `USE_DRIVE = True`. It will prompt you to authorise Google Drive (a popup) — do it.
   - Checkpoints now read/write `MyDrive/TAMPO_checkpoints/models/`.
   - The cell prints which checkpoints already exist. First session: `none`. Later
     sessions: you should see `tampo_gcn_checkpoint.pth`, etc. **If a later session prints
     `none`, stop** — Drive did not mount; re-run the cell rather than retraining from zero.
6. Run the **"Full Training Run"** cell. Leave `ENCODERS = ['gcn', 'gat', 'lstm']`,
   `SESSION_BUDGET_H = 3.3`.
   - Session 1 trains GCN until the budget runs out (say, to iteration ~300), then stops
     cleanly. You'll see per-encoder lines like `→ GCN: 300/600 ... re-run to continue.`
   - When the session ends (or you hit the budget), start a **new** Colab session and repeat
     steps 5–6. The loop resumes GCN from 300, finishes it, then moves to GAT, then LSTM.
   - Keep going, session after session, until every encoder prints **`DONE.`**
7. After any session, run the **Benchmark** cell to check progress
   (`within_episode_entropy`, makespan). Stop training an encoder early if it has plateaued.
8. When all three are done and converged, the Benchmark cell writes
   `results/run_*/benchmark_results.csv`, `action_traces.csv`, `action_distribution.csv` and
   the plots. Download them (Section 5) or back up to Drive (Section 8).

### What if Colab disconnects mid-training?
Nothing is lost beyond the last <10 iterations — the autosave already wrote to Drive. Just
start a new session and resume from step 5.

---

# CASE 2 — Single long-running VM (or Colab Pro), one session

You have enough uninterrupted compute to train all three encoders end to end.

### Option A — in the notebook
1. Setup cells (clone/deps/imports), then run the Section 2 golden-dataset cell. On a
   persistent VM the file survives, so you only need this after a fresh clone.
2. **Persistence cell:** set `USE_DRIVE = False`. Checkpoints go to local `./models/`.
   (Set it `True` if you still want a Drive backup — either works.)
3. **Full Training cell:** set `SESSION_BUDGET_H` high enough to cover everything, e.g. `48`.
   Leave `ENCODERS = ['gcn', 'gat', 'lstm']`. One execution trains all three to
   `TARGET_ITERATIONS` in sequence.
4. Run the **Benchmark** cell.

### Option B — from the command line
```bash
# from the repo root — after any fresh clone (deterministic, so re-running is safe):
python utils/generate_test_dataset.py --num_dags 500 --output data/test_dags.json
md5sum data/test_dags.json     # must match across machines/sessions

# train each encoder (main.py resumes from models/ automatically if a checkpoint exists):
#   edit configs/default_config.yaml -> algorithms.tampo.encoder_type, or drive it in code.
python main.py                       # interactive; pick an encoder and iteration count

# benchmark all three against the frozen test set:
python benchmark.py \
    --algorithms TAMPO_GCN TAMPO_GAT TAMPO_LSTM \
    --checkpoint_dir models/ \
    --dataset_path data/test_dags.json \
    --output_dir results/ \
    --seed 42
```
The `benchmark.py --seed` flag seeds evaluation; it also accepts `--use_best` to load the
lowest-meta-loss checkpoints (not generally recommended — meta-loss is a weak proxy).

Because it is one continuous process, resume never triggers; but the same determinism holds —
a fixed `SEED` makes the whole run reproducible.

---

# CASE 3 — Publication-grade multi-seed comparison

A single seed makes a run **repeatable**, not **representative**. Deep-RL final performance
swings widely on the seed alone, so a single-seed GCN-vs-GAT-vs-LSTM table cannot separate a
real architectural difference from luck. A reviewer will ask. This case is Case 1 or Case 2
repeated `k` times with a different seed, then aggregated.

Read the budget section first — this is the expensive case, and the right value of `k` is a
compute decision as much as a statistical one.

## 3.1 How many seeds? (`k ≥ 6`, not 5)

**Do not use 5 seeds if you intend to claim statistical significance.** A two-sided Wilcoxon
signed-rank test on `n` paired seeds has a hard floor on the p-value it can ever return —
even if one encoder wins on *every* seed by a mile:

| seeds `n` | best achievable two-sided p | can reach p < 0.05? |
|---|---|---|
| 4 | 0.125 | no |
| **5** | **0.0625** | **no — impossible** |
| **6** | **0.031** | yes (only if it wins on every seed) |
| 8 | 0.0078 | yes, with margin |
| 10 | 0.002 | yes, comfortably |

At `k = 5` the test is mathematically incapable of producing a significant result, so you
would be spending five full training runs on a test that cannot pass. **Use `k = 8`** if you
want significance with any margin for one encoder losing a seat; `k = 6` is the bare minimum;
`k = 10` is stronger.

`utils/seeding.py` therefore defines `SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)`.

If the compute for `k ≥ 6` is out of reach, the honest fallback is to run `k = 3–5`, report
**mean ± std only**, and explicitly state that the study is descriptive and not powered for
significance testing. That is a legitimate paper; claiming significance off 5 seeds is not.

## 3.2 The compute budget — read this before starting

Case 3 multiplies total training cost by `k`. The arithmetic:

```
total hours  =  k seeds  ×  Σ over encoders ( iterations × seconds_per_iteration ) / 3600
```

Your own measured rate on a free-tier **T4 was ~4 hours for 75 iterations (~190 s/iteration)**
on the pre-overhaul code. If that rate holds, one encoder at 600 iterations is ~32 h, three
encoders ~90 h, and `k = 8` seeds is **~700 h of T4 time** — roughly 200 free Colab sessions.
That is not feasible on the free tier.

**Do not plan from the number above — measure your own.** The training cell now prints a live
`s/it` figure. Run one session, read the real `s/it` per encoder, and put it in the formula.
Then pick your lever:

| Lever | Effect on cost | Effect on the science |
|---|---|---|
| Fewer iterations per seed | linear | Safe **only** if the encoders have already plateaued. Verify with the benchmark before cutting. |
| Lower `META_BATCH_SIZE` (6 → 3) | ~linear | Noisier gradient per step, but ~same learning per step. Usually the best first cut. |
| Lower `EPISODES_PER_TASK` (3 → 2) | ~linear | Noisier advantage estimates. |
| Drop LSTM from the multi-seed table | ~60% off | Costs you a baseline. LSTM alone is ~4× GCN/GAT (CuDNN off for MAML). |
| Fewer seeds | linear | See 3.1 — below 6 you forfeit significance testing. |
| Colab Pro / a rented A100 | — | The realistic route to `k = 8` at full iterations. |
| Raise `meta_learning_rate` | — | **No.** Measured: the policy collapses (entropy → 0). |

A defensible compromise when compute is tight: **`k = 6`, 400 iterations, `META_BATCH_SIZE = 3`,
GCN + GAT + LSTM** — provided all three have plateaued by 400 (check before committing to it).

## 3.3 Directory layout

Each seed needs its own checkpoint directory and its own results directory, or seed 1 will
silently resume seed 0's checkpoints and the run is void.

```
/content/drive/MyDrive/TAMPO_checkpoints/
├── seed_0/models/tampo_gcn_checkpoint.pth      # + gat, lstm  (and *_best.pth)
├── seed_1/models/...
└── seed_7/models/...

results/
├── seed_0/run_<timestamp>/benchmark_results.csv   # + action_traces.csv,
├── seed_1/run_<timestamp>/...                     #   action_distribution.csv, *.png
├── seed_7/run_<timestamp>/...
└── aggregate/
    ├── summary_across_seeds.csv                # mean ± std per (encoder, metric)
    └── significance_across_seeds.csv           # paired Wilcoxon per encoder pair
```

## 3.4 Running it — notebook (Colab)

For each seed `s` in `0..7`, do a complete Case 1 (or Case 2) pass:

1. **Persistence cell** (`a6608b3c`) — point it at this seed's folder:
   ```python
   DRIVE_ROOT = '/content/drive/MyDrive/TAMPO_checkpoints/seed_0'   # bump per seed
   ```
   Confirm the printed `Existing checkpoints:` line matches the seed you intend to be on —
   `none (first session)` when starting a seed, this seed's files when resuming it.
2. **Training cell** (`254e899d`) — set the matching seed:
   ```python
   SEED = 0                     # must match the seed_N in DRIVE_ROOT above
   TARGET_ITERATIONS = 600      # identical for every seed and every encoder
   META_BATCH_SIZE   = 6        # identical for every seed
   EPISODES_PER_TASK = 3        # identical for every seed
   ```
   Re-run across sessions until all three encoders print `DONE.` (Case 1, step 6).
3. **Benchmark cell** (`cdb01c`) — send this seed's results to its own directory by changing
   the `--output_dir` argument:
   ```
   --output_dir results/seed_0/
   ```
   Leave `--seed 42` as is: that seeds *evaluation*, and holding it fixed across seeds keeps
   the test conditions identical. The thing that varies between seeds is the **training**
   seed, set in step 2.
4. Back up `results/seed_0/` to Drive (Section 8) — the VM's disk does not survive.
5. Move to the next seed: repeat 1–4 with `seed_1`, `SEED = 1`, `results/seed_1/`.

> **The one mistake that voids the study:** changing `DRIVE_ROOT` but forgetting `SEED` (or
> the reverse). Then two "different" seeds train identical weights, or one seed resumes
> another's checkpoints. Change both together, and check the persistence cell's output.

## 3.5 Running it — CLI (VM)

```bash
python utils/generate_test_dataset.py --num_dags 500 --output data/test_dags.json

for s in 0 1 2 3 4 5 6 7; do
  # 1. train all three encoders under seed $s into models_seed_$s/
  #    (drive TAMPOFramework(env, cfg, model_path=..., seed=$s) from your own script —
  #     the notebook training cell is the reference implementation)

  # 2. benchmark this seed into its own results directory
  python benchmark.py \
      --algorithms TAMPO_GCN TAMPO_GAT TAMPO_LSTM \
      --checkpoint_dir models_seed_$s \
      --dataset_path data/test_dags.json \
      --output_dir results/seed_$s \
      --seed 42
done
```

## 3.6 Aggregate and test

Once every seed has a `benchmark_results.csv`:

```bash
python utils/aggregate_seeds.py --results_root results --seeds 0 1 2 3 4 5 6 7
```

It picks the newest `run_*` per seed, and writes to `results/aggregate/`:

- **`summary_across_seeds.csv`** — `mean`, `std`, `n_seeds` and the raw per-seed values for
  `avg_makespan`, `avg_energy`, `within_episode_entropy`, `degenerate_episodes`.
- **`significance_across_seeds.csv`** — paired Wilcoxon for every encoder pair on makespan and
  energy. Paired is the correct design here: every encoder sees the same seeds.

It prints a warning for any seed whose results are missing rather than silently reporting a
smaller `k`, and flags any test run on fewer than 5 seeds as under-powered.

## 3.7 What to report

- The table is **mean ± std across seeds**, never a single seed's number.
- **Report a difference only if it exceeds the across-seed spread.** If GAT beats GCN by less
  than the ± of either, the honest conclusion is "no measurable difference between encoders" —
  which is a real, publishable finding, not a failed experiment.
- State `k`, the iteration count, and that all encoders were trained on an identical task
  stream (same graphs, preferences and channel noise per seed — see the top of this doc).
- If `within_episode_entropy` is still ~0 for an encoder, it never learned to schedule; its
  makespan is not a scheduling result and must not go in the table as one.

> Every seed must be a full, converged run of **all three** encoders at matched iterations,
> `META_BATCH_SIZE`, `EPISODES_PER_TASK`, `inner_steps` and evaluation seed. A half-trained
> encoder in the table is worse than no table.

---

## Quick reference — knobs

| Knob | Where | Meaning |
|---|---|---|
| `SEED` | training cell / `experiment.seed` | Reproducibility + cross-encoder fairness. Same for all encoders in a comparison. |
| `ENCODERS` | training cell | Which encoders to train this run. |
| `TARGET_ITERATIONS` | training cell | Per-encoder ceiling. Behavioural stop is preferred. |
| `META_BATCH_SIZE` | training cell | Tasks per outer update. Smaller → more steps/hour, noisier gradient. |
| `EPISODES_PER_TASK` | training cell → config `episodes_per_task` | Rollouts per task per iteration. |
| `SESSION_BUDGET_H` | training cell | Wall-clock for THIS session. ~3.3 for free Colab; large for a VM. |
| `USE_DRIVE`, `DRIVE_ROOT` | persistence cell | Where checkpoints persist. |
| `inner_steps`, `meta_learning_rate` | config | MAML inner-loop depth / step size. Do not raise the LR. |

**Inert config keys** (present in YAML, read by no code — the training cell is the source of
truth): `num_meta_iterations`, `meta_batch_size`, `num_episodes`, `num_attention_heads`.
