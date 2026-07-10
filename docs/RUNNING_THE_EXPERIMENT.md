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
3. Run **Section 2 — Generate the Golden Test Dataset** *once, ever*. It writes
   `data/test_dags.json`. Never re-run it between comparisons — it must stay frozen.
4. (Optional) Run **Section 2.5** verification and the **Section 1** encoder unit test —
   both should print all `PASS`.

### Every session
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
1. Setup cells (clone/deps/imports) and generate the golden dataset once, as in Case 1.
2. **Persistence cell:** set `USE_DRIVE = False`. Checkpoints go to local `./models/`.
   (Set it `True` if you still want a Drive backup — either works.)
3. **Full Training cell:** set `SESSION_BUDGET_H` high enough to cover everything, e.g. `48`.
   Leave `ENCODERS = ['gcn', 'gat', 'lstm']`. One execution trains all three to
   `TARGET_ITERATIONS` in sequence.
4. Run the **Benchmark** cell.

### Option B — from the command line
```bash
# from the repo root, once:
python utils/generate_test_dataset.py --num_dags 500 --output data/test_dags.json

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

A single seed makes a run **repeatable**, not **representative**. RL final performance varies
a lot with the seed alone, so a single-seed GCN-vs-GAT-vs-LSTM table cannot separate a real
architectural difference from luck. For a paper, repeat across seeds.

1. Do Case 1 or Case 2, but run the whole train-then-benchmark pipeline once per seed in
   `utils.seeding.SEEDS = (0, 1, 2, 3, 4)`.
   - In the notebook: set `SEED = 0`, train all encoders to target, benchmark to
     `results/seed_0/`; then `SEED = 1`, retrain (delete or repoint checkpoints), benchmark
     to `results/seed_1/`; and so on. Use a distinct `CKPT_DIR` / `DRIVE_ROOT` per seed so
     checkpoints don't collide.
   - From the CLI:
     ```bash
     for s in 0 1 2 3 4; do
       # train each encoder under seed $s into models_seed_$s/ ...
       python benchmark.py --seed $s --checkpoint_dir models_seed_$s \
         --dataset_path data/test_dags.json --output_dir results/seed_$s
     done
     ```
2. Aggregate: for each `(encoder, metric)` report **mean ± std** across the 5 seeds.
3. Significance: because every encoder sees the same seeds (paired design), use a Wilcoxon
   signed-rank test on the per-seed makespan/energy. `k = 5` seeds is the practical minimum;
   `k ≥ 10` is stronger.
4. Report the encoder difference **only if** it exceeds the across-seed spread. If GAT beats
   GCN by less than the ± of either, the honest conclusion is "no measurable difference".

> Each seed must be a full, converged run of all three encoders at matched iterations. A
> half-trained encoder in the table is worse than no table.

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
