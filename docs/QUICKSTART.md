# Quickstart — Run the Experiment (Bare Bones)

The shortest path from zero to a GCN vs GAT vs LSTM comparison table.
Details live in `docs/RUNNING_THE_EXPERIMENT.md`; how to read the numbers in
`docs/READING_RESULTS.md`. This page is just: what to click, what you should see,
how long it takes.

---

## What you are doing

Training the same TAMPO agent three times — once per encoder (GCN, GAT, LSTM) — on the
same task graphs with the same seed, then testing all three on 500 graphs they have never
seen. The only difference between the three runs is the encoder. Whichever schedules
tasks with lower delay/energy on the unseen graphs is the better encoder.

---

## How to run it

Everything happens in `Colab_Test_Run.ipynb` on Google Colab with a **T4 GPU**
(Runtime → Change runtime type → T4).

**Every session, run these cells top to bottom:**

1. **Setup cells (0a–0c)** — clone the repo, install dependencies, check the GPU.
2. **Section 2 — Golden dataset** — creates the 500 test graphs. Run it every session
   (a fresh Colab machine doesn't have the file). It prints an MD5 hash — it must be the
   same every session.
3. **Persistence cell** — connects Google Drive so checkpoints survive when Colab kills
   the session. Approve the popup. It prints which checkpoints already exist:
   first session says `none`, later sessions must list your `.pth` files.
4. **Full Training Run cell** — just run it. It trains GCN → GAT → LSTM automatically,
   saves progress to Drive every 10 iterations, and stops itself cleanly before Colab's
   ~4-hour kill. When the session ends, open a new session and repeat steps 1–4 —
   it continues exactly where it stopped, as if never interrupted.
5. **Benchmark cell** — run after any session to check progress. When all three encoders
   print `DONE`, this produces the final results (CSV files + plots), which you download
   with the Section 5 cell.

That's the whole loop: **run the notebook, let it stop itself, open a new session,
run it again. Repeat until every encoder says DONE, then benchmark.**

---

## How long it takes

- One free Colab session gives you ~4 hours of GPU; the notebook uses 3.3 h and stops
  itself safely.
- The target is 600 training iterations per encoder. LSTM is ~4× slower than GCN/GAT
  (a technical limitation of meta-learning through LSTMs), so it dominates the total.
- **Exact time depends on the GPU you get.** After your first session, look at the
  printed `s/it` (seconds per iteration) and estimate:

  `sessions needed ≈ (600 × s/it, summed over the 3 encoders) ÷ 3.3 hours`

- Realistic expectation on free Colab: **several sessions spread over days** — think
  5–15 sessions total, not 1–2. A paid GPU (Colab Pro / a rented VM) can do it in one
  long run with the same notebook (set `SESSION_BUDGET_H = 48`).
- You do NOT have to reach exactly 600: after each session run the benchmark; once an
  encoder's makespan has stopped improving across two sessions, it's done. All three
  encoders must end at the SAME iteration count for a fair table.

---

## What you should see

**While training (healthy):**

- A loss that goes down over time (noisily — that's normal).
- An `[actions]` line showing a MIX of servers, e.g.
  `local=12% cloud=30% edge0=8% edge1=42% edge2=8% | entropy=0.85`.
  The entropy number is your health check: **anywhere clearly above 0 is alive;
  0.00 means the policy broke** (it sends everything to one server).

**In the benchmark (healthy):**

- `within_episode_entropy` above 0.000 — the agent actually schedules, placing
  different parts of each graph on different servers.
- The action mix shifts with the preference: when told "care about speed" (0.8/0.2)
  it should use the fast cloud more; when told "care about energy" (0.2/0.8) it should
  use cheap edge servers more.
- A results table like:

  | algorithm | avg_makespan | avg_energy | within_episode_entropy |
  |---|---|---|---|
  | TAMPO_GCN | ... | ... | > 0 |
  | TAMPO_GAT | ... | ... | > 0 |
  | TAMPO_LSTM | ... | ... | > 0 |

  plus a bar chart and a Pareto plot (energy vs delay trade-off).

**Which encoder wins is the experiment's answer, not a promise.** The setup is fair —
same data, same seed, same everything except the encoder. The graph encoders (GCN/GAT)
have the right inductive bias for this task and ~2.8× fewer encoder parameters than the
LSTM, so if they win, that's a strong result. If the LSTM keeps up anyway, that is also
a real, reportable finding.

**Bad signs (stop and investigate, don't keep training):**

- entropy `0.00` in training, or `within_episode_entropy 0.000` in the benchmark
  → policy collapsed; see `docs/READING_RESULTS.md`.
- A later session's persistence cell prints `none` for checkpoints
  → Drive didn't mount; re-run that cell before training or you'll restart from zero.
- The golden dataset MD5 changed between sessions
  → the test set moved; results before and after are not comparable.

---

## For a paper (one extra step)

One seed = one run of the experiment. For a publishable claim ("GCN beats LSTM"),
repeat the whole thing under several seeds and average — see **Case 3** in
`docs/RUNNING_THE_EXPERIMENT.md`. Short version: 6+ seeds are needed before a
significance test can even work; with fewer, report averages and honestly call the
study descriptive.
