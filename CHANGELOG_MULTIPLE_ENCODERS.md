# Simultaneous Encoders Feature: Detailed Code Breakdown

## 1. Current Status
**Yes, TAMPO can now run the LSTM encoder path and the GCN encoder path in the same interactive session with isolated checkpoints and side-by-side results.**

This feature is now backed by the corrected graph pipeline, so the comparison is no longer just two labels pointing at the same broken flattened input path.

---

## 2. What Changed

### 2A. Interactive Selection (`tampo/main.py -> get_user_input()`)
The old single `Run TAMPO?` prompt was replaced with:

* `Run TAMPO-LSTM?`
* `Run TAMPO-GCN?`

Both variants share the same `tampo_iterations` prompt, so you can enable one or both without duplicating the training-configuration questions.

### 2B. Encoder Override (`tampo/main.py -> test_tampo()`)
`test_tampo(...)` now accepts an explicit `encoder_type` argument and injects it directly into the TAMPO config before framework creation:

```python
tampo_config['encoder_type'] = encoder_type
checkpoint_path = f"models/tampo_{encoder_type}_checkpoint.pth"
```

This guarantees that:

* `TAMPO_LSTM` always instantiates the LSTM encoder
* `TAMPO_GCN` always instantiates the GCN encoder
* checkpoints stay architecture-specific

### 2C. Main Orchestration (`tampo/main.py -> main()`)
The execution loop now runs TAMPO variants independently:

```python
if user_choices['TAMPO_LSTM']:
    results['TAMPO_LSTM'] = test_tampo(..., encoder_type='lstm')

if user_choices['TAMPO_GCN']:
    results['TAMPO_GCN'] = test_tampo(..., encoder_type='gcn')
```

That means you can answer `yes` to both prompts and get both runs in one session.

### 2D. Result Isolation
Each encoder variant now writes to its own checkpoint file:

* `models/tampo_lstm_checkpoint.pth`
* `models/tampo_gcn_checkpoint.pth`

This prevents architecture cross-loading and keeps resumed training clean.

### 2E. Plotting Support
The comparison plotting code now uses dynamic colors instead of a fixed 6-color list, so running more algorithm variants in a single report does not break the charts.

### 2F. Meta-Learning Correctness
The simultaneous encoder feature now also benefits from the corrected TAMPO MAML path.

`tampo/algorithms/rl/tampo.py` no longer performs temporary live-weight assignment during `_forward_with_params(...)`. It now uses `torch.func.functional_call`, which means:

* `TAMPO_LSTM` uses a proper functional MAML inner loop
* `TAMPO_GCN` uses a proper functional MAML inner loop
* both variants now compare encoder behavior on top of the same meta-learning logic, rather than on top of the old approximation

### 2G. Paper-Standard GCN Library Path
The `TAMPO_GCN` branch now uses `torch_geometric.nn.GCNConv` instead of the earlier native dense adjacency multiplication.

That means:

* `TAMPO_LSTM` remains the sequence-encoder baseline
* `TAMPO_GCN` now runs through PyG `Data`/`Batch` graph objects and `edge_index`
* the side-by-side comparison is now closer to what readers expect from a standard GCN benchmark

---

## 3. Why This Is Safe

1. Each call to `test_tampo(...)` creates a fresh `TAMPOFramework`, so the neural modules are instantiated separately per encoder run.
2. The checkpoint path is derived from `encoder_type`, so weights do not overwrite each other.
3. The evaluator stores results under distinct keys like `TAMPO_LSTM` and `TAMPO_GCN`, which makes the output tables, JSON, and plots naturally separate the two variants.
4. The environment now preserves the selected DAG properly during TAMPO evaluation, so both variants are evaluated on real graph inputs instead of silently reverting to random scalar tasks.
5. The functional MAML path keeps the meta-gradient chain intact for both encoder variants, so the comparison is not biased by the old temporary-parameter workaround.
6. The GCN variant now uses the standard PyG graph-convolution stack instead of a custom dense kernel, which makes the LSTM-vs-GCN comparison cleaner for reporting.

---

## 4. Extra Note

Inside `tampo/algorithms/rl/tampo.py`, the encoder implementation now also supports `encoder_type: 'both'` as a fused internal branch option. The interactive CLI currently exposes the side-by-side `LSTM` and `GCN` runs explicitly, which is the safer comparison workflow for benchmarking.

The side-by-side CLI still exposes `LSTM` and `GCN` as separate benchmark entries, while `both` remains available as a direct config option for internal experimentation.

---

## Summary

You can now start `tampo/main.py`, answer `yes` to both `TAMPO-LSTM` and `TAMPO-GCN`, and the framework will:

* train both variants independently
* evaluate both variants on the loaded DAG dataset
* save separate checkpoints
* print both results side-by-side
* include both curves/points in the generated output artifacts
