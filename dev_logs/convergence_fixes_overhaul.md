# Dev Log: Convergence Fix Overhaul
**Date:** 2026-07-06  
**Scope:** All 15 convergence root-causes from `convergence_failure_investigation.md` — implemented, tested, verified.

---

## Phase 1 — Broken Signal

### RC#3 — Reward Scale 5.0 → 1.0 (`base_offloading_env.py`)
`5.0 * total_improvement` clipped to ±5 → scaled to ±1. With gamma=0.99 and 50 nodes, the old scale caused theoretical value magnitudes of ~200, exploding value loss from iteration 1.

### RC#15 — kappa 1e-28 → 1e-23 (`default_config.yaml`)
Old local energy = 1e-10 J vs transmission ~0.025 J (8 orders of magnitude apart). Energy improvement ratio was always pinned at the clip floor. New kappa: local energy ≈ 1e-5 J — learnable signal.

### RC#1 + RC#4 — mo_return sign + discounting (`tampo.py`, `_collect_task_experiences`)
Old code stored `[info['delay'], info['energy']]` (raw positive costs) as mo_return. Advantages were computed as `actual - predicted`, so lower delay (good) produced negative advantage → agent penalised for good actions.

Fix: Per episode, store `raw_delay`, `raw_energy`, `node_cycles` per step. After the full episode loop, run a backward pass:
```
G_delay = step_d_improvement + gamma * G_delay  (backward through episode)
G_energy = step_e_improvement + gamma * G_energy
mo_return = [G_delay, G_energy]  # higher = better, includes temporal discounting
```
Improvement = (local_delay - step_delay) / local_delay. Positive when agent does better than local.

---

## Phase 2 — Structural Architecture

### RC#2 — GCN/GAT encoded_tasks all-zeros (`tampo.py`, DAGEncoder.forward)
Old: `encoded_tasks = torch.zeros(B, N, hidden_dim*2)` — decoder attention K/V was zeros.
New: `encoded_tasks = context.unsqueeze(1).expand(-1, N, -1).contiguous()` — broadcasts graph-level context to every node position.

### RC#13 — Decoder discards second half of context (`tampo.py`, PreferenceConditionedDecoder)
Old: `h_t = context[:, :hidden_dim]` — discards backward LSTM encoding.
New: Added `self.context_projection = nn.Linear(hidden_dim*2, hidden_dim)` and `h_t = self.context_projection(context)`. No downstream tensor shape changes needed.

### RC#5 — Value network input mismatch (`tampo.py`, MultiObjectiveValueNetwork)
Old: Value network received only flat obs (36-dim). Policy sees rich graph features.
New: Value network input = flat_obs (36) + server_features (20) + preference (2) = 58-dim. Both loss functions extract `server_features` from batch and pass to `meta_value(state, pref, server_features=...)`. Fallback path (zeros) provided when server_features not available.

---

## Phase 3 — Stabilise Learning

### RC#8 — Advantage normalisation (`tampo.py`, both loss functions)
Added after `weighted_advantages` computation in `_compute_loss_with_params` and `_compute_loss`:
```python
adv_std = weighted_advantages.std() + 1e-8
weighted_advantages = (weighted_advantages - weighted_advantages.mean()) / adv_std
```

### RC#7 — Dropout during MAML inner loop (`tampo.py`, `inner_loop_update`)
Added `self.meta_policy.eval()` before inner loop, `self.meta_policy.train()` after. Stochastic dropout masks with `create_graph=True` produce inconsistent second-order gradients across inner steps.

### RC#6 — LR config (`default_config.yaml`)
```yaml
meta_learning_rate: 5.0e-5   # was 3.0e-4
inner_lr: 0.005               # was 0.01
inner_steps: 5                # was 3
```

### RC#11 — Exploration (`tampo.py`, `_select_action`)
Old: 95% argmax + biased exploration (action 0 excluded).
New: `Categorical(probs).sample().item()` during training; argmax only when `deterministic=True`.

### RC#10 — Sequential train/test split (`tampo.py`, `_collect_task_experiences`)
Added `random.shuffle(all_experiences)` after the backward return-computation pass and before the 80/20 split.

---

## Phase 4 — Observability & Minor

### RC#12 — stdout redirect removed (`tampo.py`, `train`)
4-line `sys.stdout = StringIO()` block deleted. Gradient diagnostics (zero-grad warnings) are now visible.

### RC#9 — HyperVolume reference point (`tampo.py`, `LowerLayerAgent.__init__`)
Changed from `[10.0, 1.0]` (raw scale) to `[2.0, 2.0]` (improvement scale). `update_performance()` now clamps improvement values to [-1,1] instead of dividing by physically-wrong reference magnitudes.

### RC#14 — Value network capacity (`tampo.py`, `TAMPOFramework.__init__`)
`hidden_dim=hidden_dim * 2` (512 instead of 256) to match policy network capacity.

---

## Test Results

| Test | Result |
|---|---|
| Syntax: tampo.py | ✅ |
| Syntax: base_offloading_env.py | ✅ |
| ValueNetwork dim (RC#5+#14) | ✅ (8,2) output |
| Decoder context_projection (RC#13) | ✅ (4,128) projection |
| GCN encoded_tasks non-zero (RC#2) | ✅ max_abs=0.2565 |
| MetaPolicyNetwork forward (LSTM) | ✅ (2,1,5) logits |
| Advantage normalisation (RC#8) | ✅ mean≈0, std≈1 |
| Categorical exploration (RC#11) | ✅ action 0 included |
| Discounted mo_return sign (RC#1+#4) | ✅ positive for faster-than-local |

## Files Modified
- `algorithms/rl/tampo.py` — RC#1,2,4,5,7,8,9,10,11,12,13,14
- `env/base_offloading_env.py` — RC#3
- `configs/default_config.yaml` — RC#6, RC#15
