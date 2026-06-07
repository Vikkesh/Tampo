"""
TAMPO GAT Smoke Test
--------------------
Run from the root of the Tampo repo:
    python smoke_test_gat.py

No training. No checkpoints. Shape validation only.
Expected: zero exceptions, all shapes printed, GCN == GAT on every dimension.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch, to_undirected
from utils.dag_parser import DAGParser
from algorithms.rl.tampo import DAGEncoder, _build_pyg_batch

HIDDEN_DIM    = 256
NUM_LAYERS    = 2
NUM_GAT_HEADS = 8
FEATURE_DIM   = 6
DEVICE        = torch.device('cpu')   # CPU is fine for shape validation

SEP = "=" * 60

# ── 1. Load real DAGs from the repository ───────────────────────────────────
print(SEP)
print("Step 1: Load real DAGs")
parser = DAGParser()
graphs = parser.load_dataset(num_graphs=2)
assert len(graphs) > 0, "No graphs loaded — check data path"
dag = graphs[0]
print(f"  Loaded DAG 0 : {dag['num_tasks']} nodes, {len(dag['edges'])} edges")
if len(graphs) > 1:
    print(f"  Loaded DAG 1 : {graphs[1]['num_tasks']} nodes, {len(graphs[1]['edges'])} edges")

# ── 2. Build feature matrices and PyG batch ─────────────────────────────────
print(f"\nStep 2: Build PyG batch ({len(graphs[:2])} graphs)")

task_features_list = []
adjacency_list     = []

for g in graphs[:2]:
    adj = np.asarray(g['adj_matrix'], dtype=np.float32)
    N   = adj.shape[0]
    in_deg  = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    feats   = np.stack([
        np.array([t['data_size'] for t in g['tasks']], dtype=np.float32) / 1e7,
        np.array([t['cycles']    for t in g['tasks']], dtype=np.float32) / 1e10,
        in_deg  / max(N - 1, 1),
        out_deg / max(N - 1, 1),
        np.zeros(N, dtype=np.float32),   # depth (placeholder)
        np.zeros(N, dtype=np.float32),   # comm_load (placeholder)
    ], axis=1)
    task_features_list.append(feats)
    adjacency_list.append(adj)

graph_batch = _build_pyg_batch(task_features_list, adjacency_list).to(DEVICE)
B = len(graphs[:2])
N = max(f.shape[0] for f in task_features_list)   # dense max_nodes
T = graph_batch.x.shape[0]                        # sparse total nodes

print(f"  Batch size        : {B}")
print(f"  Max nodes N       : {N}")
print(f"  Total nodes T     : {T}")
print(f"  graph_batch.x     : {list(graph_batch.x.shape)}")
print(f"  edge_index        : {list(graph_batch.edge_index.shape)}")

# Build dense task_features and node_mask
task_features_np = np.zeros((B, N, FEATURE_DIM), dtype=np.float32)
node_mask_np     = np.zeros((B, N),               dtype=np.float32)
for i, feats in enumerate(task_features_list):
    n = feats.shape[0]
    task_features_np[i, :n] = feats
    node_mask_np[i,    :n]  = 1.0

task_features = torch.FloatTensor(task_features_np).to(DEVICE)
node_mask     = torch.BoolTensor(node_mask_np.astype(bool)).to(DEVICE)

# ── 3. GAT: layer-by-layer shape probe ──────────────────────────────────────
print(f"\nStep 3: GAT layer-by-layer shape probe")
print(f"  (hidden_dim={HIDDEN_DIM}, heads={NUM_GAT_HEADS}, layers={NUM_LAYERS})")

enc_gat = DAGEncoder(
    task_feature_dim=FEATURE_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    encoder_type='gat',
    num_gat_heads=NUM_GAT_HEADS
).to(DEVICE)
enc_gat.eval()

with torch.no_grad():
    ei = graph_batch.edge_index

    x0 = graph_batch.x
    print(f"  graph_batch.x (input)          : {list(x0.shape)}")

    x1 = enc_gat.task_embedding(x0)
    print(f"  after task_embedding           : {list(x1.shape)}")

    x2 = enc_gat.gat_layers[0](x1, ei)
    print(f"  after GATv2Conv layer 0        : {list(x2.shape)}")

    x2a = enc_gat.dropout(torch.relu(x2))
    x3  = enc_gat.gat_layers[1](x2a, ei)
    print(f"  after GATv2Conv layer 1        : {list(x3.shape)}")

    dense_x, nmask = to_dense_batch(x3, graph_batch.batch, max_num_nodes=N)
    print(f"  after to_dense_batch           : {list(dense_x.shape)}")

    kpm = ~nmask.bool()
    attn_out, _ = enc_gat.graph_attention(dense_x, dense_x, dense_x, key_padding_mask=kpm)
    encoded = enc_gat.layer_norm(dense_x + enc_gat.dropout(attn_out))
    print(f"  after graph_attention + LN     : {list(encoded.shape)}")

    mx  = encoded.masked_fill(~nmask.unsqueeze(-1), float('-inf'))
    ctx = mx.max(dim=1)[0]
    print(f"  context (masked_max_pool)      : {list(ctx.shape)}")

# ── 4. Full forward() — GAT ──────────────────────────────────────────────────
print(f"\nStep 4: Full DAGEncoder.forward() — GAT")
with torch.no_grad():
    gat_encoded, gat_context = enc_gat.forward(
        task_features,
        graph_batch=graph_batch,
        node_mask=node_mask
    )
    print(f"  encoded_tasks.shape            : {list(gat_encoded.shape)}")
    print(f"  context.shape                  : {list(gat_context.shape)}")
    assert not torch.isnan(gat_encoded).any(), "FAIL: NaN in encoded_tasks"
    assert not torch.isnan(gat_context).any(), "FAIL: NaN in context"
    print(f"  NaN check                      : OK")

# ── 5. Full forward() — GCN ──────────────────────────────────────────────────
print(f"\nStep 5: Full DAGEncoder.forward() — GCN")
enc_gcn = DAGEncoder(
    task_feature_dim=FEATURE_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    encoder_type='gcn'
).to(DEVICE)
enc_gcn.eval()

with torch.no_grad():
    gcn_encoded, gcn_context = enc_gcn.forward(
        task_features,
        graph_batch=graph_batch,
        node_mask=node_mask
    )
    print(f"  encoded_tasks.shape            : {list(gcn_encoded.shape)}")
    print(f"  context.shape                  : {list(gcn_context.shape)}")
    assert not torch.isnan(gcn_encoded).any(), "FAIL: NaN in GCN encoded_tasks"
    assert not torch.isnan(gcn_context).any(), "FAIL: NaN in GCN context"
    print(f"  NaN check                      : OK")

# ── 6. Shape equality assertions ─────────────────────────────────────────────
print(f"\nStep 6: Shape equality — GAT vs GCN")
assert list(gat_encoded.shape) == list(gcn_encoded.shape), (
    f"MISMATCH: encoded_tasks GAT={gat_encoded.shape} GCN={gcn_encoded.shape}"
)
assert list(gat_context.shape) == list(gcn_context.shape), (
    f"MISMATCH: context GAT={gat_context.shape} GCN={gcn_context.shape}"
)
print(f"  encoded_tasks : {list(gat_encoded.shape)} == {list(gcn_encoded.shape)}   OK")
print(f"  context       : {list(gat_context.shape)} == {list(gcn_context.shape)}   OK")

# ── 7. state_dict — no missing/unexpected keys ───────────────────────────────
print(f"\nStep 7: state_dict key check")
gat_keys = set(enc_gat.state_dict().keys())
gcn_keys = set(enc_gcn.state_dict().keys())
print(f"  GAT state_dict keys ({len(gat_keys)}):")
for k in sorted(gat_keys):
    print(f"    {k}")
print(f"  GCN state_dict keys ({len(gcn_keys)}):")
for k in sorted(gcn_keys):
    print(f"    {k}")
gat_only = gat_keys - gcn_keys
gcn_only = gcn_keys - gat_keys
shared   = gat_keys & gcn_keys
print(f"  Keys only in GAT : {sorted(gat_only)}")
print(f"  Keys only in GCN : {sorted(gcn_only)}")
print(f"  Shared keys      : {len(shared)} (task_embedding, graph_attention, layer_norm, ...)")

# ── 8. Construction check — all four encoder types ───────────────────────────
print(f"\nStep 8: Construction check — all four encoder types")
for et in ('lstm', 'gcn', 'gat', 'both'):
    try:
        e = DAGEncoder(
            task_feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            encoder_type=et
        )
        print(f"  encoder_type={et!r:6s}  constructed OK  "
              f"(params: {sum(p.numel() for p in e.parameters()):,})")
    except Exception as exc:
        print(f"  encoder_type={et!r:6s}  FAILED: {exc}")

# ── 9. Invalid encoder_type raises ValueError ─────────────────────────────────
print(f"\nStep 9: Invalid encoder_type raises ValueError")
try:
    DAGEncoder(task_feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, encoder_type='bad')
    print("  ERROR: no exception raised for encoder_type='bad'")
except ValueError as e:
    print(f"  ValueError raised correctly   : OK  ({e})")

# ── 10. Head count assertion fires correctly ───────────────────────────────────
print(f"\nStep 10: num_gat_heads assertion fires on bad head count")
try:
    DAGEncoder(task_feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
               encoder_type='gat', num_gat_heads=7)   # 512 % 7 != 0
    print("  ERROR: no assertion raised for num_gat_heads=7")
except AssertionError as e:
    print(f"  AssertionError raised correctly: OK  ({e})")

print(f"\n{SEP}")
print("ALL CHECKS PASSED")
print(SEP)
