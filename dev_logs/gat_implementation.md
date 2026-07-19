# Development Log: Bi-GATv2 Encoder Implementation

**Date**: 2026-06-13
**Branch**: `gat-pyg` (branched from `gcn-pyg`)
**Scope**: Add a Bi-GATv2 encoder to TAMPO as a controlled drop-in replacement for the Bi-GCN encoder, enabling a fair apples-to-apples comparison between GCN and GAT message-passing operators within the same meta-RL framework.

---

## 1. Design Rationale

### Why a Drop-In Swap?

The existing GCN encoder (Cai et al., 2025 / GDRL) uses a specific bidirectional architecture:
- Forward stream: `GCNConv(6→16) → GCNConv(16→1) → mean pool → scalar`
- Backward stream: same on reversed DAG edges
- Both scalars concatenated with server features → FNN → context vector

To isolate the effect of the graph convolution operator, GAT is integrated by replacing only `GCNConv` with `GATv2Conv` inside the identical Bi-directional skeleton. Every other component — the reversed-edge backward stream, the mean pooling, the server-feature concatenation, and the FNN — is unchanged. This ensures the only experimental variable is the conv operator.

### Why Bi-Directional?

The GAPO paper (Zhang et al., 2025) does not define a bidirectional traversal. However, for a controlled experiment the existing GDRL architectural skeleton is preserved. Swapping only the conv operator means:
- One variable changes (GCN → GAT)
- The bidirectionality, pooling, and FNN are held constant
- Results are directly attributable to the attention mechanism vs. fixed normalisation

### Why Directed Edges Are Kept

`_build_pyg_batch` continues to use directed edges (no `to_undirected()`), consistent with the GCN baseline. The backward stream is handled inside `_apply_gat` by flipping `local_edges_fwd` — identical to how `_apply_gcn` works.

---

## 2. Architecture Mapping

| Layer | GCN (Cai et al. 2025) | GAT (Zhang et al. 2025, adapted) |
|---|---|---|
| Layer 1 fwd | `GCNConv(6, 16)` | `GATv2Conv(6, 4, heads=4, concat=True)` → `[N, 16]` |
| Layer 2 fwd | `GCNConv(16, 1)` | `GATv2Conv(16, 1, heads=1, concat=False)` → `[N, 1]` |
| Layer 1 bwd | `GCNConv(6, 16)` | `GATv2Conv(6, 4, heads=4, concat=True)` → `[N, 16]` |
| Layer 2 bwd | `GCNConv(16, 1)` | `GATv2Conv(16, 1, heads=1, concat=False)` → `[N, 1]` |
| Pooling | `mean()` → scalar | `mean()` → scalar (identical) |
| Combined | `stack([fwd, bwd])` → `[2]` | `stack([fwd, bwd])` → `[2]` (identical) |
| FNN | `Linear(server_dim+2→128→64)` → `Linear(64→64→hidden_dim*2)` | identical |

The intermediate dimension is preserved at 16 in both encoders (`gat_hidden_dim: 16`). The head split is `hidden_per_head = 16 // 4 = 4`.

`add_self_loops=True` in `GATv2Conv` matches `GCNConv`'s default, ensuring self-feature aggregation is identical in both encoders.

---

## 3. Files Changed

### `algorithms/rl/tampo.py`

| Change | What | Why |
|---|---|---|
| Import | Added `GATv2Conv` to `torch_geometric.nn` import | New conv operator |
| `DAGEncoder.__init__` | Added `num_gat_heads`, `gat_hidden_dim`, `gat_add_self_loops` params | Config-driven construction |
| `DAGEncoder.__init__` | Added `elif encoder_type == 'gat':` block with `gat1_fwd`, `gat2_fwd`, `gat1_bwd`, `gat2_bwd` and shared-structure `fnn1`/`fnn_out` | Bi-GATv2 encoder layers |
| `DAGEncoder.__init__` | Updated `encoder_type` validation to include `'gat'` | Prevent silent fallthrough |
| `DAGEncoder._apply_gat` | New method — identical data flow to `_apply_gcn`, uses `gat*` layers | GAT forward pass |
| `DAGEncoder.forward` | Changed `if encoder_type == 'gcn'` → `if encoder_type in {'gcn', 'gat'}` with inner branch | Route GAT through same FNN stack |
| `MetaPolicyNetwork.__init__` | Added `num_gat_heads`, `gat_hidden_dim`, `gat_add_self_loops` params; forwarded to `DAGEncoder` | Thread config down |
| `TAMPOFramework.__init__` | Extracts `num_gat_heads`, `gat_hidden_dim`, `gat_add_self_loops` from config; passes to `MetaPolicyNetwork` | Config → network |

### `configs/default_config.yaml`

- Updated `encoder_type` comment to include `'gat'`
- Added `num_gat_heads: 4`, `gat_hidden_dim: 16`, `gat_add_self_loops: true` under `algorithms.tampo`

### `main.py`

- Added `TAMPO_GAT` option to `get_user_input()`
- Updated all `if any([...TAMPO_GCN])` guards to include `TAMPO_GAT`
- Added execution block calling `test_tampo(..., encoder_type='gat')`

### `README.md`

- Added TAMPO-GAT row to the Implemented Algorithms table
- Added GAT config block to the "Configuring TAMPO" section

### `Papers referred.md`

- Corrected journal name from "IEEE Internet of Things Journal" to "Electronics (MDPI), vol. 14, no. 16, Art. no. 3238, 2025" (DOI prefix `10.3390/` belongs to MDPI)

---

## 4. Config Parameter Reference

```yaml
algorithms:
  tampo:
    encoder_type: 'gat'   # activate Bi-GATv2 encoder
    num_gat_heads: 4       # heads for layer 1; hidden_per_head = gat_hidden_dim // num_gat_heads
    gat_hidden_dim: 16     # intermediate node-embedding dimension (must be divisible by num_gat_heads)
    gat_add_self_loops: true  # matches GCNConv default; ensures nodes aggregate own features
```

---

## 5. Checkpoint

Trained model saved to `models/tampo_gat_checkpoint.pth` (path derived automatically from `encoder_type='gat'` in `TAMPOFramework`).

---

## 6. Historical Reference

The earlier `gat-implementation` branch (commit `533a850 Add GATv2 encoder`) contains a proof-of-concept GAT encoder based on the older pre-physics-overhaul codebase. It uses a different (non-GDRL) GCN architecture and applies `to_undirected()` in `_build_pyg_batch`. That branch is kept as historical reference only. The `gat-pyg` branch is the authoritative implementation.
