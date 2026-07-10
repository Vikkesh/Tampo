## GDRL GCN Encoder

Y. Cai, P. Cheng, Z. Chen, W. Xiang, B. Vucetic, and Y. Li,
"Graphic Deep Reinforcement Learning for Dynamic Resource Allocation in
Space-Air-Ground Integrated Networks,"
IEEE Journal on Selected Areas in Communications,
vol. 43, no. 1, pp. 334–349, Jan. 2025.
https://ieeexplore.ieee.org/document/10712792

*Used for:* The two-layer `GCNConv` graph feature extractor and the
"GNN node embeddings ⊕ scalar system state → two-block FNN" head, adapted from
`CustomFeaturesExtractor` in the authors' `Feature.py`.

*Faithful to the reference:*
- Two stacked `GCNConv` layers with `ReLU` + `Dropout` between them.
- Per-node outputs preserved through the second conv (GDRL's `gnn2(...).squeeze(2)`
  yields one value per node; it does **not** pool).
- The graph embedding is concatenated with the non-graph state vector and passed
  through `fnn1: Linear(→128→64)` then `fnn_out: Linear(64→64→out)`.
- Directed edges (no `to_undirected()`).

*Documented deviations (see `dev_logs/graph_encoder_and_observability_overhaul.md` §2):*
1. **Bidirectional streams.** GDRL runs a single forward pass over the graph. TAMPO runs
   a second stream over the reversed edge index and concatenates the two. This is our
   addition, not GDRL's, and is applied identically to the GCN and GAT encoders so the
   operator remains the only variable between them.
2. **Variable graph size.** GDRL assumes a fixed `U + L + N = 35` node graph and can
   therefore concatenate the full `[35]` per-node vector into the FNN. TAMPO's DAGs vary
   from 10 to 50 nodes, so the graph-level context uses a mean readout over node
   embeddings instead of a fixed-length concatenation.
3. **Per-node embedding width.** GDRL's second conv maps to 1 channel because its
   downstream FNN consumes the whole `[35]` vector. TAMPO's decoder attends over node
   embeddings, so the second conv maps to `hidden_dim` channels per direction.
4. **Sequential decoder.** GDRL feeds the extracted features to a TRPO actor-critic that
   emits a full allocation in one shot. TAMPO decodes node-by-node with a pointer-style
   attention decoder (see Vinyals et al. below), because server queue state evolves as
   nodes are placed.

*Citation guidance:* Cite Cai et al. (2025) for the **graph feature extractor** — that is
what is reused. Do **not** describe the implementation as a reproduction of GDRL: it is a
GDRL-derived encoder embedded in a different agent (MAML + PPO, not TRPO), operating on
variable-size DAGs, with a sequential decoder. Suggested phrasing:

> "The GCN encoder follows the two-layer `GCNConv` feature extractor of Cai et al. [x],
> adapted for variable-size DAGs by replacing their fixed-length node concatenation with a
> mean readout, and extended with a reversed-edge stream. Node embeddings are consumed by
> a pointer-style attention decoder [y] rather than a single-shot actor head."

---

## GAPO GATv2 Encoder

Y. Zhang, X. Wang, Y. Wang, W. Liu, and H. Wang,
"GAPO: A Graph Attention-Based Reinforcement Learning Algorithm for Congestion-Aware Task Offloading in Multi-Hop Vehicular Edge Computing,"
Electronics (MDPI), vol. 14, no. 16, Art. no. 3238, 2025.
https://doi.org/10.3390/electronics14163238

*Used for:* Replacing the graph convolution operator with `GATv2Conv`, so node
aggregation is attention-weighted rather than degree-normalised.

*Controlled-comparison note:* The GAT encoder is a drop-in swap of the conv operator
inside the identical Bi-directional skeleton used by the GCN encoder — same layer count,
same widths, same readout, same FNN head, same decoder, same `add_self_loops=True`. The
conv operator is the **only** difference, which is what licenses attributing any measured
GCN-vs-GAT gap to the attention mechanism.

*Documented deviation:* GAPO does not define a bidirectional traversal; the reversed-edge
stream is TAMPO's, applied symmetrically to both encoders.

---

## Graph Convolution Operator

T. N. Kipf and M. Welling,
"Semi-Supervised Classification with Graph Convolutional Networks,"
International Conference on Learning Representations (ICLR), 2017.
https://arxiv.org/abs/1609.02907

*Used for:* The underlying `GCNConv` operator (via `torch_geometric.nn.GCNConv`).

---

## Graph Attention Operator

S. Brody, U. Alon, and E. Yahav,
"How Attentive are Graph Attention Networks?,"
International Conference on Learning Representations (ICLR), 2022.
https://arxiv.org/abs/2105.14491

*Used for:* The `GATv2Conv` operator (via `torch_geometric.nn.GATv2Conv`), which fixes the
static-attention limitation of the original GAT.

---

## Pointer-Style Attention Decoder

O. Vinyals, M. Fortunato, and N. Jaitly,
"Pointer Networks,"
Advances in Neural Information Processing Systems (NeurIPS), 2015.
https://arxiv.org/abs/1506.03134

*Used for:* Indexing the embedding of the node currently being scheduled out of the
encoder's node-embedding matrix, and using it as the attention query for the decision
head. This is what makes the policy a function of *which* node is being placed rather
than of the graph as a whole.

---

## MAML Meta-Learning

C. Finn, P. Abbeel, and S. Levine,
"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,"
International Conference on Machine Learning (ICML), 2017.
https://arxiv.org/abs/1703.03400

*Used for:* The inner-loop / outer-loop structure of `LowerLayerAgent.inner_loop_update`
and `HigherLayerMetaLearner.meta_update`, including second-order gradients through the
adaptation graph (`create_graph=True`).

---

## PPO Clipped Surrogate Objective

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov,
"Proximal Policy Optimization Algorithms,"
arXiv:1707.06347, 2017.
https://arxiv.org/abs/1707.06347

*Used for:* The clipped importance-ratio objective in `LowerLayerAgent._ppo_policy_loss`.
The MAML inner loop takes several gradient steps on a single on-policy batch, so from the
second step onward the data is off-policy with respect to the adapted parameters; the
clipped ratio bounds the resulting update.

---
