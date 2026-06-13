
## GDRL GCN Encoder

Y. Cai, P. Cheng, Z. Chen, W. Xiang, B. Vucetic, and Y. Li,
"Graphic Deep Reinforcement Learning for Dynamic Resource Allocation in
Space-Air-Ground Integrated Networks,"
IEEE Journal on Selected Areas in Communications,
vol. 43, no. 1, pp. 334–349, Jan. 2025.
https://ieeexplore.ieee.org/document/10712792

*Used for:* Adapting the `CustomFeaturesExtractor` two-layer GCN architecture
(`GCNConv(6,16) → GCNConv(16,1) → FNN`) as TAMPO's DAG Encoder, replacing
the BiLSTM path.  Directed edges are preserved as per the reference implementation.

---


## GAPO GATv2 Encoder

Y. Zhang, X. Wang, Y. Wang, W. Liu, and H. Wang,
"GAPO: A Graph Attention-Based Reinforcement Learning Algorithm for Congestion-Aware Task Offloading in Multi-Hop Vehicular Edge Computing,"
Electronics (MDPI), vol. 14, no. 16, Art. no. 3238, 2025.
https://doi.org/10.3390/electronics14163238

*Used for:* Adapting the DAG encoder from a two-layer GCN architecture to a two-layer Graph Attention Network architecture (`GATv2Conv → GATv2Conv → FNN`) within TAMPO's `CustomFeaturesExtractor`. The attention mechanism learns task/node importance during message passing, enabling weighted aggregation of neighboring node features rather than the fixed normalization used by GCN. This paper serves as the primary architectural reference for the GCN-to-GAT migration while retaining the existing PPO training pipeline and DAG graph representation.

---