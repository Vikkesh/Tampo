
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


