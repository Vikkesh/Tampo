# TAMPO Core Logic & Architecture Concepts Explained

*This document captures the detailed conceptual logic behind the TAMPO meta-RL framework, specifically addressing evaluation profiling, episodic execution, physics trade-offs, network architecture, and dataset separation.*

---

## 1. Multi-Objective Evaluation (User Preference Profiles)

TAMPO is a Multi-Objective Meta-RL (MORL) framework designed to adapt to conflicting user goals: minimizing Latency (Makespan) vs. minimizing Battery usage (Energy).

**Testing the Convergence:**
When the benchmark script evaluates the algorithm against the 500-graph Golden Test Dataset, it does not just test one preference. It splits the workload across three distinct user profiles:
1.  **Performance Mode** `[w_delay: 0.8, w_energy: 0.2]` (First third of graphs)
2.  **Balanced Mode** `[w_delay: 0.5, w_energy: 0.5]` (Second third of graphs)
3.  **Battery Saver Mode** `[w_delay: 0.2, w_energy: 0.8]` (Final third of graphs)

**The Result:** The final output metric (e.g., Average Makespan and Average Energy plotted on the Pareto Front) is a consolidated average across all these profiles. Because TAMPO produces a single data point that sits favorably on the Pareto front, it proves that the *single trained meta-agent has successfully converged* and can generalize its behavior on the fly to meet wildly different user demands without getting stuck in a local minimum.

---

## 2. Episodic Execution Definition

In TAMPO's reinforcement learning loop, **1 Episode = Processing exactly 1 complete DAG workflow.**

**The Flow of an Episode:**
1.  The environment loads a single Directed Acyclic Graph (DAG) (e.g., a workflow containing 20 interdependent tasks) and assigns a random preference vector.
2.  The agent analyzes the graph and chooses a server for the very first node in the topological sequence. The environment calculates the start and finish time for that specific node based on network and server availability.
3.  The agent proceeds to choose a server for the second node, and continues sequentially through the graph.
4.  Once the final node in the DAG is assigned and successfully processed, the `done` flag is triggered. The environment calculates the total *Makespan* (the absolute time from the start of the first node to the end of the last node) for the entire graph. The episode is then complete.

---

## 3. The Edge vs. Cloud Physics Trade-off

The physics engine models a "Star" topology where the user's mobile device sits in the center and communicates directly with both Edge servers and the Cloud server. There is no multi-hop (Device → Edge → Cloud) relay.

**The Trade-offs:**
*   **The Edge:**
    *   **Pros:** Physically closer. The physics engine applies a `1.5x` bandwidth multiplier, meaning uploading data is extremely fast (Low Network Latency). It also costs less battery (`edge_power_tx = 0.3W`).
    *   **Cons:** Computationally slower CPU capacity (`5.0 GHz`).
*   **The Cloud:**
    *   **Pros:** Massive CPU capacity (`10.0 GHz`—twice as fast as the Edge).
    *   **Cons:** Physically distant. It lacks the bandwidth multiplier (High Network Latency) and costs significantly more battery to reach (`cloud_power_tx = 0.5W`).

**Agent Decision Making:** If a task is "Data-Heavy" (e.g., uploading a 4K video) but "Computation-Light", the agent learns to send it to the Edge to avoid the massive upload delay and battery drain. If a task is "Data-Light" (e.g., an array of floats) but "Computation-Heavy" (e.g., running an AI inference model), the agent learns to send it to the Cloud because the 10.0 GHz processor will finish the job instantly, entirely offsetting the slightly slower upload time.

---

## 4. Parallel GCN Encoding vs. Sequential RL Decoding

TAMPO utilizes a hybrid architecture that blends parallel graph processing with sequential decision making.

*   **The GCN Encoder (Parallel):** When a new DAG arrives, the Graph Convolutional Network (GCN) processes the *entire* graph structure simultaneously. By passing messages across the edges, it identifies bottlenecks and heavy child nodes before any decisions are made. It gives the agent the "full picture" instantly.
*   **The RL Decoder (Sequential/Topological):** Despite the GCN seeing everything at once, the agent *must* assign tasks to servers one-by-one in strict topological order (parents before children). This is required for two strict physics reasons:
    1.  **Dynamic Server Loads:** If the agent assigned all 20 nodes to the Cloud simultaneously, it would be blind to the fact that the Cloud's queue was filling up. By assigning sequentially, the environment updates the `server_available` timeline after every single decision, forcing the agent to dynamically react to its own past actions and balance the load.
    2.  **Communication Penalties:** The engine cannot calculate the cross-server data transfer penalty for a Child node unless it knows exactly which server the Parent node was assigned to. Sorting topologically guarantees that the Parent is securely locked into a server before the Child is evaluated.

---

## 5. Dataset Separation (Preventing Data Leaks)

A strict separation between training and testing data is enforced to ensure the benchmark measures the agent's **Zero-Shot Adaptability** (generalization to unseen workflows).

*   **Training Data (`main.py`):** The training loop uses `DAGParser` to pull graphs directly from the raw data generation folders (e.g., `data/meta_offloading_n/offload_random20`). The agent practices on these to learn its meta-policy.
*   **The Golden Test Dataset (`benchmark.py`):** The `data/test_dags.json` file is a frozen, immutable snapshot containing an equal mix of 10, 20, 30, 40, and 50-node graphs. The agent *never* sees this exact file during its training loop. It is solely reserved for final evaluation. By testing against varying sizes, the benchmark proves the algorithm didn't just overfit to 20-node graphs, but learned the fundamental physics of offloading.
