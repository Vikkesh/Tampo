---
name: task-offloading-algorithm-benchmarking
description: Guidelines and instructions for implementing, integrating, and evaluating comparative DRL algorithms within the Task Offloading (TO) framework for fair performance benchmarking.
---

# Task Offloading (TO) Benchmarking & Algorithm Integration Skill

You are assisting the user in building a comparative analysis report of multiple Deep Reinforcement Learning (DRL) algorithms against the Task Offloading (TO) framework. The user plans to implement the algorithms **one by one** and will execute the intensive training and evaluation phases in **Google Colab**.

Follow these exact architectural rules and workflows whenever you are asked to implement or benchmark a new algorithm.

## 1. Strict Requirement: No Scratch Implementations

> [!IMPORTANT]
> **CRITICAL RULE:** Do NOT implement algorithms from scratch.
> You MUST include the algorithm strictly based on the reference paper or the source code files provided by the user.

- If the user asks you to implement a baseline (e.g., D3QN, TPTO, ReSACO), but does not provide a specific reference paper or source code repo, **STOP and explicitly ask the user for the reference.**
- The purpose of this benchmark is to accurately reproduce and cite specific papers.
- After successfully implementing a baseline from a provided reference, you MUST automatically update the [Papers referred.md](file:///Papers%20referred.md) file with the full citation of the paper you used.

## 2. The Ground Truth: The Common Environment

To ensure a fair, mathematically rigorous comparison, the underlying physics engine must never change.
- **Rule:** You MUST use TaskOffloadingEnv as the absolute ground truth.
- **Why?** It handles all physical calculations: tracking server loads, calculating transmission/computation delays, and computing energy costs.
- **Rule:** NO algorithm is allowed to calculate its own latency or energy. Every algorithm must interface via calling `env.step(action)` and accept the latency/energy tuple returned by the environment. This ensures 100% fairness in the physics.

## 3. Environment Adapters (Gym Wrappers)

Different algorithms expect different observation spaces. Instead of modifying the core environment, use the Adapter Pattern. When implementing a new algorithm, you must wrap the core environment depending on what the algorithm expects:
- **You can create new gym wrappers for the algorithms based on the environment.**
- **Algorithms using the TAMPO algorithm (e.g., TAMPO with GCN/GAT):** Use the `TAMPOWrapper` (or native environment). It expects complex graph structures (Nodes, Edges, Adjacency Matrix) and multi-objective preferences.
- **Standalone Baselines (e.g., D3QN, SAC, MAPPO):** You must implement and use a `FlatVectorWrapper`. This wrapper takes the DAG output from the environment, flattens all task features (e.g., computation requirements, communication size, memory) into a fixed-size 1D array, and pads it with zeros to a maximum task limit. These algorithms panic if handed a raw graph.
- **Sequence Models (e.g., TPTO, MTD3):** You must implement and use a `SequenceWrapper`. This orders the tasks topologically (parent tasks ordered before child tasks) into a sequence format.

## 4. Algorithm Architecture Setup

Keep the codebase strictly organized into two categories:

### Category 1: Integrated TAMPO Algorithm Tweaks
Algorithms that modify the TAMPO algorithm's internal mechanics (e.g., GNN, PPO objective, Reptile, MAPPO, Lyapunov, FDRL).
- **Location:** Integrate these directly into [tampo.py](file:///algorithms/rl/tampo.py).
- **Toggle:** Expose them as configurable toggles in [default_config.yaml](file:///configs/default_config.yaml) (e.g., `meta_updater: reptile`, `reward_function: lyapunov`).

### Category 2: Standalone Baselines
Competing architectures that do not use the TAMPO algorithm's core mechanics (e.g., D3QN, ReSACO, TPTO, MTD3).
- **Location:** Create a dedicated folder: `algorithms/baselines/`.
- **Structure:** Create a separate file for each (e.g., `d3qn.py`, `resaco.py`). All baselines MUST inherit from a common `BaseAgent` interface to ensure standardized `train()` and `predict()` methods.

## 5. The Workflow: One-by-One Implementation & Colab Execution

The user is implementing these algorithms sequentially and training them off-machine in Google Colab.

### Step 1: The Golden Dataset
Before training, ensure a script [generate_test_dataset.py](file:///utils/generate_test_dataset.py) generates a fixed, immutable dataset of random DAG workflows (e.g., 500 workflows) and saves them to `test_dags.json`. Every algorithm will be evaluated against this exact same file for zero-shot testing.

### Step 2: Implement One-by-One
When the user asks to implement an algorithm:
1. Verify you have the source paper/code (Stop and ask if not).
2. Write the algorithm code (either as a TAMPO algorithm tweak or a Standalone Baseline).
3. Write the necessary Environment Wrapper if it's a baseline.
4. Update the [Papers referred.md](file:///Papers%20referred.md) file with the citation.
5. Provide a clear, isolated training script (or Jupyter Notebook cell format) specifically for that algorithm so the user can easily copy/upload it to Google Colab.

### Step 3: Colab Training & Zero-Shot Evaluation
Since execution happens in Colab:
- Ensure your code is modular so it can be zipped or cloned easily into a Colab environment.
- Create an evaluation script [benchmark.py](file:///benchmark.py) that:
  1. Loads `test_dags.json`.
  2. Disables exploration (greedy/deterministic actions only).
  3. Loops through the tasks for the algorithm and records the exact `Total Delay` and `Total Energy` for every task into a `results.csv`.
- The benchmark script should generate:
  - **Bar Charts:** Average Latency and Average Energy per algorithm.
  - **Scatter Plot (Pareto Front):** Energy (X-axis) vs. Latency (Y-axis) to visually show the trade-offs (Hypervolume) between algorithms like Lyapunov (which focuses on energy) vs standard PPO (which might focus on speed).
  - The plots should be saved in a format that the user can download directly from Colab for their analysis report.

## 6. Project Documentation & README Updates

### Note
**Keep a detailed/ unshortened version of all the changes that are made in that particular lasrge scale update in the folder "dev_logs" and the file "dev_logs/ .md" in complete details. If 2 log files are associated have only a single "dev_log .md" file and merge them.**

Whenever you implement a new algorithm, wrapper, or framework component, you MUST update the global [README.md](file:///README.md) file to explain everything about the project as it evolves in a brief, effecient manner.

Specifically:
- Maintain a dedicated section in [README.md](file:///README.md) detailing **how to switch between the implemented algorithms** and **how to run all of them** (both training in Colab and local benchmarking).
- Document new baselines, configuration toggles, wrappers, and dataset paths.


