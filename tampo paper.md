3 Proposed Methodology
In this chapter, we will discuss various models, architectural diagram and algorithmic
approach of the proposed research work.
3.1 Network, Task, and Energy Consumption Models
3.1.1 Network Models
We consider a MEC system consisting of 𝑁 mobile devices and a single MEC server
located at the base station. The wireless channel is modeled as a time-varying system.
The uplink data rate 𝑅𝑢𝑙 and downlink data rate 𝑅𝑑𝑙 for a device are given by Shannon’s
capacity:
𝑅 = 𝐵log2 1+ 𝑃𝑡𝑥 · ℎ
𝑁0
(3.1)
where 𝐵 is the channel bandwidth, 𝑃𝑡𝑥 is the transmission power, ℎ is the channel gain
(including path loss and fading), and 𝑁0 is the noise power spectral density.
3.1.2 Task Model
Each computation task is represented as a Directed Acyclic Graph (DAG) 𝐺 = (𝑉, 𝐸).
• 𝑉 ={𝑣1,𝑣2,...,𝑣𝑀} is the set of sub-tasks. Each sub-task 𝑣𝑖 is characterized by its
data size 𝐷𝑖 (in bits) and computational workload 𝐶𝑖 (in CPU cycles).
• 𝐸 ⊆𝑉×𝑉 istheset of dependencies. An edge (𝑖, 𝑗) ∈ 𝐸 implies that task 𝑣𝑖 must
be completed before 𝑣𝑗 can start. If 𝑣𝑖 and 𝑣𝑗 are executed on different nodes (e.g.,
one local, one remote), a data transfer of size 𝑑𝑖,𝑗 is required.
3.1.3 Energy Consumption Model
The total energy consumption is the sum of local processing energy and transmission
energy.
8
1. Local Processing Energy: For a task 𝑣𝑖 executed locally, the energy consumption
is modeled as:
𝐸𝑙𝑜𝑐𝑎𝑙,𝑖 = 𝜅 · 𝐶𝑖 · 𝑓2
𝑈𝐸
(3.2)
where 𝜅 is the effective switched capacitance coefficient (dependent on chip archi
tecture, typically 10−26), and 𝑓𝑈𝐸 is the CPU frequency of the user equipment.
2. Transmission Energy For a task 𝑣𝑖 offloaded to the MEC server, the energy cost is
primarily due to wireless transmission:
𝐸𝑜𝑓 𝑓𝑙𝑜𝑎𝑑,𝑖 = 𝑃𝑡𝑥 · 𝐷𝑖𝑛,𝑖
𝑅𝑢𝑙 
+ 𝐷𝑜𝑢𝑡,𝑖
𝑅𝑑𝑙
(3.3)
where 𝐷𝑖𝑛,𝑖 is the input data size and 𝐷𝑜𝑢𝑡,𝑖 is the result data size. Note that the
computation energy at the MEC server is usually not a concern for the mobile user’s
battery life.
3.2 Proposed Architecture: TAMPO
Threshold-AdaptiveMeta-ReinforcementLearningforPareto-OptimalOffloading(TAMPO)
employs a hierarchical architecture to balance the benefits of centralized meta-learning
with the efficiency of distributed execution, as illustrated in Figure 3.1.
3.2.1 Hierarchical Multi-Agent Structure
The system is divided into two layers:
1. Higher Layer (Meta-Learner): Resides on the MEC server or Cloud. It main
tains the global meta-policy parameters 𝜃𝑚𝑒𝑡𝑎. Its goal is to learn a generalized
initialization that can be quickly adapted.
2. Lower Layer (Device Agents): Reside on individual mobile devices. Each agent
𝑘 has a local copy of the policy 𝜃𝑘. Agents execute tasks, collect experiences, and
perform local updates.
3.2.2 Preference-Conditioned Policy Network
To handle multi-objective optimization, we modify the standard Seq2Seq architecture.
• Input: The state 𝑠𝑡 includes the attributes of the current sub-task (workload, data
size) and the status of its predecessors.
10
• Preference Injection: The user preference vector 𝑤 is projected into a high
dimensional embedding space using a dense layer:
ℎ𝑝𝑟𝑒𝑓 = tanh(𝑊𝑝 · 𝑤 + 𝑏𝑝)
(3.4)
• Encoder: A Bidirectional LSTM processes the topological sort of the task DAG.
The preference embedding ℎ𝑝𝑟𝑒𝑓 is added to the initial hidden state or concatenated
with the inputs of the LSTM cells.
• Decoder: An LSTM with an attention mechanism generates the binary offloading
decision 𝑎𝑡. The attention mechanism allows the decoder to focus on relevant parts
of the task graph (e.g., critical path dependencies).
3.2.3 Threshold-Adaptive Communication
To reduce the communication overhead of sending gradients to the Higher Layer, TAMPO
uses a performance-based trigger.
1. Each agent maintains a moving average of its Hypervolume metric (𝐻𝑉𝑎𝑣𝑔), which
measures the quality of the Pareto front it is achieving.
2. Athreshold 𝜏 is defined.
3. Trigger Condition:
• If 𝐻𝑉𝑎𝑣𝑔 < 𝜏: The agent is performing poorly. It sends its accumulated
gradients to the Higher Layer to request a meta-update (help from the global
knowledge).
• If 𝐻𝑉𝑎𝑣𝑔 ≥ 𝜏: The agent is performing well. It continues with local updates
only, saving bandwidth.
3.3 Meta-Training Algorithm
The training process is based on the Model-Agnostic Meta-Learning (MAML) approach,
which essentially teaches the model how to learn new tasks quickly, rather than learning
one fixed solution.
In our case, multiple device agents each face slightly different offloading scenarios. Every
agent first learns on its own local task, and only when its performance falls below a certain
threshold does it communicate its updates to the central controller.
11
The controller then combines these useful updates to improve a shared “meta-policy,”
which is like a common brain that helps every agent start learning faster in the future. This
way, the system avoids unnecessary communication, reduces overhead, and still learns a
policy that adapts efficiently to many types of environments.
Algorithm 1 TAMPO Distributed Training with Threshold Mechanism
1: Initialize meta-policy parameters 𝜃 on Higher Layer
2: Initialize performance threshold 𝜏
3: while not converged do
4:
5:
6:
7:
8:
9:
10:
11:
12:
13:
14:
15:
16:
17:
18:
19:
20:
21:
22:
23:
24:
25:
for each Device Agent 𝑖 in parallel do
Sample task T𝑖 and preference 𝑤𝑖
Local Adaptation:
𝜃′
𝑖 ←𝜃
for 𝑘 = 1 to 𝐾 do
Compute loss 𝐿T𝑖
(𝜃′
𝑖)
Update 𝜃′
𝑖 ← 𝜃′
𝑖 − 𝛼∇𝜃′
𝑖
𝐿T𝑖
(𝜃′
𝑖)
end for
Collect trajectory using 𝜋𝜃′
𝑖
Calculate Hypervolume 𝐻𝑉𝑖 and update moving average ¯
if ¯
𝐻𝑉𝑖 < 𝜏 then
// Trigger Condition Met: Participate in Meta-Update
Send gradients ∇𝜃𝐿T𝑖
(𝜃′
𝑖) to Higher Layer
else
// Performance Satisfactory: Local Update Only
(No communication with Higher Layer)
end if
end for
Meta-Update (Higher Layer):
𝐻𝑉𝑖
Collect gradients from subset of agents S where trigger met
Update 𝜃 ← 𝜃 − 𝛽 𝑖∈S∇𝜃𝐿T𝑖
(𝜃′
𝑖)
Broadcast updated 𝜃 to agents in S
26: end while
12
3.4 Performance Metrics
3.4.1 Weighted Cost Function
Let 𝐴 = {𝑎1,𝑎2,...,𝑎𝑀} be the offloading decision vector, where 𝑎𝑖 ∈ {0,1}. 𝑎𝑖 = 0
denotes local execution, and 𝑎𝑖 = 1 denotes offloading. The problem is to find a policy 𝜋
that minimizes the weighted cost function:
𝐽(𝐴) = 𝑤𝑑𝑒𝑙𝑎𝑦 ·𝑇𝑡𝑜𝑡𝑎𝑙(𝐴) + 𝑤𝑒𝑛𝑒𝑟𝑔𝑦 · 𝐸𝑡𝑜𝑡𝑎𝑙(𝐴)
(3.5)
subject to the dependency constraints defined by 𝐺. Here, 𝑤 = [𝑤𝑑𝑒𝑙𝑎𝑦,𝑤𝑒𝑛𝑒𝑟𝑔𝑦] is the
user preference vector, where 𝑤𝑑𝑒𝑙𝑎𝑦 + 𝑤𝑒𝑛𝑒𝑟𝑔𝑦 = 1. Instead of solving for a fixed 𝑤,
TAMPO aims to learn a policy 𝜋(𝑎|𝑠,𝑤) that is optimal for any given 𝑤.
3.4.2 Hypervolume Calculation
TheHypervolume indicator is used as the primary metric for multi-objective performance.
For a set of solutions 𝑆 in the objective space (Delay, Energy), the hypervolume 𝐻𝑉(𝑆) is
the measure of the region dominated by 𝑆 and bounded by a reference point 𝑟𝑟𝑒𝑓.
𝐻𝑉(𝑆) = Λ
{𝑧|𝑠 ≺ 𝑧 ≺ 𝑟𝑟𝑒𝑓}
𝑠∈𝑆
(3.6)
where Λ is the Lebesgue measure (area in 2D). Maximizing hypervolume is equivalent
to finding a set of solutions that are both close to the true Pareto front (convergence) and
diverse (spread).
13

