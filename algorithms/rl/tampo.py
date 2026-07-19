import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.distributions import Categorical
from torch.func import functional_call
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import to_dense_batch
from collections import deque
import copy
import os
import time
import random

def _pad_graph_batch(
    task_features_list: List[np.ndarray],
    adjacency_list: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-size graphs into a dense batch plus node mask."""
    batch_size = len(task_features_list)
    max_nodes = max(features.shape[0] for features in task_features_list)
    feature_dim = task_features_list[0].shape[1]

    task_batch = np.zeros((batch_size, max_nodes, feature_dim), dtype=np.float32)
    adj_batch = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_nodes), dtype=np.float32)

    for idx, features in enumerate(task_features_list):
        num_nodes = features.shape[0]
        task_batch[idx, :num_nodes] = features
        node_mask[idx, :num_nodes] = 1.0

        if adjacency_list is not None and adjacency_list[idx] is not None:
            adj = np.asarray(adjacency_list[idx], dtype=np.float32)
        else:
            adj = np.eye(num_nodes, dtype=np.float32)
        adj_batch[idx, :num_nodes, :num_nodes] = adj

    return task_batch, adj_batch, node_mask

def _build_pyg_batch(
    task_features_list: List[np.ndarray],
    adjacency_list: Optional[List[np.ndarray]] = None
) -> Batch:
    """Build a PyG batch using edge_index, which is the standard GCN input format."""
    data_list = []

    for idx, features in enumerate(task_features_list):
        x = torch.as_tensor(features, dtype=torch.float32)
        num_nodes = x.size(0)

        edge_index = torch.empty((2, 0), dtype=torch.long)
        if adjacency_list is not None and adjacency_list[idx] is not None:
            adj = torch.as_tensor(adjacency_list[idx], dtype=torch.float32)
            edge_positions = torch.nonzero(adj > 0, as_tuple=False)
            if edge_positions.numel() > 0:
                edge_index = edge_positions.t().contiguous().long()
                # Removed to_undirected to keep directed edges as per GDRL Feature.py

        data_list.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes))

    return Batch.from_data_list(data_list)


def action_name(action: int) -> str:
    """Human-readable name for an offloading action index."""
    if action == 0:
        return "local"
    if action == 1:
        return "cloud"
    return f"edge{action - 2}"


class DAGEncoder(nn.Module):
    """
    Enhanced encoder for DAG structure using LSTM or GCN approach.
    """
    
    def __init__(
        self,
        task_feature_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        encoder_type: str = 'lstm',
        server_feature_dim: int = 20,
        num_gat_heads: int = 4,
        gat_hidden_dim: int = 16,
        gat_add_self_loops: bool = True,
        gnn_hidden_dim: int = 16
    ):
        super(DAGEncoder, self).__init__()

        self.encoder_type = encoder_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.1)

        self.task_embedding = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        if self.encoder_type in {'lstm', 'both'}:
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        # ── GNN node-embedding width ───────────────────────────────────────────────
        # Each directional stream emits `hidden_dim` per node; forward ⊕ backward gives
        # `hidden_dim * 2`, matching the decoder's attention embed_dim and the LSTM
        # encoder's BiLSTM output width. This keeps GCN / GAT / LSTM interchangeable.
        #
        # GDRL's Feature.py ends with GCNConv(16, 1) and concatenates the resulting
        # [num_nodes] vector with the scalar state — it keeps one value PER NODE. The
        # previous code here collapsed that to `x.mean()`, a single scalar for the whole
        # graph, then broadcast it to every node slot. Attention then ran over N
        # identical keys, so no per-node information reached the decoder at all. We keep
        # per-node outputs (as GDRL does) and widen the final conv from 1 to hidden_dim
        # so the decoder has something to attend over.
        if self.encoder_type == 'gcn':
            self.gnn1_fwd = GCNConv(task_feature_dim, gnn_hidden_dim)
            self.gnn2_fwd = GCNConv(gnn_hidden_dim, hidden_dim)

            self.gnn1_bwd = GCNConv(task_feature_dim, gnn_hidden_dim)
            self.gnn2_bwd = GCNConv(gnn_hidden_dim, hidden_dim)
        elif self.encoder_type == 'gat':
            # Bi-GATv2 — drop-in for Bi-GCN, identical skeleton, only the conv differs.
            hidden_per_head = gat_hidden_dim // num_gat_heads
            self.gat1_fwd = GATv2Conv(
                task_feature_dim, hidden_per_head, heads=num_gat_heads,
                concat=True, dropout=0.1, add_self_loops=gat_add_self_loops
            )
            self.gat2_fwd = GATv2Conv(
                gat_hidden_dim, hidden_dim, heads=1,
                concat=False, dropout=0.1, add_self_loops=gat_add_self_loops
            )
            self.gat1_bwd = GATv2Conv(
                task_feature_dim, hidden_per_head, heads=num_gat_heads,
                concat=True, dropout=0.1, add_self_loops=gat_add_self_loops
            )
            self.gat2_bwd = GATv2Conv(
                gat_hidden_dim, hidden_dim, heads=1,
                concat=False, dropout=0.1, add_self_loops=gat_add_self_loops
            )

        if self.encoder_type in {'gcn', 'gat'}:
            # Graph-level context: mean readout over node embeddings, concatenated with
            # the server state, through GDRL's two-block FNN head. The readout replaces
            # GDRL's fixed-size concatenation, which assumes a constant node count (35).
            fnn_in = server_feature_dim + hidden_dim * 2
            self.fnn1 = nn.Sequential(
                nn.Linear(fnn_in, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.fnn_out = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_dim * 2),
                nn.ReLU()
            )
            self.node_norm = nn.LayerNorm(hidden_dim * 2)
        elif self.encoder_type == 'both':
            # Legacy both branch, kept for structural integrity if used elsewhere
            self.gcn_layers = nn.ModuleList([
                GCNConv(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim * 2)
                for i in range(num_layers)
            ])
        if self.encoder_type not in {'lstm', 'gcn', 'gat', 'both'}:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
        
        if self.encoder_type == 'both':
            self.hybrid_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 2)
            )
        
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def _masked_max_pool(self, x, node_mask):
        """Max pool only across valid nodes."""
        if node_mask is None:
            return x.max(dim=1)[0]
        masked_x = x.masked_fill(~node_mask.unsqueeze(-1), float('-inf'))
        pooled = masked_x.max(dim=1)[0]
        pooled[torch.isinf(pooled)] = 0.0
        return pooled

    def _apply_lstm(self, x, node_mask=None):
        """BiLSTM sequence encoder over topologically ordered node features."""
        if node_mask is None:
            # CuDNN LSTM kernels do not support the double backward needed by MAML.
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = self.lstm(x)
            return out

        lengths = node_mask.sum(dim=1).clamp(min=1).to(dtype=torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        with torch.backends.cudnn.flags(enabled=False):
            packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        return out

    def _apply_gcn(self, graph_batch, max_num_nodes: int):
        """GDRL GCN forward pass (Cai et al. 2025, Feature.py)."""
        if graph_batch is None:
            raise ValueError("graph_batch is required for GCN-based encoders")

        if self.encoder_type == 'both':
            x = self.task_embedding(graph_batch.x)
            edge_index = graph_batch.edge_index
            for i, layer in enumerate(self.gcn_layers):
                x = layer(x, edge_index)
                if i < len(self.gcn_layers) - 1:
                    x = self.dropout(torch.relu(x))
            dense_x, node_mask = to_dense_batch(
                x,
                graph_batch.batch,
                max_num_nodes=max_num_nodes
            )
            return dense_x, node_mask

        # GDRL 'gcn' path — bidirectional, per-node outputs (no pooling)
        return self._apply_bidirectional_gnn(
            graph_batch, max_num_nodes,
            self.gnn1_fwd, self.gnn2_fwd, self.gnn1_bwd, self.gnn2_bwd
        )

    def _apply_gat(self, graph_batch, max_num_nodes: int):
        """Bi-GATv2 forward pass — drop-in replacement for Bi-GCN (GAPO, Zhang et al. 2025)."""
        if graph_batch is None:
            raise ValueError("graph_batch is required for GAT-based encoders")

        return self._apply_bidirectional_gnn(
            graph_batch, max_num_nodes,
            self.gat1_fwd, self.gat2_fwd, self.gat1_bwd, self.gat2_bwd
        )

    def _apply_bidirectional_gnn(self, graph_batch, max_num_nodes, l1_fwd, l2_fwd, l1_bwd, l2_bwd):
        """
        Shared Bi-GNN body for the 'gcn' and 'gat' paths — the ONLY difference between
        them is the conv operator passed in, which is what makes the GCN-vs-GAT
        comparison a controlled experiment.

        Runs two streams over the whole PyG batch at once (the previous per-graph Python
        loop was correct but O(batch_size) sequential conv calls), then scatters back to
        a dense [B, N_max, hidden*2] tensor.

        Returns:
            dense_nodes: [B, N_max, hidden_dim * 2] per-node embeddings
            node_mask:   [B, N_max] True where a real node lives
        """
        import torch.nn.functional as F

        x = graph_batch.x
        edge_fwd = graph_batch.edge_index          # parent → child (DAG direction)
        edge_bwd = edge_fwd.flip(0)                # child → parent

        h_fwd = F.dropout(F.relu(l1_fwd(x, edge_fwd)), training=self.training)
        h_fwd = l2_fwd(h_fwd, edge_fwd)            # [total_nodes, hidden_dim]

        h_bwd = F.dropout(F.relu(l1_bwd(x, edge_bwd)), training=self.training)
        h_bwd = l2_bwd(h_bwd, edge_bwd)            # [total_nodes, hidden_dim]

        h = torch.cat([h_fwd, h_bwd], dim=-1)      # [total_nodes, hidden_dim * 2]

        dense_nodes, node_mask = to_dense_batch(
            h, graph_batch.batch, max_num_nodes=max_num_nodes
        )
        return dense_nodes, node_mask

    def forward(self, task_features, adjacency_matrix=None, node_mask=None, graph_batch=None, server_features=None):
        """
        Args:
            task_features: [batch, num_tasks, feature_dim]
            adjacency_matrix: [batch, num_tasks, num_tasks] optional
            node_mask: [batch, num_tasks] optional
            graph_batch: PyG Batch for GCN-based encoders
            server_features: [batch, server_feature_dim] optional, required for 'gcn'
        
        Returns:
            encoded_tasks: [batch, num_tasks, hidden_dim * 2]
            context: [batch, hidden_dim * 2]
        """
        if self.encoder_type in {'gcn', 'gat'}:
            if server_features is None:
                raise ValueError("server_features is required for GCN/GAT path")
            apply_gnn = self._apply_gcn if self.encoder_type == 'gcn' else self._apply_gat
            encoded_tasks, gnn_mask = apply_gnn(graph_batch, task_features.size(1))

            if node_mask is None:
                node_mask = gnn_mask
            valid = node_mask.unsqueeze(-1).float()
            encoded_tasks = self.node_norm(encoded_tasks) * valid

            # Mean readout over real nodes only (padding must not dilute the mean).
            readout = (encoded_tasks * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

            combined = torch.cat([server_features, readout], dim=-1)
            context = self.fnn_out(self.fnn1(combined))          # [batch, hidden_dim*2]
            return encoded_tasks, context

        embedded = self.task_embedding(task_features)
        if node_mask is not None:
            embedded = embedded * node_mask.unsqueeze(-1).float()

        encoded_streams = []
        if self.encoder_type in {'lstm', 'both'}:
            encoded_streams.append(self._apply_lstm(embedded, node_mask))
        if self.encoder_type == 'both':
            gcn_out, gcn_mask = self._apply_gcn(graph_batch, task_features.size(1))
            if node_mask is None:
                node_mask = gcn_mask
            encoded_streams.append(gcn_out)

        if self.encoder_type == 'both':
            out = self.hybrid_fusion(torch.cat(encoded_streams, dim=-1))
        else:
            out = encoded_streams[0]

        key_padding_mask = None
        if node_mask is not None:
            key_padding_mask = ~node_mask.bool()

        attn_out, _ = self.graph_attention(
            out, out, out, key_padding_mask=key_padding_mask
        )
        encoded = self.layer_norm(out + self.dropout(attn_out))
        if node_mask is not None:
            encoded = encoded * node_mask.unsqueeze(-1).float()

        context = self._masked_max_pool(encoded, node_mask.bool() if node_mask is not None else None)
        
        return encoded, context

class PreferenceConditionedDecoder(nn.Module):
    """
    Decoder that makes sequential offloading decisions conditioned on user preferences
    """
    
    def __init__(self, hidden_dim: int, num_resources: int, preference_dim: int = 2):
        super(PreferenceConditionedDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_resources = num_resources
        
        # Preference encoding
        self.preference_encoder = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # RC#13: Project full context [hidden_dim*2] → [hidden_dim] so both halves
        # of the bidirectional encoding initialise the decoder LSTM state.
        self.context_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Embedding of the node currently being scheduled, indexed out of encoded_tasks.
        # Without this the decoder has no idea WHICH node it is placing: every input it
        # receives (graph, servers, preference) is constant across an episode, so it
        # emitted identical logits at every step and placed the whole DAG on one server.
        self.current_node_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # LSTM decoder — input is [h_t, attn_context, pref, current_node]
        self.lstm_cell = nn.LSTMCell(hidden_dim * 4, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            batch_first=True
        )
        
        # Decision head with separate objective prediction
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_resources)
        )

    def forward(self, encoded_tasks, context, preference, num_steps,
                node_mask=None, current_node_idx=None):
        """
        Sequential decoding with preference conditioning.

        Args:
            current_node_idx: [B] index of the node being scheduled this step. When None,
                falls back to node 0 — only correct for single-node/independent tasks.
        """
        batch_size = encoded_tasks.size(0)
        key_padding_mask = None
        if node_mask is not None:
            key_padding_mask = ~node_mask.bool()

        # Encode preference
        pref_encoded = self.preference_encoder(preference)

        # Pull out the embedding of the node actually being placed (pointer-style
        # indexing, cf. Vinyals et al. 2015). This is what makes the policy a function
        # of the current node rather than of the graph alone.
        if current_node_idx is None:
            current_node_idx = torch.zeros(batch_size, dtype=torch.long, device=encoded_tasks.device)
        current_node_idx = current_node_idx.clamp(0, encoded_tasks.size(1) - 1)
        cur_node = encoded_tasks[torch.arange(batch_size, device=encoded_tasks.device), current_node_idx]
        cur_encoded = self.current_node_projection(cur_node)      # [B, hidden_dim]

        # Initialize LSTM state
        # RC#13: Use full context via projection (was discarding second half with [:hidden_dim])
        h_t = self.context_projection(context)  # [B, hidden_dim]
        c_t = torch.zeros_like(h_t)

        action_logits_list = []

        for step in range(num_steps):
            # Attention query — conditioned on the current node so the decoder can look
            # up that node's parents/children among the encoded nodes.
            query = torch.cat([cur_encoded, pref_encoded], dim=-1).unsqueeze(1)

            attn_out, attn_weights = self.attention(
                query, encoded_tasks, encoded_tasks, key_padding_mask=key_padding_mask
            )

            attn_context = attn_out.squeeze(1)

            # LSTM update
            lstm_input = torch.cat([
                h_t,
                attn_context[:, :self.hidden_dim],
                pref_encoded,
                cur_encoded
            ], dim=-1)

            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))

            # Decision
            decision_input = torch.cat([
                h_t,
                attn_context[:, :self.hidden_dim],
                pref_encoded,
                cur_encoded
            ], dim=-1)

            logits = self.decision_head(decision_input)
            action_logits_list.append(logits)

        action_logits = torch.stack(action_logits_list, dim=1)
        
        return action_logits

class MetaPolicyNetwork(nn.Module):
    """
    Meta-policy network for TAM-PO
    """
    
    def __init__(
        self,
        task_feature_dim: int,
        server_feature_dim: int,
        num_resources: int,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        encoder_type: str = 'lstm',
        num_gat_heads: int = 4,
        gat_hidden_dim: int = 16,
        gat_add_self_loops: bool = True,
        gnn_hidden_dim: int = 16
    ):
        super(MetaPolicyNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.encoder = DAGEncoder(
            task_feature_dim, hidden_dim, num_encoder_layers, encoder_type=encoder_type,
            server_feature_dim=server_feature_dim,
            num_gat_heads=num_gat_heads,
            gat_hidden_dim=gat_hidden_dim,
            gat_add_self_loops=gat_add_self_loops,
            gnn_hidden_dim=gnn_hidden_dim
        )
        
        self.decoder = PreferenceConditionedDecoder(
            hidden_dim, num_resources, preference_dim=2
        )
        
        # Server state encoder
        self.server_encoder = nn.Sequential(
            nn.Linear(server_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2)
        )
        
    def forward(
        self,
        task_features,
        server_features,
        preference,
        num_decisions: int = 1,
        adjacency=None,
        node_mask=None,
        graph_batch=None,
        current_node_idx=None
    ):
        """
        Forward pass

        Args:
            current_node_idx: [B] LongTensor — which node each batch element is placing.
        """
        encoded_tasks, context = self.encoder(
            task_features,
            adjacency,
            node_mask,
            graph_batch=graph_batch,
            server_features=server_features
        )

        server_encoded = self.server_encoder(server_features)
        combined_context = context + server_encoded
        action_logits = self.decoder(
            encoded_tasks, combined_context, preference, num_decisions,
            node_mask=node_mask, current_node_idx=current_node_idx
        )

        return action_logits

class MultiObjectiveValueNetwork(nn.Module):
    """
    Separate value functions for delay and energy objectives.
    RC#5: Accepts server_features in addition to the flat obs state so it has
    the same representational information as the policy network.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, server_feature_dim: int = 20):
        super(MultiObjectiveValueNetwork, self).__init__()
        
        # Input: flat obs (36) + server features (20) + preference (2)
        input_dim = state_dim + server_feature_dim + 2
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Separate value heads for each objective
        self.delay_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state, preference, server_features=None):
        """
        Returns: [batch, 2] - [V_delay, V_energy]
        Args:
            state:          [B, state_dim] — flat environment observation
            preference:     [B, 2]         — w_delay, w_energy
            server_features:[B, 20]        — structured server features (RC#5)
        """
        if server_features is not None:
            x = torch.cat([state, server_features, preference], dim=-1)
        else:
            # Fallback: pad with zeros so dimension is always correct
            pad = torch.zeros(state.size(0), 20, device=state.device)
            x = torch.cat([state, pad, preference], dim=-1)
        features = self.shared(x)
        
        delay_val = self.delay_head(features)
        energy_val = self.energy_head(features)
        
        return torch.cat([delay_val, energy_val], dim=-1)

class HypervolumeCalculator:
    """
    Hypervolume indicator calculation for Pareto front quality
    """
    
    def __init__(self, reference_point: np.ndarray):
        self.reference_point = reference_point
    
    def calculate(self, solutions: np.ndarray) -> float:
        """
        Calculate 2D hypervolume using WFG algorithm approximation
        """
        if len(solutions) == 0:
            return 0.0
        
        # Get Pareto front
        pareto_front = self._get_pareto_front(solutions)
        
        if len(pareto_front) == 0:
            return 0.0
        
        # Sort by first objective
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
        
        hv = 0.0
        prev_x = 0.0
        
        for x, y in sorted_front:
            if x >= self.reference_point[0] or y >= self.reference_point[1]:
                continue
            
            width = x - prev_x
            height = self.reference_point[1] - y
            
            if width > 0 and height > 0:
                hv += width * height
            
            prev_x = x
        
        # Final rectangle
        if len(sorted_front) > 0:
            last_x, last_y = sorted_front[-1]
            if last_x < self.reference_point[0] and last_y < self.reference_point[1]:
                width = self.reference_point[0] - last_x
                height = self.reference_point[1] - last_y
                if width > 0 and height > 0:
                    hv += width * height
        
        return hv
    
    def _get_pareto_front(self, solutions: np.ndarray) -> np.ndarray:
        """Extract non-dominated solutions (Pareto front)"""
        is_pareto = np.ones(len(solutions), dtype=bool)
        
        for i in range(len(solutions)):
            if not is_pareto[i]:
                continue
            for j in range(i + 1, len(solutions)):
                if not is_pareto[j]:
                    continue
                # Check dominance
                if np.all(solutions[j] <= solutions[i]) and np.any(solutions[j] < solutions[i]):
                    is_pareto[i] = False
                    break
                elif np.all(solutions[i] <= solutions[j]) and np.any(solutions[i] < solutions[j]):
                    is_pareto[j] = False
        
        return solutions[is_pareto]


class _LossCallCounter:
    """Module-level counter for throttled diagnostic logging in _compute_loss_with_params."""
    _n: int = 0


class LowerLayerAgent:

    """
    Lower-layer device agent with local adaptation (MAML inner loop)
    """
    
    def __init__(
        self,
        agent_id: int,
        meta_policy: MetaPolicyNetwork,
        value_network: MultiObjectiveValueNetwork,
        config: Dict,
        device: torch.device
    ):
        self.agent_id = agent_id
        self.device = device
        self.config = config
        
        # Store references to meta parameters (NOT deep copy)
        # This is critical for gradient flow
        self.meta_policy = meta_policy
        self.meta_value = value_network
        
        # Local adapted parameters (these will be different from meta)
        self.adapted_params = None
        
        # Local optimizer (SGD for fast adaptation as per MAML)
        self.inner_lr = config.get('inner_lr', 0.01)

        # PPO surrogate clipping. The inner loop takes `inner_steps` gradient steps on a
        # single batch collected under the meta-policy, so after the first step the data
        # is off-policy. A clipped importance ratio bounds how far each step can move the
        # policy on stale data; without it those steps are uncorrected vanilla PG.
        self.clip_eps = config.get('ppo_clip_eps', 0.2)
        self.value_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)

        # Hypervolume tracking — the threshold-adaptive communication state of the TAMPO
        # paper (§3.2.3): each agent tracks a moving average of its hypervolume and
        # requests a meta-update only when it drops below hv_threshold. This block was
        # accidentally stranded after a `return` in _old_log_probs when the PPO methods
        # were inserted mid-class, so none of these attributes ever existed and every
        # call into the trigger path raised AttributeError.
        self.hv_threshold = config.get('hypervolume_threshold', 0.5)
        self.hv_window = config.get('moving_average_window', 50)
        self.hv_calculator = HypervolumeCalculator(
            # RC#9: Reference point updated to [2.0, 2.0].
            # After RC#1/#4 fix, mo_return values are normalised improvements in [-1, 1].
            # A reference of 2.0 is safely above the maximum possible value of 1.0.
            reference_point=np.array([2.0, 2.0])
        )

        self.performance_buffer = deque(maxlen=200)
        self.hv_history = deque(maxlen=self.hv_window)

        # Communication trigger
        self.update_needed = False
        self.update_package = None

    def _ppo_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Clipped surrogate objective (Schulman et al., 2017)."""
        ratio = torch.exp(log_probs - old_log_probs)
        surr_unclipped = ratio * advantages
        surr_clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        return -torch.min(surr_unclipped, surr_clipped).mean()

    @staticmethod
    def _old_log_probs(batch: List[Dict], device: torch.device) -> torch.Tensor:
        """
        Behaviour-policy log-probs for the batch.

        Falls back to the current policy's log-probs (ratio == 1, i.e. plain policy
        gradient) for experiences collected before `old_log_prob` was recorded, so
        older checkpoints and replay buffers keep working.
        """
        if 'old_log_prob' not in batch[0]:
            return None
        return torch.FloatTensor(
            np.array([exp['old_log_prob'] for exp in batch], dtype=np.float32)
        ).to(device)

    def _prepare_policy_batch(
        self,
        batch: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Batch]:
        """Pad graph batches for dense paths and build a PyG batch for GCN paths."""
        task_features_list = [
            np.asarray(exp['task_features'], dtype=np.float32) for exp in batch
        ]
        adjacency_list = [
            np.asarray(exp.get('adj_matrix'), dtype=np.float32)
            if exp.get('adj_matrix') is not None else None
            for exp in batch
        ]
        task_features_np, adjacency_np, node_mask_np = _pad_graph_batch(
            task_features_list,
            adjacency_list
        )
        task_features = torch.FloatTensor(task_features_np).to(self.device)
        adjacency = torch.FloatTensor(adjacency_np).to(self.device)
        node_mask = torch.BoolTensor(node_mask_np.astype(bool)).to(self.device)
        graph_batch = _build_pyg_batch(task_features_list, adjacency_list).to(self.device)

        server_features = torch.FloatTensor(
            np.array([exp['server_features'] for exp in batch], dtype=np.float32)
        ).to(self.device)

        current_node_idx = torch.LongTensor(
            [int(exp.get('current_node_idx', 0)) for exp in batch]
        ).to(self.device)

        return task_features, server_features, adjacency, node_mask, graph_batch, current_node_idx

    def _forward_with_params(
        self,
        params_dict,
        task_features,
        server_features,
        preference,
        num_decisions: int = 1,
        adjacency=None,
        node_mask=None,
        graph_batch=None,
        current_node_idx=None
    ):
        """
        Forward pass using specific parameters (for functional gradient computation)
        This keeps the full inner-loop update chain differentiable for MAML.
        """
        buffer_dict = dict(self.meta_policy.named_buffers())

        output = functional_call(
            self.meta_policy,
            (params_dict, buffer_dict),
            (task_features, server_features, preference),
            {
                'num_decisions': num_decisions,
                'adjacency': adjacency,
                'node_mask': node_mask,
                'graph_batch': graph_batch,
                'current_node_idx': current_node_idx
            }
        )

        return output
        
    def inner_loop_update(self, experiences: List[Dict], num_steps: int, create_graph: bool = False):
        """
        Fast adaptation using local experiences (MAML inner loop)
        Based on MRLCO's approach - uses functional gradients
        
        Args:
            experiences: List of experience dictionaries
            num_steps: Number of gradient steps
            create_graph: Whether to create computational graph (needed for meta-learning)
        
        Returns:
            Adapted parameters dictionary
        """
        if len(experiences) == 0:
            return {}
        
        # RC#7: Disable dropout during inner-loop adaptation.
        # Stochastic dropout masks during create_graph=True produce inconsistent
        # second-order gradients across inner steps, destabilising MAML.
        # Restore the PREVIOUS mode on exit rather than forcing train(): meta_update
        # holds the policy in eval() across the whole task loop so the outer test-loss
        # forward sees the same dropout-free regime as the inner loop and as the
        # old_log_prob recorded at collection time. Forcing train() here re-enabled
        # dropout for the outer loss, injecting noise into the PPO importance ratio
        # (measured: same batch, two outer-loss evals differed 0.338 vs 0.276).
        was_training = self.meta_policy.training
        self.meta_policy.eval()
        
        # Start with meta-policy parameters
        adapted_params = {name: param.clone() for name, param in self.meta_policy.named_parameters()}
        
        losses = []
        
        for step in range(num_steps):
            # Sample mini-batch
            batch_size = min(32, len(experiences))
            if len(experiences) > batch_size:
                batch_indices = np.random.choice(len(experiences), batch_size, replace=False)
                batch = [experiences[i] for i in batch_indices]
            else:
                batch = experiences
            
            # Compute loss with current adapted parameters
            loss = self._compute_loss_with_params(adapted_params, batch)
            losses.append(loss.item() if not create_graph else loss)
            
            # Compute gradients w.r.t. adapted parameters
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph,
                retain_graph=create_graph or (step < num_steps - 1),
                allow_unused=True
            )
            
            # Update adapted parameters using gradient descent
            # CRITICAL: We create new tensors to maintain the computation graph
            new_adapted_params = {}
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    if create_graph:
                        # Keep in computation graph for meta-learning
                        new_adapted_params[name] = param - self.inner_lr * grad
                    else:
                        # Detach for normal training
                        new_adapted_params[name] = (param - self.inner_lr * grad).detach()
                else:
                    new_adapted_params[name] = param
            
            adapted_params = new_adapted_params
        
        self.adapted_params = adapted_params
        # RC#7: Restore the caller's mode (see note at the top of this method).
        if was_training:
            self.meta_policy.train()
        return adapted_params
    
    def _compute_loss_with_params(self, params_dict: Dict, batch: List[Dict]) -> torch.Tensor:
        """
        Compute loss using specific parameters (functional style)
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        (task_features, server_features, adjacency, node_mask,
         graph_batch, current_node_idx) = self._prepare_policy_batch(batch)

        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        preferences = torch.FloatTensor(np.array([exp['preference'] for exp in batch])).to(self.device)
        mo_returns = torch.FloatTensor(np.array([exp['mo_return'] for exp in batch])).to(self.device)

        # Forward pass with adapted parameters
        action_logits = self._forward_with_params(
            params_dict,
            task_features,
            server_features,
            preferences,
            num_decisions=1,
            adjacency=adjacency,
            node_mask=node_mask,
            graph_batch=graph_batch,
            current_node_idx=current_node_idx
        )
        
        # Policy loss
        logits = action_logits[:, 0, :]
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Value prediction (RC#5: pass server_features)
        server_feat_batch = torch.FloatTensor(
            np.array([exp['server_features'] for exp in batch], dtype=np.float32)
        ).to(self.device)
        values = self.meta_value(states, preferences, server_features=server_feat_batch)
        
        # Multi-objective advantages
        advantages = mo_returns - values.detach()
        weighted_advantages = (advantages * preferences).sum(dim=-1)
        
        # RC#8: Advantage normalisation — zero-mean, unit-variance per mini-batch
        adv_std = weighted_advantages.std() + 1e-8
        adv_mean = weighted_advantages.mean()
        weighted_advantages = (weighted_advantages - adv_mean) / adv_std

        # Combined loss
        # NOTE: .mean(dim=-1) not .sum(dim=-1) for value_loss.
        # mo_returns / values are shape (batch, 2). .sum() over dim=-1 multiplies
        # the MSE by the number of objectives, ballooning value_loss to dominate
        # the gradient and causing divergence. .mean() keeps both objectives
        # equally weighted and in the same scale as policy_loss.
        old_log_probs = self._old_log_probs(batch, self.device)
        if old_log_probs is None:
            policy_loss = -(log_probs * weighted_advantages).mean()
        else:
            policy_loss = self._ppo_policy_loss(log_probs, old_log_probs, weighted_advantages)
        value_loss   = ((mo_returns - values) ** 2).mean(dim=-1).mean()
        entropy_loss = -entropy.mean()

        # ── Diagnostic: log component breakdown every 50 loss computations ──────────
        # Reveals which term dominates and whether returns are in a sane range.
        # Guarded by a module-level counter so it doesn't spam on every inner step.
        _LossCallCounter._n = getattr(_LossCallCounter, '_n', 0) + 1
        if _LossCallCounter._n % 50 == 1:
            avg_ret_d = mo_returns[:, 0].abs().mean().item()
            avg_ret_e = mo_returns[:, 1].abs().mean().item()
            print(
                f"    [diag] returns |G_d|={avg_ret_d:.3f}  |G_e|={avg_ret_e:.3f}"
                f"  policy={policy_loss.item():.4f}"
                f"  value={value_loss.item():.4f}"
                f"  entropy={entropy_loss.item():.4f}"
            )

        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        return total_loss

    def _compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """
        Compute combined policy and value loss (using current meta-policy)
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        (task_features, server_features, adjacency, node_mask,
         graph_batch, current_node_idx) = self._prepare_policy_batch(batch)

        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        preferences = torch.FloatTensor(np.array([exp['preference'] for exp in batch])).to(self.device)
        mo_returns = torch.FloatTensor(np.array([exp['mo_return'] for exp in batch])).to(self.device)

        # Forward pass
        action_logits = self.meta_policy(
            task_features,
            server_features,
            preferences,
            num_decisions=1,
            adjacency=adjacency,
            node_mask=node_mask,
            graph_batch=graph_batch,
            current_node_idx=current_node_idx
        )
        
        # Policy loss
        logits = action_logits[:, 0, :]
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Value prediction (RC#5: pass server_features)
        server_feat_batch = torch.FloatTensor(
            np.array([exp['server_features'] for exp in batch], dtype=np.float32)
        ).to(self.device)
        values = self.meta_value(states, preferences, server_features=server_feat_batch)
        
        # Multi-objective advantages
        advantages = mo_returns - values.detach()
        weighted_advantages = (advantages * preferences).sum(dim=-1)
        
        # RC#8: Advantage normalisation — zero-mean, unit-variance per mini-batch
        adv_std = weighted_advantages.std() + 1e-8
        adv_mean = weighted_advantages.mean()
        weighted_advantages = (weighted_advantages - adv_mean) / adv_std

        # Combined loss — same .mean(dim=-1) fix as _compute_loss_with_params
        old_log_probs = self._old_log_probs(batch, self.device)
        if old_log_probs is None:
            policy_loss = -(log_probs * weighted_advantages).mean()
        else:
            policy_loss = self._ppo_policy_loss(log_probs, old_log_probs, weighted_advantages)
        value_loss   = ((mo_returns - values) ** 2).mean(dim=-1).mean()
        entropy_loss = -entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        return total_loss
    
    def update_performance(self, delay: float, energy: float):
        """
        Update performance buffer using normalised improvement values.
        RC#9: Now normalises to the same [-1,1] scale as mo_return so HV is meaningful.
        """
        # Values are normalised improvements (higher = better) after RC#1.
        # HypervolumeCalculator uses a MINIMISATION convention (its Pareto filter keeps
        # points with smaller coordinates, and HV grows as points move below the
        # reference), so store the NEGATED improvements as costs in [-1, 1]. Without the
        # negation the semantics invert: a uniformly poor agent (improvement -0.9)
        # scored HV ~8.4 against reference [2,2] while a good one scored ~1.2, so the
        # paper's "HV_avg < tau -> request meta-update" trigger could never fire for
        # the agents that actually needed help.
        norm_delay_cost = -float(np.clip(delay, -1.0, 1.0))
        norm_energy_cost = -float(np.clip(energy, -1.0, 1.0))

        self.performance_buffer.append([norm_delay_cost, norm_energy_cost])
        
        # Calculate hypervolume
        if len(self.performance_buffer) >= 20:
            solutions = np.array(list(self.performance_buffer))
            hv = self.hv_calculator.calculate(solutions)
            self.hv_history.append(hv)
            
            # Check threshold
            if len(self.hv_history) >= self.hv_window:
                moving_avg = np.mean(list(self.hv_history))
                
                if moving_avg < self.hv_threshold:
                    self.update_needed = True
                    self._prepare_update_package()
    
    def _prepare_update_package(self):
        """Prepare update package for meta-learner"""
        # Collect gradients
        policy_grads = {}
        for name, param in self.meta_policy.named_parameters():
            if param.grad is not None:
                policy_grads[name] = param.grad.clone().cpu()
        
        self.update_package = {
            'agent_id': self.agent_id,
            'policy_gradients': policy_grads,
            'recent_performance': list(self.performance_buffer)[-50:],
            'hypervolume_history': list(self.hv_history)[-20:],
            'hv_moving_avg': np.mean(list(self.hv_history)) if len(self.hv_history) > 0 else 0.0
        }
    
    def get_update_package(self) -> Optional[Dict]:
        """Get update package if triggered"""
        if self.update_needed:
            self.update_needed = False
            return self.update_package
        return None

class HigherLayerMetaLearner:
    """
    Higher-layer meta-learner implementing MAML-style meta-learning
    Based on meta-rl offloading MRLCO implementation
    """
    
    def __init__(
        self,
        meta_policy: MetaPolicyNetwork,
        value_network: MultiObjectiveValueNetwork,
        config: Dict,
        device: torch.device
    ):
        self.device = device
        self.config = config
        
        self.meta_policy = meta_policy.to(device)
        self.value_network = value_network.to(device)
        
        # Meta-optimizer (Adam for outer loop)
        self.meta_lr = config.get('meta_learning_rate', 1e-4)
        self.meta_optimizer = optim.Adam(
            list(self.meta_policy.parameters()) + list(self.value_network.parameters()),
            lr=self.meta_lr,
            betas=(0.9, 0.999)
        )
        
        self.lower_agents: List[LowerLayerAgent] = []
        
    def create_lower_agent(self, agent_id: int) -> LowerLayerAgent:
        """Create new lower-layer agent"""
        agent = LowerLayerAgent(
            agent_id=agent_id,
            meta_policy=self.meta_policy,
            value_network=self.value_network,
            config=self.config,
            device=self.device
        )
        self.lower_agents.append(agent)
        return agent

    def meta_update(self, task_batch: List[Dict]):
        """
        MAML meta-update: train on adapted policies

        Memory-efficient implementation using per-task backward (gradient
        accumulation).  Instead of summing all task losses and calling one
        big backward() — which keeps ALL tasks' computation graphs live
        simultaneously — we call backward() immediately after each task and
        let PyTorch accumulate the gradients into .grad attributes.  This
        reduces peak GPU memory from O(num_tasks × graph_size) to O(1 × graph_size).
        """
        valid_tasks = 0
        encoder_type = getattr(self.meta_policy.encoder, 'encoder_type', 'lstm')

        # Zero gradients before meta-update
        self.meta_optimizer.zero_grad()

        # LSTM inner loops require the CuDNN-incompatible autograd backend.
        # GCN / GAT do not need this flag and run faster with CuDNN enabled.
        cudnn_enabled = encoder_type not in {'lstm', 'both'}

        # All encoder types use the same inner_steps from config.
        # The per-task backward() fix above already reduces peak GPU memory
        # from O(num_tasks × graph) to O(1 × graph), so LSTM no longer needs
        # fewer inner steps than GCN/GAT.  Using different inner_steps per
        # encoder would make the benchmark comparison unfair.
        # If LSTM still OOMs, reduce meta_batch_size in the training cell
        # (e.g. 8 instead of 15) — that affects gradient variance, not
        # per-task adaptation quality.
        inner_steps = self.config.get('inner_steps', 5)

        running_loss_sum = 0.0

        # Hold the policy in eval() for the entire MAML computation — inner-loop
        # adaptation AND the outer test-loss forward. Every log-prob the PPO ratio
        # compares (collection-time old_log_prob, inner-step log-probs, outer-loss
        # log-probs) is then computed under the identical dropout-free regime, and
        # the second-order gradients stay consistent across the whole graph.
        # Exploration comes from Categorical sampling at collection time, not dropout.
        # (inner_loop_update also sets eval() and now restores the mode it found,
        # so it no longer flips the policy back to train() mid-loop.)
        self.meta_policy.eval()

        with torch.backends.cudnn.flags(enabled=cudnn_enabled):
            for task_data in task_batch:
                if len(task_data['test_experiences']) == 0:
                    continue

                # Create temporary agent sharing meta-policy reference
                temp_agent = LowerLayerAgent(
                    agent_id=-1,
                    meta_policy=self.meta_policy,
                    value_network=self.value_network,
                    config=self.config,
                    device=self.device
                )

                # Inner loop adaptation
                adapted_params = None
                if len(task_data['train_experiences']) > 0:
                    adapted_params = temp_agent.inner_loop_update(
                        task_data['train_experiences'],
                        num_steps=inner_steps,
                        create_graph=True  # second-order gradients for proper MAML
                    )

                if len(task_data['test_experiences']) > 0 and adapted_params is not None:
                    # Normalise by total valid tasks so the scale is consistent
                    # regardless of how many tasks had experiences.
                    n_tasks = max(sum(
                        1 for t in task_batch
                        if len(t['test_experiences']) > 0 and len(t.get('train_experiences', [])) > 0
                    ), 1)

                    task_loss = temp_agent._compute_loss_with_params(
                        adapted_params,
                        task_data['test_experiences']
                    ) / n_tasks

                    # ── KEY FIX: backward immediately so the graph is freed ──
                    # retain_graph=False (default) lets PyTorch free the computation
                    # graph as soon as backward() completes.  Gradients accumulate
                    # in .grad attributes across tasks just like mini-batch accumulation.
                    task_loss.backward()

                    running_loss_sum += task_loss.item()
                    valid_tasks += 1

                    # Free cached allocations between tasks so fragmentation
                    # doesn't cause false OOMs on the NEXT task's forward pass.
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

        # Leave the module in train() mode between meta-updates (the conventional
        # resting state); every consumer — collection, inner loop, this method,
        # benchmark — explicitly sets the mode it needs before any forward pass.
        self.meta_policy.train()

        if valid_tasks == 0:
            print("WARNING: No valid tasks for meta-update!")
            return 0.0

        avg_meta_loss = running_loss_sum  # already divided by n_tasks per task

        # Gradient diagnostics — only printed every 10 iterations to reduce noise.
        total_grad_norm = sum(
            p.grad.norm().item()
            for p in self.meta_policy.parameters()
            if p.grad is not None
        )
        any_grad = total_grad_norm > 0

        if not any_grad:
            print("WARNING: No gradients detected in meta-update! Check inner loop.")
        else:
            print(f"  [meta] grad_norm_total={total_grad_norm:.4f}  tasks={valid_tasks}  inner_steps={inner_steps}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.meta_policy.parameters()) + list(self.value_network.parameters()),
            max_norm=1.0
        )

        # Meta-optimizer step
        self.meta_optimizer.step()

        return avg_meta_loss


    def collect_updates(self) -> List[Dict]:
        """Collect threshold-triggered updates"""
        updates = []
        for agent in self.lower_agents:
            package = agent.get_update_package()
            if package is not None:
                updates.append(package)
        return updates
    
    def distribute_meta_policy(self):
        """
        Distribute updated meta-policy
        Note: In this implementation, agents already share the meta-policy reference
        """
        pass
    
    def refine_with_agent_updates(self, updates: List[Dict]):
        """
        Policy refinement using agent feedback
        """
        if len(updates) == 0:
            return
        
        # Aggregate gradients
        aggregated_grads = {}
        
        for update in updates:
            for name, grad in update['policy_gradients'].items():
                if grad is not None:
                    if name not in aggregated_grads:
                        aggregated_grads[name] = []
                    aggregated_grads[name].append(grad.to(self.device))
        
        # Apply aggregated gradients
        if len(aggregated_grads) > 0:
            self.meta_optimizer.zero_grad()
            
            for name, param in self.meta_policy.named_parameters():
                if name in aggregated_grads and len(aggregated_grads[name]) > 0:
                    avg_grad = torch.stack(aggregated_grads[name]).mean(dim=0)
                    if param.grad is None:
                        param.grad = avg_grad
                    else:
                        param.grad += avg_grad
            
            # Optimization step
            torch.nn.utils.clip_grad_norm_(
                self.meta_policy.parameters(),
                max_norm=1.0
            )
            self.meta_optimizer.step()

class TAMPOFramework:
    """
    SIMPLIFIED TAM-PO Framework - focus on fast learning
    """
    
    def __init__(self, env, config: Dict, model_path: Optional[str] = None,
                 seed: Optional[int] = None):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔧 Using device: {self.device}")

        # Seed BEFORE building the networks. Weight initialisation consumes the torch RNG,
        # so it must be seeded here or the initial weights differ from run to run (that was
        # a real bug: seeding inside train() left W0 dependent on prior global RNG state).
        # torch init does not touch the numpy RNG, so after this call every encoder — GCN,
        # GAT, LSTM — enters training with an identical numpy stream and therefore faces the
        # identical sequence of graphs, preferences and channel conditions, even though
        # their weight inits (torch) legitimately differ. On resume, load() overwrites the
        # weights and captures the saved RNG for train() to restore, so this seed is moot.
        self._seed = seed if seed is not None else config.get('seed', None)
        if self._seed is not None:
            from utils.seeding import set_seed
            set_seed(self._seed, deterministic_torch=config.get('deterministic_torch', False))

        # Network dimensions — read the width from the env rather than hardcoding it,
        # so adding a node feature cannot silently desync the encoder's input layer.
        task_feature_dim = getattr(env, 'task_feature_dim', 9)
        server_feature_dim = 20
        num_resources = env.action_space.n
        hidden_dim = config.get('hidden_dim', config.get('hidden_dims', [128])[0])
        encoder_type = config.get('encoder_type', 'lstm')
        self.encoder_type = encoder_type

        # Per-episode action histogram, for the training-time policy-collapse diagnostic
        self._action_counts: Dict[int, int] = {}

        # GAT-specific config (ignored for non-GAT encoders)
        num_gat_heads = config.get('num_gat_heads', 4)
        gat_hidden_dim = config.get('gat_hidden_dim', 16)
        gat_add_self_loops = config.get('gat_add_self_loops', True)
        gnn_hidden_dim = config.get('gnn_hidden_dim', 16)

        # Initialize networks
        meta_policy = MetaPolicyNetwork(
            task_feature_dim=task_feature_dim,
            server_feature_dim=server_feature_dim,
            num_resources=num_resources,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            num_gat_heads=num_gat_heads,
            gat_hidden_dim=gat_hidden_dim,
            gat_add_self_loops=gat_add_self_loops,
            gnn_hidden_dim=gnn_hidden_dim
        )
        
        value_network = MultiObjectiveValueNetwork(
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim * 2,          # RC#14: doubled for sufficient value capacity
            server_feature_dim=server_feature_dim
        )
        
        # Explicit decoder initialization hook.
        self._initialize_policy_bias(meta_policy, num_resources)
        
        self.meta_learner = HigherLayerMetaLearner(
            meta_policy=meta_policy,
            value_network=value_network,
            config=config,
            device=self.device
        )
        
        # Create agents
        self.num_agents = config.get('num_agents', 1)
        self.agents = []
        for i in range(self.num_agents):
            agent = self.meta_learner.create_lower_agent(agent_id=i)
            self.agents.append(agent)
        
        # Training history
        self.training_history = {
            'losses': [],
            'iterations': 0,
            'best_loss': float('inf')
        }

        # RNG state captured from a resumed checkpoint, consumed once at the start of the
        # next train() call. None on a fresh model. This is what lets a run split across
        # sessions reproduce a single continuous run bit-for-bit: the exact position in
        # the numpy/python/torch RNG streams is restored, not merely re-seeded.
        self._checkpoint_rng = None
        self._resumed = False

        # Automatic checkpoint loading
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            self._resumed = True
            print(f"✓ Resuming from checkpoint: {model_path}")
        else:
            print("✓ Initialized new TAM-PO model with neutral decoder initialization")
    
    def _initialize_policy_bias(self, policy: MetaPolicyNetwork, num_resources: int):
        """
        Keep decoder initialization explicit but neutral so the learned policy
        reflects the data and preference conditioning instead of hand-crafted
        action priors.
        """
        for name, param in policy.decoder.decision_head.named_parameters():
            if 'weight' in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                with torch.no_grad():
                    param.zero_()

    def train(self, num_iterations: int, meta_batch_size: int, checkpoint_path: str = None,
              time_budget_s: float = None):
        """
        Main training loop with proper MAML meta-learning and checkpointing.

        Args:
            time_budget_s: Optional wall-clock budget. Training stops cleanly at the first
                iteration boundary after the budget is exhausted, saves a checkpoint, and
                returns. Intended for Colab free-tier sessions, which are killed at ~4h —
                without this the run dies mid-iteration and loses everything since the last
                10-iteration autosave.

        Seeding is handled in __init__ (before weight init). Here we only RESTORE the RNG
        stream when resuming, so a run split across sessions is bit-identical to one
        continuous run — iterations N..M draw exactly the tasks they would have drawn had
        the run never stopped.
        """
        if checkpoint_path is None:
            checkpoint_path = f"models/tampo_{self.encoder_type}_checkpoint.pth"

        episodes_per_task = int(self.config.get('episodes_per_task', 5))

        # ── RNG: restore an in-flight stream when resuming ────────────────────────────
        if self._checkpoint_rng is not None:
            random.setstate(self._checkpoint_rng['python'])
            np.random.set_state(self._checkpoint_rng['numpy'])
            torch.set_rng_state(self._checkpoint_rng['torch'])
            if self._checkpoint_rng.get('torch_cuda') is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(self._checkpoint_rng['torch_cuda'])
                except Exception:
                    pass  # resumed on a different GPU count; CPU/numpy streams still exact
            self._checkpoint_rng = None   # consume once
            rng_mode = "restored from checkpoint (exact continuation)"
        elif self._seed is not None:
            rng_mode = f"fresh, seeded {self._seed} at construction"
        else:
            rng_mode = "unseeded (not reproducible)"

        print(f"\n{'='*60}")
        print(f"🚀 TAM-PO Meta-Training")
        print(f"{'='*60}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Meta-batch size: {meta_batch_size}")
        print(f"  Episodes per task: {episodes_per_task}")
        print(f"  Starting from iteration: {self.training_history['iterations']}")
        print(f"  Inner LR: {self.config.get('inner_lr', 0.01)}")
        print(f"  Meta LR: {self.config.get('meta_learning_rate', 1e-4)}")
        print(f"  RNG: {rng_mode}")
        if time_budget_s:
            print(f"  Wall-clock budget: {time_budget_s / 3600:.2f} h (stops cleanly, checkpoint saved)")
        print(f"{'='*60}\n")

        train_start = time.time()
        stopped_early = False

        cudnn_enabled = not (self.encoder_type in {'lstm', 'both'} and self.device.type == 'cuda')
        if not cudnn_enabled:
            print("  CuDNN disabled for TAMPO LSTM-style meta-training to support second-order gradients.")
        
        with torch.backends.cudnn.flags(enabled=cudnn_enabled):
            # Create checkpoint directory (dirname may be '' for a bare filename)
            os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
            
            start_iter = self.training_history['iterations']
            
            for iteration in range(num_iterations):
                current_iter = start_iter + iteration
                
                # Sample tasks
                if hasattr(self.env, 'sample_tasks'):
                    task_ids = self.env.sample_tasks(meta_batch_size)
                else:
                    task_ids = list(range(min(meta_batch_size, 10)))
                
                task_batch = []
                self._action_counts = {}   # reset the per-iteration action histogram

                # Collect experiences (silent mode)
                for task_id in task_ids:
                    if hasattr(self.env, 'set_task'):
                        self.env.set_task(task_id)

                    train_exp, test_exp = self._collect_task_experiences(
                        task_id, num_episodes=episodes_per_task
                    )

                    task_batch.append({
                        'task_id': task_id,
                        'train_experiences': train_exp,
                        'test_experiences': test_exp
                    })

                action_report = self._format_action_distribution(self._action_counts)
                meta_loss = self.meta_learner.meta_update(task_batch)
                
                # Store loss
                self.training_history['losses'].append(meta_loss)
                self.training_history['iterations'] = current_iter + 1
                
                # Update best loss
                if meta_loss < self.training_history['best_loss']:
                    self.training_history['best_loss'] = meta_loss
                    best_path = checkpoint_path.replace('.pth', '_best.pth')
                    self._save_checkpoint(best_path)
                
                # Progress reporting every 5 iterations
                if (iteration + 1) % 5 == 0 or iteration == 0:
                    avg_loss_10 = np.mean(self.training_history['losses'][-10:])
                    elapsed = time.time() - train_start
                    sec_per_iter = elapsed / (iteration + 1)
                    print(f"  Iter {current_iter + 1:3d}/{start_iter + num_iterations} | "
                          f"Loss: {meta_loss:.4f} | "
                          f"Avg(10): {avg_loss_10:.4f} | "
                          f"Best: {self.training_history['best_loss']:.4f} | "
                          f"{sec_per_iter:.1f}s/it")
                    # Which servers is the policy actually choosing?  A distribution that
                    # collapses onto one action means the policy has degenerated — the
                    # single most useful signal that training has gone wrong.
                    print(f"  [actions] {action_report}")

                # Save checkpoint every 10 iterations
                if (iteration + 1) % 10 == 0:
                    self._save_checkpoint(checkpoint_path)

                # Wall-clock budget: stop at an iteration boundary with a saved checkpoint
                # rather than being killed mid-iteration by the Colab session limit.
                if time_budget_s and (time.time() - train_start) >= time_budget_s:
                    self._save_checkpoint(checkpoint_path)
                    stopped_early = True
                    done = iteration + 1
                    print(f"\n⏱  Wall-clock budget reached after {done}/{num_iterations} "
                          f"iterations this session ({(time.time() - train_start) / 3600:.2f} h).")
                    print(f"   Checkpoint saved. Re-run the same cell to continue from "
                          f"iteration {self.training_history['iterations']}.")
                    break

            # Final save
            self._save_checkpoint(checkpoint_path)

        elapsed = time.time() - train_start
        print(f"\n✓ Training {'stopped on budget' if stopped_early else 'complete'}!")
        print(f"  Total iterations: {self.training_history['iterations']}")
        print(f"  This session: {elapsed / 3600:.2f} h "
              f"({elapsed / max(iteration + 1, 1):.1f}s per iteration)")
        print(f"  Final loss: {meta_loss:.4f}")
        print(f"  Best loss: {self.training_history['best_loss']:.4f}")
        print(f"  Model saved to: {checkpoint_path}")
    
    def _save_checkpoint(self, path: str):
        """
        Save checkpoint with training history AND full RNG state.

        The RNG snapshot (python / numpy / torch, plus CUDA if present) is what makes a
        run resumable *exactly*. On reload, train() restores these instead of re-seeding,
        so iterations 251..500 of a resumed run draw the identical task sequence,
        preferences and channel gains they would have drawn in one continuous 500-iteration
        run. Writing is atomic (temp file + os.replace) so a session killed mid-write
        cannot leave a truncated, unloadable checkpoint.
        """
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        payload = {
            'meta_policy': self.meta_learner.meta_policy.state_dict(),
            'value_network': self.meta_learner.value_network.state_dict(),
            'meta_optimizer': self.meta_learner.meta_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'rng_state': rng_state,
        }
        tmp = f"{path}.tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)   # atomic on POSIX; never a half-written checkpoint
        except OSError:
            # Google Drive's FUSE mount can reject cross-name rename; fall back to a
            # direct write. Slightly less crash-safe, but Drive is the durable copy and
            # the local ./models autosave still provides atomicity when both are used.
            torch.save(payload, path)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    
    def _collect_task_experiences(self, task_id: int, num_episodes: int = 10):
        """Collect MORE experiences for better learning"""
        all_experiences = []
        
        for ep in range(num_episodes):  # Increased from 5 to 10
            preference = self._sample_preference()
            
            state = self.env.reset(preference_vector=preference)
            done = False
            
            step_count = 0
            num_nodes = len(self.env.current_task['tasks']) if self.env._is_dag_task() else 50
            max_steps = num_nodes
            
            while not done and step_count < max_steps:
                task_features = self._extract_task_features(state)
                server_features = self._extract_server_features(state)
                adj_matrix = self._extract_adjacency_matrix()
                current_node_idx = self._current_node_idx()

                action, old_log_prob = self._select_action(
                    task_features,
                    server_features,
                    preference,
                    adj_matrix,
                    deterministic=False,
                    return_log_prob=True,
                    current_node_idx=current_node_idx
                )

                next_state, reward, done, info = self.env.step(action)
                self._action_counts[action] = self._action_counts.get(action, 0) + 1

                exp = {
                    'state': state,
                    'task_features': task_features,
                    'server_features': server_features,
                    'adj_matrix': adj_matrix,
                    'current_node_idx': current_node_idx,
                    'action': action,
                    'reward': reward,
                    'preference': preference,
                    # Behaviour-policy log-prob, for the PPO importance ratio.  The inner
                    # loop takes several gradient steps on this batch, so from step 2
                    # onward the data is off-policy w.r.t. the adapted params.
                    'old_log_prob': old_log_prob,
                    # Per-objective step rewards straight from the env.  r_delay already
                    # carries the congestion + communication penalties; both are clipped
                    # to [-1, 1].  Consumed by the backward pass below.
                    'r_delay': float(info.get('r_delay', 0.0)),
                    'r_energy': float(info.get('r_energy', 0.0)),
                    # Placeholder; will be overwritten by backward pass
                    'mo_return': np.zeros(2, dtype=np.float32),
                    'next_state': next_state,
                    'done': done
                }
                all_experiences.append(exp)
                
                state = next_state
                step_count += 1

        # ── RC#1 + RC#4: Compute per-objective discounted returns (backward pass) ──────
        # mo_return must be a BENEFIT (higher = better), not a raw physical cost.
        #
        # The per-step signals are the env's own reward components (info['r_delay'],
        # info['r_energy']).  Deriving them here from raw delay/energy instead — as this
        # used to — silently discards the congestion and communication penalties, so the
        # agent was never taught to avoid queueing even though makespan, the metric it is
        # scored on, is almost entirely queue-driven.
        #
        # IMPORTANT: G MUST reset between episodes. Without resetting, returns from
        # episode N bleed backwards into episode N-1, inflating returns by up to
        # num_episodes * correct_return and making the value network impossible to train.
        gamma = self.config.get('gamma', 0.99)

        # Split experiences back into per-episode groups by detecting done=True
        episodes: list = []
        current_ep: list = []
        for exp in all_experiences:
            current_ep.append(exp)
            if exp['done']:          # episode boundary
                episodes.append(current_ep)
                current_ep = []
        if current_ep:               # last episode if env didn't set done
            episodes.append(current_ep)

        # Compute discounted returns within each episode independently
        for ep_exps in episodes:
            G_delay, G_energy = 0.0, 0.0          # reset per episode
            for exp in reversed(ep_exps):
                G_delay  = exp.pop('r_delay')  + gamma * G_delay
                G_energy = exp.pop('r_energy') + gamma * G_energy
                exp['mo_return'] = np.array([G_delay, G_energy], dtype=np.float32)

        # Flatten all episodes back into a single list for the train/test split
        all_experiences = [exp for ep_exps in episodes for exp in ep_exps]

        # ── RC#10: Shuffle before train/test split to break sequential bias ────────────
        import random as _random
        _random.shuffle(all_experiences)

        if len(all_experiences) <= 1:
            return all_experiences, all_experiences

        split = max(1, int(len(all_experiences) * 0.8))
        if split >= len(all_experiences):
            split = len(all_experiences) - 1

        train_experiences = all_experiences[:split]
        test_experiences  = all_experiences[split:]
        return train_experiences, test_experiences
    
    def _format_action_distribution(self, counts: Dict[int, int]) -> str:
        """
        Render an action histogram plus its normalised entropy.

        entropy = 1.00 → the policy spreads evenly over all servers.
        entropy = 0.00 → the policy has collapsed onto a single server, which is the
        degenerate solution this framework is prone to. Watch this number.
        """
        total = sum(counts.values())
        if total == 0:
            return "no actions recorded"

        n = self.env.action_space.n
        probs = np.array([counts.get(a, 0) / total for a in range(n)])
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(n)) if n > 1 else 0.0

        parts = " ".join(
            f"{action_name(a)}={probs[a] * 100:4.1f}%" for a in range(n)
        )
        return f"{parts} | entropy={entropy:.2f} (0=collapsed, 1=uniform) | n={total}"

    def _sample_preference(self) -> np.ndarray:
        """Sample preference vector"""
        w_delay = np.random.uniform(0.2, 0.8)
        w_energy = 1.0 - w_delay
        return np.array([w_delay, w_energy])
    
    def _extract_task_features(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract graph node features for the active DAG."""
        if hasattr(self.env, 'get_task_feature_matrix'):
            features = self.env.get_task_feature_matrix()
            if features is not None:
                return np.asarray(features, dtype=np.float32)

        # Fallbacks must match the width the encoder was built for (env.task_feature_dim,
        # 9 since the 2026-07-10 overhaul) — a 6-wide matrix crashes the GNN input layer.
        dim = getattr(self.env, 'task_feature_dim', 9)
        if state is None:
            return np.zeros((1, dim), dtype=np.float32)

        row = np.asarray(state[:6], dtype=np.float32)
        if len(row) < dim:
            row = np.pad(row, (0, dim - len(row)))
        return row[:dim].reshape(1, -1)
    
    def _extract_server_features(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract structured server features."""
        if hasattr(self.env, 'get_server_features'):
            features = self.env.get_server_features()
            if features is not None:
                return np.asarray(features, dtype=np.float32)

        if state is None:
            return np.zeros(20, dtype=np.float32)

        server_slice = state[6:26]
        if len(server_slice) < 20:
            server_slice = np.pad(server_slice, (0, 20 - len(server_slice)))
        return np.asarray(server_slice[:20], dtype=np.float32)

    def _extract_adjacency_matrix(self) -> np.ndarray:
        """Return the active adjacency matrix or a single-node identity fallback."""
        if hasattr(self.env, 'get_adjacency_matrix'):
            adj_matrix = self.env.get_adjacency_matrix()
            if adj_matrix is not None:
                return np.asarray(adj_matrix, dtype=np.float32)

        node_count = self._extract_task_features().shape[0]
        return np.eye(node_count, dtype=np.float32)

    def select_action(self, state: np.ndarray, preference: np.ndarray, deterministic: bool = True) -> int:
        """Public deterministic action helper used by evaluation."""
        task_features = self._extract_task_features(state)
        server_features = self._extract_server_features(state)
        adj_matrix = self._extract_adjacency_matrix()
        return self._select_action(
            task_features,
            server_features,
            preference,
            adj_matrix=adj_matrix,
            deterministic=deterministic
        )
    
    def _select_action(
        self,
        task_features,
        server_features,
        preference,
        adj_matrix=None,
        deterministic: bool = False,
        return_log_prob: bool = False,
        current_node_idx: int = None
    ):
        """
        Select a graph-conditioned offloading action.

        Returns the action, or `(action, log_prob)` when `return_log_prob=True`.
        The log-prob is the behaviour policy's, needed for the PPO importance ratio
        during the MAML inner loop.

        The policy always runs in eval() mode here. Dropout during action selection
        would (a) make `deterministic=True` non-deterministic — benchmark actions were
        argmaxes over dropout-corrupted logits — and (b) break the PPO ratio, since
        `old_log_prob` would come from a different dropout mask than the one the loss
        recomputes under. Exploration comes from Categorical sampling, not from dropout.
        """
        policy = self.agents[0].meta_policy
        was_training = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                task_tensor = torch.FloatTensor(task_features).unsqueeze(0).to(self.device)
                server_tensor = torch.FloatTensor(server_features).unsqueeze(0).to(self.device)
                pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
                node_mask = torch.ones((1, task_features.shape[0]), dtype=torch.bool, device=self.device)

                if adj_matrix is None:
                    adj_matrix = np.eye(task_features.shape[0], dtype=np.float32)
                adj_tensor = torch.FloatTensor(adj_matrix).unsqueeze(0).to(self.device)
                graph_batch = _build_pyg_batch(
                    [np.asarray(task_features, dtype=np.float32)],
                    [np.asarray(adj_matrix, dtype=np.float32)]
                ).to(self.device)

                if current_node_idx is None:
                    current_node_idx = self._current_node_idx()
                cur_idx = torch.LongTensor([current_node_idx]).to(self.device)

                logits = policy(
                    task_tensor,
                    server_tensor,
                    pref_tensor,
                    num_decisions=1,
                    adjacency=adj_tensor,
                    node_mask=node_mask,
                    graph_batch=graph_batch,
                    current_node_idx=cur_idx
                )

                probs = torch.softmax(logits[:, 0, :], dim=-1)

                if deterministic:
                    # RC#11: Greedy argmax only during evaluation — not during training.
                    action = torch.argmax(probs, dim=-1).item()
                else:
                    # RC#11: Use Categorical sampling for training exploration.
                    # This naturally weights all actions by their policy probability,
                    # includes local (action 0), and avoids the biased epsilon-greedy.
                    action = Categorical(probs).sample().item()

                if return_log_prob:
                    log_prob = float(torch.log(probs[0, action].clamp_min(1e-8)).item())
                    return action, log_prob

            return action
        finally:
            if was_training:
                policy.train()

    def _current_node_idx(self) -> int:
        """Index (into the task-feature matrix) of the node the env is about to schedule."""
        env = self.env
        if getattr(env, 'topo_order', None) is None:
            return 0
        idx = getattr(env, 'current_node_idx', 0)
        if idx >= len(env.topo_order):
            return 0
        return int(env.topo_order[idx])
    
    def evaluate(self, num_episodes: int = 20):
        """Evaluate framework"""
        print(f"\n{'='*60}")
        print(f"📊 Evaluating TAM-PO")
        print(f"{'='*60}")
        
        delays = []
        energies = []
        
        # Test with different preferences
        preferences = [
            np.array([0.8, 0.2]),  # Delay-focused
            np.array([0.5, 0.5]),  # Balanced
            np.array([0.2, 0.8])   # Energy-focused
        ]
        
        for pref in preferences:
            for _ in range(num_episodes // 3):
                state = self.env.reset(preference_vector=pref)
                
                done = False
                episode_delay = 0
                episode_energy = 0
                
                step_count = 0
                num_nodes = len(self.env.current_task['tasks']) if self.env._is_dag_task() else 50
                max_steps = num_nodes
                
                while not done and step_count < max_steps:
                    action = self.select_action(state, pref, deterministic=True)
                    
                    state, reward, done, info = self.env.step(action)
                    
                    step_count += 1
                
                episode_delay = info.get('makespan', self.env.total_delay)
                episode_energy = info.get('total_energy', self.env.total_energy)
                
                delays.append(episode_delay)
                energies.append(episode_energy)
        
        results = {
            'avg_delay': np.mean(delays),
            'avg_energy': np.mean(energies),
            'std_delay': np.std(delays),
            'std_energy': np.std(energies)
        }
        
        print(f"  Avg Delay: {results['avg_delay']:.4f}s")
        print(f"  Avg Energy: {results['avg_energy']:.4f}J")
        
        return results
    
    def save(self, path: str):
        """
        Save model. Delegates to _save_checkpoint so the RNG state is included — a plain
        torch.save here would silently drop it and break exact resume, since the notebook
        calls save() at the end of every session.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._save_checkpoint(path)
        print(f"✓ Model saved to {path}")

    def load(self, path: str):
        """Load model, including the RNG snapshot (captured, applied later by train())."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.meta_learner.meta_policy.load_state_dict(checkpoint['meta_policy'])
        except RuntimeError as err:
            raise RuntimeError(
                f"Checkpoint '{path}' is incompatible with the current network.\n"
                f"The 2026-07-10 encoder overhaul changed the node feature width (6 -> 9) "
                f"and made the GNN emit per-node embeddings instead of a pooled scalar, so "
                f"pre-overhaul checkpoints cannot be loaded. Delete models/*.pth and retrain.\n"
                f"Original error: {err}"
            ) from err
        self.meta_learner.value_network.load_state_dict(checkpoint['value_network'])
        self.meta_learner.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])

        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        # Capture (do NOT apply) the RNG snapshot. Applying it here would be undone by any
        # RNG draw between construction and the training loop; train() applies it at the
        # exact right moment. Older checkpoints without it fall back to seed-based init.
        self._checkpoint_rng = checkpoint.get('rng_state', None)
        if self._checkpoint_rng is None:
            print("  ⚠ Checkpoint has no RNG state (pre-2026-07-11). Resume is approximate: "
                  "weights continue, but the task stream restarts from the seed.")

        print(f"✓ Model loaded from {path}  (iteration {self.training_history['iterations']})")
