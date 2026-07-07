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
        gat_add_self_loops: bool = True
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
        if self.encoder_type == 'gcn':
            # GDRL Feature.py architecture mapping (Upgraded to Bi-Directional)
            self.gnn1_fwd = GCNConv(task_feature_dim, 16)
            self.gnn2_fwd = GCNConv(16, 1)

            self.gnn1_bwd = GCNConv(task_feature_dim, 16)
            self.gnn2_bwd = GCNConv(16, 1)

            fnn_in = server_feature_dim + 2
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
        elif self.encoder_type == 'gat':
            # Bi-GATv2 architecture — drop-in for Bi-GCN (GAPO, Zhang et al. 2025)
            # Layer 1: concat=True  → out = hidden_per_head * heads = gat_hidden_dim
            # Layer 2: heads=1, concat=False → out = 1 → squeeze → mean pool → scalar
            hidden_per_head = gat_hidden_dim // num_gat_heads
            self.gat1_fwd = GATv2Conv(
                task_feature_dim, hidden_per_head, heads=num_gat_heads,
                concat=True, dropout=0.1, add_self_loops=gat_add_self_loops
            )
            self.gat2_fwd = GATv2Conv(
                gat_hidden_dim, 1, heads=1,
                concat=False, dropout=0.1, add_self_loops=gat_add_self_loops
            )
            self.gat1_bwd = GATv2Conv(
                task_feature_dim, hidden_per_head, heads=num_gat_heads,
                concat=True, dropout=0.1, add_self_loops=gat_add_self_loops
            )
            self.gat2_bwd = GATv2Conv(
                gat_hidden_dim, 1, heads=1,
                concat=False, dropout=0.1, add_self_loops=gat_add_self_loops
            )

            fnn_in = server_feature_dim + 2
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

        # GDRL 'gcn' path
        import torch.nn.functional as F
        x_list = []
        # graph_batch.batch contains graph ID per node
        unique_graphs = torch.unique(graph_batch.batch, sorted=True)
        for graph_id in unique_graphs:
            mask = graph_batch.batch == graph_id
            x_i = graph_batch.x[mask]                      # [N_i, 6]

            # Extract local edge_index for this graph
            node_offset = mask.nonzero(as_tuple=False)[0].item()
            local_edges_fwd = graph_batch.edge_index[:, 
                (graph_batch.edge_index[0] >= node_offset) & 
                (graph_batch.edge_index[0] < node_offset + x_i.size(0))
            ] - node_offset
            
            local_edges_bwd = local_edges_fwd[[1, 0]]

            # Forward pass
            x_fwd = self.gnn1_fwd(x_i, local_edges_fwd)
            x_fwd = F.relu(x_fwd)
            x_fwd = F.dropout(x_fwd, training=self.training)
            x_fwd = self.gnn2_fwd(x_fwd, local_edges_fwd).squeeze(1)  # [N_i]
            
            # Backward pass
            x_bwd = self.gnn1_bwd(x_i, local_edges_bwd)
            x_bwd = F.relu(x_bwd)
            x_bwd = F.dropout(x_bwd, training=self.training)
            x_bwd = self.gnn2_bwd(x_bwd, local_edges_bwd).squeeze(1)  # [N_i]

            # Mean pool both streams and combine
            summary = torch.stack([x_fwd.mean(), x_bwd.mean()]) # [2]
            x_list.append(summary)

        # Return the aggregated graphs, node_mask is None since we bypass attention
        return torch.stack(x_list).unsqueeze(1), None

    def _apply_gat(self, graph_batch, max_num_nodes: int):
        """Bi-GATv2 forward pass — drop-in replacement for Bi-GCN (GAPO, Zhang et al. 2025)."""
        if graph_batch is None:
            raise ValueError("graph_batch is required for GAT-based encoders")

        import torch.nn.functional as F
        x_list = []
        unique_graphs = torch.unique(graph_batch.batch, sorted=True)
        for graph_id in unique_graphs:
            mask = graph_batch.batch == graph_id
            x_i = graph_batch.x[mask]                      # [N_i, task_feature_dim]

            node_offset = mask.nonzero(as_tuple=False)[0].item()
            local_edges_fwd = graph_batch.edge_index[:,
                (graph_batch.edge_index[0] >= node_offset) &
                (graph_batch.edge_index[0] < node_offset + x_i.size(0))
            ] - node_offset

            local_edges_bwd = local_edges_fwd[[1, 0]]

            # Forward stream (DAG direction: parent → child)
            x_fwd = self.gat1_fwd(x_i, local_edges_fwd)   # [N_i, gat_hidden_dim]
            x_fwd = F.relu(x_fwd)
            x_fwd = F.dropout(x_fwd, training=self.training)
            x_fwd = self.gat2_fwd(x_fwd, local_edges_fwd).squeeze(1)  # [N_i]

            # Backward stream (reversed DAG: child → parent)
            x_bwd = self.gat1_bwd(x_i, local_edges_bwd)   # [N_i, gat_hidden_dim]
            x_bwd = F.relu(x_bwd)
            x_bwd = F.dropout(x_bwd, training=self.training)
            x_bwd = self.gat2_bwd(x_bwd, local_edges_bwd).squeeze(1)  # [N_i]

            # Mean pool both streams → [2] graph summary
            summary = torch.stack([x_fwd.mean(), x_bwd.mean()])
            x_list.append(summary)

        return torch.stack(x_list).unsqueeze(1), None

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
            if self.encoder_type == 'gcn':
                graph_summary, _ = self._apply_gcn(graph_batch, task_features.size(1))
            else:
                graph_summary, _ = self._apply_gat(graph_batch, task_features.size(1))
            graph_summary = graph_summary.squeeze(1)             # [batch, 2]
            combined = torch.cat([server_features, graph_summary], dim=-1)  # [batch, server_dim+2]
            out = self.fnn1(combined)                            # [batch, 64]
            context = self.fnn_out(out)                          # [batch, hidden_dim*2]

            # Decoder only uses context; encoded_tasks must be shape-compatible
            # RC#2: Broadcast context to every node position so decoder attention has
            # meaningful content instead of all-zeros. Every node gets the graph-level
            # summary; per-node differentiation is handled by the decoder's attention
            # over this uniform-but-non-zero key/value tensor.
            encoded_tasks = context.unsqueeze(1).expand(
                -1, task_features.size(1), -1
            ).contiguous()
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
        
        # LSTM decoder
        self.lstm_cell = nn.LSTMCell(hidden_dim * 3, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            batch_first=True
        )
        
        # Decision head with separate objective prediction
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_resources)
        )
        
    def forward(self, encoded_tasks, context, preference, num_steps, node_mask=None):
        """
        Sequential decoding with preference conditioning
        """
        batch_size = encoded_tasks.size(0)
        key_padding_mask = None
        if node_mask is not None:
            key_padding_mask = ~node_mask.bool()
        
        # Encode preference
        pref_encoded = self.preference_encoder(preference)
        
        # Initialize LSTM state
        # RC#13: Use full context via projection (was discarding second half with [:hidden_dim])
        h_t = self.context_projection(context)  # [B, hidden_dim]
        c_t = torch.zeros_like(h_t)
        
        action_logits_list = []
        
        for step in range(num_steps):
            # Attention query
            query = torch.cat([h_t, pref_encoded], dim=-1).unsqueeze(1)
            
            attn_out, attn_weights = self.attention(
                query, encoded_tasks, encoded_tasks, key_padding_mask=key_padding_mask
            )
            
            attn_context = attn_out.squeeze(1)
            
            # LSTM update
            lstm_input = torch.cat([
                h_t,
                attn_context[:, :self.hidden_dim],
                pref_encoded
            ], dim=-1)
            
            h_t, c_t = self.lstm_cell(lstm_input, (h_t, c_t))
            
            # Decision
            decision_input = torch.cat([
                h_t,
                attn_context[:, :self.hidden_dim],
                pref_encoded
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
        gat_add_self_loops: bool = True
    ):
        super(MetaPolicyNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.encoder = DAGEncoder(
            task_feature_dim, hidden_dim, num_encoder_layers, encoder_type=encoder_type,
            server_feature_dim=server_feature_dim,
            num_gat_heads=num_gat_heads,
            gat_hidden_dim=gat_hidden_dim,
            gat_add_self_loops=gat_add_self_loops
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
        graph_batch=None
    ):
        """
        Forward pass
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
            encoded_tasks, combined_context, preference, num_decisions, node_mask=node_mask
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
        
        # Hypervolume tracking
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

        return task_features, server_features, adjacency, node_mask, graph_batch

    def _forward_with_params(
        self,
        params_dict,
        task_features,
        server_features,
        preference,
        num_decisions: int = 1,
        adjacency=None,
        node_mask=None,
        graph_batch=None
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
                'graph_batch': graph_batch
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
        # RC#7: Restore training mode after inner-loop adaptation.
        self.meta_policy.train()
        return adapted_params
    
    def _compute_loss_with_params(self, params_dict: Dict, batch: List[Dict]) -> torch.Tensor:
        """
        Compute loss using specific parameters (functional style)
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        task_features, server_features, adjacency, node_mask, graph_batch = self._prepare_policy_batch(batch)
            
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
            graph_batch=graph_batch
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
        policy_loss = -(log_probs * weighted_advantages).mean()
        value_loss = ((mo_returns - values) ** 2).sum(dim=-1).mean()
        entropy_loss = -entropy.mean()
        
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        return total_loss

    def _compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """
        Compute combined policy and value loss (using current meta-policy)
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        task_features, server_features, adjacency, node_mask, graph_batch = self._prepare_policy_batch(batch)
            
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
            graph_batch=graph_batch
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
        policy_loss = -(log_probs * weighted_advantages).mean()
        value_loss = ((mo_returns - values) ** 2).sum(dim=-1).mean()
        entropy_loss = -entropy.mean()
        
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        return total_loss
    
    def update_performance(self, delay: float, energy: float):
        """
        Update performance buffer using normalised improvement values.
        RC#9: Now normalises to the same [-1,1] scale as mo_return so HV is meaningful.
        """
        # Values are already normalised improvements after RC#1 fix;
        # clamp to [-1,1] for safety and flip sign so higher = better for HV.
        norm_delay_imp = float(np.clip(delay, -1.0, 1.0))
        norm_energy_imp = float(np.clip(energy, -1.0, 1.0))
        
        self.performance_buffer.append([norm_delay_imp, norm_energy_imp])
        
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

        if valid_tasks == 0:
            print("WARNING: No valid tasks for meta-update!")
            return 0.0

        avg_meta_loss = running_loss_sum  # already divided by n_tasks per task

        # Gradient diagnostics — only printed every 10 iterations to reduce noise.
        # The training loop already prints loss/avg every 5 iterations.
        total_grad_norm = sum(
            p.grad.norm().item()
            for p in self.meta_policy.parameters()
            if p.grad is not None
        )
        any_grad = total_grad_norm > 0

        if not any_grad:
            print("WARNING: No gradients detected in meta-update! Check inner loop.")
        else:
            # Compact single-line summary instead of per-parameter spam
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
    
    def __init__(self, env, config: Dict, model_path: Optional[str] = None):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Using device: {self.device}")
        
        # Network dimensions
        task_feature_dim = 6
        server_feature_dim = 20
        num_resources = env.action_space.n
        hidden_dim = config.get('hidden_dim', config.get('hidden_dims', [128])[0])
        encoder_type = config.get('encoder_type', 'lstm')
        self.encoder_type = encoder_type

        # GAT-specific config (ignored for non-GAT encoders)
        num_gat_heads = config.get('num_gat_heads', 4)
        gat_hidden_dim = config.get('gat_hidden_dim', 16)
        gat_add_self_loops = config.get('gat_add_self_loops', True)

        # Initialize networks
        meta_policy = MetaPolicyNetwork(
            task_feature_dim=task_feature_dim,
            server_feature_dim=server_feature_dim,
            num_resources=num_resources,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            num_gat_heads=num_gat_heads,
            gat_hidden_dim=gat_hidden_dim,
            gat_add_self_loops=gat_add_self_loops
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
        
        # Automatic checkpoint loading
        if model_path and os.path.exists(model_path):
            self.load(model_path)
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

    def train(self, num_iterations: int, meta_batch_size: int, checkpoint_path: str = None):
        """
        Main training loop with proper MAML meta-learning and checkpointing
        """
        if checkpoint_path is None:
            checkpoint_path = f"models/tampo_{self.encoder_type}_checkpoint.pth"
        print(f"\n{'='*60}")
        print(f"🚀 TAM-PO Meta-Training")
        print(f"{'='*60}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Meta-batch size: {meta_batch_size}")
        print(f"  Starting from iteration: {self.training_history['iterations']}")
        print(f"  Inner LR: {self.config.get('inner_lr', 0.01)}")
        print(f"  Meta LR: {self.config.get('meta_learning_rate', 1e-4)}")
        print(f"{'='*60}\n")

        cudnn_enabled = not (self.encoder_type in {'lstm', 'both'} and self.device.type == 'cuda')
        if not cudnn_enabled:
            print("  CuDNN disabled for TAMPO LSTM-style meta-training to support second-order gradients.")
        
        with torch.backends.cudnn.flags(enabled=cudnn_enabled):
            # Create checkpoint directory
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            start_iter = self.training_history['iterations']
            
            for iteration in range(num_iterations):
                current_iter = start_iter + iteration
                
                # Sample tasks
                if hasattr(self.env, 'sample_tasks'):
                    task_ids = self.env.sample_tasks(meta_batch_size)
                else:
                    task_ids = list(range(min(meta_batch_size, 10)))
                
                task_batch = []
                
                # Collect experiences (silent mode)
                for task_id in task_ids:
                    if hasattr(self.env, 'set_task'):
                        self.env.set_task(task_id)
                    
                    train_exp, test_exp = self._collect_task_experiences(task_id, num_episodes=5)
                    
                    task_batch.append({
                        'task_id': task_id,
                        'train_experiences': train_exp,
                        'test_experiences': test_exp
                    })
                
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
                    print(f"  Iter {current_iter + 1:3d}/{start_iter + num_iterations} | "
                          f"Loss: {meta_loss:.4f} | "
                          f"Avg(10): {avg_loss_10:.4f} | "
                          f"Best: {self.training_history['best_loss']:.4f}")
                
                # Save checkpoint every 10 iterations
                if (iteration + 1) % 10 == 0:
                    self._save_checkpoint(checkpoint_path)
            
            # Final save
            self._save_checkpoint(checkpoint_path)

        print(f"\n✓ Training complete!")
        print(f"  Total iterations: {self.training_history['iterations']}")
        print(f"  Final loss: {meta_loss:.4f}")
        print(f"  Best loss: {self.training_history['best_loss']:.4f}")
        print(f"  Model saved to: {checkpoint_path}")
    
    def _save_checkpoint(self, path: str):
        """Save checkpoint with training history"""
        torch.save({
            'meta_policy': self.meta_learner.meta_policy.state_dict(),
            'value_network': self.meta_learner.value_network.state_dict(),
            'meta_optimizer': self.meta_learner.meta_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
    
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
                
                action = self._select_action(
                    task_features,
                    server_features,
                    preference,
                    adj_matrix,
                    deterministic=False
                )
                
                next_state, reward, done, info = self.env.step(action)
                
                # RC#1 + RC#4: Store raw delay/energy and node_cycles so the
                # post-episode backward pass can compute signed discounted returns.
                node_id = self.env.topo_order[step_count] if (self.env._is_dag_task() and step_count < len(self.env.topo_order)) else 0
                node_cycles = 0.0
                if self.env._is_dag_task() and self.env.current_task is not None:
                    tasks_list = self.env.current_task.get('tasks', [])
                    if node_id < len(tasks_list):
                        node_cycles = float(tasks_list[node_id].get('cycles', 1e9))
                
                exp = {
                    'state': state,
                    'task_features': task_features,
                    'server_features': server_features,
                    'adj_matrix': adj_matrix,
                    'action': action,
                    'reward': reward,
                    'preference': preference,
                    # Temporary fields: removed in backward return-computation pass below
                    'raw_delay': float(info.get('delay', 0)),
                    'raw_energy': float(info.get('energy', 0)),
                    'node_cycles': node_cycles,
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
        # We compute the improvement over local execution per node, then discount backward
        # through the episode so the agent learns to plan across the full DAG sequence.
        gamma = self.config.get('gamma', 0.99)
        env_local_freq = self.env.local_freq
        env_kappa = self.env.kappa
        
        # Annotate each experience with per-step improvements (needs cycles from env)
        # info['delay'] = comp_delay for this node; we need local_delay for comparison.
        # We stored 'node_cycles' below; compute baseline here.
        G_delay, G_energy = 0.0, 0.0
        for exp in reversed(all_experiences):
            cycles = exp.pop('node_cycles', 1e9)  # retrieved from stored value
            local_delay = cycles / max(env_local_freq, 1e-9)
            local_energy = env_kappa * cycles * (env_local_freq ** 2)
            # Step delay/energy improvement (positive = better than local)
            step_d_imp = (local_delay - exp.pop('raw_delay', 0)) / max(local_delay, 1e-9)
            step_e_imp = (local_energy - exp.pop('raw_energy', 0)) / max(local_energy, 1e-9)
            # Accumulate discounted returns backward
            G_delay = step_d_imp + gamma * G_delay
            G_energy = step_e_imp + gamma * G_energy
            exp['mo_return'] = np.array([G_delay, G_energy], dtype=np.float32)
        
        # ── RC#10: Shuffle before train/test split to break sequential bias ────────────
        import random as _random
        _random.shuffle(all_experiences)
        
        if len(all_experiences) <= 1:
            return all_experiences, all_experiences

        split = max(1, int(len(all_experiences) * 0.8))
        if split >= len(all_experiences):
            split = len(all_experiences) - 1

        train_experiences = all_experiences[:split]
        test_experiences = all_experiences[split:]
        return train_experiences, test_experiences
    
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

        if state is None:
            return np.zeros((1, 6), dtype=np.float32)

        return np.asarray(state[:6], dtype=np.float32).reshape(1, -1)
    
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
        deterministic: bool = False
    ):
        """Select a graph-conditioned offloading action."""
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
            
            logits = self.agents[0].meta_policy(
                task_tensor,
                server_tensor,
                pref_tensor,
                num_decisions=1,
                adjacency=adj_tensor,
                node_mask=node_mask,
                graph_batch=graph_batch
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
        
        return action
    
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
        """Save model"""
        torch.save({
            'meta_policy': self.meta_learner.meta_policy.state_dict(),
            'value_network': self.meta_learner.value_network.state_dict(),
            'meta_optimizer': self.meta_learner.meta_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.meta_learner.meta_policy.load_state_dict(checkpoint['meta_policy'])
        self.meta_learner.value_network.load_state_dict(checkpoint['value_network'])
        self.meta_learner.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"✓ Model loaded from {path}")
