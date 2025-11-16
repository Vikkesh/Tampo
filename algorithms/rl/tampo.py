import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.distributions import Categorical
from collections import deque
import copy
import os

class DAGEncoder(nn.Module):
    """
    Enhanced encoder for DAG structure using GNN-inspired approach
    """
    
    def __init__(self, task_feature_dim: int, hidden_dim: int, num_layers: int = 2):
        super(DAGEncoder, self).__init__()
        
        self.task_embedding = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # BiLSTM for sequential processing
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Graph attention for dependency modeling
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, task_features, adjacency_matrix=None):
        """
        Args:
            task_features: [batch, num_tasks, feature_dim]
            adjacency_matrix: [batch, num_tasks, num_tasks] optional
        
        Returns:
            encoded_tasks: [batch, num_tasks, hidden_dim * 2]
            context: [batch, hidden_dim * 2]
        """
        batch_size, num_tasks, _ = task_features.shape
        
        # Embed tasks
        embedded = self.task_embedding(task_features)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Graph attention if adjacency provided
        if adjacency_matrix is not None:
            attn_out, _ = self.graph_attention(
                lstm_out, lstm_out, lstm_out
            )
            encoded = self.layer_norm(lstm_out + attn_out)
        else:
            encoded = self.layer_norm(lstm_out)
        
        # Context from final hidden states
        context = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
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
        
    def forward(self, encoded_tasks, context, preference, num_steps):
        """
        Sequential decoding with preference conditioning
        """
        batch_size = encoded_tasks.size(0)
        
        # Encode preference
        pref_encoded = self.preference_encoder(preference)
        
        # Initialize LSTM state
        h_t = context[:, :self.hidden_dim]
        c_t = torch.zeros_like(h_t)
        
        action_logits_list = []
        
        for step in range(num_steps):
            # Attention query
            query = torch.cat([h_t, pref_encoded], dim=-1).unsqueeze(1)
            
            attn_out, attn_weights = self.attention(
                query, encoded_tasks, encoded_tasks
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
        num_encoder_layers: int = 2
    ):
        super(MetaPolicyNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.encoder = DAGEncoder(
            task_feature_dim, hidden_dim, num_encoder_layers
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
        
    def forward(self, task_features, server_features, preference, num_tasks, adjacency=None):
        """
        Forward pass
        """
        # Encode DAG
        encoded_tasks, context = self.encoder(task_features, adjacency)
        
        # Encode server state
        server_encoded = self.server_encoder(server_features)
        
        # Combine contexts
        combined_context = context + server_encoded
        
        # Decode decisions
        action_logits = self.decoder(
            encoded_tasks, combined_context, preference, num_tasks
        )
        
        return action_logits

class MultiObjectiveValueNetwork(nn.Module):
    """
    Separate value functions for delay and energy objectives
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(MultiObjectiveValueNetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),
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
        
    def forward(self, state, preference):
        """
        Returns: [batch, 2] - [V_delay, V_energy]
        """
        x = torch.cat([state, preference], dim=-1)
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
            reference_point=np.array([10.0, 1.0])
        )
        
        self.performance_buffer = deque(maxlen=200)
        self.hv_history = deque(maxlen=self.hv_window)
        
        # Communication trigger
        self.update_needed = False
        self.update_package = None
    
    def _forward_with_params(self, params_dict, task_features, server_features, preference, num_tasks):
        """
        Forward pass using specific parameters (for functional gradient computation)
        This is the KEY to making MAML work - we use functional programming style
        """
        # This is a simplified version - you'd need to implement proper functional forward
        # For now, we'll use a workaround with temporary parameter assignment
        
        # Save original parameters
        original_params = {}
        for name, param in self.meta_policy.named_parameters():
            original_params[name] = param.data.clone()
        
        # Temporarily set adapted parameters
        for name, param in self.meta_policy.named_parameters():
            if name in params_dict:
                param.data = params_dict[name]
        
        # Forward pass
        output = self.meta_policy(task_features, server_features, preference, num_tasks)
        
        # Restore original parameters
        for name, param in self.meta_policy.named_parameters():
            param.data = original_params[name]
        
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
        return adapted_params
    
    def _compute_loss_with_params(self, params_dict: Dict, batch: List[Dict]) -> torch.Tensor:
        """
        Compute loss using specific parameters (functional style)
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        
        task_features_list = [exp['task_features'] for exp in batch]
        task_features = torch.FloatTensor(np.array(task_features_list)).to(self.device)
        if len(task_features.shape) == 4:
            task_features = task_features.squeeze(1)
        
        server_features = torch.FloatTensor(np.array([exp['server_features'] for exp in batch])).to(self.device)
        if len(server_features.shape) == 3:
            server_features = server_features.squeeze(1)
            
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        preferences = torch.FloatTensor(np.array([exp['preference'] for exp in batch])).to(self.device)
        mo_returns = torch.FloatTensor(np.array([exp['mo_return'] for exp in batch])).to(self.device)
        
        # Forward pass with adapted parameters
        num_tasks = task_features.size(1)
        action_logits = self._forward_with_params(
            params_dict, task_features, server_features, preferences, num_tasks
        )
        
        # Policy loss
        logits = action_logits[:, 0, :]
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Value prediction (using meta value network for simplicity)
        values = self.meta_value(states, preferences)
        
        # Multi-objective advantages
        advantages = mo_returns - values.detach()
        weighted_advantages = (advantages * preferences).sum(dim=-1)
        
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
        
        task_features_list = [exp['task_features'] for exp in batch]
        task_features = torch.FloatTensor(np.array(task_features_list)).to(self.device)
        if len(task_features.shape) == 4:
            task_features = task_features.squeeze(1)
        
        server_features = torch.FloatTensor(np.array([exp['server_features'] for exp in batch])).to(self.device)
        if len(server_features.shape) == 3:
            server_features = server_features.squeeze(1)
            
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        preferences = torch.FloatTensor(np.array([exp['preference'] for exp in batch])).to(self.device)
        mo_returns = torch.FloatTensor(np.array([exp['mo_return'] for exp in batch])).to(self.device)
        
        # Forward pass
        num_tasks = task_features.size(1)
        action_logits = self.meta_policy(task_features, server_features, preferences, num_tasks)
        
        # Policy loss
        logits = action_logits[:, 0, :]
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Value prediction
        values = self.meta_value(states, preferences)
        
        # Multi-objective advantages
        advantages = mo_returns - values.detach()
        weighted_advantages = (advantages * preferences).sum(dim=-1)
        
        # Combined loss
        policy_loss = -(log_probs * weighted_advantages).mean()
        value_loss = ((mo_returns - values) ** 2).sum(dim=-1).mean()
        entropy_loss = -entropy.mean()
        
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        return total_loss
    
    def update_performance(self, delay: float, energy: float):
        """
        Update performance and check threshold
        """
        # Normalize objectives
        norm_delay = delay / 10.0
        norm_energy = energy / 1.0
        
        self.performance_buffer.append([norm_delay, norm_energy])
        
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
        
        This implements the proper MAML algorithm following MRLCO:
        1. For each task, perform inner loop adaptation with create_graph=True
        2. Compute meta-loss on test set with adapted parameters
        3. Backpropagate through entire computation graph to meta-policy
        4. Update meta-policy parameters
        """
        meta_loss = 0.0
        valid_tasks = 0
        
        # Zero gradients before meta-update
        self.meta_optimizer.zero_grad()
        
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
            
            # Inner loop adaptation with gradient graph creation
            adapted_params = None
            if len(task_data['train_experiences']) > 0:
                adapted_params = temp_agent.inner_loop_update(
                    task_data['train_experiences'],
                    num_steps=self.config.get('inner_steps', 3),
                    create_graph=True  # CRITICAL: Enables second-order gradients
                )
            
            # Compute test loss with adapted parameters
            if len(task_data['test_experiences']) > 0 and adapted_params is not None:
                test_loss = temp_agent._compute_loss_with_params(
                    adapted_params,
                    task_data['test_experiences']
                )
                meta_loss = meta_loss + test_loss
                valid_tasks += 1
        
        if valid_tasks == 0:
            print("WARNING: No valid tasks for meta-update!")
            return 0.0
        
        # Average meta-loss across tasks
        meta_loss = meta_loss / valid_tasks
        
        # Backpropagate through entire computation graph
        meta_loss.backward()
        
        # Check if gradients are flowing properly
        print("\n=== Meta-Gradient Check ===")
        any_grad = False
        total_grad_norm = 0.0
        for name, p in self.meta_policy.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if grad_norm > 1e-8:  # Only print non-negligible gradients
                    print(f"  {name}: grad_norm = {grad_norm:.6f}")
                total_grad_norm += grad_norm
                any_grad = True
        
        print(f"  Total grad norm: {total_grad_norm:.6f}")
        print(f"  Any meta grads? {any_grad}")
        print("=" * 30 + "\n")
        
        if not any_grad:
            print("WARNING: No gradients detected! Check:")
            print("  1. Are train/test experiences non-empty?")
            print("  2. Is create_graph=True in inner loop?")
            print("  3. Are losses computed correctly?")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.meta_policy.parameters()) + list(self.value_network.parameters()),
            max_norm=1.0
        )
        
        # Meta-optimizer step
        self.meta_optimizer.step()
        
        return meta_loss.item()

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
    Complete TAM-PO Framework with proper MAML meta-learning
    Automatic checkpoint loading - no user prompts
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
        hidden_dim = config.get('hidden_dim', 256)
        
        # Initialize networks
        meta_policy = MetaPolicyNetwork(
            task_feature_dim=task_feature_dim,
            server_feature_dim=server_feature_dim,
            num_resources=num_resources,
            hidden_dim=hidden_dim
        )
        
        value_network = MultiObjectiveValueNetwork(
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim
        )
        
        # Create meta-learner
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
        
        # Automatic checkpoint loading - no prompts
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"✓ Resuming from checkpoint: {model_path}")
            print(f"  Previous iterations: {self.training_history['iterations']}")
            print(f"  Best loss: {self.training_history['best_loss']:.4f}")
        else:
            print("✓ Initialized new TAM-PO model")
    
    def train(self, num_iterations: int, meta_batch_size: int, checkpoint_path: str = "models/tampo_checkpoint.pth"):
        """
        Main training loop with proper MAML meta-learning and checkpointing
        """
        print(f"\n{'='*60}")
        print(f"🚀 TAM-PO Meta-Training")
        print(f"{'='*60}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Meta-batch size: {meta_batch_size}")
        print(f"  Starting from iteration: {self.training_history['iterations']}")
        print(f"  Inner LR: {self.config.get('inner_lr', 0.01)}")
        print(f"  Meta LR: {self.config.get('meta_learning_rate', 1e-4)}")
        print(f"{'='*60}\n")
        
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
                
                train_exp, test_exp = self._collect_task_experiences(task_id)
                
                task_batch.append({
                    'task_id': task_id,
                    'train_experiences': train_exp,
                    'test_experiences': test_exp
                })
            
            # Meta-update (suppressed output)
            import sys
            from io import StringIO
            
            # Capture verbose output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            meta_loss = self.meta_learner.meta_update(task_batch)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Store loss
            self.training_history['losses'].append(meta_loss)
            self.training_history['iterations'] = current_iter + 1
            
            # Update best loss
            if meta_loss < self.training_history['best_loss']:
                self.training_history['best_loss'] = meta_loss
                # Save best model
                best_path = checkpoint_path.replace('.pth', '_best.pth')
                self._save_checkpoint(best_path)
            
            # Threshold-triggered updates (silent)
            if iteration % 10 == 0:
                updates = self.meta_learner.collect_updates()
                if len(updates) > 0:
                    self.meta_learner.refine_with_agent_updates(updates)
            
            # Progress reporting every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration == 0:
                avg_loss_10 = np.mean(self.training_history['losses'][-10:])
                print(f"  Iter {current_iter + 1:3d}/{start_iter + num_iterations} | "
                      f"Loss: {meta_loss:.4f} | "
                      f"Avg(10): {avg_loss_10:.4f} | "
                      f"Best: {self.training_history['best_loss']:.4f}")
            
            # Save checkpoint every 20 iterations
            if (iteration + 1) % 20 == 0:
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
    
    def _collect_task_experiences(self, task_id: int, num_episodes: int = 3):
        """Collect training and test experiences (silent mode)"""
        train_experiences = []
        test_experiences = []
        
        for ep in range(num_episodes):
            preference = self._sample_preference()
            
            state = self.env.reset(preference_vector=preference)
            done = False
            episode_exp = []
            
            step_count = 0
            max_steps = 50
            
            while not done and step_count < max_steps:
                task_features = self._extract_task_features(state)
                server_features = self._extract_server_features(state)
                
                action = self._select_action(task_features, server_features, preference)
                
                next_state, reward, done, info = self.env.step(action)
                
                exp = {
                    'state': state,
                    'task_features': task_features,
                    'server_features': server_features,
                    'action': action,
                    'reward': reward,
                    'preference': preference,
                    'mo_return': np.array([info.get('delay', 0), info.get('energy', 0)]),
                    'next_state': next_state,
                    'done': done
                }
                episode_exp.append(exp)
                
                state = next_state
                step_count += 1
            
            # Split train/test
            if len(episode_exp) > 0:
                split = max(1, len(episode_exp) // 2)
                if ep < num_episodes - 1:
                    train_experiences.extend(episode_exp[:split])
                else:
                    test_experiences.extend(episode_exp[split:])
        
        return train_experiences, test_experiences
    
    def _sample_preference(self) -> np.ndarray:
        """Sample preference vector"""
        w_delay = np.random.uniform(0.2, 0.8)
        w_energy = 1.0 - w_delay
        return np.array([w_delay, w_energy])
    
    def _extract_task_features(self, state: np.ndarray) -> np.ndarray:
        """Extract task features"""
        task_feat = state[:6].reshape(1, 1, -1)
        return task_feat
    
    def _extract_server_features(self, state: np.ndarray) -> np.ndarray:
        """Extract server features"""
        server_feat = state[6:26].reshape(1, -1)
        return server_feat
    
    def _select_action(self, task_features, server_features, preference):
        """Select action using policy"""
        with torch.no_grad():
            task_tensor = torch.FloatTensor(task_features).to(self.device)
            server_tensor = torch.FloatTensor(server_features).to(self.device)
            pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
            
            logits = self.agents[0].meta_policy(
                task_tensor, server_tensor, pref_tensor, num_tasks=1
            )
            
            probs = torch.softmax(logits[:, 0, :], dim=-1)
            action = torch.multinomial(probs, 1).item()
        
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
                max_steps = 50
                
                while not done and step_count < max_steps:
                    task_features = self._extract_task_features(state)
                    server_features = self._extract_server_features(state)
                    
                    action = self._select_action(task_features, server_features, pref)
                    
                    state, reward, done, info = self.env.step(action)
                    
                    episode_delay += info.get('delay', 0)
                    episode_energy += info.get('energy', 0)
                    
                    step_count += 1
                
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