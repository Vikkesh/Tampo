import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.distributions import Categorical
import copy
import os

class HistogramEncoder(nn.Module):
    """
    Histogram-based workload encoder for server states
    Discretizes continuous workload into histogram bins
    """
    
    def __init__(self, num_bins: int = 10, hidden_dim: int = 128):
        super(HistogramEncoder, self).__init__()
        
        self.num_bins = num_bins
        
        # Network to process histogram features
        self.encoder = nn.Sequential(
            nn.Linear(num_bins, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, workloads: torch.Tensor) -> torch.Tensor:
        """
        Args:
            workloads: [batch, num_servers] - continuous workload values
        
        Returns:
            encoded: [batch, hidden_dim] - encoded histogram features
        """
        batch_size, num_servers = workloads.shape
        
        # Create histogram for each sample
        histograms = []
        for i in range(batch_size):
            # Discretize workloads into bins
            hist, _ = np.histogram(
                workloads[i].cpu().numpy(),
                bins=self.num_bins,
                range=(0, 10)  # Adjusted range for server loads
            )
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-8)
            histograms.append(hist)
        
        hist_tensor = torch.FloatTensor(np.array(histograms)).to(workloads.device)
        
        return self.encoder(hist_tensor)

class PreferenceConditionedActorNetwork(nn.Module):
    """
    Actor network conditioned on preference vector
    Uses histogram encoding for server workloads
    Discrete-SAC architecture for discrete action spaces
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int,
        num_servers: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super(PreferenceConditionedActorNetwork, self).__init__()
        
        self.num_servers = num_servers
        self.preference_dim = 2  # [w_delay, w_energy]
        
        # Histogram encoder for server workloads
        self.histogram_encoder = HistogramEncoder(num_bins=10, hidden_dim=128)
        
        # Observation encoder (excludes server workload part)
        # obs_dim includes: task features (2) + preference (2) + server freqs + server loads + channel gains
        obs_without_servers = obs_dim - num_servers  # Remove server loads dimension
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_without_servers, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0])
        )
        
        # Preference encoder
        self.pref_encoder = nn.Sequential(
            nn.Linear(self.preference_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Combined feature processing
        combined_dim = hidden_dims[0] + 128 + 64  # obs + histogram + preference
        
        layers = []
        prev_dim = combined_dim
        
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.policy_head = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, obs_dim] - state observation
            preference: [batch, 2] - [w_delay, w_energy]
        
        Returns:
            action_logits: [batch, action_dim]
        """
        batch_size = obs.size(0)
        
        # Extract server workloads from observation
        # Assuming server loads are in a specific position in obs
        # Based on env structure: [task(2), pref(2), server_freqs(num_servers), server_loads(num_servers), channels(num_servers)]
        start_idx = 2 + 2 + self.num_servers  # After task + pref + freqs
        server_workloads = obs[:, start_idx:start_idx + self.num_servers]
        
        # Remove server loads from obs for encoder
        obs_parts = [
            obs[:, :start_idx],  # Everything before server loads
            obs[:, start_idx + self.num_servers:]  # Everything after server loads
        ]
        obs_without_servers = torch.cat(obs_parts, dim=-1)
        
        # Encode components
        obs_encoded = self.obs_encoder(obs_without_servers)
        hist_encoded = self.histogram_encoder(server_workloads)
        pref_encoded = self.pref_encoder(preference)
        
        # Concatenate all features
        combined = torch.cat([obs_encoded, hist_encoded, pref_encoded], dim=-1)
        
        # Generate action logits
        logits = self.policy_head(combined)
        
        return logits

class MultiObjectiveCriticNetwork(nn.Module):
    """
    Critic network with separate value heads for each objective
    Estimates Q(s, a, ω) for delay and energy separately using Discrete-SAC
    """
    
    def __init__(
        self, 
        obs_dim: int,
        action_dim: int,
        num_servers: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super(MultiObjectiveCriticNetwork, self).__init__()
        
        self.num_servers = num_servers
        self.num_objectives = 2
        self.action_dim = action_dim
        
        # Histogram encoder
        self.histogram_encoder = HistogramEncoder(num_bins=10, hidden_dim=128)
        
        # Observation encoder
        obs_without_servers = obs_dim - num_servers
        
        # Preference encoder
        self.pref_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Shared layers
        shared_layers = []
        prev_dim = obs_without_servers + 128 + 64
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Q-value heads for each action (Discrete-SAC style)
        # Output is [batch, action_dim, 2] for delay and energy Q-values per action
        self.delay_q_head = nn.Linear(prev_dim, action_dim)
        self.energy_q_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, obs: torch.Tensor, preference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, obs_dim]
            preference: [batch, 2]
        
        Returns:
            q_values: [batch, action_dim, 2] - Q-values for each action and objective
        """
        batch_size = obs.size(0)
        
        # Extract server workloads
        start_idx = 2 + 2 + self.num_servers
        server_workloads = obs[:, start_idx:start_idx + self.num_servers]
        
        obs_parts = [
            obs[:, :start_idx],
            obs[:, start_idx + self.num_servers:]
        ]
        obs_without_servers = torch.cat(obs_parts, dim=-1)
        
        # Encode
        hist_encoded = self.histogram_encoder(server_workloads)
        pref_encoded = self.pref_encoder(preference)
        
        # Combine
        combined = torch.cat([obs_without_servers, hist_encoded, pref_encoded], dim=-1)
        
        # Shared features
        shared_features = self.shared_network(combined)
        
        # Get Q-values for each objective and action
        q_delay = self.delay_q_head(shared_features)  # [batch, action_dim]
        q_energy = self.energy_q_head(shared_features)  # [batch, action_dim]
        
        # Stack to [batch, action_dim, 2]
        q_values = torch.stack([q_delay, q_energy], dim=-1)
        
        return q_values

class CoverageSet:
    """
    Maintains diverse coverage set of preference vectors
    Implements preference selection mechanism from GMORL paper
    """
    
    def __init__(self, num_objectives: int = 2, min_distance: float = 0.1):
        self.num_objectives = num_objectives
        self.min_distance = min_distance
        self.preferences = []
        
        # Initialize with corner preferences and balanced
        self.preferences.append(np.array([1.0, 0.0]))  # Only delay
        self.preferences.append(np.array([0.0, 1.0]))  # Only energy
        self.preferences.append(np.array([0.5, 0.5]))  # Balanced
        
        # Add intermediate preferences
        for alpha in [0.25, 0.75]:
            self.preferences.append(np.array([alpha, 1.0 - alpha]))
    
    def sample_preference(self) -> np.ndarray:
        """Sample a preference ensuring diversity"""
        if len(self.preferences) < 10 and np.random.rand() < 0.3:
            # Explore: sample random preference
            w_delay = np.random.uniform(0, 1)
            w_energy = 1.0 - w_delay
            new_pref = np.array([w_delay, w_energy])
            
            # Check if sufficiently different from existing
            if self._is_diverse(new_pref):
                self.preferences.append(new_pref)
            
            return new_pref
        else:
            # Exploit: sample from existing preferences
            return self.preferences[np.random.randint(len(self.preferences))]
    
    def _is_diverse(self, new_pref: np.ndarray) -> bool:
        """Check if preference is sufficiently different"""
        for existing_pref in self.preferences:
            distance = np.linalg.norm(new_pref - existing_pref)
            if distance < self.min_distance:
                return False
        return True

class GMORLAgent:
    """GMORL with aggressive learning"""
    
    def __init__(self, env, config: Dict, model_path: Optional[str] = None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Using device: {self.device}")
        
        # AGGRESSIVE hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-3)  # Higher
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)  # Faster target update
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.num_objectives = 2
        
        # Discrete-SAC specific
        self.alpha = config.get('alpha', 0.01)  # Very low temperature
        self.target_entropy = -np.log(1.0 / env.action_space.n) * 0.5
        
        # Environment dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        num_servers = env.num_servers
        hidden_dims = config.get('hidden_dims', [128, 128])  # Smaller
        
        # Networks
        self.actor = PreferenceConditionedActorNetwork(
            obs_dim, action_dim, num_servers, hidden_dims
        ).to(self.device)
        
        # EXTREME initialization
        self._initialize_with_bias(action_dim)
        
        self.critic1 = MultiObjectiveCriticNetwork(
            obs_dim, action_dim, num_servers, hidden_dims
        ).to(self.device)
        
        self.critic2 = MultiObjectiveCriticNetwork(
            obs_dim, action_dim, num_servers, hidden_dims
        ).to(self.device)
        
        # Target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.learning_rate
        )
        
        # Temperature
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        
        # Coverage set
        self.coverage_set = CoverageSet()
        
        # Replay buffer
        self.buffer_size = config.get('buffer_size', 100000)
        self.buffer = []
        self.batch_size = config.get('batch_size', 128)  # Larger batches
        
        # Training history
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'alpha': [],
            'episodes': 0,
            'best_loss': float('inf')
        }
        
        # Load checkpoint
        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
                print(f"✓ Resuming from checkpoint")
            except Exception as e:
                print(f"⚠️  Starting fresh")
        else:
            print("✓ Initialized GMORL with EXTREME offloading bias")
    
    def _initialize_with_bias(self, action_dim: int):
        """EXTREME initialization"""
        with torch.no_grad():
            final_layer = self.actor.policy_head[-1]
            # Destroy local
            final_layer.weight[0, :] *= 0.01
            final_layer.bias[0] = -10.0
            
            # Boost cloud massively
            final_layer.weight[1, :] *= 5.0
            final_layer.bias[1] = 5.0
            
            # Boost edges
            for i in range(2, action_dim):
                final_layer.weight[i, :] *= 4.0
                final_layer.bias[i] = 4.0

    def select_action(
        self, 
        state: np.ndarray, 
        preference: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, np.ndarray]:
        """
        Select action based on current policy
        
        Args:
            state: Current state
            preference: [w_delay, w_energy]
            deterministic: Whether to select deterministically
            
        Returns:
            (action, log_prob, q_values)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.actor(state_tensor, pref_tensor)
            action_probs = torch.softmax(logits, dim=-1)
            
            # Get Q-values
            q_values1 = self.critic1(state_tensor, pref_tensor)  # [1, action_dim, 2]
            q_values2 = self.critic2(state_tensor, pref_tensor)
            q_values = torch.min(q_values1, q_values2)  # Conservative estimate
            
            # Weight Q-values by preference
            weighted_q = (q_values * pref_tensor.unsqueeze(1)).sum(dim=-1)  # [1, action_dim]
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-8)
        
        return action.item(), log_prob.item(), weighted_q.squeeze(0).cpu().numpy()
    
    def update(self):
        """Perform Discrete-SAC update"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self._sample_batch()
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        preferences = torch.FloatTensor(batch['preferences']).to(self.device)
        mo_rewards = torch.FloatTensor(batch['rewards']).to(self.device)  # [batch, 2]
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Scalarize rewards using preferences
        rewards = (mo_rewards * preferences).sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # Update critics
        with torch.no_grad():
            # Get next action probabilities
            next_logits = self.actor(next_states, preferences)
            next_probs = torch.softmax(next_logits, dim=-1)
            
            # Get next Q-values from target networks
            next_q1 = self.critic1_target(next_states, preferences)  # [batch, action_dim, 2]
            next_q2 = self.critic2_target(next_states, preferences)
            next_q = torch.min(next_q1, next_q2)
            
            # Weight by preference and get expected value
            weighted_next_q = (next_q * preferences.unsqueeze(1)).sum(dim=-1)  # [batch, action_dim]
            
            # Entropy term
            next_value = (next_probs * (weighted_next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=-1, keepdim=True)
            
            target_q = rewards + self.gamma * (1 - dones.unsqueeze(1)) * next_value
        
        # Current Q-values
        current_q1 = self.critic1(states, preferences)  # [batch, action_dim, 2]
        current_q2 = self.critic2(states, preferences)
        
        # Weight by preference
        weighted_q1 = (current_q1 * preferences.unsqueeze(1)).sum(dim=-1)  # [batch, action_dim]
        weighted_q2 = (current_q2 * preferences.unsqueeze(1)).sum(dim=-1)
        
        # Select Q-values for taken actions
        current_q1_a = weighted_q1.gather(1, actions.unsqueeze(1))
        current_q2_a = weighted_q2.gather(1, actions.unsqueeze(1))
        
        # Critic loss
        critic1_loss = nn.MSELoss()(current_q1_a, target_q)
        critic2_loss = nn.MSELoss()(current_q2_a, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # Update actor
        logits = self.actor(states, preferences)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        
        # Get current Q-values
        q1 = self.critic1(states, preferences)
        q2 = self.critic2(states, preferences)
        q = torch.min(q1, q2)
        weighted_q = (q * preferences.unsqueeze(1)).sum(dim=-1)
        
        # Actor loss (maximize Q - alpha * entropy)
        actor_loss = (probs * (self.alpha * log_probs - weighted_q)).sum(dim=-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        # Store metrics
        self.training_history['policy_loss'].append(actor_loss.item())
        self.training_history['value_loss'].append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.training_history['alpha'].append(self.alpha)
        
        # Update best loss
        current_loss = actor_loss.item() + (critic1_loss.item() + critic2_loss.item()) / 2
        if current_loss < self.training_history['best_loss']:
            self.training_history['best_loss'] = current_loss
    
    def _soft_update(self, source, target):
        """Soft update of target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def _sample_batch(self):
        """Sample a batch from replay buffer"""
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch_data = [self.buffer[i] for i in indices]
        
        batch = {
            'states': np.array([d['state'] for d in batch_data]),
            'actions': np.array([d['action'] for d in batch_data]),
            'preferences': np.array([d['preference'] for d in batch_data]),
            'rewards': np.array([d['reward'] for d in batch_data]),
            'next_states': np.array([d['next_state'] for d in batch_data]),
            'dones': np.array([d['done'] for d in batch_data])
        }
        
        return batch
    
    def _save_checkpoint(self, path: str):
        """Save checkpoint with training history"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'coverage_set': self.coverage_set.preferences,
            'buffer': self.buffer[-10000:],  # Save last 10k experiences
            'training_history': self.training_history
        }, path)
    
    def train(self, num_episodes: int, checkpoint_path: str = "models/gmorl_checkpoint.pth"):
        """
        Train the agent with diverse preferences and checkpointing
        
        Args:
            num_episodes: Number of training episodes
            checkpoint_path: Path to save checkpoints
        """
        print(f"\n{'='*60}")
        print(f"🚀 GMORL Training")
        print(f"{'='*60}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Starting from episode: {self.training_history['episodes']}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Target entropy: {self.target_entropy:.4f}")
        print(f"{'='*60}\n")
        
        # Create checkpoint directory
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        start_episode = self.training_history['episodes']
        
        for episode in range(num_episodes):
            current_episode = start_episode + episode
            
            # Sample preference from coverage set
            preference = self.coverage_set.sample_preference()
            
            state = self.env.reset(preference_vector=preference)
            episode_delay = 0
            episode_energy = 0
            done = False
            
            while not done:
                action, log_prob, _ = self.select_action(state, preference)
                next_state, reward, done, info = self.env.step(action)
                
                # Store multi-objective reward
                mo_reward = np.array([-info['delay'], -info['energy']])  # Negative for minimization
                
                # Add to buffer
                self.buffer.append({
                    'state': state,
                    'action': action,
                    'preference': preference,
                    'reward': mo_reward,
                    'next_state': next_state,
                    'done': done
                })
                
                # Limit buffer size
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                
                episode_delay += info['delay']
                episode_energy += info['energy']
                
                state = next_state
                
                # Update networks
                if len(self.buffer) >= self.batch_size:
                    self.update()
            
            # Update episode count
            self.training_history['episodes'] = current_episode + 1
            
            # Progress reporting - FIX: Use scientific notation for energy
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_policy_loss = np.mean(self.training_history['policy_loss'][-100:]) if len(self.training_history['policy_loss']) > 0 else 0
                avg_value_loss = np.mean(self.training_history['value_loss'][-100:]) if len(self.training_history['value_loss']) > 0 else 0
                print(f"  Episode {current_episode + 1:3d}/{start_episode + num_episodes} | "
                      f"Pref: [{preference[0]:.2f}, {preference[1]:.2f}] | "
                      f"Delay: {episode_delay:.2f}s | Energy: {episode_energy:.6f}J | "  # Changed from .2f to .6f
                      f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
                      f"Alpha: {self.alpha:.4f}")
            
            # Save checkpoint every 20 episodes
            if (episode + 1) % 20 == 0:
                self._save_checkpoint(checkpoint_path)
            
            # Save best model
            if len(self.training_history['policy_loss']) > 0:
                current_loss = self.training_history['policy_loss'][-1] + self.training_history['value_loss'][-1]
                if current_loss < self.training_history['best_loss']:
                    best_path = checkpoint_path.replace('.pth', '_best.pth')
                    self._save_checkpoint(best_path)
        
        # Final save
        self._save_checkpoint(checkpoint_path)
        print(f"\n✓ Training complete!")
        print(f"  Total episodes: {self.training_history['episodes']}")
        print(f"  Best loss: {self.training_history['best_loss']:.4f}")
        print(f"  Model saved to: {checkpoint_path}")
    
    def save(self, path: str):
        """Save model"""
        self._save_checkpoint(path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.coverage_set.preferences = checkpoint['coverage_set']
        
        # Load buffer if available
        if 'buffer' in checkpoint:
            self.buffer = checkpoint['buffer']
        
        # Load training history
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"✓ Model loaded from {path}")