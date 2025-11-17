import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.distributions import Categorical
import os

class PolicyNetwork(nn.Module):
    """Actor network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (no softmax here - we'll use it in forward with log_softmax)
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, x):
        """
        Forward pass - returns action probabilities
        
        Args:
            x: State tensor
            
        Returns:
            Action probabilities
        """
        features = self.network(x)
        action_logits = self.action_head(features)
        # Use softmax to get valid probability distribution
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    """Critic network for PPO"""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

class PPOAgent:
    """PPO with aggressive learning for offloading"""
    
    def __init__(self, env, config: Dict, model_path: Optional[str] = None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Using device: {self.device}")
        
        # AGGRESSIVE hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-3)  # Higher LR
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.3)  # Larger clip
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.001)  # Less entropy
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.ppo_epochs = config.get('ppo_epochs', 10)  # More epochs
        self.batch_size = config.get('batch_size', 64)
        self.rollout_steps = config.get('rollout_steps', 2048)
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dims = config.get('hidden_dims', [128, 128])  # Smaller/faster
        
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_dims).to(self.device)
        
        # EXTREME initialization
        self._initialize_with_bias(action_dim)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.learning_rate
        )
        
        # Storage
        self.reset_storage()
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_delays': [],
            'episode_energies': [],
            'policy_losses': [],
            'value_losses': [],
            'episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf')
        }
        
        # Load checkpoint
        if model_path and os.path.exists(model_path):
            try:
                self.load(model_path)
                print(f"✓ Resuming from checkpoint")
            except Exception as e:
                print(f"⚠️  Starting fresh")
        else:
            print("✓ Initialized PPO with EXTREME offloading bias")
    
    def _initialize_with_bias(self, action_dim: int):
        """EXTREME bias toward offloading"""
        with torch.no_grad():
            # Destroy local action completely
            self.policy.action_head.weight[:, 0] *= 0.01
            self.policy.action_head.bias[0] = -10.0
            
            # Massively boost cloud
            self.policy.action_head.weight[:, 1] *= 5.0
            self.policy.action_head.bias[1] = 5.0
            
            # Boost edges
            for i in range(2, action_dim):
                self.policy.action_head.weight[:, i] *= 4.0
                self.policy.action_head.bias[i] = 4.0

    def reset_storage(self):
        """Reset experience storage for new rollout"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action based on current policy
        
        Args:
            state: Current state
            deterministic: Whether to select deterministically (for evaluation)
            
        Returns:
            action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
            value = self.value(state_tensor)
        
        if deterministic:
            # Greedy action selection
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
        
        # Calculate log probability
        log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-10)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in rollout buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        From ppo.txt:
        - delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        - A_t = delta_t + (gamma * lambda) * A_{t+1}
        - V_target_t = A_t + V(s_t)
        
        Args:
            next_value: Value of the next state
            
        Returns:
            advantages, returns (value targets)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_val = next_value
        
        # Process in reverse order (from T to 1)
        for t in reversed(range(len(self.rewards))):
            # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            
            # GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            # Insert at beginning (since we're going backwards)
            advantages.insert(0, gae)
            
            # Value target: V_target_t = A_t + V(s_t)
            returns.insert(0, gae + self.values[t])
            
            # Update next_val for previous timestep
            next_val = self.values[t]
        
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)
    
    def update(self, next_value: float):
        """
        Update policy and value networks using PPO-Clip algorithm
        
        From ppo.txt:
        - Compute GAE advantages
        - For K epochs:
            - Calculate probability ratio: r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
            - Clipped surrogate: L_CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
            - Value loss: L_VF = (V(s) - V_target)^2
            - Entropy bonus: L_S = -Entropy[pi_theta]
            - Total loss: L = -L_CLIP + c1 * L_VF - c2 * L_S
        
        Args:
            next_value: Value of the next state
        """
        # Step 3: Advantage Calculation (GAE)
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages (improves training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Step 4: Optimization (Multiple Epochs)
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        epoch_policy_losses = []
        epoch_value_losses = []
        
        for epoch in range(self.ppo_epochs):
            # Shuffle data for each epoch
            np.random.shuffle(indices)
            
            # Process in mini-batches
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute current policy and values
                action_probs = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                values = self.value(batch_states)
                
                # 1. Policy Loss (Clipped Surrogate Objective L_CLIP)
                # Probability ratio: r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Take minimum (pessimistic bound)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Value Loss (L_VF)
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 3. Entropy Bonus (L_S) - encourage exploration
                entropy_loss = -entropy
                
                # 4. Total Loss
                # L_TOTAL = Actor_Loss + c1 * Critic_Loss + c2 * Entropy_Bonus
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Store losses
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
        
        # Store average losses
        if len(epoch_policy_losses) > 0:
            self.training_history['policy_losses'].append(np.mean(epoch_policy_losses))
            self.training_history['value_losses'].append(np.mean(epoch_value_losses))
        
        # Clear storage after update
        self.reset_storage()
    
    def _save_checkpoint(self, path: str):
        """Save checkpoint with training history"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }, path)
    
    def train(self, num_episodes: int, checkpoint_path: str = "models/ppo_checkpoint.pth"):
        """
        Train the agent using PPO algorithm with checkpointing
        
        Args:
            num_episodes: Number of training episodes
            checkpoint_path: Path to save checkpoints
        """
        print(f"\n{'='*60}")
        print(f"🚀 PPO Training")
        print(f"{'='*60}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Starting from episode: {self.training_history['episodes']}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Clip epsilon: {self.clip_epsilon}")
        print(f"{'='*60}\n")
        
        # Create checkpoint directory
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        start_episode = self.training_history['episodes']
        episode_count = start_episode
        
        while episode_count < start_episode + num_episodes:
            # Step 2: Data Collection (On-Policy Rollout)
            state = self.env.reset()
            episode_reward = 0
            episode_delay = 0
            episode_energy = 0
            done = False
            steps_in_episode = 0
            
            # Collect rollout_steps of experience
            while not done and steps_in_episode < self.rollout_steps:
                # Select action using current policy
                action, log_prob, value = self.select_action(state, deterministic=False)
                
                # Execute action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.store_transition(state, action, reward, value, log_prob, done)
                
                # Update metrics
                episode_reward += reward
                episode_delay += info['delay']
                episode_energy += info['energy']
                
                state = next_state
                steps_in_episode += 1
                self.training_history['total_steps'] += 1
            
            # Calculate final value for GAE
            if done:
                next_value = 0.0
            else:
                _, _, next_value = self.select_action(next_state, deterministic=False)
            
            # Update policy after collecting enough data
            if len(self.states) >= self.batch_size:
                self.update(next_value)
            
            # Track episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_delays'].append(episode_delay)
            self.training_history['episode_energies'].append(episode_energy)
            
            # Update episode count
            episode_count += 1
            self.training_history['episodes'] = episode_count
            
            # Update best reward
            if episode_reward > self.training_history['best_reward']:
                self.training_history['best_reward'] = episode_reward
                # Save best model
                best_path = checkpoint_path.replace('.pth', '_best.pth')
                self._save_checkpoint(best_path)
            
            # Print progress every 10 episodes
            if episode_count % 10 == 0 or episode_count == start_episode + 1:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_delay = np.mean(self.training_history['episode_delays'][-10:])
                avg_energy = np.mean(self.training_history['episode_energies'][-10:])
                
                avg_policy_loss = np.mean(self.training_history['policy_losses'][-10:]) if len(self.training_history['policy_losses']) > 0 else 0
                avg_value_loss = np.mean(self.training_history['value_losses'][-10:]) if len(self.training_history['value_losses']) > 0 else 0
                
                print(f"  Episode {episode_count:3d}/{start_episode + num_episodes} | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Delay: {avg_delay:.2f}s | "
                      f"Energy: {avg_energy:.2f}J | "
                      f"Policy Loss: {avg_policy_loss:.4f} | "
                      f"Value Loss: {avg_value_loss:.4f}")
            
            # Save checkpoint every 20 episodes
            if episode_count % 20 == 0:
                self._save_checkpoint(checkpoint_path)
        
        # Final save
        self._save_checkpoint(checkpoint_path)
        
        print(f"\n✓ Training complete!")
        print(f"  Total episodes: {self.training_history['episodes']}")
        print(f"  Total steps: {self.training_history['total_steps']}")
        print(f"  Best reward: {self.training_history['best_reward']:.4f}")
        print(f"  Model saved to: {checkpoint_path}")
        
        # Print final statistics
        print(f"\nFinal Training Statistics:")
        print(f"  Average Reward (last 100): {np.mean(self.training_history['episode_rewards'][-100:]):.4f}")
        print(f"  Average Delay (last 100): {np.mean(self.training_history['episode_delays'][-100:]):.4f}s")
        print(f"  Average Energy (last 100): {np.mean(self.training_history['episode_energies'][-100:]):.4f}J")
    
    def evaluate(self, num_episodes: int = 20):
        """
        Evaluate the trained agent
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        delays = []
        energies = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_delay = 0
            episode_energy = 0
            
            while not done:
                # Use deterministic policy for evaluation
                action, _, _ = self.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
                episode_delay += info['delay']
                episode_energy += info['energy']
            
            delays.append(episode_delay)
            energies.append(episode_energy)
        
        return {
            'avg_delay': np.mean(delays),
            'avg_energy': np.mean(energies),
            'std_delay': np.std(delays),
            'std_energy': np.std(energies)
        }
    
    def save(self, path: str):
        """Save model"""
        self._save_checkpoint(path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"✓ Model loaded from {path}")