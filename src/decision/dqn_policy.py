"""Deep Q-Network policy for learning optimal threat response actions."""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from .rl_env import create_threat_decision_env


class DQNNetwork(nn.Module):
    """Deep Q-Network for threat decision making."""
    
    def __init__(self, 
                 state_dim: int = 7,
                 action_dim: int = 3,
                 hidden_dim: int = 128):
        """Initialize DQN network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension  
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Dueling DQN components
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        logger.info(f"DQNNetwork initialized: {state_dim} -> {action_dim}")
    
    def forward(self, state: torch.Tensor, dueling: bool = True) -> torch.Tensor:
        """Forward pass through DQN.
        
        Args:
            state: Input state tensor
            dueling: Whether to use dueling architecture
            
        Returns:
            Q-values for all actions
        """
        if dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_head(state)
            advantage = self.advantage_head(state)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values
        else:
            # Standard DQN
            return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences.
        
        Args:
            batch_size: Batch size to sample
            
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNPolicy:
    """DQN-based policy for threat response decisions."""
    
    def __init__(self, config):
        """Initialize DQN policy.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # RL hyperparameters
        self.learning_rate = config.get("rl.learning_rate", 1e-4)
        self.gamma = config.get("rl.gamma", 0.99)
        self.eps_start = config.get("rl.eps_start", 1.0)
        self.eps_end = config.get("rl.eps_end", 0.01)
        self.eps_decay = config.get("rl.eps_decay", 0.995)
        self.target_update = config.get("rl.target_update", 100)
        self.memory_size = config.get("rl.memory_size", 10000)
        
        # Networks
        self.q_network = DQNNetwork()
        self.target_network = DQNNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Training state
        self.epsilon = self.eps_start
        self.steps_done = 0
        self.episode = 0
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'q_values': []
        }
        
        logger.info("DQNPolicy initialized")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Random action
            return random.randrange(3)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step.
        
        Args:
            batch_size: Training batch size
            
        Returns:
            Training loss or None if insufficient data
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.steps_done += 1
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Record training statistics
        self.training_history['losses'].append(loss.item())
        self.training_history['q_values'].append(current_q_values.mean().item())
        
        return loss.item()
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """Train for one episode.
        
        Args:
            env: Training environment
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        losses = []
        
        for step in range(max_steps):
            # Select and perform action
            action = self.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Training step
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        self.episode += 1
        
        # Record episode statistics
        episode_stats = {
            'episode': self.episode,
            'total_reward': total_reward,
            'steps': steps,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory)
        }
        
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(steps)
        
        return episode_stats
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate policy performance.
        
        Args:
            env: Evaluation environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        total_rewards = []
        episode_lengths = []
        action_counts = {0: 0, 1: 0, 2: 0}  # monitor, isolate, block
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < 1000:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                
                total_reward += reward
                steps += 1
                action_counts[action] += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'action_distribution': {
                k: v / sum(action_counts.values()) 
                for k, v in action_counts.items()
            }
        }
    
    def save_checkpoint(self, filepath: str):
        """Save policy checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode': self.episode,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"DQN checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load policy checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode = checkpoint['episode']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"DQN checkpoint loaded from {filepath}")


def create_dqn_policy(config) -> DQNPolicy:
    """Create DQN policy from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured DQNPolicy
    """
    return DQNPolicy(config)


def train_dqn_policy(config, num_episodes: int = 1000) -> DQNPolicy:
    """Train DQN policy in threat decision environment.
    
    Args:
        config: System configuration
        num_episodes: Number of training episodes
        
    Returns:
        Trained DQNPolicy
    """
    env = create_threat_decision_env(config)
    policy = create_dqn_policy(config)
    
    logger.info(f"Starting DQN training for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        stats = policy.train_episode(env)
        
        if (episode + 1) % 100 == 0:
            eval_stats = policy.evaluate(env, num_episodes=5)
            logger.info(f"Episode {episode + 1}: "
                       f"reward={stats['total_reward']:.2f}, "
                       f"eval_reward={eval_stats['avg_reward']:.2f}, "
                       f"epsilon={stats['epsilon']:.3f}")
    
    logger.info("DQN training completed")
    return policy
