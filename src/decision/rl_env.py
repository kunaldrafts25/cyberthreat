"""Reinforcement Learning environment for threat response decision making."""

from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger


class ThreatDecisionEnvironment(gym.Env):
    """RL environment for learning optimal threat response actions."""
    
    def __init__(self, config):
        """Initialize threat decision environment.
        
        Args:
            config: System configuration
        """
        super().__init__()
        
        self.config = config
        
        # Action space: 0=monitor, 1=isolate, 2=block
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [threat_prob, severity, confidence, response_time, 
        #                    num_similar_threats, time_of_day, network_load]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # Environment state
        self.current_threat = None
        self.step_count = 0
        self.episode_length = 100
        
        # Threat database for sampling
        self.threat_database = self._initialize_threat_database()
        
        # Reward parameters
        self.reward_params = {
            'correct_action_bonus': 10.0,
            'wrong_action_penalty': -5.0,
            'overreaction_penalty': -3.0,
            'underreaction_penalty': -7.0,
            'action_costs': [0.1, 2.0, 5.0],  # monitor, isolate, block costs
            'false_positive_penalty': -8.0,
            'false_negative_penalty': -15.0
        }
        
        logger.info("ThreatDecisionEnvironment initialized")
    
    def _initialize_threat_database(self) -> List[Dict]:
        """Initialize database of threat scenarios.
        
        Returns:
            List of threat scenario dictionaries
        """
        scenarios = []
        
        # Benign scenarios
        for i in range(50):
            scenarios.append({
                'threat_probability': np.random.uniform(0.0, 0.3),
                'severity_score': np.random.uniform(0.0, 0.4),
                'confidence': np.random.uniform(0.6, 1.0),
                'response_time': np.random.uniform(0.1, 0.5),
                'similar_threats': np.random.randint(0, 3),
                'time_of_day': np.random.uniform(0.0, 1.0),
                'network_load': np.random.uniform(0.2, 0.8),
                'true_label': 0,  # benign
                'optimal_action': 0  # monitor
            })
        
        # Suspicious scenarios
        for i in range(30):
            scenarios.append({
                'threat_probability': np.random.uniform(0.3, 0.7),
                'severity_score': np.random.uniform(0.3, 0.7),
                'confidence': np.random.uniform(0.4, 0.8),
                'response_time': np.random.uniform(0.3, 0.8),
                'similar_threats': np.random.randint(1, 5),
                'time_of_day': np.random.uniform(0.0, 1.0),
                'network_load': np.random.uniform(0.3, 0.9),
                'true_label': 1,  # suspicious
                'optimal_action': 1  # isolate
            })
        
        # Malicious scenarios
        for i in range(20):
            scenarios.append({
                'threat_probability': np.random.uniform(0.7, 1.0),
                'severity_score': np.random.uniform(0.6, 1.0),
                'confidence': np.random.uniform(0.5, 0.9),
                'response_time': np.random.uniform(0.6, 1.0),
                'similar_threats': np.random.randint(2, 8),
                'time_of_day': np.random.uniform(0.0, 1.0),
                'network_load': np.random.uniform(0.5, 1.0),
                'true_label': 2,  # malicious
                'optimal_action': 2  # block
            })
        
        return scenarios
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        self.step_count = 0
        self.current_threat = np.random.choice(self.threat_database)
        
        observation = self._get_observation()
        info = {'threat_scenario': self.current_threat.copy()}
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return new state.
        
        Args:
            action: Action to take (0=monitor, 1=isolate, 2=block)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        terminated = False  # Task doesn't have natural termination
        truncated = self.step_count >= self.episode_length
        
        # Sample new threat for next step
        self.current_threat = np.random.choice(self.threat_database)
        observation = self._get_observation()
        
        info = {
            'action_taken': action,
            'reward_breakdown': self._get_reward_breakdown(action),
            'threat_scenario': self.current_threat.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from threat scenario.
        
        Returns:
            Observation array
        """
        threat = self.current_threat
        return np.array([
            threat['threat_probability'],
            threat['severity_score'],
            threat['confidence'],
            threat['response_time'],
            min(threat['similar_threats'] / 10.0, 1.0),  # Normalize
            threat['time_of_day'],
            threat['network_load']
        ], dtype=np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for taking action in current state.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
        """
        threat = self.current_threat
        optimal_action = threat['optimal_action']
        true_label = threat['true_label']
        
        reward = 0.0
        
        # Base reward for correct action
        if action == optimal_action:
            reward += self.reward_params['correct_action_bonus']
        else:
            reward += self.reward_params['wrong_action_penalty']
            
            # Additional penalties for specific types of mistakes
            if action > optimal_action:  # Overreaction
                reward += self.reward_params['overreaction_penalty']
            else:  # Underreaction
                reward += self.reward_params['underreaction_penalty']
        
        # Action cost
        reward -= self.reward_params['action_costs'][action]
        
        # False positive/negative penalties
        if true_label == 0 and action > 0:  # Benign but took action
            reward += self.reward_params['false_positive_penalty']
        elif true_label == 2 and action == 0:  # Malicious but only monitoring
            reward += self.reward_params['false_negative_penalty']
        
        # Bonus for high-confidence correct decisions
        if action == optimal_action:
            confidence_bonus = threat['confidence'] * 2.0
            reward += confidence_bonus
        
        # Penalty for acting on low-confidence predictions
        if action > 0 and threat['confidence'] < 0.5:
            reward -= 3.0
        
        return reward
    
    def _get_reward_breakdown(self, action: int) -> Dict[str, float]:
        """Get detailed breakdown of reward components.
        
        Args:
            action: Action taken
            
        Returns:
            Dictionary of reward components
        """
        threat = self.current_threat
        optimal_action = threat['optimal_action']
        true_label = threat['true_label']
        
        breakdown = {
            'action_cost': -self.reward_params['action_costs'][action],
            'correctness_reward': 0.0,
            'false_positive_penalty': 0.0,
            'false_negative_penalty': 0.0,
            'confidence_bonus': 0.0,
            'low_confidence_penalty': 0.0
        }
        
        # Correctness reward
        if action == optimal_action:
            breakdown['correctness_reward'] = self.reward_params['correct_action_bonus']
            breakdown['confidence_bonus'] = threat['confidence'] * 2.0
        else:
            breakdown['correctness_reward'] = self.reward_params['wrong_action_penalty']
        
        # False positive/negative
        if true_label == 0 and action > 0:
            breakdown['false_positive_penalty'] = self.reward_params['false_positive_penalty']
        elif true_label == 2 and action == 0:
            breakdown['false_negative_penalty'] = self.reward_params['false_negative_penalty']
        
        # Low confidence penalty
        if action > 0 and threat['confidence'] < 0.5:
            breakdown['low_confidence_penalty'] = -3.0
        
        return breakdown
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render environment state.
        
        Args:
            mode: Render mode
            
        Returns:
            String representation if mode is 'ansi'
        """
        threat = self.current_threat
        obs = self._get_observation()
        
        output = f"""
Threat Decision Environment - Step {self.step_count}
==============================================
Threat Probability: {obs[0]:.3f}
Severity Score:     {obs[1]:.3f}
Confidence:         {obs[2]:.3f}
Response Time:      {obs[3]:.3f}
Similar Threats:    {obs[4]:.3f}
Time of Day:        {obs[5]:.3f}
Network Load:       {obs[6]:.3f}

True Label:         {threat['true_label']} ({'benign' if threat['true_label']==0 else 'suspicious' if threat['true_label']==1 else 'malicious'})
Optimal Action:     {threat['optimal_action']} ({'monitor' if threat['optimal_action']==0 else 'isolate' if threat['optimal_action']==1 else 'block'})
"""
        
        if mode == 'ansi':
            return output
        elif mode == 'human':
            print(output)
        
        return None


def create_threat_decision_env(config) -> ThreatDecisionEnvironment:
    """Create threat decision environment from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ThreatDecisionEnvironment
    """
    return ThreatDecisionEnvironment(config)
