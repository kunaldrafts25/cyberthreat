"""Simulated Annealing optimizer as practical substitute for quantum optimization."""

import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger


class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimizer for complex decision policies."""
    
    def __init__(self, 
                 config,
                 objective_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 initial_temperature: Optional[float] = None,
                 cooling_rate: Optional[float] = None,
                 max_iterations: Optional[int] = None):
        """Initialize Simulated Annealing optimizer.
        
        Args:
            config: System configuration
            objective_function: Function to optimize
            bounds: List of (min, max) bounds for each parameter
            initial_temperature: Initial temperature (defaults from config)
            cooling_rate: Temperature cooling rate (defaults from config)
            max_iterations: Maximum iterations (defaults from config)
        """
        self.config = config
        self.objective_function = objective_function
        self.bounds = bounds
        self.dimension = len(bounds)
        
        # SA parameters
        self.initial_temperature = initial_temperature or config.get("analytics.sa_temperature_init", 1000.0)
        self.cooling_rate = cooling_rate or config.get("analytics.sa_cooling_rate", 0.95)
        self.max_iterations = max_iterations or config.get("analytics.sa_iterations", 1000)
        
        # Current state
        self.current_solution = None
        self.current_energy = float('inf')
        self.best_solution = None
        self.best_energy = float('inf')
        
        # Optimization history
        self.history = {
            'energies': [],
            'temperatures': [],
            'acceptance_rates': []
        }
        
        logger.info(f"SimulatedAnnealingOptimizer initialized: {self.dimension}D problem")
    
    def _generate_neighbor(self, solution: np.ndarray, temperature: float) -> np.ndarray:
        """Generate neighboring solution with temperature-dependent step size.
        
        Args:
            solution: Current solution
            temperature: Current temperature
            
        Returns:
            New neighboring solution
        """
        neighbor = solution.copy()
        
        # Temperature-dependent step size
        step_scale = temperature / self.initial_temperature
        
        for i in range(self.dimension):
            # Gaussian perturbation scaled by temperature
            noise = np.random.normal(0, step_scale * 0.1)
            neighbor[i] += noise
            
            # Ensure bounds are respected
            neighbor[i] = np.clip(neighbor[i], self.bounds[i][0], self.bounds[i][1])
        
        return neighbor
    
    def _acceptance_probability(self, current_energy: float, 
                              new_energy: float, 
                              temperature: float) -> float:
        """Calculate acceptance probability for new solution.
        
        Args:
            current_energy: Energy of current solution
            new_energy: Energy of candidate solution
            temperature: Current temperature
            
        Returns:
            Acceptance probability [0, 1]
        """
        if new_energy < current_energy:
            return 1.0  # Always accept better solutions
        
        if temperature <= 0:
            return 0.0
        
        try:
            return math.exp(-(new_energy - current_energy) / temperature)
        except OverflowError:
            return 0.0
    
    def optimize(self, 
                 initial_solution: Optional[np.ndarray] = None,
                 callback: Optional[Callable] = None) -> Dict[str, Union[np.ndarray, float, List]]:
        """Run simulated annealing optimization.
        
        Args:
            initial_solution: Starting solution (random if None)
            callback: Optional callback function called each iteration
            
        Returns:
            Optimization results dictionary
        """
        # Initialize solution
        if initial_solution is None:
            self.current_solution = np.array([
                np.random.uniform(bounds[0], bounds[1]) 
                for bounds in self.bounds
            ])
        else:
            self.current_solution = initial_solution.copy()
        
        self.current_energy = self.objective_function(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_energy = self.current_energy
        
        # Reset history
        self.history = {
            'energies': [self.current_energy],
            'temperatures': [self.initial_temperature],
            'acceptance_rates': []
        }
        
        temperature = self.initial_temperature
        accepted_moves = 0
        total_moves = 0
        
        logger.info(f"Starting SA optimization: initial energy = {self.current_energy:.6f}")
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            candidate_solution = self._generate_neighbor(self.current_solution, temperature)
            candidate_energy = self.objective_function(candidate_solution)
            
            total_moves += 1
            
            # Accept or reject candidate
            acceptance_prob = self._acceptance_probability(
                self.current_energy, candidate_energy, temperature
            )
            
            if random.random() < acceptance_prob:
                self.current_solution = candidate_solution
                self.current_energy = candidate_energy
                accepted_moves += 1
                
                # Update best solution if needed
                if candidate_energy < self.best_energy:
                    self.best_solution = candidate_solution.copy()
                    self.best_energy = candidate_energy
            
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Record history
            self.history['energies'].append(self.current_energy)
            self.history['temperatures'].append(temperature)
            
            # Calculate acceptance rate over last 100 moves
            if total_moves >= 100:
                recent_acceptance_rate = accepted_moves / total_moves
                self.history['acceptance_rates'].append(recent_acceptance_rate)
                accepted_moves = 0
                total_moves = 0
            
            # Callback
            if callback:
                callback(iteration, self.current_solution, self.current_energy, temperature)
            
            # Early stopping if temperature is very low
            if temperature < 1e-8:
                logger.info(f"Early stopping at iteration {iteration}: temperature too low")
                break
            
            # Progress logging
            if (iteration + 1) % 100 == 0:
                logger.debug(f"SA iteration {iteration + 1}: energy = {self.current_energy:.6f}, "
                           f"best = {self.best_energy:.6f}, temp = {temperature:.6f}")
        
        logger.info(f"SA optimization completed: best energy = {self.best_energy:.6f}")
        
        return {
            'best_solution': self.best_solution,
            'best_energy': self.best_energy,
            'final_solution': self.current_solution,
            'final_energy': self.current_energy,
            'history': self.history,
            'iterations': min(iteration + 1, self.max_iterations)
        }


class ThreatDecisionOptimizer:
    """Optimizer for threat response decision policies using SA."""
    
    def __init__(self, config):
        """Initialize threat decision optimizer.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Decision parameters to optimize
        # [monitor_threshold, isolate_threshold, block_threshold, 
        #  response_time_weight, severity_weight, confidence_weight]
        self.parameter_bounds = [
            (0.1, 0.9),  # monitor_threshold
            (0.3, 0.8),  # isolate_threshold  
            (0.5, 0.95), # block_threshold
            (0.0, 1.0),  # response_time_weight
            (0.0, 1.0),  # severity_weight
            (0.0, 1.0),  # confidence_weight
        ]
        
        # Metrics for optimization
        self.optimization_history = []
        
    def create_objective_function(self, 
                                training_data: List[Dict],
                                ground_truth_actions: List[int]) -> Callable[[np.ndarray], float]:
        """Create objective function for decision policy optimization.
        
        Args:
            training_data: List of threat detection samples
            ground_truth_actions: Optimal actions for each sample
            
        Returns:
            Objective function that takes parameters and returns cost
        """
        def objective(params: np.ndarray) -> float:
            monitor_thresh, isolate_thresh, block_thresh, rt_weight, sev_weight, conf_weight = params
            
            total_cost = 0.0
            num_samples = len(training_data)
            
            for i, sample in enumerate(training_data):
                # Extract sample features
                threat_prob = sample.get('threat_probability', 0.5)
                severity = sample.get('severity_score', 0.5)
                confidence = sample.get('confidence', 0.5)
                response_time = sample.get('response_time', 1.0)
                
                # Compute decision score
                decision_score = (
                    threat_prob * 0.4 +
                    severity * sev_weight * 0.3 +
                    confidence * conf_weight * 0.2 +
                    (1.0 - response_time) * rt_weight * 0.1
                )
                
                # Make decision based on thresholds
                if decision_score >= block_thresh:
                    predicted_action = 2  # block
                elif decision_score >= isolate_thresh:
                    predicted_action = 1  # isolate
                elif decision_score >= monitor_thresh:
                    predicted_action = 0  # monitor
                else:
                    predicted_action = 0  # default to monitor
                
                # Cost function
                if i < len(ground_truth_actions):
                    true_action = ground_truth_actions[i]
                    
                    # Penalty for wrong decisions
                    if predicted_action != true_action:
                        # Higher penalty for under-responding to threats
                        if predicted_action < true_action:
                            penalty = (true_action - predicted_action) * 2.0
                        else:
                            penalty = (predicted_action - true_action) * 1.0
                        total_cost += penalty
                    
                    # Cost for actions themselves
                    action_costs = [0.1, 0.5, 1.0]  # monitor, isolate, block
                    total_cost += action_costs[predicted_action] * 0.1
            
            return total_cost / num_samples
        
        return objective
    
    def optimize_decision_policy(self, 
                               training_data: List[Dict],
                               ground_truth_actions: List[int]) -> Dict[str, Union[np.ndarray, float]]:
        """Optimize decision policy parameters using simulated annealing.
        
        Args:
            training_data: Training samples
            ground_truth_actions: Optimal actions
            
        Returns:
            Optimized policy parameters
        """
        objective_function = self.create_objective_function(training_data, ground_truth_actions)
        
        optimizer = SimulatedAnnealingOptimizer(
            config=self.config,
            objective_function=objective_function,
            bounds=self.parameter_bounds
        )
        
        # Initial solution (reasonable defaults)
        initial_solution = np.array([0.3, 0.5, 0.7, 0.2, 0.6, 0.8])
        
        def optimization_callback(iteration, solution, energy, temperature):
            if iteration % 50 == 0:
                logger.debug(f"Policy optimization iter {iteration}: cost = {energy:.6f}")
        
        results = optimizer.optimize(
            initial_solution=initial_solution,
            callback=optimization_callback
        )
        
        # Extract optimized parameters
        optimized_params = results['best_solution']
        param_names = [
            'monitor_threshold', 'isolate_threshold', 'block_threshold',
            'response_time_weight', 'severity_weight', 'confidence_weight'
        ]
        
        optimized_policy = {
            name: float(param) for name, param in zip(param_names, optimized_params)
        }
        
        logger.info("Optimized decision policy parameters:")
        for name, value in optimized_policy.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return {
            'policy_parameters': optimized_policy,
            'optimization_cost': results['best_energy'],
            'optimization_history': results['history']
        }


def create_threat_decision_optimizer(config) -> ThreatDecisionOptimizer:
    """Create threat decision optimizer from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ThreatDecisionOptimizer
    """
    return ThreatDecisionOptimizer(config)
