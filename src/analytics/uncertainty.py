"""Uncertainty quantification and confidence estimation for threat detection."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


class UncertaintyEngine(nn.Module):
    """Advanced uncertainty engine with multiple uncertainty types."""
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_classes: int = 3):
        """Initialize uncertainty engine.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Epistemic uncertainty estimator (model uncertainty)
        self.epistemic_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Aleatoric uncertainty estimator (data uncertainty)
        self.aleatoric_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Softplus()  # Ensure positive values
        )
        
        # Confidence calibration network
        self.calibration_network = nn.Sequential(
            nn.Linear(input_dim + num_classes + 2, 128),  # +2 for uncertainty types
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        logger.info("UncertaintyEngine initialized")
    
    def forward(self, 
                features: torch.Tensor,
                predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through uncertainty engine.
        
        Args:
            features: Input features [batch_size, input_dim]
            predictions: Model predictions [batch_size, num_classes]
            
        Returns:
            Dictionary containing various uncertainty measures
        """
        batch_size = features.shape[0]
        
        # Epistemic uncertainty (model confidence)
        epistemic = self.epistemic_network(features).squeeze(-1)
        
        # Aleatoric uncertainty (per-class data uncertainty)
        aleatoric_logits = self.aleatoric_network(features)
        aleatoric = torch.mean(aleatoric_logits, dim=-1)  # Average across classes
        
        # Predictive entropy
        pred_probs = F.softmax(predictions, dim=-1)
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=-1)
        
        # Max probability (inverse uncertainty)
        max_prob = torch.max(pred_probs, dim=-1)[0]
        
        # Combined uncertainty
        total_uncertainty = torch.sqrt(epistemic ** 2 + aleatoric ** 2)
        
        # Confidence calibration
        calib_input = torch.cat([
            features,
            pred_probs,
            epistemic.unsqueeze(-1),
            aleatoric.unsqueeze(-1)
        ], dim=-1)
        
        calibrated_confidence = self.calibration_network(calib_input).squeeze(-1)
        
        return {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total_uncertainty,
            'entropy': entropy,
            'max_probability': max_prob,
            'calibrated_confidence': calibrated_confidence,
            'uncertainty': total_uncertainty  # Primary uncertainty measure
        }


class ConfidenceEngine:
    """Engine for computing and analyzing prediction confidence."""
    
    def __init__(self, config):
        """Initialize confidence engine.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.uncertainty_threshold = config.get("analytics.uncertainty_threshold", 0.5)
        
    def compute_confidence_metrics(self, 
                                 predictions: torch.Tensor,
                                 uncertainties: torch.Tensor,
                                 labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute comprehensive confidence metrics.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            uncertainties: Uncertainty estimates [batch_size]
            labels: True labels [batch_size] (optional)
            
        Returns:
            Dictionary of confidence metrics
        """
        pred_probs = F.softmax(predictions, dim=-1)
        max_probs = torch.max(pred_probs, dim=-1)[0]
        pred_classes = torch.argmax(pred_probs, dim=-1)
        
        metrics = {}
        
        # Basic confidence statistics
        metrics['mean_max_prob'] = torch.mean(max_probs).item()
        metrics['std_max_prob'] = torch.std(max_probs).item()
        metrics['mean_uncertainty'] = torch.mean(uncertainties).item()
        metrics['std_uncertainty'] = torch.std(uncertainties).item()
        
        # High confidence predictions
        high_conf_mask = uncertainties < self.uncertainty_threshold
        metrics['high_confidence_ratio'] = torch.mean(high_conf_mask.float()).item()
        
        if labels is not None:
            # Accuracy stratified by confidence
            correct = (pred_classes == labels).float()
            
            metrics['overall_accuracy'] = torch.mean(correct).item()
            
            if high_conf_mask.sum() > 0:
                metrics['high_conf_accuracy'] = torch.mean(correct[high_conf_mask]).item()
            else:
                metrics['high_conf_accuracy'] = 0.0
            
            low_conf_mask = ~high_conf_mask
            if low_conf_mask.sum() > 0:
                metrics['low_conf_accuracy'] = torch.mean(correct[low_conf_mask]).item()
            else:
                metrics['low_conf_accuracy'] = 0.0
            
            # Calibration metrics
            metrics.update(self._compute_calibration_metrics(pred_probs, labels))
        
        return metrics
    
    def _compute_calibration_metrics(self, 
                                   pred_probs: torch.Tensor,
                                   labels: torch.Tensor,
                                   n_bins: int = 10) -> Dict[str, float]:
        """Compute calibration metrics like Expected Calibration Error.
        
        Args:
            pred_probs: Predicted probabilities [batch_size, num_classes]
            labels: True labels [batch_size]
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of calibration metrics
        """
        max_probs = torch.max(pred_probs, dim=-1)[0]
        pred_classes = torch.argmax(pred_probs, dim=-1)
        correct = (pred_classes == labels).float()
        
        # Expected Calibration Error
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'expected_calibration_error': ece.item(),
            'avg_confidence': torch.mean(max_probs).item(),
            'avg_accuracy': torch.mean(correct).item()
        }
    
    def identify_uncertain_samples(self, 
                                 uncertainties: torch.Tensor,
                                 threshold: Optional[float] = None) -> torch.Tensor:
        """Identify samples that need human review due to high uncertainty.
        
        Args:
            uncertainties: Uncertainty scores [batch_size]
            threshold: Uncertainty threshold (uses config default if None)
            
        Returns:
            Boolean mask indicating uncertain samples
        """
        if threshold is None:
            threshold = self.uncertainty_threshold
        
        return uncertainties > threshold


def create_uncertainty_engine(config) -> UncertaintyEngine:
    """Create uncertainty engine from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured UncertaintyEngine
    """
    fusion_dim = config.get("models.fusion.hidden_dim", 512)
    
    engine = UncertaintyEngine(
        input_dim=fusion_dim,
        hidden_dim=256,
        num_classes=3
    )
    
    return engine


def create_confidence_engine(config) -> ConfidenceEngine:
    """Create confidence engine from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ConfidenceEngine
    """
    return ConfidenceEngine(config)
