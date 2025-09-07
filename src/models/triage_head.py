"""Rapid threat triage head with Monte Carlo Dropout for uncertainty estimation."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer that stays active during inference."""
    
    def __init__(self, p: float = 0.5):
        """Initialize MC Dropout.
        
        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with always-active dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with dropout applied
        """
        return F.dropout(x, p=self.p, training=True)  # Always training=True for MC


class UncertaintyEstimator(nn.Module):
    """Epistemic and aleatoric uncertainty estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """Initialize uncertainty estimator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Epistemic uncertainty (model uncertainty)
        self.epistemic_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            MCDropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            MCDropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Aleatoric uncertainty (data uncertainty)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive values
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through uncertainty estimator.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary containing uncertainty estimates
        """
        epistemic = self.epistemic_head(x)
        aleatoric = self.aleatoric_head(x)
        
        # Combined uncertainty
        total_uncertainty = torch.sqrt(epistemic ** 2 + aleatoric ** 2)
        
        return {
            'epistemic_uncertainty': epistemic.squeeze(-1),
            'aleatoric_uncertainty': aleatoric.squeeze(-1),
            'total_uncertainty': total_uncertainty.squeeze(-1)
        }


class ThreatTriageHead(nn.Module):
    """Rapid threat triage classification head with uncertainty quantification."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int = 3,
                 hidden_dim: int = 256,
                 dropout_samples: int = 100,
                 dropout_rate: float = 0.3):
        """Initialize threat triage head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of threat classes (benign, suspicious, malicious)
            hidden_dim: Hidden layer dimension
            dropout_samples: Number of MC dropout samples for uncertainty
            dropout_rate: Dropout rate for MC sampling
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_samples = dropout_samples
        
        # Main classification head with MC dropout
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            MCDropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            MCDropout(dropout_rate * 0.7),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            MCDropout(dropout_rate * 0.5),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(input_dim, hidden_dim)
        
        # Confidence calibration
        self.calibration_layer = nn.Sequential(
            nn.Linear(num_classes + 3, 32),  # +3 for uncertainty measures
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Threat severity scorer
        self.severity_scorer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"ThreatTriageHead initialized: {input_dim} -> {num_classes} classes, "
                   f"{dropout_samples} MC samples")
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through threat triage head.
        
        Args:
            x: Input features [batch_size, input_dim]
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Dictionary containing predictions, probabilities, and uncertainties
        """
        batch_size = x.shape[0]
        
        # Single forward pass for base prediction
        logits = self.classifier(x)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        # Compute uncertainty estimates
        uncertainty_dict = self.uncertainty_estimator(x)
        
        # Threat severity score
        severity_scores = self.severity_scorer(x).squeeze(-1)
        
        outputs = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'severity_scores': severity_scores,
            **uncertainty_dict
        }
        
        if return_uncertainty:
            # Monte Carlo sampling for epistemic uncertainty
            mc_predictions = []
            
            for _ in range(self.dropout_samples):
                mc_logits = self.classifier(x)
                mc_probs = F.softmax(mc_logits, dim=-1)
                mc_predictions.append(mc_probs)
            
            # Stack MC predictions
            mc_predictions = torch.stack(mc_predictions, dim=0)  # [samples, batch_size, num_classes]
            
            # Compute statistics
            mc_mean = mc_predictions.mean(dim=0)  # [batch_size, num_classes]
            mc_std = mc_predictions.std(dim=0)    # [batch_size, num_classes]
            
            # Predictive entropy (epistemic + aleatoric uncertainty)
            mc_entropy = -torch.sum(mc_mean * torch.log(mc_mean + 1e-8), dim=-1)
            
            # Mutual information (epistemic uncertainty)
            entropy_of_expected = -torch.sum(mc_mean * torch.log(mc_mean + 1e-8), dim=-1)
            expected_entropy = -torch.mean(
                torch.sum(mc_predictions * torch.log(mc_predictions + 1e-8), dim=-1),
                dim=0
            )
            mutual_info = entropy_of_expected - expected_entropy
            
            # Confidence calibration
            calib_input = torch.cat([
                logits,
                uncertainty_dict['epistemic_uncertainty'].unsqueeze(-1),
                uncertainty_dict['aleatoric_uncertainty'].unsqueeze(-1),
                mc_entropy.unsqueeze(-1)
            ], dim=-1)
            
            calibrated_confidence = self.calibration_layer(calib_input).squeeze(-1)
            
            # Update outputs with MC uncertainty
            outputs.update({
                'mc_predictions': mc_predictions,
                'mc_mean_probabilities': mc_mean,
                'mc_std': mc_std,
                'predictive_entropy': mc_entropy,
                'mutual_information': mutual_info,
                'epistemic_uncertainty_mc': mutual_info,
                'calibrated_confidence': calibrated_confidence,
                'uncertainty': mc_entropy  # Primary uncertainty measure
            })
        
        # Threat-specific classifications
        threat_info = self._classify_threat_type(probabilities, severity_scores)
        outputs.update(threat_info)
        
        return outputs
    
    def _classify_threat_type(self, 
                             probabilities: torch.Tensor, 
                             severity_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify specific threat types based on probabilities.
        
        Args:
            probabilities: Class probabilities [batch_size, num_classes]
            severity_scores: Threat severity scores [batch_size]
            
        Returns:
            Dictionary containing threat classifications
        """
        batch_size = probabilities.shape[0]
        
        # Threat level determination
        threat_levels = torch.argmax(probabilities, dim=-1)
        
        # Alert generation logic
        suspicious_prob = probabilities[:, 1] if self.num_classes > 1 else torch.zeros(batch_size)
        malicious_prob = probabilities[:, 2] if self.num_classes > 2 else torch.zeros(batch_size)
        
        # Generate alerts based on thresholds
        should_alert = (suspicious_prob > 0.3) | (malicious_prob > 0.1)
        alert_priority = torch.where(
            malicious_prob > 0.5, 3,  # Critical
            torch.where(
                malicious_prob > 0.2, 2,  # High
                torch.where(suspicious_prob > 0.5, 1, 0)  # Medium
            )
        )
        
        # Risk assessment
        risk_scores = (
            probabilities[:, 0] * 0.1 +  # Benign contributes little
            probabilities[:, 1] * 0.5 +  # Suspicious moderate risk
            probabilities[:, 2] * 1.0    # Malicious high risk
        )
        risk_scores = risk_scores * severity_scores  # Weight by severity
        
        return {
            'threat_levels': threat_levels,
            'should_alert': should_alert,
            'alert_priority': alert_priority,
            'risk_scores': risk_scores,
            'is_malicious': malicious_prob > 0.5,
            'is_suspicious': suspicious_prob > 0.3,
            'needs_investigation': (suspicious_prob > 0.2) | (malicious_prob > 0.1)
        }
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor, 
                                confidence_threshold: float = 0.7) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty-aware decision making.
        
        Args:
            x: Input features
            confidence_threshold: Threshold for high-confidence predictions
            
        Returns:
            Dictionary containing uncertainty-aware predictions
        """
        outputs = self.forward(x, return_uncertainty=True)
        
        # Uncertainty-based decision making
        uncertainty = outputs['uncertainty']
        calibrated_confidence = outputs['calibrated_confidence']
        
        # High uncertainty samples need human review
        needs_human_review = (uncertainty > 0.8) | (calibrated_confidence < confidence_threshold)
        
        # Adjust predictions based on uncertainty
        adjusted_predictions = outputs['predictions'].clone()
        
        # Conservative: if high uncertainty and predicted benign, escalate to suspicious
        high_uncertainty_benign = needs_human_review & (outputs['predictions'] == 0)
        adjusted_predictions[high_uncertainty_benign] = 1
        
        # Decision confidence
        decision_confidence = torch.where(
            needs_human_review,
            torch.tensor(0.0, device=x.device),  # No confidence for human review cases
            calibrated_confidence
        )
        
        return {
            **outputs,
            'adjusted_predictions': adjusted_predictions,
            'needs_human_review': needs_human_review,
            'decision_confidence': decision_confidence,
            'high_confidence_predictions': calibrated_confidence > confidence_threshold
        }


def create_threat_triage_head(config) -> ThreatTriageHead:
    """Create threat triage head from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ThreatTriageHead
    """
    triage_config = config.get("models.triage", {})
    fusion_dim = config.get("models.fusion.hidden_dim", 512)
    
    triage_head = ThreatTriageHead(
        input_dim=fusion_dim,
        num_classes=3,  # benign, suspicious, malicious
        hidden_dim=triage_config.get("hidden_dim", 256),
        dropout_samples=triage_config.get("dropout_samples", 100),
        dropout_rate=triage_config.get("dropout", 0.3)
    )
    
    return triage_head
