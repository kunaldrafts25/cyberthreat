"""Neuro-symbolic fusion and validation combining ML predictions with rule-based reasoning."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

from .knowledge_graph import create_threat_knowledge_graph
from .rules import create_threat_rule_engine


class NeuroSymbolicFusion(nn.Module):
    """Fusion module combining neural predictions with symbolic reasoning."""
    
    def __init__(self, config):
        """Initialize neuro-symbolic fusion.
        
        Args:
            config: System configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize symbolic components
        self.knowledge_graph = create_threat_knowledge_graph(config)
        self.rule_engine = create_threat_rule_engine(config)
        
        # Neural components for fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(9, 64),  # 3 neural probs + 3 rule scores + 3 KG scores
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Output probabilities
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(9 + 3, 32),  # Input features + fusion output
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # neural, rules, KG
        
        logger.info("NeuroSymbolicFusion initialized")
    
    def forward(self, 
                neural_output: Dict[str, torch.Tensor],
                sample_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Forward pass through neuro-symbolic fusion.
        
        Args:
            neural_output: Output from neural network
            sample_data: Raw sample data for symbolic reasoning
            
        Returns:
            Fused predictions and explanations
        """
        batch_size = neural_output['probabilities'].shape[0]
        device = neural_output['probabilities'].device
        
        # Extract neural predictions
        neural_probs = neural_output['probabilities']  # [batch_size, 3]
        neural_uncertainty = neural_output.get('uncertainty', torch.zeros(batch_size, device=device))
        
        # Apply symbolic reasoning
        rule_scores = []
        kg_scores = []
        explanations = []
        
        for i in range(batch_size):
            if i < len(sample_data):
                sample = sample_data[i]
                
                # Rule-based evaluation
                rule_result = self.rule_engine.evaluate_sample(sample)
                rule_score = [
                    rule_result['scores'].get('benign', 0.0),
                    rule_result['scores'].get('suspicious', 0.0),
                    rule_result['scores'].get('malicious', 0.0)
                ]
                rule_scores.append(rule_score)
                
                # Knowledge graph reasoning
                kg_score = self._kg_reasoning(sample)
                kg_scores.append(kg_score)
                
                # Combine explanations
                explanation = {
                    'rule_explanations': rule_result['explanations'],
                    'fired_rules': rule_result['fired_rules'],
                    'kg_entities': self._extract_kg_entities(sample)
                }
                explanations.append(explanation)
            else:
                # Fallback for missing sample data
                rule_scores.append([0.33, 0.33, 0.34])
                kg_scores.append([0.33, 0.33, 0.34])
                explanations.append({'rule_explanations': [], 'fired_rules': [], 'kg_entities': []})
        
        # Convert to tensors
        rule_scores = torch.tensor(rule_scores, device=device, dtype=torch.float32)
        kg_scores = torch.tensor(kg_scores, device=device, dtype=torch.float32)
        
        # Neural-symbolic fusion
        fusion_input = torch.cat([neural_probs, rule_scores, kg_scores], dim=-1)
        fused_probs = self.fusion_network(fusion_input)
        
        # Confidence estimation
        confidence_input = torch.cat([fusion_input, fused_probs], dim=-1)
        fused_confidence = self.confidence_estimator(confidence_input).squeeze(-1)
        
        # Weighted combination approach (alternative to learned fusion)
        weights = torch.softmax(self.fusion_weights, dim=0)
        weighted_probs = (
            weights[0] * neural_probs +
            weights[1] * rule_scores +
            weights[2] * kg_scores
        )
        
        # Use learned fusion as primary, weighted as backup
        final_probs = fused_probs
        final_confidence = fused_confidence
        
        # Generate final predictions
        final_predictions = torch.argmax(final_probs, dim=-1)
        
        # Validation and consistency checks
        validation_results = self._validate_predictions(
            neural_probs, rule_scores, kg_scores, final_probs, sample_data
        )
        
        return {
            'probabilities': final_probs,
            'predictions': final_predictions,
            'confidence': final_confidence,
            'neural_probs': neural_probs,
            'rule_scores': rule_scores,
            'kg_scores': kg_scores,
            'weighted_probs': weighted_probs,
            'fusion_weights': weights,
            'explanations': explanations,
            'validation': validation_results,
            'uncertainty': neural_uncertainty  # Pass through neural uncertainty
        }
    
    def _kg_reasoning(self, sample: Dict) -> List[float]:
        """Apply knowledge graph reasoning to sample.
        
        Args:
            sample: Sample data dictionary
            
        Returns:
            List of threat scores [benign, suspicious, malicious]
        """
        scores = [0.5, 0.3, 0.2]  # Default distribution
        
        try:
            # Extract entities from sample
            entities = []
            
            # IP addresses
            for ip_field in ['SourceAddress', 'DestinationAddress']:
                if ip_field in sample and sample[ip_field]:
                    entities.append(f"IP:{sample[ip_field]}")
            
            # Port
            if 'Port' in sample:
                entities.append(f"PORT:port_{sample['Port']}")
            
            # Protocol
            if 'Protocol' in sample:
                entities.append(f"PROTOCOL:{str(sample['Protocol']).lower()}")
            
            # Geolocation
            if 'Geolocation' in sample:
                entities.append(f"GEOLOCATION:{sample['Geolocation']}")
            
            # Query knowledge graph for each entity
            threat_level_counts = {'benign': 0, 'suspicious': 0, 'malicious': 0}
            
            for entity in entities:
                threat_level = self.knowledge_graph.get_threat_level(entity)
                if threat_level:
                    threat_level_counts[threat_level] += 1
            
            # Convert counts to probabilities
            total_counts = sum(threat_level_counts.values())
            if total_counts > 0:
                scores = [
                    threat_level_counts['benign'] / total_counts,
                    threat_level_counts['suspicious'] / total_counts,
                    threat_level_counts['malicious'] / total_counts
                ]
            
        except Exception as e:
            logger.warning(f"KG reasoning error: {e}")
        
        return scores
    
    def _extract_kg_entities(self, sample: Dict) -> List[str]:
        """Extract knowledge graph entities from sample.
        
        Args:
            sample: Sample data dictionary
            
        Returns:
            List of relevant entities
        """
        entities = []
        
        # IP entities
        for ip_field in ['SourceAddress', 'DestinationAddress']:
            if ip_field in sample and sample[ip_field]:
                entities.append(f"IP:{sample[ip_field]}")
        
        # Port entity
        if 'Port' in sample:
            entities.append(f"PORT:port_{sample['Port']}")
        
        # Protocol entity
        if 'Protocol' in sample:
            entities.append(f"PROTOCOL:{str(sample['Protocol']).lower()}")
        
        return entities
    
    def _validate_predictions(self, 
                            neural_probs: torch.Tensor,
                            rule_scores: torch.Tensor,
                            kg_scores: torch.Tensor,
                            final_probs: torch.Tensor,
                            sample_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Validate consistency between different reasoning approaches.
        
        Args:
            neural_probs: Neural network probabilities
            rule_scores: Rule-based scores
            kg_scores: Knowledge graph scores
            final_probs: Final fused probabilities
            sample_data: Raw sample data
            
        Returns:
            Validation results dictionary
        """
        batch_size = neural_probs.shape[0]
        
        # Compute disagreement scores
        neural_pred = torch.argmax(neural_probs, dim=-1)
        rule_pred = torch.argmax(rule_scores, dim=-1)
        kg_pred = torch.argmax(kg_scores, dim=-1)
        final_pred = torch.argmax(final_probs, dim=-1)
        
        # Disagreement indicators
        neural_rule_agree = (neural_pred == rule_pred).float()
        neural_kg_agree = (neural_pred == kg_pred).float()
        rule_kg_agree = (rule_pred == kg_pred).float()
        
        # Overall consensus
        consensus_score = (neural_rule_agree + neural_kg_agree + rule_kg_agree) / 3
        
        # Prediction shifts (how much fusion changed neural predictions)
        prob_shift = torch.norm(final_probs - neural_probs, dim=-1)
        
        # High-disagreement samples need review
        needs_review = (consensus_score < 0.5) | (prob_shift > 0.5)
        
        return {
            'neural_rule_agreement': neural_rule_agree,
            'neural_kg_agreement': neural_kg_agree,
            'rule_kg_agreement': rule_kg_agree,
            'consensus_score': consensus_score,
            'probability_shift': prob_shift,
            'needs_review': needs_review,
            'high_disagreement': consensus_score < 0.3
        }
    
    def explain_prediction(self, 
                          idx: int, 
                          neural_output: Dict[str, torch.Tensor],
                          sample_data: Dict) -> Dict[str, any]:
        """Generate detailed explanation for a prediction.
        
        Args:
            idx: Sample index
            neural_output: Neural network output
            sample_data: Raw sample data
            
        Returns:
            Detailed explanation dictionary
        """
        # Rule-based explanation
        rule_result = self.rule_engine.evaluate_sample(sample_data)
        
        # Knowledge graph explanation
        kg_entities = self._extract_kg_entities(sample_data)
        kg_explanations = []
        for entity in kg_entities:
            explanations = self.knowledge_graph.explain_classification(entity)
            kg_explanations.extend(explanations)
        
        # Neural network contribution
        neural_prob = neural_output['probabilities'][idx]
        neural_pred = torch.argmax(neural_prob).item()
        
        explanation = {
            'final_prediction': neural_pred,
            'neural_contribution': {
                'probabilities': neural_prob.tolist(),
                'prediction': neural_pred,
                'confidence': neural_output.get('confidence', [1.0])[idx].item()
            },
            'rule_contribution': {
                'scores': rule_result['scores'],
                'fired_rules': rule_result['fired_rules'],
                'explanations': rule_result['explanations']
            },
            'kg_contribution': {
                'entities': kg_entities,
                'explanations': kg_explanations
            },
            'consensus': {
                'all_agree': len(set([
                    neural_pred,
                    max(rule_result['scores'], key=rule_result['scores'].get),
                    # Add KG prediction here
                ])) == 1
            }
        }
        
        return explanation


def create_neurosymbolic_fusion(config) -> NeuroSymbolicFusion:
    """Create neuro-symbolic fusion module from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured NeuroSymbolicFusion module
    """
    return NeuroSymbolicFusion(config)
