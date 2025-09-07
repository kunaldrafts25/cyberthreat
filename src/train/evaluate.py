"""Model evaluation utilities for threat detection system."""

from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from loguru import logger

from ..common.config import get_config
from ..common.metrics import MetricsTracker, threat_detection_metrics
from ..models.threat_system import create_threat_detection_system
from ..neuro_symbolic.fusion_validate import create_neurosymbolic_fusion


class ThreatModelEvaluator:
    """Evaluator for threat detection models."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = create_threat_detection_system(config_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize neuro-symbolic fusion
        self.neuro_symbolic = create_neurosymbolic_fusion(self.config)
        
        logger.info(f"Model loaded from {model_path}")
    
    def evaluate_model(self, test_loader: DataLoader, sample_data: list = None) -> Dict:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            sample_data: Raw sample data for neuro-symbolic reasoning
            
        Returns:
            Comprehensive evaluation metrics
        """
        metrics_tracker = MetricsTracker(3, ['benign', 'suspicious', 'malicious'])
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move to device
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(batch, mode="inference")
                
                # Apply neuro-symbolic fusion if sample data available
                if sample_data and batch_idx < len(sample_data):
                    batch_sample_data = sample_data[batch_idx * batch['label'].size(0):(batch_idx + 1) * batch['label'].size(0)]
                    if batch_sample_data:
                        ns_outputs = self.neuro_symbolic(outputs, batch_sample_data)
                        final_probs = ns_outputs['probabilities']
                    else:
                        final_probs = torch.softmax(outputs['logits'], dim=1)
                else:
                    final_probs = torch.softmax(outputs['logits'], dim=1)
                
                predictions = torch.argmax(final_probs, dim=1)
                
                # Update metrics
                metrics_tracker.update(
                    predictions, 
                    batch['label'], 
                    final_probs,
                    outputs.get('uncertainty', None)
                )
                
                # Store for detailed analysis
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probabilities.extend(final_probs.cpu().numpy())
        
        # Compute standard metrics
        standard_metrics = metrics_tracker.compute()
        
        # Compute threat-specific metrics
        import numpy as np
        threat_metrics = threat_detection_metrics(
            np.array(all_labels),
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        
        # Combine all metrics
        evaluation_results = {
            **standard_metrics,
            **threat_metrics,
            'total_samples': len(all_predictions)
        }
        
        logger.info("Evaluation completed")
        logger.info(f"Overall Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"Macro F1: {evaluation_results['f1_macro']:.4f}")
        logger.info(f"Threat Detection Rate: {evaluation_results.get('threat_detection_rate', 0):.4f}")
        
        return evaluation_results


def create_evaluator(model_path: str, config_path: Optional[str] = None) -> ThreatModelEvaluator:
    """Create model evaluator."""
    return ThreatModelEvaluator(model_path, config_path)
