"""Inference engine for threat detection serving."""

from typing import Dict, List, Optional

import torch
import numpy as np
from loguru import logger

from ..common.config import get_config
from ..common.utils import get_device
from ..models.threat_system import create_threat_detection_system
from ..neuro_symbolic.fusion_validate import create_neurosymbolic_fusion
from ..data.preprocessing import ThreatFeatureProcessor


class ThreatInferenceEngine:
    """Inference engine for threat detection predictions."""
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None):
        """Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.device = get_device(self.config.get("system.device", "auto"))
        
        # Load model
        self.model = create_threat_detection_system(config_path)
        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.neuro_symbolic = create_neurosymbolic_fusion(self.config)
        self.feature_processor = ThreatFeatureProcessor(self.config)
        
        logger.info("ThreatInferenceEngine initialized")
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
    
    def predict_single(self, sample_data: Dict) -> Dict:
        """Make prediction for a single sample.
        
        Args:
            sample_data: Raw sample data dictionary
            
        Returns:
            Prediction results dictionary
        """
        return self.predict_batch([sample_data])[0]
    
    def predict_batch(self, samples: List[Dict]) -> List[Dict]:
        """Make predictions for a batch of samples.
        
        Args:
            samples: List of raw sample data dictionaries
            
        Returns:
            List of prediction results dictionaries
        """
        # Process features
        import pandas as pd
        df = pd.DataFrame(samples)
        features = self.feature_processor.process_features(df)
        
        # Convert to tensors
        batch = {}
        batch_size = len(samples)
        
        for modality, feature_array in features.items():
            batch[modality] = torch.tensor(feature_array, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Neural prediction
            outputs = self.model(batch, mode="inference")
            
            # Neuro-symbolic fusion
            ns_outputs = self.neuro_symbolic(outputs, samples)
            
        # Process results
        results = []
        for i in range(batch_size):
            result = {
                'prediction': int(ns_outputs['predictions'][i].item()),
                'probabilities': ns_outputs['probabilities'][i].cpu().numpy().tolist(),
                'confidence': float(ns_outputs['confidence'][i].item()),
                'uncertainty': float(ns_outputs['uncertainty'][i].item()),
                'threat_level': ['benign', 'suspicious', 'malicious'][int(ns_outputs['predictions'][i].item())],
                'explanations': ns_outputs['explanations'][i] if i < len(ns_outputs['explanations']) else {},
                'validation': {
                    'consensus_score': float(ns_outputs['validation']['consensus_score'][i].item()),
                    'needs_review': bool(ns_outputs['validation']['needs_review'][i].item())
                }
            }
            results.append(result)
        
        return results


def create_inference_engine(model_path: str, config_path: Optional[str] = None) -> ThreatInferenceEngine:
    """Create inference engine from model checkpoint.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        
    Returns:
        Configured inference engine
    """
    return ThreatInferenceEngine(model_path, config_path)
