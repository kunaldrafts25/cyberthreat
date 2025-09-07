"""Complete threat detection system integrating all components."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger

from ..common.config import get_config
from ..common.utils import get_device
from .vit_small import create_vit_small
from .gcn import create_network_gcn
from .transformer_text import create_threat_text_processor
from .fusion import create_multimodal_fusion
from .triage_head import create_threat_triage_head


class ThreatDetectionSystem(nn.Module):
    """Complete end-to-end threat detection system."""
    
    def __init__(self, config):
        """Initialize threat detection system.
        
        Args:
            config: System configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize all modality encoders
        self.vit_encoder = create_vit_small(config)
        self.gcn_encoder = create_network_gcn(config)
        self.text_encoder = create_threat_text_processor(config)
        
        # Tabular feature processor
        tabular_dim = 256  # From preprocessing
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        # Temporal feature processor
        temporal_dim = 8  # From preprocessing
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Multi-modal fusion
        self.fusion_network = create_multimodal_fusion(config)
        
        # Rapid triage head
        self.triage_head = create_threat_triage_head(config)
        
        # Self-supervised learning components
        self.ssl_enabled = config.get("ssl.enabled", True)
        if self.ssl_enabled:
            self._init_ssl_components()
        
        logger.info("ThreatDetectionSystem initialized with all components")
    
    def _init_ssl_components(self):
        """Initialize self-supervised learning components."""
        fusion_dim = self.config.get("models.fusion.hidden_dim", 512)
        
        # Contrastive learning head for SSL
        self.ssl_projector = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Masked language modeling head for text
        text_dim = self.config.get("models.transformer.d_model", 384)
        vocab_size = 30522  # BERT vocab size
        
        self.mlm_head = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.GELU(),
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, vocab_size)
        )
        
        logger.info("SSL components initialized")
    
    def forward(self, 
                batch: Dict[str, torch.Tensor], 
                mode: str = "inference") -> Dict[str, torch.Tensor]:
        """Forward pass through the complete threat detection system.
        
        Args:
            batch: Input batch dictionary containing all modalities
            mode: Operation mode ("inference", "training", "ssl")
            
        Returns:
            Dictionary containing predictions, features, and auxiliary outputs
        """
        outputs = {}
        modal_features = {}
        
        # Process each modality
        if 'tabular' in batch:
            tabular_out = self.tabular_encoder(batch['tabular'])
            modal_features['tabular'] = tabular_out
        
        if 'text' in batch:
            text_out = self.text_encoder(batch['text'])
            modal_features['text'] = text_out['embeddings']
            outputs['text_features'] = text_out
        
        if 'graph' in batch:
            graph_out = self.gcn_encoder(batch['graph'])
            modal_features['graph'] = graph_out['embeddings']
            outputs['graph_features'] = graph_out
        
        if 'image' in batch:
            image_out = self.vit_encoder(batch['image'])
            modal_features['image'] = image_out['embeddings']
            outputs['image_features'] = image_out
        
        if 'temporal' in batch:
            temporal_out = self.temporal_encoder(batch['temporal'])
            modal_features['temporal'] = temporal_out
        
        # Multi-modal fusion
        fusion_output = self.fusion_network(modal_features)
        fused_features = fusion_output['fused_embedding']
        
        outputs.update({
            'modal_features': modal_features,
            'fusion_output': fusion_output,
            'fused_features': fused_features
        })
        
        if mode == "ssl" and self.ssl_enabled:
            # Self-supervised learning mode
            ssl_projections = self.ssl_projector(fused_features)
            outputs['ssl_projections'] = ssl_projections
            
            # MLM for text modality
            if 'text_features' in outputs:
                mlm_logits = self.mlm_head(outputs['text_features']['sequence_output'])
                outputs['mlm_logits'] = mlm_logits
        
        elif mode in ["inference", "training"]:
            # Main threat detection task
            triage_output = self.triage_head(fused_features, return_uncertainty=(mode=="inference"))
            outputs.update(triage_output)
        
        return outputs
    
    def ssl_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Self-supervised learning forward pass."""
        return self.forward(batch, mode="ssl")
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make predictions with uncertainty quantification."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, mode="inference")
        return outputs
    
    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract fused embeddings for downstream tasks."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, mode="training")
            return outputs['fused_features']


def create_threat_detection_system(config_path: Optional[str] = None) -> ThreatDetectionSystem:
    """Create complete threat detection system from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ThreatDetectionSystem
    """
    config = get_config(config_path)
    system = ThreatDetectionSystem(config)
    
    device = get_device(config.get("system.device", "auto"))
    system = system.to(device)
    
    return system
