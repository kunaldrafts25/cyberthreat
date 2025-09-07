"""Transformer encoder for processing textual threat data."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TransformerTextEncoder(nn.Module):
    """Transformer encoder for processing text-based threat features."""
    
    def __init__(self,
                 vocab_size: int = 30522,  # BERT vocab size as default
                 d_model: int = 384,
                 nhead: int = 6,
                 num_layers: int = 4,
                 dim_feedforward: int = 1536,
                 dropout: float = 0.1,
                 max_len: int = 512):
        """Initialize transformer text encoder.
        
        Args:
            vocab_size: Vocabulary size for embeddings
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Pooling strategies
        self.pooling_methods = nn.ModuleDict({
            'cls': nn.Identity(),  # Use CLS token if available
            'mean': nn.AdaptiveAvgPool1d(1),
            'max': nn.AdaptiveMaxPool1d(1),
            'attention': self._create_attention_pooling()
        })
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # 3 for mean+max+attention pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        logger.info(f"TransformerTextEncoder initialized: {d_model}D, {num_layers} layers, {nhead} heads")
    
    def _create_attention_pooling(self) -> nn.Module:
        """Create attention-based pooling mechanism.
        
        Returns:
            Attention pooling module
        """
        return nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer text encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing various text representations
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch_size, d_model]
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Create padding mask for transformer
        src_key_padding_mask = (attention_mask == 0)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        encoded = self.norm(encoded)
        
        # Apply different pooling strategies
        pooled_outputs = {}
        
        # Mean pooling (considering attention mask)
        masked_encoded = encoded * attention_mask.unsqueeze(-1)
        mean_pooled = masked_encoded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled_outputs['mean'] = mean_pooled
        
        # Max pooling
        masked_encoded_max = masked_encoded.clone()
        masked_encoded_max[attention_mask == 0] = -float('inf')
        max_pooled = masked_encoded_max.max(dim=1)[0]
        # Handle case where all tokens are masked
        max_pooled = torch.where(
            attention_mask.sum(dim=1, keepdim=True) > 0,
            max_pooled,
            torch.zeros_like(max_pooled)
        )
        pooled_outputs['max'] = max_pooled
        
        # Attention pooling
        attention_weights = self.pooling_methods['attention'](encoded)
        attention_weights = attention_weights * attention_mask.unsqueeze(-1)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        attention_pooled = (encoded * attention_weights).sum(dim=1)
        pooled_outputs['attention'] = attention_pooled
        
        # Combine all pooling methods
        combined_features = torch.cat([
            pooled_outputs['mean'],
            pooled_outputs['max'],
            pooled_outputs['attention']
        ], dim=-1)
        
        # Final projection
        final_embeddings = self.output_projection(combined_features)
        
        return {
            'sequence_output': encoded,  # [batch_size, seq_len, d_model]
            'pooled_output': final_embeddings,  # [batch_size, d_model]
            'mean_pooled': pooled_outputs['mean'],
            'max_pooled': pooled_outputs['max'],
            'attention_pooled': pooled_outputs['attention'],
            'attention_weights': attention_weights.squeeze(-1),  # [batch_size, seq_len]
            'embeddings': final_embeddings
        }


class ThreatTextProcessor(nn.Module):
    """Specialized text processor for cybersecurity threat data."""
    
    def __init__(self, config):
        """Initialize threat text processor.
        
        Args:
            config: System configuration
        """
        super().__init__()
        transformer_config = config.get("models.transformer", {})
        data_config = config.get("data", {})
        
        # Determine vocab size from tokenizer or use default
        vocab_size = 30522  # BERT default
        max_seq_length = data_config.get("max_seq_length", 512)
        
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            d_model=transformer_config.get("d_model", 384),
            nhead=transformer_config.get("nhead", 6),
            num_layers=transformer_config.get("num_layers", 4),
            dim_feedforward=transformer_config.get("d_model", 384) * 4,
            dropout=transformer_config.get("dropout", 0.1),
            max_len=max_seq_length
        )
        
        # Threat-specific text classification heads
        d_model = transformer_config.get("d_model", 384)
        
        self.threat_classifiers = nn.ModuleDict({
            'malware_family': nn.Linear(d_model, 10),  # Common malware families
            'attack_type': nn.Linear(d_model, 8),      # Common attack types
            'severity': nn.Linear(d_model, 3),         # Low/Medium/High severity
        })
        
    def forward(self, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through threat text processor.
        
        Args:
            text_features: Tokenized text features [batch_size, seq_len]
            
        Returns:
            Dictionary containing text embeddings and classifications
        """
        # Handle both integer tokens and float features
        if text_features.dtype == torch.float:
            # Convert TF-IDF or similar features to "token" representation
            # This is a simplified approach - in practice, you'd use actual tokens
            input_ids = (text_features * 1000).long().clamp(0, 30521)
        else:
            input_ids = text_features.long()
        
        # Ensure proper shape
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Forward through text encoder
        text_output = self.text_encoder(input_ids)
        
        # Additional threat-specific classifications
        embeddings = text_output['pooled_output']
        threat_classifications = {}
        
        for classifier_name, classifier in self.threat_classifiers.items():
            threat_classifications[classifier_name] = classifier(embeddings)
        
        return {
            **text_output,
            'threat_classifications': threat_classifications,
            'embeddings': embeddings
        }


def create_threat_text_processor(config) -> ThreatTextProcessor:
    """Create a threat text processor from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured ThreatTextProcessor model
    """
    return ThreatTextProcessor(config)
