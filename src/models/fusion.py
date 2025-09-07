"""Cross-modal attention fusion for combining multi-modal threat features."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different modalities."""
    
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        """Initialize cross-modal attention.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        
        # Query, Key, Value projections for each modality
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through cross-modal attention.
        
        Args:
            query: Query features [batch_size, seq_len_q, dim]
            key: Key features [batch_size, seq_len_k, dim]
            value: Value features [batch_size, seq_len_v, dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        K = K.transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        V = V.transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.dim
        )
        
        # Output projection
        output = self.output_proj(attended)
        
        # Return mean attention weights across heads
        mean_attention = attention_weights.mean(dim=1)
        
        return output, mean_attention


class ModalityFusionBlock(nn.Module):
    """Fusion block for combining multiple modalities with cross-attention."""
    
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        """Initialize modality fusion block.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        
        # Cross-modal attention layers
        self.cross_attention = CrossModalAttention(dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                primary: torch.Tensor, 
                context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through fusion block.
        
        Args:
            primary: Primary modality features [batch_size, seq_len, dim]
            context: Context modality features [batch_size, seq_len, dim]
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        # Cross-modal attention (primary queries context)
        attended, attention_weights = self.cross_attention(primary, context, context)
        
        # Residual connection and normalization
        fused = self.norm1(primary + attended)
        
        # Feed-forward with residual connection
        output = self.norm2(fused + self.ffn(fused))
        
        return {
            'fused_features': output,
            'attention_weights': attention_weights
        }


class MultiModalFusionNetwork(nn.Module):
    """Multi-modal fusion network for threat detection."""
    
    def __init__(self, 
                 modalities: Dict[str, int],
                 fusion_dim: int = 512,
                 num_heads: int = 8,
                 num_fusion_layers: int = 3,
                 dropout: float = 0.15):
        """Initialize multi-modal fusion network.
        
        Args:
            modalities: Dictionary mapping modality names to their dimensions
            fusion_dim: Common dimension for fusion
            num_heads: Number of attention heads
            num_fusion_layers: Number of fusion layers
            dropout: Dropout probability
        """
        super().__init__()
        self.modalities = modalities
        self.fusion_dim = fusion_dim
        self.num_fusion_layers = num_fusion_layers
        
        # Projection layers to common dimension
        self.modality_projections = nn.ModuleDict()
        for modality, dim in modalities.items():
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Fusion blocks
        self.fusion_blocks = nn.ModuleList([
            ModalityFusionBlock(fusion_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # Modality importance weights
        self.modality_weights = nn.Parameter(
            torch.ones(len(modalities)) / len(modalities)
        )
        
        # Final fusion layers
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * len(modalities), fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        logger.info(f"MultiModalFusionNetwork initialized: {len(modalities)} modalities -> {fusion_dim}D")
        
    def forward(self, modal_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-modal fusion network.
        
        Args:
            modal_features: Dictionary of modality features
            
        Returns:
            Dictionary containing fused representations and attention maps
        """
        batch_size = next(iter(modal_features.values())).shape[0]
        
        # Project all modalities to common dimension
        projected_features = {}
        for modality, features in modal_features.items():
            if modality in self.modality_projections:
                # Handle different input shapes
                if features.dim() == 2:
                    features = features.unsqueeze(1)  # Add sequence dimension
                elif features.dim() == 3:
                    pass  # Already has sequence dimension
                else:
                    features = features.view(batch_size, -1).unsqueeze(1)
                
                projected = self.modality_projections[modality](features)
                projected_features[modality] = projected
        
        # Iterative fusion with cross-modal attention
        fused_modalities = {}
        attention_maps = {}
        
        modality_names = list(projected_features.keys())
        
        # Initialize with projected features
        for modality in modality_names:
            fused_modalities[modality] = projected_features[modality]
        
        # Apply fusion blocks
        for layer_idx, fusion_block in enumerate(self.fusion_blocks):
            updated_modalities = {}
            layer_attention = {}
            
            for i, primary_modality in enumerate(modality_names):
                # Fuse primary modality with all others
                primary_features = fused_modalities[primary_modality]
                
                # Create context by combining other modalities
                context_features = []
                for j, context_modality in enumerate(modality_names):
                    if i != j:
                        context_features.append(fused_modalities[context_modality])
                
                if context_features:
                    # Average context features
                    context = torch.stack(context_features).mean(dim=0)
                    
                    # Apply fusion block
                    fusion_output = fusion_block(primary_features, context)
                    updated_modalities[primary_modality] = fusion_output['fused_features']
                    layer_attention[f"{primary_modality}_layer_{layer_idx}"] = fusion_output['attention_weights']
                else:
                    updated_modalities[primary_modality] = primary_features
            
            fused_modalities = updated_modalities
            attention_maps.update(layer_attention)
        
        # Final fusion with learned modality weights
        normalized_weights = F.softmax(self.modality_weights, dim=0)
        
        # Weighted combination of modalities
        weighted_features = []
        for i, modality in enumerate(modality_names):
            features = fused_modalities[modality].squeeze(1)  # Remove sequence dim
            weighted = features * normalized_weights[i]
            weighted_features.append(weighted)
        
        # Concatenate and apply final fusion
        concatenated = torch.cat(weighted_features, dim=-1)
        final_embedding = self.final_fusion(concatenated)
        
        # Compute modality contributions
        modality_contributions = {
            modality: normalized_weights[i].item()
            for i, modality in enumerate(modality_names)
        }
        
        return {
            'fused_embedding': final_embedding,
            'modality_features': fused_modalities,
            'attention_maps': attention_maps,
            'modality_weights': normalized_weights,
            'modality_contributions': modality_contributions,
            'embeddings': final_embedding
        }


def create_multimodal_fusion(config) -> MultiModalFusionNetwork:
    """Create multi-modal fusion network from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured MultiModalFusionNetwork
    """
    fusion_config = config.get("models.fusion", {})
    
    # Define modality dimensions based on model configurations
    modalities = {
        'tabular': 256,  # From preprocessing
        'text': config.get("models.transformer.d_model", 384),
        'graph': config.get("models.gcn.hidden_dim", 256),
        'image': config.get("models.vit.dim", 384) * 2,  # CLS + global features
        'temporal': 8  # From temporal preprocessing
    }
    
    fusion_network = MultiModalFusionNetwork(
        modalities=modalities,
        fusion_dim=fusion_config.get("hidden_dim", 512),
        num_heads=fusion_config.get("num_heads", 8),
        num_fusion_layers=3,
        dropout=fusion_config.get("dropout", 0.15)
    )
    
    return fusion_network
