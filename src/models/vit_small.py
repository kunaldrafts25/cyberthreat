"""Small Vision Transformer for byte-matrix processing in threat detection."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class PatchEmbedding(nn.Module):
    """Convert byte matrices to patch embeddings."""
    
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 1, embed_dim: int = 384):
        """Initialize patch embedding layer.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through patch embedding.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        x = self.projection(x)  # [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize multi-head attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        out = torch.matmul(attention_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Transformed tensor [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformerSmall(nn.Module):
    """Small Vision Transformer for processing byte matrices."""
    
    def __init__(self, 
                 img_size: int = 64,
                 patch_size: int = 8,
                 in_channels: int = 1,
                 dim: int = 384,
                 depth: int = 6,
                 heads: int = 6,
                 mlp_dim: int = 1024,
                 dropout: float = 0.1):
        """Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            in_channels: Number of input channels
            dim: Embedding dimension
            depth: Number of transformer blocks
            heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
        logger.info(f"VisionTransformerSmall initialized: {img_size}x{img_size} -> {num_patches} patches, {dim}D")
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through Vision Transformer.
        
        Args:
            x: Input tensor [batch_size, height*width] (flattened byte matrix)
            
        Returns:
            Dictionary containing features and attention weights
        """
        batch_size = x.shape[0]
        
        # Reshape to image format
        x = x.view(batch_size, 1, self.img_size, self.img_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, num_patches, dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Store attention weights for interpretability
        attention_weights = []
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract CLS token and patch features
        cls_features = x[:, 0]  # [batch_size, dim]
        patch_features = x[:, 1:]  # [batch_size, num_patches, dim]
        
        # Global average pooling of patch features
        global_features = patch_features.mean(dim=1)  # [batch_size, dim]
        
        return {
            'cls_features': cls_features,
            'global_features': global_features,
            'patch_features': patch_features,
            'embeddings': torch.cat([cls_features, global_features], dim=-1)  # Combined features
        }


def create_vit_small(config) -> VisionTransformerSmall:
    """Create a small Vision Transformer from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured VisionTransformerSmall model
    """
    vit_config = config.get("models.vit", {})
    data_config = config.get("data", {})
    
    model = VisionTransformerSmall(
        img_size=data_config.get("image_size", 64),
        patch_size=vit_config.get("patch_size", 8),
        in_channels=1,  # Byte data is single channel
        dim=vit_config.get("dim", 384),
        depth=vit_config.get("depth", 6),
        heads=vit_config.get("heads", 6),
        mlp_dim=vit_config.get("mlp_dim", 1024),
        dropout=vit_config.get("dropout", 0.1)
    )
    
    return model
