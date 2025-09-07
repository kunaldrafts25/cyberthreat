"""Graph Convolutional Network for processing network topology features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import Dict, Optional, Tuple
from loguru import logger


class GraphConvBlock(nn.Module):
    """Graph convolutional block with normalization and activation."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        """Initialize graph conv block.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.conv = GCNConv(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph conv block.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
class GlobalPoolWrapper(nn.Module):
    def __init__(self, pool_fn):
        super().__init__()
        self.pool_fn = pool_fn

    def forward(self, x, batch):
        return self.pool_fn(x, batch)


class GraphConvolutionalNetwork(nn.Module):
    """Graph Convolutional Network for network topology analysis."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None):
        """Initialize GCN.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of graph conv layers
            dropout: Dropout probability
            output_dim: Output dimension (defaults to hidden_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or hidden_dim
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GraphConvBlock(input_dim, hidden_dim, dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GraphConvBlock(hidden_dim, hidden_dim, dropout))
        
        # Output layer
        if num_layers > 1:
            self.conv_layers.append(GraphConvBlock(hidden_dim, self.output_dim, dropout))
        
        # Global pooling layers
        self.global_pooling = nn.ModuleDict({
            'mean': GlobalPoolWrapper(global_mean_pool),
            'max': GlobalPoolWrapper(global_max_pool),
})

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        logger.info(f"GCN initialized: {input_dim} -> {hidden_dim} -> {self.output_dim}, {num_layers} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through GCN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for each node [num_nodes] (optional)
            
        Returns:
            Dictionary containing node and graph level features
        """
        node_features = x
        
        # Pass through graph conv layers
        for conv_layer in self.conv_layers:
            node_features = conv_layer(node_features, edge_index)
        
        # Global pooling for graph-level features
        if batch is None:
            # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_mean = global_mean_pool(node_features, batch)
        graph_max = global_max_pool(node_features, batch)
        
        # Combine pooled features
        graph_features = torch.cat([graph_mean, graph_max], dim=-1)
        graph_features = self.projection(graph_features)
        
        return {
            'node_features': node_features,
            'graph_features': graph_features,
            'embeddings': graph_features
        }


class NetworkTopologyGCN(nn.Module):
    """Specialized GCN for network topology in cybersecurity context."""
    
    def __init__(self, config):
        """Initialize network topology GCN.
        
        Args:
            config: System configuration
        """
        super().__init__()
        gcn_config = config.get("models.gcn", {})
        
        # Fixed input dimension based on preprocessing
        # (node features: centrality measures + metadata)
        input_dim = 25  # From preprocessing: tabular graph features
        
        self.gcn = GraphConvolutionalNetwork(
            input_dim=input_dim,
            hidden_dim=gcn_config.get("hidden_dim", 256),
            num_layers=gcn_config.get("num_layers", 3),
            dropout=gcn_config.get("dropout", 0.2)
        )
        
        # Attention mechanism for important nodes
        self.node_attention = nn.Sequential(
            nn.Linear(gcn_config.get("hidden_dim", 256), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, graph_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through network topology GCN.
        
        Args:
            graph_features: Preprocessed graph features [batch_size, feature_dim]
            
        Returns:
            Dictionary containing graph embeddings and attention weights
        """
        batch_size, feature_dim = graph_features.shape
        
        # Create simple node features and edges for each sample
        # In a full implementation, this would use actual network topology
        # Here we create a simplified representation
        
        outputs = []
        attention_weights = []
        
        for i in range(batch_size):
            # Create dummy graph structure (in practice, use real topology)
            num_nodes = min(10, feature_dim // 2)  # Simplified
            
            # Node features from input
            node_feats = graph_features[i:i+1].repeat(num_nodes, 1)
            if node_feats.shape[1] > 25:
                node_feats = node_feats[:, :25]
            elif node_feats.shape[1] < 25:
                padding = torch.zeros(num_nodes, 25 - node_feats.shape[1], device=node_feats.device)
                node_feats = torch.cat([node_feats, padding], dim=1)
            
            # Create simple edge structure (star topology)
            edge_index = torch.tensor([
                [0] * (num_nodes - 1) + list(range(1, num_nodes)),
                list(range(1, num_nodes)) + [0] * (num_nodes - 1)
            ], device=graph_features.device)
            
            # Forward through GCN
            gcn_output = self.gcn(node_feats, edge_index)
            
            # Compute node attention
            node_attn = self.node_attention(gcn_output['node_features'])
            
            # Weighted aggregation of node features
            weighted_nodes = gcn_output['node_features'] * node_attn
            sample_embedding = weighted_nodes.mean(dim=0, keepdim=True)
            
            outputs.append(sample_embedding)
            attention_weights.append(node_attn.mean().item())
        
        # Stack batch outputs
        embeddings = torch.cat(outputs, dim=0)
        
        return {
            'embeddings': embeddings,
            'graph_features': embeddings,
            'attention_weights': torch.tensor(attention_weights, device=graph_features.device)
        }


def create_network_gcn(config) -> NetworkTopologyGCN:
    """Create a network topology GCN from configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Configured NetworkTopologyGCN model
    """
    return NetworkTopologyGCN(config)
