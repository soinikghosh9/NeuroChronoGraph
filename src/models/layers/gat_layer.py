"""
Graph Attention Layer (GATv2) for Brain Connectivity.

This module implements the Graph Attention layer for processing
brain connectivity graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.nn import GATv2Conv, MessagePassing
    from torch_geometric.utils import add_self_loops, softmax
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


if HAS_TORCH_GEOMETRIC:
    class BrainGATLayer(nn.Module):
        """
        Graph Attention Layer for brain connectivity processing.
        
        Uses GATv2Conv for dynamic attention computation.
        """
        
        def __init__(self,
                     in_channels: int,
                     out_channels: int,
                     heads: int = 4,
                     concat: bool = True,
                     dropout: float = 0.2,
                     edge_dim: int = 1,
                     add_self_loops: bool = True):
            """
            Initialize the Brain GAT Layer.
            
            Args:
                in_channels: Input feature dimension
                out_channels: Output feature dimension
                heads: Number of attention heads
                concat: Whether to concatenate or average head outputs
                dropout: Dropout rate
                edge_dim: Edge feature dimension
                add_self_loops: Whether to add self-loops
            """
            super().__init__()
            
            self.gat = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=add_self_loops
            )
            
            self.heads = heads
            self.concat = concat
            self.out_dim = out_channels * heads if concat else out_channels
            
        def forward(self, 
                    x: torch.Tensor,
                    edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Node features (n_nodes, in_channels)
                edge_index: Edge indices (2, n_edges)
                edge_attr: Edge features (n_edges, edge_dim)
                
            Returns:
                Updated node features (n_nodes, out_dim)
            """
            return self.gat(x, edge_index, edge_attr=edge_attr)
        
        def get_attention_weights(self,
                                   x: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Get attention weights.
            
            Args:
                x: Node features
                edge_index: Edge indices
                edge_attr: Edge features
                
            Returns:
                (output, attention_weights) tuple
            """
            return self.gat(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)


    class BrainGATBlock(nn.Module):
        """
        Multi-layer Graph Attention Block for brain networks.
        
        Consists of two GAT layers with normalization and residual connections.
        """
        
        def __init__(self,
                     in_channels: int,
                     hidden_channels: int,
                     out_channels: int,
                     heads: int = 4,
                     dropout: float = 0.3,
                     edge_dim: int = 1):
            """
            Initialize the Brain GAT Block.
            
            Args:
                in_channels: Input feature dimension
                hidden_channels: Hidden layer dimension
                out_channels: Output feature dimension
                heads: Number of attention heads
                dropout: Dropout rate
                edge_dim: Edge feature dimension
            """
            super().__init__()
            
            # First GAT layer
            self.gat1 = GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim
            )
            
            # Second GAT layer
            self.gat2 = GATv2Conv(
                in_channels=hidden_channels * heads,
                out_channels=out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim
            )
            
            # Normalization layers
            self.norm1 = nn.LayerNorm(hidden_channels * heads)
            self.norm2 = nn.LayerNorm(out_channels)
            
            # Dropout
            self.dropout = nn.Dropout(dropout)
            
            # Projection for residual if needed
            if in_channels != out_channels:
                self.residual_proj = nn.Linear(in_channels, out_channels)
            else:
                self.residual_proj = None
            
        def forward(self,
                    x: torch.Tensor,
                    edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Node features (n_nodes, in_channels)
                edge_index: Edge indices (2, n_edges)
                edge_attr: Edge features (n_edges, edge_dim)
                
            Returns:
                Updated node features (n_nodes, out_channels)
            """
            identity = x
            
            # First GAT layer
            x = self.gat1(x, edge_index, edge_attr=edge_attr)
            x = self.norm1(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Second GAT layer
            x = self.gat2(x, edge_index, edge_attr=edge_attr)
            x = self.norm2(x)
            
            # Residual connection
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            
            x = x + identity
            x = F.relu(x)
            
            return x

else:
    # Fallback implementations without torch_geometric
    class BrainGATLayer(nn.Module):
        """Fallback GAT layer without torch_geometric."""
        
        def __init__(self, in_channels, out_channels, heads=4, **kwargs):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels * heads)
            self.out_dim = out_channels * heads
            
        def forward(self, x, edge_index, edge_attr=None):
            return self.linear(x)
    
    class BrainGATBlock(nn.Module):
        """Fallback GAT block without torch_geometric."""
        
        def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
            super().__init__()
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, out_channels)
            self.norm = nn.LayerNorm(out_channels)
            
        def forward(self, x, edge_index, edge_attr=None):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.norm(x)
            return x
