"""
Dual-Track Neuro-Conditioned Graph Transformer.

This module implements the main deep learning model for AD/FTD classification,
combining sensor-level and source-level graph neural networks with
temporal modeling and clinical conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

try:
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

from .layers.gat_layer import BrainGATBlock
from .layers.film_layer import FiLMLayer, MetadataEncoder
from .layers.temporal_encoder import TemporalTransformerEncoder, TemporalAttentionPooling
from ..config.config import MODEL_CONFIG, N_CHANNELS


class SensorLevelGNN(nn.Module):
    """
    Graph Neural Network for sensor-level (19-channel) connectivity.
    
    Processes the connectivity graph between EEG channels.
    """
    
    def __init__(self,
                 node_features: int = 32,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.3,
                 edge_dim: int = 3):
        """
        Initialize sensor-level GNN.
        
        Args:
            node_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Edge feature dimension (number of frequency bands)
        """
        super().__init__()
        
        self.n_nodes = N_CHANNELS  # 19
        
        # Node feature projection
        self.node_proj = nn.Linear(node_features, hidden_dim)
        
        # Graph Attention Block
        self.gat_block = BrainGATBlock(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            heads=n_heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (n_nodes * batch_size, node_features)
            edge_index: Edge indices (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            batch: Batch indices (n_nodes * batch_size,)
            
        Returns:
            Graph-level embedding (batch_size, output_dim)
        """
        # Project node features
        x = self.node_proj(x)
        x = F.relu(x)
        
        # Apply GAT block
        x = self.gat_block(x, edge_index, edge_attr)
        
        # Global pooling
        if HAS_TORCH_GEOMETRIC:
            x = global_mean_pool(x, batch)
        else:
            # Fallback: simple mean
            x = x.mean(dim=0, keepdim=True)
        
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class SourceLevelGNN(nn.Module):
    """
    Graph Neural Network for source-level (8-ROI) connectivity.
    
    Processes the connectivity graph between source-level brain regions.
    """
    
    def __init__(self,
                 node_features: int = 32,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.3,
                 edge_dim: int = 3,
                 n_rois: int = 8):
        """
        Initialize source-level GNN.
        
        Args:
            node_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Edge feature dimension
            n_rois: Number of ROIs (brain regions)
        """
        super().__init__()
        
        self.n_nodes = n_rois
        
        # Node feature projection
        self.node_proj = nn.Linear(node_features, hidden_dim)
        
        # Graph Attention Block
        self.gat_block = BrainGATBlock(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            heads=n_heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (n_nodes * batch_size, node_features)
            edge_index: Edge indices (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            batch: Batch indices
            
        Returns:
            Graph-level embedding (batch_size, output_dim)
        """
        # Project node features
        x = self.node_proj(x)
        x = F.relu(x)
        
        # Apply GAT block
        x = self.gat_block(x, edge_index, edge_attr)
        
        # Global pooling
        if HAS_TORCH_GEOMETRIC:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class DualTrackNeuroGNN(nn.Module):
    """
    Dual-Track Neuro-Conditioned Graph Neural Network.
    
    Combines sensor-level and source-level analysis with temporal
    modeling and clinical conditioning for dementia classification.
    """
    
    def __init__(self,
                 sensor_node_features: int = None,
                 source_node_features: int = None,
                 hidden_dim: int = None,
                 n_heads: int = None,
                 n_temporal_steps: int = None,
                 metadata_dim: int = None,
                 n_classes: int = None,
                 dropout: float = None,
                 use_sensor: bool = True,
                 use_source: bool = True):
        """
        Initialize the dual-track model.
        
        Args:
            sensor_node_features: Sensor-level node feature dimension
            source_node_features: Source-level node feature dimension
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_temporal_steps: Number of temporal windows
            metadata_dim: Clinical metadata dimension (Age, Sex, MMSE = 3)
            n_classes: Number of output classes (AD, FTD, CN = 3)
            dropout: Dropout rate
            use_sensor: Whether to use sensor-level track
            use_source: Whether to use source-level track
        """
        super().__init__()
        
        # Use defaults from config if not specified
        sensor_node_features = sensor_node_features or MODEL_CONFIG['sensor_node_features']
        source_node_features = source_node_features or MODEL_CONFIG['source_node_features']
        hidden_dim = hidden_dim or MODEL_CONFIG['hidden_dim']
        n_heads = n_heads or MODEL_CONFIG['n_heads']
        n_temporal_steps = n_temporal_steps or MODEL_CONFIG['n_temporal_steps']
        metadata_dim = metadata_dim or MODEL_CONFIG['metadata_dim']
        n_classes = n_classes or MODEL_CONFIG['n_classes']
        dropout = dropout or MODEL_CONFIG['dropout']
        
        self.hidden_dim = hidden_dim
        self.use_sensor = use_sensor
        self.use_source = use_source
        
        # Determine combined dimension
        n_tracks = int(use_sensor) + int(use_source)
        combined_dim = hidden_dim * n_tracks
        
        # Sensor-level GNN
        if use_sensor:
            self.sensor_gnn = SensorLevelGNN(
                node_features=sensor_node_features,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # Source-level GNN
        if use_source:
            self.source_gnn = SourceLevelGNN(
                node_features=source_node_features,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Temporal Transformer Encoder
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=2,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            max_len=n_temporal_steps
        )
        
        # Temporal attention pooling
        self.temporal_pooling = TemporalAttentionPooling(
            d_model=hidden_dim,
            n_heads=n_heads
        )
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            input_dim=metadata_dim,
            output_dim=32,
            hidden_dim=64,
            dropout=dropout
        )
        
        # FiLM conditioning
        self.film = FiLMLayer(
            feature_dim=hidden_dim,
            condition_dim=32,
            hidden_dim=64
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def forward(self,
                sensor_data: Optional[Dict] = None,
                source_data: Optional[Dict] = None,
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sensor_data: Dictionary with 'graphs' key containing list of
                        sensor-level graph data for each time step
            source_data: Dictionary with 'graphs' key containing list of
                        source-level graph data for each time step
            metadata: Clinical metadata tensor (batch, 3)
            
        Returns:
            Class logits (batch, n_classes)
        """
        temporal_features = []
        
        # Get number of time steps
        if sensor_data is not None:
            n_steps = len(sensor_data['graphs'])
        elif source_data is not None:
            n_steps = len(source_data['graphs'])
        else:
            raise ValueError("At least one of sensor_data or source_data must be provided")
        
        # Process each time step
        for t in range(n_steps):
            features = []
            
            # Sensor-level processing
            if self.use_sensor and sensor_data is not None:
                sg = sensor_data['graphs'][t]
                sensor_feat = self.sensor_gnn(
                    sg.x, sg.edge_index, sg.edge_attr, sg.batch
                )
                features.append(sensor_feat)
            
            # Source-level processing
            if self.use_source and source_data is not None:
                src_g = source_data['graphs'][t]
                source_feat = self.source_gnn(
                    src_g.x, src_g.edge_index, src_g.edge_attr, src_g.batch
                )
                features.append(source_feat)
            
            # Concatenate and fuse
            combined = torch.cat(features, dim=1)
            fused = self.fusion(combined)
            
            temporal_features.append(fused)
        
        # Stack temporal features: (batch, n_steps, hidden_dim)
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Apply temporal transformer
        temporal_output = self.temporal_encoder(temporal_features)
        
        # Pool temporal dimension
        temporal_repr = self.temporal_pooling(temporal_output)
        
        # Apply FiLM conditioning if metadata provided
        if metadata is not None:
            condition = self.metadata_encoder(metadata)
            temporal_repr = self.film(temporal_repr, condition)
        
        # Classification
        logits = self.classifier(temporal_repr)
        
        return logits
    
    def get_attention_weights(self):
        """Get attention weights for explainability."""
        # This would be implemented to extract attention weights
        # from GAT and temporal transformer layers
        pass


class SingleTrackGNN(nn.Module):
    """
    Single-track GNN model (sensor-only or source-only).
    
    Simplified version when only one track is needed.
    """
    
    def __init__(self,
                 node_features: int = 32,
                 hidden_dim: int = 64,
                 n_heads: int = 4,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 n_nodes: int = 19,
                 use_metadata: bool = True):
        """
        Initialize single-track model.
        
        Args:
            node_features: Node feature dimension
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_classes: Number of output classes
            dropout: Dropout rate
            n_nodes: Number of graph nodes
            use_metadata: Whether to use metadata conditioning
        """
        super().__init__()
        
        self.use_metadata = use_metadata
        
        # Node projection
        self.node_proj = nn.Linear(node_features, hidden_dim)
        
        # GAT block
        self.gat_block = BrainGATBlock(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=n_heads,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Metadata encoder and FiLM
        if use_metadata:
            self.metadata_encoder = MetadataEncoder(input_dim=3, output_dim=32)
            self.film = FiLMLayer(hidden_dim, 32)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch indices
            metadata: Optional metadata
            
        Returns:
            Class logits
        """
        # Project and process
        x = F.relu(self.node_proj(x))
        x = self.gat_block(x, edge_index, edge_attr)
        
        # Pool
        if HAS_TORCH_GEOMETRIC:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        x = self.norm(x)
        
        # Apply FiLM if metadata provided
        if self.use_metadata and metadata is not None:
            condition = self.metadata_encoder(metadata)
            x = self.film(x, condition)
        
        # Classify
        return self.classifier(x)
