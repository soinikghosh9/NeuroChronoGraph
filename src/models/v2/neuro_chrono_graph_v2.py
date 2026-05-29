"""
NeuroChronoGraph v2 - Advanced Architecture.

This is the main model combining:
- Adaptive Graph Learning
- Modular Brain Transformer
- Cross-Band Attention
- Hierarchical Clinical Conditioning
- Uncertainty Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .adaptive_graph import AdaptiveGraphLearning, GatedGraphConvolution
from .modular_brain_transformer import ModularBrainTransformer, TemporalGraphTransformer
from .clinical_conditioning import HierarchicalFiLMConditioner, UncertaintyHead


class EEGEncoder(nn.Module):
    """
    EEG signal encoder for extracting initial features.
    """
    
    def __init__(self,
                 n_channels: int = 19,
                 n_times: int = 2000,
                 hidden_dim: int = 64,
                 output_dim: int = 128):
        super().__init__()
        
        # Temporal convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(hidden_dim * 2, output_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50)  # Fixed output length
        )
        
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode EEG signals.
        
        Args:
            x: EEG data [batch, n_channels, n_times]
            
        Returns:
            Encoded features [batch, output_dim, 50]
        """
        return self.temporal_conv(x)


class ChannelWiseEncoder(nn.Module):
    """
    Encode each EEG channel independently to preserve spatial information.
    
    This is critical for graph-based processing - each channel (node) must have
    distinct features for the graph structure to be meaningful.
    """
    
    def __init__(self,
                 n_channels: int = 19,
                 n_times: int = 2000,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        
        # Per-channel temporal convolution (shared across channels for efficiency)
        self.channel_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, kernel_size=25, stride=1, padding=12),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool to single value per channel
        )
        
        # Channel-specific embeddings (learnable position encoding for each electrode)
        self.channel_embeddings = nn.Parameter(torch.randn(n_channels, hidden_dim) * 0.02)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each channel independently.
        
        Args:
            x: EEG data [batch, n_channels, n_times]
            
        Returns:
            Node features [batch, n_channels, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Process each channel separately
        # [B, C, T] -> [B*C, 1, T]
        x_channels = x.view(batch_size * self.n_channels, 1, -1)
        
        # Apply shared conv [B*C, 1, T] -> [B*C, hidden, 1]
        channel_features = self.channel_conv(x_channels)
        
        # Reshape [B*C, hidden, 1] -> [B, C, hidden]
        channel_features = channel_features.squeeze(-1).view(batch_size, self.n_channels, self.hidden_dim)
        
        # Add channel-specific embeddings
        channel_features = torch.cat([
            channel_features,
            self.channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        ], dim=-1)
        
        # Project to final dimension
        node_features = self.output_proj(channel_features)
        
        return node_features


class NeuroChronoGraphV2(nn.Module):
    """
    NeuroChronoGraph Version 2.
    
    Advanced architecture with:
    - Adaptive graph learning
    - Modular brain analysis
    - Cross-band attention
    - Hierarchical clinical conditioning
    - Multi-objective outputs
    """
    
    def __init__(self,
                 n_channels: int = 19,
                 n_times: int = 2000,
                 n_classes: int = 3,
                 n_bands: int = 5,
                 hidden_dim: int = 128,
                 n_graph_layers: int = 3,
                 n_temporal_layers: int = 4,
                 n_heads: int = 8,
                 dropout: float = 0.3,
                 use_clinical: bool = True,
                 use_uncertainty: bool = True,
                 feature_streams: Optional[List[str]] = None,
                 stream_dim: int = 64,
                 **kwargs):
        """
        Initialize NeuroChronoGraph v2.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time samples
            n_classes: Number of output classes (3: AD, FTD, CN)
            n_bands: Number of frequency bands
            hidden_dim: Hidden dimension
            n_graph_layers: Number of graph attention layers
            n_temporal_layers: Number of temporal transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_clinical: Whether to use clinical conditioning
            use_clinical: Whether to use clinical conditioning
            use_uncertainty: Whether to estimate uncertainty
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.use_clinical = use_clinical
        self.use_uncertainty = use_uncertainty
        
        # ===== STAGE 1: Initial Encoding =====
        self.eeg_encoder = EEGEncoder(
            n_channels=n_channels,
            n_times=n_times,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim
        )
        

        
        # Channel-wise encoder for proper node features (critical for graph processing)
        self.channel_encoder = ChannelWiseEncoder(
            n_channels=n_channels,
            n_times=n_times,
            hidden_dim=hidden_dim
        )
        
        # ===== STAGE 2: Adaptive Graph Learning =====
        self.adaptive_graph = AdaptiveGraphLearning(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            n_heads=n_heads // 2,
            dropout=dropout,
            use_prior=True
        )
        

        
        # ===== STAGE 4: Graph Attention Layers =====
        self.gat_layers = nn.ModuleList([
            GatedGraphConvolution(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(n_graph_layers)
        ])
        
        # ===== STAGE 5: Modular Brain Transformer =====
        self.modular_transformer = ModularBrainTransformer(
            node_dim=hidden_dim,
            module_dim=hidden_dim,
            n_layers=n_graph_layers,
            n_heads=n_heads // 2,
            dropout=dropout
        )
        
        # ===== STAGE 6: Temporal Transformer =====
        self.temporal_transformer = TemporalGraphTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            n_layers=n_temporal_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # ===== STAGE 7: Clinical Conditioning =====
        if use_clinical:
            # GAT Conditioner: Handles the graph layers (all have dim = hidden_dim)
            # hierarchy: Age -> MMSE -> Combined within the graph layers
            self.gat_conditioner = HierarchicalFiLMConditioner(
                feature_dim=hidden_dim,
                n_layers=n_graph_layers,
                use_uncertainty=False,  # Uncertainty only at final output
                combined_dim=hidden_dim
            )
            self.gat_conditioner.clinical_dropout = kwargs.get('clinical_dropout', 0.2)
            
            # Final Conditioner: Handles the fusion layer (dim = hidden_dim * 4)
            # Always uses Combined conditioning
            self.final_conditioner = HierarchicalFiLMConditioner(
                feature_dim=hidden_dim * 4,
                n_layers=1,  # Single layer
                use_uncertainty=use_uncertainty,
                combined_dim=hidden_dim * 4
            )
            self.final_conditioner.clinical_dropout = kwargs.get('clinical_dropout', 0.2)
        
        # ===== Hand-crafted biomarker stream fusion (optional) =====
        # Gated fusion of spectral / connectivity / complexity / microstate
        # streams into the 4*hidden_dim fused embedding. When None, the model
        # behaves identically to the original V2.
        self.feature_streams = feature_streams
        if feature_streams:
            from src.models.v3.feature_stream_fusion import FeatureStreamFusion
            self.stream_fusion = FeatureStreamFusion(
                embed_dim=hidden_dim * 4,
                stream_dim=stream_dim,
                streams=feature_streams,
                dropout=dropout,
                stream_dropout=kwargs.get('stream_dropout', 0.1),
            )
        else:
            self.stream_fusion = None

        # ===== STAGE 8: Output Heads (Hierarchical) =====
        # Input dimension: distinct from hidden_dim
        classifier_input_dim = hidden_dim * 4
        
        # Head 1: Screening (Healthy vs Impaired)
        # Classes: 0=Healthy(CN), 1=Impaired(AD/FTD/MCI)
        self.screening_head = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Head 2: Staging (MCI vs Dementia)
        # Classes: 0=MCI, 1=Dementia(AD/FTD)
        # Typically trained only on Impaired samples
        self.staging_head = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Head 3: Subtyping (AD vs FTD)
        # Classes: 0=AD, 1=FTD
        # Typically trained only on Dementia samples
        self.subtype_head = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # MMSE regression head (for clinical validation)
        self.mmse_regressor = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uncertainty head
        if use_uncertainty:
            # Uncertainty for Screening is most critical
            self.uncertainty_head = UncertaintyHead(
                input_dim=classifier_input_dim,
                n_classes=2 # Screening uncertainty
            )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self,
                x: torch.Tensor,
                adj: Optional[torch.Tensor] = None,
                band_features: Optional[Dict[str, torch.Tensor]] = None,
                clinical_data: Optional[Dict[str, torch.Tensor]] = None,
                feature_streams: Optional[Dict[str, torch.Tensor]] = None,
                stream_mask: Optional[Dict[str, bool]] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: EEG data [batch, n_channels, n_times]
            adj: Adjacency matrix (optional) [batch, n_channels, n_channels]
            band_features: Band-specific features (optional)
            clinical_data: Clinical metadata (age, mmse, sex)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with outputs (logits, mmse_pred, uncertainty, etc.)
        """
        batch_size = x.shape[0]
        outputs = {}
        
        # ===== STAGE 1: Initial Encoding =====
        # Temporal encoding (for temporal transformer branch)
        temporal_enc = self.eeg_encoder(x)  # [B, hidden, T']
        temporal_enc = temporal_enc.transpose(1, 2)  # [B, T', hidden]
        
        # Channel-wise node features (for graph processing)
        # Each channel gets distinct features - critical for meaningful graph structure
        node_features = self.channel_encoder(x)  # [B, n_channels, hidden]
        

        
        # ===== STAGE 2: Adaptive Graph Learning =====
        learned_adj, adj_attn = self.adaptive_graph(node_features, adj)
        outputs['learned_adjacency'] = learned_adj
        
        # ===== STAGE 3: Graph Attention Layers =====
        h = node_features
        layer_idx = 0
        
        for gat_layer in self.gat_layers:
            h = gat_layer(h, learned_adj)
            
            # Apply clinical conditioning if available
            if self.use_clinical and clinical_data is not None:
                # Use GAT conditioner for these layers
                h, _ = self.gat_conditioner(h, layer_idx, clinical_data)
                layer_idx += 1
        
        # ===== STAGE 4: Modular Brain Transformer =====
        modular_out, modular_info = self.modular_transformer(h, learned_adj)
        outputs['module_coupling'] = modular_info['coupling_matrix']
        
        # ===== STAGE 5: Temporal Processing =====
        temporal_out, temporal_weights = self.temporal_transformer(temporal_enc)
        outputs['temporal_weights'] = temporal_weights
        
        # ===== STAGE 6: Feature Fusion =====
        # Combine graph and temporal features
        graph_pooled = h.mean(dim=1)  # [B, hidden]
        combined = torch.cat([modular_out, temporal_out, graph_pooled], dim=-1)
        
        # Final clinical conditioning
        if self.use_clinical and clinical_data is not None:
            # Use final conditioner (single layer, index 0)
            combined, final_unc = self.final_conditioner(
                combined.unsqueeze(1), 0, clinical_data
            )
            combined = combined.squeeze(1)
            outputs['conditioning_uncertainty'] = final_unc

        # ===== Feature-stream fusion (optional) =====
        # Inject hand-crafted biomarker streams via gated bottleneck. The
        # ``stream_mask`` dict allows zeroing individual streams at inference
        # time for the leave-one-out / only-one ablation conditions.
        if self.stream_fusion is not None:
            combined = self.stream_fusion(combined, feature_streams, stream_mask)
            outputs['stream_gates'] = self.stream_fusion.gate_values()

        # ===== STAGE 7: Output Heads =====
        # Always expose the fused representation for downstream use (e.g. DANN)
        outputs['embedding'] = combined

        # Classification

        # 1. Screening: Healthy (0=CN) vs Impaired (1)
        logits_screen = self.screening_head(combined)
        outputs['logits_screen'] = logits_screen
        outputs['probs_screen'] = F.softmax(logits_screen, dim=-1)
        
        # 2. Staging: MCI (0) vs Dementia (1)
        # Note: In training we mask this validation loss.
        logits_stage = self.staging_head(combined)
        outputs['logits_stage'] = logits_stage
        outputs['probs_stage'] = F.softmax(logits_stage, dim=-1)
        
        # 3. Subtype: AD (0) vs FTD (1)
        logits_subtype = self.subtype_head(combined)
        outputs['logits_subtype'] = logits_subtype
        outputs['probs_subtype'] = F.softmax(logits_subtype, dim=-1)
        
        # MMSE regression
        mmse_pred = self.mmse_regressor(combined)
        outputs['mmse_pred'] = mmse_pred.squeeze(-1)
        
        # Uncertainty
        if self.use_uncertainty:
            unc_outputs = self.uncertainty_head(combined)
            outputs['uncertainty'] = unc_outputs['uncertainty']
            outputs['evidential_alpha'] = unc_outputs['alpha']
        
        # Return embeddings for analysis
        if return_embeddings:
            outputs['embeddings'] = {
                'node_features': node_features,
                'graph_pooled': graph_pooled,
                'modular': modular_out,
                'temporal': temporal_out,
                'combined': combined
            }
        
        return outputs
    
    def get_explainability(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract explainability information from forward outputs.
        """
        return {
            'adjacency': outputs.get('learned_adjacency'),
            'temporal_weights': outputs.get('temporal_weights'),
            'module_coupling': outputs.get('module_coupling'),
            'band_coupling': outputs.get('band_coupling'),
            'uncertainty': outputs.get('uncertainty')
        }


def create_neuro_chrono_graph_v2(config: Dict = None) -> NeuroChronoGraphV2:
    """
    Factory function to create NeuroChronoGraph v2.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    # Valid parameters for NeuroChronoGraphV2.__init__
    valid_params = {
        'n_channels', 'n_times', 'n_classes', 'n_bands',
        'hidden_dim', 'n_graph_layers', 'n_temporal_layers',
        'n_heads', 'dropout', 'clinical_dropout', 'use_clinical', 'use_uncertainty',
        'feature_streams', 'stream_dim',
    }

    default_config = {
        'n_channels': 19,
        'n_times': 2000,
        'n_classes': 3,
        'n_bands': 5,
        'hidden_dim': 128,
        'n_graph_layers': 3,
        'n_temporal_layers': 4,
        'n_heads': 8,
        'dropout': 0.3,
        'clinical_dropout': 0.2,
        'use_clinical': True,
        'use_uncertainty': True,
        'feature_streams': None,
        'stream_dim': 64,
    }
    
    if config:
        # Only include valid parameters, filter out GPU/training config
        filtered_config = {k: v for k, v in config.items() if k in valid_params}
        default_config.update(filtered_config)
    
    return NeuroChronoGraphV2(**default_config)
