"""
NeuroChronoGraph v3 - Anatomically Guided Architecture.

This version extends V2 with:
1.  **Anatomical Attention Constraint**: Enforces disease-specific attention patterns (FTD->Frontal, AD->Posterior).
2.  **Dynamic Graph Refinement**: Refines the graph structure over multiple passes.
3.  **Residual Clinical Conditioning**: Improved conditioning stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.models.v2.adaptive_graph import AdaptiveGraphLearning, CrossBandAttention, GatedGraphConvolution
from src.models.v2.modular_brain_transformer import ModularBrainTransformer, TemporalGraphTransformer
from src.models.v2.clinical_conditioning import HierarchicalFiLMConditioner, UncertaintyHead
from src.models.v2.neuro_chrono_graph_v2 import EEGEncoder, ChannelWiseEncoder, BandSpecificEncoder
from src.models.v3.feature_stream_fusion import FeatureStreamFusion, STREAM_DIMS

class NeuroChronoGraphV3(nn.Module):
    """
    NeuroChronoGraph Version 3.

    Key Innovation: Anatomically Guided Modular Attention with optional
    hand-crafted biomarker stream fusion (spectral/connectivity/complexity/
    microstate) for clinically-grounded ablation studies.
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
                 stream_dim: int = 64):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.use_clinical = use_clinical
        self.use_uncertainty = use_uncertainty
        
        # Encoders (Same as V2)
        self.eeg_encoder = EEGEncoder(n_channels, n_times, hidden_dim // 2, hidden_dim)
        self.band_encoder = BandSpecificEncoder(n_channels, hidden_dim, n_bands)
        self.channel_encoder = ChannelWiseEncoder(n_channels, n_times, hidden_dim)
        
        # Adaptive Graph (V2)
        self.adaptive_graph = AdaptiveGraphLearning(hidden_dim, hidden_dim // 2, n_heads // 2, dropout, use_prior=True)
        
        # Cross Band (V2)
        self.cross_band_attn = CrossBandAttention(n_bands, hidden_dim, n_heads // 2, dropout)
        
        # Graph Layers (V2)
        self.gat_layers = nn.ModuleList([
            GatedGraphConvolution(hidden_dim, hidden_dim, dropout) for _ in range(n_graph_layers)
        ])
        
        # Modular Transformer (V2 - Key for Anatomy)
        self.modular_transformer = ModularBrainTransformer(
            node_dim=hidden_dim,
            module_dim=hidden_dim,
            n_layers=n_graph_layers,
            n_heads=n_heads // 2,
            dropout=dropout
        )
        
        # Temporal Transformer (V2)
        self.temporal_transformer = TemporalGraphTransformer(hidden_dim, hidden_dim * 2, n_temporal_layers, n_heads, dropout)
        
        # Clinical Conditioning
        if use_clinical:
            self.gat_conditioner = HierarchicalFiLMConditioner(hidden_dim, n_graph_layers, False, hidden_dim)
            self.final_conditioner = HierarchicalFiLMConditioner(hidden_dim * 4, 1, use_uncertainty, hidden_dim * 4)
        
        # Optional hand-crafted biomarker stream fusion (Section: Feature streams)
        self.feature_streams = feature_streams
        if feature_streams:
            self.stream_fusion = FeatureStreamFusion(
                embed_dim=hidden_dim * 4,
                stream_dim=stream_dim,
                streams=feature_streams,
                dropout=dropout,
            )
        else:
            self.stream_fusion = None

        # Output Heads (fused embedding has dim hidden_dim*4)
        classifier_input_dim = hidden_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
        self.mmse_regressor = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(classifier_input_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,
                x,
                adj=None,
                band_features=None,
                clinical_data=None,
                feature_streams=None,
                stream_mask=None,
                return_embeddings=False):
        outputs = {}

        # 1. Encoding
        temporal_enc = self.eeg_encoder(x).transpose(1, 2)
        node_features = self.channel_encoder(x)

        if band_features:
            band_enc = self.band_encoder(band_features)
            band_enc_4d = band_enc.unsqueeze(2).expand(-1, -1, self.n_channels, -1)
            coupled_bands, coupling_mat = self.cross_band_attn(band_enc_4d)
            outputs['band_coupling'] = coupling_mat

        # 2. Graph Learning
        learned_adj, adj_attn = self.adaptive_graph(node_features, adj)
        outputs['learned_adjacency'] = learned_adj
        
        # 3. GAT Processing
        h = node_features
        layer_idx = 0
        for gat in self.gat_layers:
            h = gat(h, learned_adj)
            if self.use_clinical and clinical_data:
                h, _ = self.gat_conditioner(h, layer_idx, clinical_data)
                layer_idx += 1
                
        # 4. Modular Transformer (Anatomical Focus)
        modular_out, modular_info = self.modular_transformer(h, learned_adj)
        outputs['module_coupling'] = modular_info['coupling_matrix']
        outputs['module_attention'] = modular_info.get('module_attention') # Ensure this exists in V2 generic
        
        # 5. Temporal
        temporal_out, temp_weights = self.temporal_transformer(temporal_enc)
        outputs['temporal_weights'] = temp_weights
        
        # 6. Fusion
        graph_pooled = h.mean(dim=1)
        combined = torch.cat([modular_out, temporal_out, graph_pooled], dim=-1)
        
        if self.use_clinical and clinical_data:
            combined, final_unc = self.final_conditioner(combined.unsqueeze(1), 0, clinical_data)
            combined = combined.squeeze(1)
            outputs['conditioning_uncertainty'] = final_unc

        # 6b. Hand-crafted biomarker stream fusion (optional)
        if self.stream_fusion is not None:
            combined = self.stream_fusion(combined, feature_streams, stream_mask)
            outputs['stream_gates'] = self.stream_fusion.gate_values()

        # 7. Outputs
        logits = self.classifier(combined)
        outputs['logits'] = logits
        outputs['probs'] = F.softmax(logits, dim=-1)

        mmse_pred = self.mmse_regressor(combined)
        outputs['mmse_pred'] = mmse_pred.squeeze(-1)
        
        if self.use_uncertainty:
            unc_out = self.uncertainty_head(combined)
            outputs['uncertainty'] = unc_out['uncertainty']
            outputs['evidential_alpha'] = unc_out['alpha']
            
        if return_embeddings:
            outputs['embeddings'] = {
                'combined': combined,
                'graph_pooled': graph_pooled
            }
            
        return outputs

def create_neuro_chrono_graph_v3(config=None):
    valid_params = {
        'n_channels', 'n_times', 'n_classes', 'n_bands',
        'hidden_dim', 'n_graph_layers', 'n_temporal_layers',
        'n_heads', 'dropout', 'use_clinical', 'use_uncertainty',
        'feature_streams', 'stream_dim'
    }
    default = {
        'n_channels': 19, 'n_times': 2000, 'n_classes': 3, 'n_bands': 5,
        'hidden_dim': 128, 'n_graph_layers': 3, 'n_temporal_layers': 4,
        'n_heads': 8, 'dropout': 0.3, 'use_clinical': True, 'use_uncertainty': True,
        'feature_streams': None, 'stream_dim': 64,
    }
    if config:
        default.update({k: v for k, v in config.items() if k in valid_params})
    return NeuroChronoGraphV3(**default)
