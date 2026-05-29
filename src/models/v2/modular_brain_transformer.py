"""
Modular Brain Transformer.

This module implements modular brain network analysis inspired by mBrainGT,
processing brain regions as functional modules with intra- and inter-module attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# Default brain module definitions based on 10-20 system
# Channel order from config.py CHANNEL_NAMES:
# Fp1=0, Fp2=1, F3=2, F4=3, C3=4, C4=5, P3=6, P4=7, O1=8, O2=9,
# F7=10, F8=11, T3=12, T4=13, T5=14, T6=15, Fz=16, Cz=17, Pz=18
DEFAULT_BRAIN_MODULES = {
    'frontal': [0, 1, 2, 3, 10, 11, 16],   # Fp1, Fp2, F3, F4, F7, F8, Fz
    'central': [4, 5, 17],                  # C3, C4, Cz
    'temporal': [12, 13, 14, 15],           # T3, T4, T5, T6
    'parietal': [6, 7, 18],                 # P3, P4, Pz
    'occipital': [8, 9]                     # O1, O2
}

# Channel name to index mapping (matches config.py CHANNEL_NAMES)
CHANNEL_TO_IDX = {
    'Fp1': 0, 'Fp2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 
    'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11,
    'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'Fz': 16, 'Cz': 17, 'Pz': 18
}


class IntraModuleAttention(nn.Module):
    """
    Attention within a brain module (e.g., within frontal lobe).
    """
    
    def __init__(self,
                 node_dim: int,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(node_dim)
        
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply intra-module attention.
        
        Args:
            x: Node features within module [batch, n_module_nodes, node_dim]
            adj: Optional adjacency within module
            
        Returns:
            Updated features
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        # FFN
        x = self.layer_norm2(x + self.ffn(x))
        
        return x


class InterModuleAttention(nn.Module):
    """
    Cross-attention between brain modules.
    
    Captures inter-lobe connectivity (e.g., fronto-parietal, temporo-parietal).
    """
    
    def __init__(self,
                 module_dim: int,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=module_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(module_dim)
        
        # Coupling strength estimator
        self.coupling_mlp = nn.Sequential(
            nn.Linear(module_dim * 2, module_dim),
            nn.ReLU(),
            nn.Linear(module_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                module_embeddings: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply inter-module cross-attention.
        
        Args:
            module_embeddings: Dict of module name -> embedding [batch, module_dim]
            
        Returns:
            Updated module embeddings and coupling matrix
        """
        module_names = list(module_embeddings.keys())
        n_modules = len(module_names)
        batch_size = module_embeddings[module_names[0]].shape[0]
        module_dim = module_embeddings[module_names[0]].shape[-1]
        
        # Stack into tensor
        stacked = torch.stack([module_embeddings[name] for name in module_names], dim=1)
        # [batch, n_modules, module_dim]
        
        # Self-attention across modules
        attended, attn_weights = self.cross_attention(
            stacked, stacked, stacked,
            need_weights=True
        )
        
        # Residual
        stacked = self.layer_norm(stacked + attended)
        
        # Compute coupling matrix
        coupling = torch.zeros(batch_size, n_modules, n_modules, device=stacked.device)
        
        for i in range(n_modules):
            for j in range(n_modules):
                if i != j:
                    pair = torch.cat([stacked[:, i], stacked[:, j]], dim=-1)
                    coupling[:, i, j] = self.coupling_mlp(pair).squeeze(-1)
        
        # Convert back to dict
        updated = {name: stacked[:, i] for i, name in enumerate(module_names)}
        
        return updated, coupling


class ModularBrainTransformer(nn.Module):
    """
    Modular Brain Graph Transformer.
    
    Based on mBrainGT (2024), this model:
    1. Partitions brain into functional modules
    2. Applies intra-module attention
    3. Applies inter-module cross-attention
    4. Aggregates for final representation
    """
    
    def __init__(self,
                 node_dim: int,
                 module_dim: int = 128,
                 n_layers: int = 3,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 brain_modules: Optional[Dict[str, List[int]]] = None):
        """
        Initialize ModularBrainTransformer.
        
        Args:
            node_dim: Dimension of input node features
            module_dim: Dimension of module embeddings
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            brain_modules: Dict mapping module name to channel indices
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.module_dim = module_dim
        self.n_layers = n_layers
        
        self.brain_modules = brain_modules or DEFAULT_BRAIN_MODULES
        self.module_names = list(self.brain_modules.keys())
        n_modules = len(self.module_names)
        
        # Input projection
        self.input_proj = nn.Linear(node_dim, module_dim)
        
        # Module-specific encoders
        self.module_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(module_dim, module_dim),
                nn.LayerNorm(module_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for name in self.module_names
        })
        
        # Intra-module attention layers
        self.intra_attn_layers = nn.ModuleList([
            IntraModuleAttention(module_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Inter-module attention layers
        self.inter_attn_layers = nn.ModuleList([
            InterModuleAttention(module_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Module aggregation
        self.module_pool = nn.Sequential(
            nn.Linear(module_dim, module_dim),
            nn.Tanh()
        )
        
        # Final aggregation across modules
        self.final_attn = nn.MultiheadAttention(
            embed_dim=module_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(module_dim * n_modules, module_dim)
        
        # Learnable module position embeddings
        self.module_pos_embed = nn.Parameter(torch.randn(n_modules, module_dim))
        
    def forward(self,
                x: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Process brain graph through modular transformer.
        
        Args:
            x: Node features [batch, n_nodes, node_dim]
            adj: Adjacency matrix [batch, n_nodes, n_nodes]
            
        Returns:
            output: Final representation [batch, module_dim]
            info: Dictionary with module embeddings and coupling
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)  # [B, N, module_dim]
        
        # Split into modules
        module_features = {}
        for name, indices in self.brain_modules.items():
            # Handle case where indices might be out of range
            valid_indices = [i for i in indices if i < x.shape[1]]
            if valid_indices:
                module_x = x[:, valid_indices, :]  # [B, n_module, module_dim]
                module_x = self.module_encoders[name](module_x)
                module_features[name] = module_x
            else:
                # Create dummy module if no valid indices
                module_features[name] = torch.zeros(batch_size, 1, self.module_dim, device=x.device)
        
        # Store coupling matrices
        all_couplings = []
        
        # Apply layers
        for layer_idx in range(self.n_layers):
            # Intra-module attention
            for name in self.module_names:
                module_features[name] = self.intra_attn_layers[layer_idx](
                    module_features[name]
                )
            
            # Pool modules to get module-level representations
            module_embeddings = {
                name: self.module_pool(feat.mean(dim=1))  # [B, module_dim]
                for name, feat in module_features.items()
            }
            
            # Inter-module attention
            module_embeddings, coupling = self.inter_attn_layers[layer_idx](
                module_embeddings
            )
            all_couplings.append(coupling)
            
            # Broadcast back to node level
            for name in self.module_names:
                n_nodes_in_module = module_features[name].shape[1]
                expanded = module_embeddings[name].unsqueeze(1).expand(-1, n_nodes_in_module, -1)
                module_features[name] = module_features[name] + 0.1 * expanded
        
        # Final module embeddings
        final_module_embeds = {
            name: feat.mean(dim=1) + self.module_pos_embed[i]
            for i, (name, feat) in enumerate(module_features.items())
        }
        
        # Stack and apply final attention
        stacked = torch.stack([final_module_embeds[name] for name in self.module_names], dim=1)
        attended, _ = self.final_attn(stacked, stacked, stacked)
        
        # Flatten and project
        output = attended.reshape(batch_size, -1)  # [B, n_modules * module_dim]
        output = self.output_proj(output)  # [B, module_dim]
        
        # Average coupling across layers
        avg_coupling = torch.stack(all_couplings, dim=0).mean(dim=0)
        
        info = {
            'module_embeddings': final_module_embeds,
            'coupling_matrix': avg_coupling,
            'module_names': self.module_names
        }
        
        return output, info


class TemporalGraphTransformer(nn.Module):
    """
    Temporal Transformer for dynamic graph sequences.
    
    Captures how brain network states evolve over time.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Temporal pooling
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process temporal sequence of graph embeddings.
        
        Args:
            x: Graph embeddings over time [batch, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            output: Aggregated temporal representation [batch, hidden_dim]
            temporal_weights: Attention weights over time [batch, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Temporal attention pooling
        attn_scores = self.temporal_attn(x).squeeze(-1)  # [B, T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        temporal_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum
        output = torch.bmm(temporal_weights.unsqueeze(1), x).squeeze(1)  # [B, hidden_dim]
        output = self.output_proj(output)
        
        return output, temporal_weights
