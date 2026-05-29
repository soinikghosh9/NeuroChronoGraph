"""
Adaptive Graph Learning Module.

This module implements dynamic, learnable graph structure generation
that adapts to the data rather than relying on fixed connectivity matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AdaptiveGraphLearning(nn.Module):
    """
    Learn optimal graph adjacency from node features dynamically.
    
    Combines learned structure with functional connectivity prior (wPLI).
    Based on AGGCN (2023) architecture.
    """
    
    def __init__(self,
                 node_dim: int,
                 hidden_dim: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 use_prior: bool = True,
                 prior_weight: float = 0.3):
        """
        Initialize AdaptiveGraphLearning.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden dimension for attention
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_prior: Whether to use FC prior
            prior_weight: Weight for FC prior (1-prior_weight for learned)
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_prior = use_prior
        self.prior_weight = prior_weight
        
        # Query and Key projections for attention-based adjacency
        self.query_proj = nn.Linear(node_dim, hidden_dim * n_heads)
        self.key_proj = nn.Linear(node_dim, hidden_dim * n_heads)
        
        # Gated combination of local and global structure
        self.gate_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Fixed prior blend weight (post-sigmoid)
        # NOTE: alpha was originally nn.Parameter(prior_weight) passed through
        # sigmoid at runtime, but gradient through the gated multi-scale path
        # is near-zero — it never moved from init across all 5 CV folds.
        # The Q/K attention and per-node gate_mlp already provide full
        # adaptivity; a single scalar adds no capacity.  Fixed at the
        # value that was empirically used throughout all experiments.
        if use_prior:
            alpha_val = torch.sigmoid(torch.tensor(prior_weight))
            self.register_buffer('alpha', alpha_val)
        
        # Multi-scale aggregation
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.dropout = nn.Dropout(dropout)
        
        # Edge feature transformation
        self.edge_transform = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                node_features: torch.Tensor,
                fc_prior: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate adaptive adjacency matrix.
        
        Args:
            node_features: Node features [batch, n_nodes, node_dim]
            fc_prior: Functional connectivity prior [batch, n_nodes, n_nodes]
            mask: Optional mask for valid nodes
            
        Returns:
            adjacency: Learned adjacency matrix [batch, n_nodes, n_nodes]
            attention_weights: Attention weights for explainability
        """
        batch_size, n_nodes, _ = node_features.shape
        
        # Multi-head attention for adjacency learning
        Q = self.query_proj(node_features)  # [B, N, H*D]
        K = self.key_proj(node_features)     # [B, N, H*D]
        
        # Reshape for multi-head
        Q = Q.view(batch_size, n_nodes, self.n_heads, self.hidden_dim)
        K = K.view(batch_size, n_nodes, self.n_heads, self.hidden_dim)
        
        # Compute attention scores
        # [B, H, N, N]
        attn_scores = torch.einsum('bnhd,bmhd->bhnm', Q, K) / math.sqrt(self.hidden_dim)
        
        # Average across heads
        attn_scores = attn_scores.mean(dim=1)  # [B, N, N]
        
        # Apply softmax to get probabilities
        A_learned = F.softmax(attn_scores, dim=-1)
        
        # Symmetrize
        A_learned = (A_learned + A_learned.transpose(-1, -2)) / 2
        
        # Remove self-loops
        eye = torch.eye(n_nodes, device=A_learned.device).unsqueeze(0)
        A_learned = A_learned * (1 - eye)
        
        # Combine with prior if available
        if self.use_prior and fc_prior is not None:
            # Normalize prior
            fc_prior = fc_prior / (fc_prior.max(dim=-1, keepdim=True)[0] + 1e-8)
            
            # Fixed blend (alpha is pre-computed sigmoid value stored as buffer)
            A_combined = self.alpha * fc_prior + (1 - self.alpha) * A_learned
        else:
            A_combined = A_learned
        
        # Multi-scale structure (local, medium, global)
        A_local = A_combined
        A_medium = torch.bmm(A_combined, A_combined)  # 2-hop
        A_global = torch.bmm(A_medium, A_combined)     # 3-hop
        
        # Normalize scale weights
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        A_multiscale = (scale_weights[0] * A_local + 
                        scale_weights[1] * A_medium + 
                        scale_weights[2] * A_global)
        
        # Gated combination per node
        gate = self.gate_mlp(node_features).squeeze(-1)  # [B, N]
        gate = gate.unsqueeze(-1) * gate.unsqueeze(-2)    # [B, N, N]
        
        # Apply gating (balance local vs global)
        A_final = gate * A_local + (1 - gate) * A_multiscale
        
        # Normalize
        A_final = A_final / (A_final.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply dropout
        A_final = self.dropout(A_final)
        
        return A_final, attn_scores


class CrossBandAttention(nn.Module):
    """
    Attention mechanism across frequency bands.
    
    Models cross-frequency coupling (e.g., theta-gamma) which is
    critical for understanding AD-related communication breakdown.
    """
    
    def __init__(self,
                 n_bands: int = 5,
                 band_dim: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize CrossBandAttention.
        
        Args:
            n_bands: Number of frequency bands
            band_dim: Dimension of band embeddings
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_bands = n_bands
        self.band_dim = band_dim
        self.n_heads = n_heads
        
        # Band embeddings (learnable)
        self.band_embeddings = nn.Parameter(torch.randn(n_bands, band_dim))
        
        # Cross-band attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=band_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Band-specific projections
        self.band_proj = nn.ModuleList([
            nn.Linear(band_dim, band_dim) for _ in range(n_bands)
        ])
        
        # Coupling strength estimator
        self.coupling_mlp = nn.Sequential(
            nn.Linear(band_dim * 2, band_dim),
            nn.ReLU(),
            nn.Linear(band_dim, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(band_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                band_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-band attention.
        
        Args:
            band_features: Features per band [batch, n_bands, n_nodes, band_dim]
            
        Returns:
            coupled_features: Cross-band coupled features
            coupling_matrix: Band coupling strengths [batch, n_bands, n_bands]
        """
        batch_size, n_bands, n_nodes, band_dim = band_features.shape
        
        # Pool across nodes for band-level representation
        band_repr = band_features.mean(dim=2)  # [B, n_bands, band_dim]
        
        # Add positional band embeddings
        band_repr = band_repr + self.band_embeddings.unsqueeze(0)
        
        # Cross-band attention
        attended, attn_weights = self.cross_attn(
            band_repr, band_repr, band_repr,
            need_weights=True
        )
        
        # Residual connection
        band_repr = self.layer_norm(band_repr + self.dropout(attended))
        
        # Compute coupling strengths
        coupling_matrix = torch.zeros(batch_size, n_bands, n_bands, device=band_repr.device)
        
        for i in range(n_bands):
            for j in range(n_bands):
                if i != j:
                    pair = torch.cat([band_repr[:, i], band_repr[:, j]], dim=-1)
                    coupling_matrix[:, i, j] = self.coupling_mlp(pair).squeeze(-1)
        
        # Apply cross-band modulation to original features
        coupled_features = band_features.clone()
        
        for i in range(n_bands):
            # Modulate band i by information from other bands
            modulation = torch.zeros_like(band_features[:, i])
            for j in range(n_bands):
                if i != j:
                    strength = coupling_matrix[:, j, i].unsqueeze(-1).unsqueeze(-1)
                    modulation = modulation + strength * band_features[:, j]
            
            coupled_features[:, i] = band_features[:, i] + 0.1 * modulation
        
        return coupled_features, coupling_matrix


class GatedGraphConvolution(nn.Module):
    """
    Gated Graph Convolution with adaptive weighting.
    
    Dynamically adjusts the contribution of different spatial scales.
    """
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float = 0.1):
        """
        Initialize GatedGraphConvolution.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Transformation
        self.linear = nn.Linear(in_dim, out_dim)
        
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.Sigmoid()
        )
        
        # Update
        self.update = nn.GRUCell(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        Apply gated graph convolution.
        
        Args:
            x: Node features [batch, n_nodes, in_dim]
            adj: Adjacency matrix [batch, n_nodes, n_nodes]
            
        Returns:
            Updated node features [batch, n_nodes, out_dim]
        """
        batch_size, n_nodes, _ = x.shape
        
        # Message passing
        messages = torch.bmm(adj, x)  # [B, N, in_dim]
        
        # Transform
        h = self.linear(messages)  # [B, N, out_dim]
        h = F.relu(h)
        
        # Compute gate
        gate_input = torch.cat([x, h], dim=-1)
        g = self.gate(gate_input)
        
        # Apply gate
        h = g * h
        
        # GRU update (if dimensions match)
        if self.in_dim == self.out_dim:
            # Use reshape instead of view for non-contiguous tensors
            h_flat = h.reshape(batch_size * n_nodes, self.out_dim)
            x_flat = x.reshape(batch_size * n_nodes, self.out_dim)
            h_flat = self.update(h_flat, x_flat)
            h = h_flat.reshape(batch_size, n_nodes, self.out_dim)
        
        # Normalize and dropout
        h = self.layer_norm(h)
        h = self.dropout(h)
        
        return h
