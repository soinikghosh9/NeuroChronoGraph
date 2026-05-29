"""
Feature-wise Linear Modulation (FiLM) Layer.

This module implements FiLM for conditioning neural network features
on clinical metadata (Age, Sex, MMSE).
"""

import torch
import torch.nn as nn
from typing import Optional


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    
    Applies learned affine transformation to features based on 
    conditioning information (e.g., clinical metadata).
    
    FiLM(F | γ, β) = γ * F + β
    
    where γ and β are learned functions of the conditioning input.
    """
    
    def __init__(self,
                 feature_dim: int,
                 condition_dim: int,
                 hidden_dim: int = 32,
                 use_bias: bool = True):
        """
        Initialize FiLM layer.
        
        Args:
            feature_dim: Dimension of features to modulate
            condition_dim: Dimension of conditioning input
            hidden_dim: Hidden layer dimension for gamma/beta networks
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        
        # Gamma (scale) network
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=use_bias)
        )
        
        # Beta (shift) network
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=use_bias)
        )
        
        # Initialize gamma to ones and beta to zeros
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for identity modulation at start."""
        # Initialize gamma output to produce ones
        nn.init.ones_(self.gamma_net[-1].weight.data.mean(dim=1).unsqueeze(1))
        nn.init.zeros_(self.gamma_net[-1].bias)
        
        # Initialize beta output to produce zeros
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
    
    def forward(self, 
                features: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            features: Input features (batch, feature_dim) or (batch, seq, feature_dim)
            condition: Conditioning input (batch, condition_dim)
            
        Returns:
            Modulated features with same shape as input
        """
        # Compute gamma and beta
        gamma = self.gamma_net(condition)  # (batch, feature_dim)
        beta = self.beta_net(condition)     # (batch, feature_dim)
        
        # Handle different input shapes
        if features.dim() == 3:
            # (batch, seq, feature_dim)
            gamma = gamma.unsqueeze(1)  # (batch, 1, feature_dim)
            beta = beta.unsqueeze(1)     # (batch, 1, feature_dim)
        
        # Apply modulation
        modulated = gamma * features + beta
        
        return modulated


class FiLMBlock(nn.Module):
    """
    FiLM Block with residual connection.
    
    Applies FiLM modulation with an optional transformation
    and residual connection.
    """
    
    def __init__(self,
                 feature_dim: int,
                 condition_dim: int,
                 hidden_dim: int = 32,
                 dropout: float = 0.1,
                 use_residual: bool = True):
        """
        Initialize FiLM block.
        
        Args:
            feature_dim: Feature dimension
            condition_dim: Conditioning dimension
            hidden_dim: Hidden dimension for modulation networks
            dropout: Dropout rate
            use_residual: Whether to add residual connection
        """
        super().__init__()
        
        self.film = FiLMLayer(feature_dim, condition_dim, hidden_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
    
    def forward(self,
                features: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features
            condition: Conditioning input
            
        Returns:
            Modulated and normalized features
        """
        identity = features
        
        x = self.film(features, condition)
        x = self.norm(x)
        x = self.dropout(x)
        
        if self.use_residual:
            x = x + identity
        
        return x


class MetadataEncoder(nn.Module):
    """
    Encoder for clinical metadata.
    
    Encodes Age, Sex, and MMSE into a conditioning vector
    for FiLM layers.
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 output_dim: int = 32,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        """
        Initialize metadata encoder.
        
        Args:
            input_dim: Number of metadata features (Age, Sex, MMSE = 3)
            output_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Encode metadata.
        
        Args:
            metadata: (batch, input_dim) tensor of normalized metadata
                     [Age_normalized, Sex_encoded, MMSE_normalized]
                     
        Returns:
            (batch, output_dim) conditioning embedding
        """
        return self.encoder(metadata)
