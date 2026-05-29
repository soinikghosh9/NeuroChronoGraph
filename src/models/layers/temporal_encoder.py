"""
Temporal Encoder using Transformer.

This module implements the temporal encoding component
for processing sequences of graph representations.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """
    
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 500,
                 dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for temporal sequences.
    """
    
    def __init__(self,
                 d_model: int,
                 max_len: int = 500,
                 dropout: float = 0.1):
        """
        Initialize learnable positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.positions = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_encoding = self.positions(positions)
        
        x = x + pos_encoding.unsqueeze(0)
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal modeling of brain states.
    
    Processes sequences of graph embeddings to capture
    temporal dynamics of brain connectivity.
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 use_learnable_pe: bool = False):
        """
        Initialize temporal transformer encoder.
        
        Args:
            d_model: Model dimension (input/output feature size)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_learnable_pe: Use learnable vs sinusoidal positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        if use_learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode temporal sequence.
        
        Args:
            x: Input sequence (batch, seq_len, d_model)
            mask: Optional attention mask (seq_len, seq_len)
            
        Returns:
            Encoded sequence (batch, seq_len, d_model)
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x, mask=mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def get_sequence_embedding(self,
                                x: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                pooling: str = 'mean') -> torch.Tensor:
        """
        Get single embedding for entire sequence.
        
        Args:
            x: Input sequence (batch, seq_len, d_model)
            mask: Optional attention mask
            pooling: Pooling method ('mean', 'max', 'cls', 'last')
            
        Returns:
            Sequence embedding (batch, d_model)
        """
        encoded = self.forward(x, mask)
        
        if pooling == 'mean':
            return encoded.mean(dim=1)
        elif pooling == 'max':
            return encoded.max(dim=1)[0]
        elif pooling == 'cls':
            return encoded[:, 0, :]  # First token
        elif pooling == 'last':
            return encoded[:, -1, :]  # Last token
        else:
            return encoded.mean(dim=1)


class TemporalAttentionPooling(nn.Module):
    """
    Attention-based pooling for temporal sequences.
    
    Learns to weight different time points based on their
    relevance to the classification task.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4):
        """
        Initialize temporal attention pooling.
        
        Args:
            d_model: Feature dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence using attention.
        
        Args:
            x: Input sequence (batch, seq_len, d_model)
            
        Returns:
            Pooled representation (batch, d_model)
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Apply attention
        attn_output, _ = self.attention(query, x, x)
        
        # Normalize and squeeze
        output = self.norm(attn_output).squeeze(1)
        
        return output
