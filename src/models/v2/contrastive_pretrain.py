"""
Contrastive Pretraining Framework.

This module implements self-supervised contrastive learning for EEG:
- Multi-view data augmentation (time, frequency, spatial)
- Barlow Twins loss (no negative samples needed)
- Subject-invariant representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import random


class EEGAugmentation:
    """
    Multi-view augmentation strategies for EEG data.
    
    Creates different "views" of the same sample for contrastive learning.
    """
    
    def __init__(self,
                 time_mask_ratio: float = 0.1,
                 channel_mask_ratio: float = 0.1,
                 noise_std: float = 0.1,
                 freq_shift_max: float = 0.1,
                 time_shift_max: int = 50):
        """
        Initialize augmentation parameters.
        
        Args:
            time_mask_ratio: Fraction of time points to mask
            channel_mask_ratio: Fraction of channels to mask
            noise_std: Standard deviation of additive noise
            freq_shift_max: Max frequency shift ratio
            time_shift_max: Max time shift in samples
        """
        self.time_mask_ratio = time_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.noise_std = noise_std
        self.freq_shift_max = freq_shift_max
        self.time_shift_max = time_shift_max
        
    def time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random time segments."""
        batch, channels, time = x.shape
        mask_len = int(time * self.time_mask_ratio)
        
        x_aug = x.clone()
        for i in range(batch):
            start = random.randint(0, time - mask_len)
            x_aug[i, :, start:start+mask_len] = 0
        
        return x_aug
    
    def channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random channels (spatial dropout)."""
        batch, channels, time = x.shape
        n_mask = max(1, int(channels * self.channel_mask_ratio))
        
        x_aug = x.clone()
        for i in range(batch):
            mask_idx = random.sample(range(channels), n_mask)
            x_aug[i, mask_idx, :] = 0
        
        return x_aug
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift signal in time (circular)."""
        batch, channels, time = x.shape
        
        x_aug = x.clone()
        for i in range(batch):
            shift = random.randint(-self.time_shift_max, self.time_shift_max)
            x_aug[i] = torch.roll(x[i], shifts=shift, dims=-1)
        
        return x_aug
    
    def amplitude_scale(self, x: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Random amplitude scaling."""
        batch = x.shape[0]
        scales = torch.empty(batch, 1, 1, device=x.device).uniform_(*scale_range)
        return x * scales
    
    def frequency_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random bandstop filter (simulate missing frequency)."""
        # Simple approximation: remove random frequency component
        fft = torch.fft.rfft(x, dim=-1)
        
        batch, channels, freq_bins = fft.shape
        
        for i in range(batch):
            # Random bandstop
            center = random.randint(freq_bins // 4, 3 * freq_bins // 4)
            width = random.randint(1, freq_bins // 10)
            start = max(0, center - width)
            end = min(freq_bins, center + width)
            fft[i, :, start:end] *= 0.5  # Attenuate, don't zero
        
        return torch.fft.irfft(fft, n=x.shape[-1], dim=-1)
    
    def get_view1(self, x: torch.Tensor) -> torch.Tensor:
        """Generate view 1 (time-domain perturbations)."""
        x = self.time_mask(x)
        x = self.time_shift(x)
        x = self.add_noise(x)
        return x
    
    def get_view2(self, x: torch.Tensor) -> torch.Tensor:
        """Generate view 2 (frequency + spatial perturbations)."""
        x = self.channel_mask(x)
        x = self.frequency_filter(x)
        x = self.amplitude_scale(x)
        return x


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps encoder output to embedding space where contrastive loss is applied.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 output_dim: int = 256,
                 n_layers: int = 3):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Loss for self-supervised learning.
    
    Advantages over SimCLR:
    - No need for negative samples
    - No need for large batch sizes
    - More stable training
    
    Loss = Σᵢ(1 - Cᵢᵢ)² + λ·Σᵢ Σⱼ≠ᵢ Cᵢⱼ²
    where C is the cross-correlation matrix between embeddings of two views.
    """
    
    def __init__(self,
                 embedding_dim: int,
                 lambda_param: float = 0.005):
        """
        Initialize Barlow Twins loss.
        
        Args:
            embedding_dim: Dimension of embeddings
            lambda_param: Weight for off-diagonal terms (redundancy reduction)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.lambda_param = lambda_param
        
        # Batch normalization for embeddings
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)
        
    def forward(self,
                z1: torch.Tensor,
                z2: torch.Tensor) -> torch.Tensor:
        """
        Compute Barlow Twins loss.
        
        Args:
            z1: Embeddings from view 1 [batch, embedding_dim]
            z2: Embeddings from view 2 [batch, embedding_dim]
            
        Returns:
            Loss value
        """
        batch_size = z1.shape[0]
        
        # Normalize embeddings
        z1_norm = self.bn(z1)
        z2_norm = self.bn(z2)
        
        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / batch_size  # [D, D]
        
        # Diagonal loss (invariance)
        diag = torch.diagonal(c)
        invariance_loss = ((diag - 1) ** 2).sum()
        
        # Off-diagonal loss (redundancy reduction)
        off_diag_mask = ~torch.eye(self.embedding_dim, dtype=torch.bool, device=c.device)
        redundancy_loss = (c[off_diag_mask] ** 2).sum()
        
        loss = invariance_loss + self.lambda_param * redundancy_loss
        
        return loss


class ContrastivePretrainer(nn.Module):
    """
    Self-supervised contrastive pretraining for EEG encoder.
    
    Learns robust representations without labels by:
    1. Generating two views of each sample
    2. Encoding both views
    3. Projecting to embedding space
    4. Applying Barlow Twins loss
    """
    
    def __init__(self,
                 encoder: nn.Module,
                 encoder_dim: int,
                 projection_dim: int = 256,
                 hidden_dim: int = 512):
        """
        Initialize ContrastivePretrainer.
        
        Args:
            encoder: The encoder model to pretrain
            encoder_dim: Output dimension of encoder
            projection_dim: Dimension of projection space
            hidden_dim: Hidden dimension for projection head
        """
        super().__init__()
        
        self.encoder = encoder
        
        # Projection head
        self.projector = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
        
        # Loss function
        self.loss_fn = BarlowTwinsLoss(projection_dim)
        
        # Augmentation
        self.augmentor = EEGAugmentation()
        
    def forward(self,
                x: torch.Tensor,
                return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass for pretraining.
        
        Args:
            x: Input EEG [batch, channels, time]
            return_embedding: If True, return encoder output instead of loss
            
        Returns:
            Loss value or encoder embedding
        """
        if return_embedding:
            return self.encoder(x)
        
        # Generate two views
        view1 = self.augmentor.get_view1(x)
        view2 = self.augmentor.get_view2(x)
        
        # Encode
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)
        
        # Handle different encoder output types
        if isinstance(h1, tuple):
            h1 = h1[0]
            h2 = h2[0]
        
        # Flatten if needed
        # Flatten if needed
        if h1.dim() > 2:
            h1 = h1.mean(dim=-1)  # Pool over time/nodes
            h2 = h2.mean(dim=-1)
        
        # Project
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # Compute loss
        loss = self.loss_fn(z1, z2)
        
        return loss


class SubjectInvariantLoss(nn.Module):
    """
    Loss for learning subject-invariant features.
    
    Encourages features to be similar for same diagnosis, different for different.
    Uses MMD (Maximum Mean Discrepancy) to align distributions.
    """
    
    def __init__(self,
                 kernel: str = 'rbf',
                 sigma: float = 1.0):
        super().__init__()
        
        self.kernel = kernel
        self.sigma = sigma
        
    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between x and y."""
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        yy = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy = torch.mm(x, y.T)
        
        dist = xx + yy.T - 2 * xy
        
        return torch.exp(-dist / (2 * self.sigma ** 2))
    
    def forward(self,
                features: torch.Tensor,
                subjects: torch.Tensor) -> torch.Tensor:
        """
        Compute subject invariance loss.
        
        Minimizes MMD between features from different subjects.
        
        Args:
            features: Feature vectors [batch, dim]
            subjects: Subject IDs [batch]
            
        Returns:
            MMD loss
        """
        unique_subjects = torch.unique(subjects)
        
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Sample two subjects
        idx = torch.randperm(len(unique_subjects))[:2]
        s1, s2 = unique_subjects[idx[0]], unique_subjects[idx[1]]
        
        f1 = features[subjects == s1]
        f2 = features[subjects == s2]
        
        if len(f1) == 0 or len(f2) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # MMD
        k_xx = self._rbf_kernel(f1, f1).mean()
        k_yy = self._rbf_kernel(f2, f2).mean()
        k_xy = self._rbf_kernel(f1, f2).mean()
        
        mmd = k_xx + k_yy - 2 * k_xy
        
        return mmd


class DisentangledEncoder(nn.Module):
    """
    Encoder that disentangles subject-specific and disease-specific features.
    
    Key for generalization to unseen subjects.
    """
    
    def __init__(self,
                 base_encoder: nn.Module,
                 encoder_dim: int,
                 shared_dim: int = 128,
                 private_dim: int = 64):
        super().__init__()
        
        self.base_encoder = base_encoder
        
        # Shared (disease-specific) encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(encoder_dim, shared_dim * 2),
            nn.ReLU(),
            nn.Linear(shared_dim * 2, shared_dim)
        )
        
        # Private (subject-specific) encoder
        self.private_encoder = nn.Sequential(
            nn.Linear(encoder_dim, private_dim * 2),
            nn.ReLU(),
            nn.Linear(private_dim * 2, private_dim)
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim + private_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )
        
    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input into shared and private representations.
        
        Returns:
            shared: Disease-specific features (used for classification)
            private: Subject-specific features
            reconstructed: Reconstructed base features
        """
        # Base encoding
        base = self.base_encoder(x)
        if isinstance(base, tuple):
            base = base[0]
        if base.dim() > 2:
            base = base.mean(dim=1)
        
        # Disentangle
        shared = self.shared_encoder(base)
        private = self.private_encoder(base)
        
        # Reconstruct
        combined = torch.cat([shared, private], dim=-1)
        reconstructed = self.decoder(combined)
        
        return shared, private, reconstructed
    
    def compute_loss(self,
                     shared: torch.Tensor,
                     private: torch.Tensor,
                     reconstructed: torch.Tensor,
                     base: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement losses.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, base)
        
        # Orthogonality constraint (shared and private should be independent)
        ortho_loss = torch.abs(torch.mm(shared.T, private)).mean()
        
        return {
            'reconstruction': recon_loss,
            'orthogonality': ortho_loss,
            'total': recon_loss + 0.1 * ortho_loss
        }
