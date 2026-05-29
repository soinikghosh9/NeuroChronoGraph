"""
Domain Adaptation and Generalization for Cross-Subject EEG Classification.

This module implements techniques to learn subject-invariant features:
1. Gradient Reversal Layer (GRL) - For adversarial domain adaptation
2. Domain Discriminator - Tries to identify which subject an EEG came from
3. MMD Loss - Maximum Mean Discrepancy for distribution alignment
4. Combined Domain-Invariant Training utilities

References:
- Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016)
- Long et al., "Learning Transferable Features with Deep Adaptation Networks" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for Domain Adversarial Neural Networks.
    
    During forward pass: Identity function
    During backward pass: Multiplies gradients by -lambda
    
    This forces the feature extractor to learn domain-invariant features
    by fooling the domain classifier.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradients with scaling factor
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for GRL."""
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """Update lambda (useful for scheduling)."""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Domain (Subject) Discriminator for DANN.
    
    Tries to predict which subject an embedding belongs to.
    When combined with GRL, this forces the feature extractor
    to produce subject-invariant representations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_subjects: int = 88):
        super().__init__()
        
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Efficient 2-layer discriminator
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_subjects)
        )
    
    def forward(self, x, apply_grl: bool = True):
        """
        Args:
            x: Feature embeddings [batch, dim]
            apply_grl: Whether to apply gradient reversal
        """
        if apply_grl:
            x = self.grl(x)
        return self.classifier(x)
    
    def set_lambda(self, lambda_: float):
        """Update GRL lambda for curriculum learning."""
        self.grl.set_lambda(lambda_)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss.
    
    Measures the distance between two distributions in a reproducing
    kernel Hilbert space (RKHS). Used to align feature distributions
    across different subjects/domains.
    
    Uses Gaussian (RBF) kernel with multiple bandwidths for robustness.
    """
    
    def __init__(self, kernel_type: str = 'rbf', kernel_mul: float = 2.0, 
                 kernel_num: int = 5, fix_sigma: Optional[float] = None):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
    
    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-bandwidth Gaussian kernel."""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        
        # Pairwise squared distances
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # Multi-bandwidth kernel
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
        
        # Create kernel with multiple bandwidths
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        
        kernel_val = sum([torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list])
        return kernel_val
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target distributions.
        
        Args:
            source: Features from one domain [batch1, dim]
            target: Features from another domain [batch2, dim]
        """
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)
        
        batch_size = min(source.size(0), target.size(0))
        
        # Subsample if needed for efficiency
        if source.size(0) > batch_size:
            idx = torch.randperm(source.size(0))[:batch_size]
            source = source[idx]
        if target.size(0) > batch_size:
            idx = torch.randperm(target.size(0))[:batch_size]
            target = target[idx]
        
        kernels = self.gaussian_kernel(source, target)
        
        n_s = source.size(0)
        n_t = target.size(0)
        
        # MMD = E[k(s,s)] + E[k(t,t)] - 2*E[k(s,t)]
        XX = kernels[:n_s, :n_s].mean()
        YY = kernels[n_s:, n_s:].mean()
        XY = kernels[:n_s, n_s:].mean()
        
        mmd = XX + YY - 2 * XY
        return mmd


class DomainInvariantLoss(nn.Module):
    """
    Combined loss for domain-invariant learning.
    
    total_loss = classification_loss + lambda_dann * dann_loss + lambda_mmd * mmd_loss
    
    Where:
    - classification_loss: Standard cross-entropy or focal loss
    - dann_loss: Domain adversarial loss (subject classification)
    - mmd_loss: Distribution alignment across subjects
    """
    
    def __init__(self, n_subjects: int, embedding_dim: int,
                 lambda_dann: float = 0.1, lambda_mmd: float = 0.05):
        super().__init__()
        
        self.domain_discriminator = DomainDiscriminator(
            input_dim=embedding_dim,
            hidden_dim=min(256, embedding_dim),
            n_subjects=n_subjects
        )
        self.mmd_loss = MMDLoss()
        
        self.lambda_dann = lambda_dann
        self.lambda_mmd = lambda_mmd
        
        # Subject classification loss
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor, subject_ids: torch.Tensor,
                compute_mmd: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Compute domain-invariant losses.
        
        Args:
            embeddings: Feature embeddings [batch, dim]
            subject_ids: Subject indices [batch]
            compute_mmd: Whether to compute MMD (can skip for efficiency)
            
        Returns:
            total_loss: Combined domain loss
            loss_dict: Individual loss components for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=embeddings.device)
        
        # 1. DANN Loss (Subject Classification with GRL)
        domain_logits = self.domain_discriminator(embeddings, apply_grl=True)
        dann_loss = self.domain_criterion(domain_logits, subject_ids)
        total_loss = total_loss + self.lambda_dann * dann_loss
        loss_dict['dann_loss'] = dann_loss.item()
        
        # Domain accuracy (for monitoring - should approach 1/n_subjects)
        domain_acc = (domain_logits.argmax(dim=1) == subject_ids).float().mean()
        loss_dict['domain_acc'] = domain_acc.item()
        
        # 2. MMD Loss (Distribution Alignment)
        if compute_mmd and self.lambda_mmd > 0:
            unique_subjects = torch.unique(subject_ids)
            if len(unique_subjects) >= 2:
                # Sample pairs of subjects for MMD
                idx1, idx2 = torch.randperm(len(unique_subjects))[:2]
                subj1, subj2 = unique_subjects[idx1], unique_subjects[idx2]
                
                mask1 = subject_ids == subj1
                mask2 = subject_ids == subj2
                
                if mask1.sum() > 0 and mask2.sum() > 0:
                    mmd = self.mmd_loss(embeddings[mask1], embeddings[mask2])
                    total_loss = total_loss + self.lambda_mmd * mmd
                    loss_dict['mmd_loss'] = mmd.item()
        
        return total_loss, loss_dict
    
    def update_lambda(self, progress: float):
        """
        Schedule lambda values (curriculum learning).
        
        Args:
            progress: Training progress [0, 1]
        """
        # Gradually increase domain adversarial strength
        p = progress
        lambda_p = 2.0 / (1.0 + np.exp(-10 * p)) - 1  # Sigmoid schedule [0, 1]
        
        self.domain_discriminator.set_lambda(lambda_p)
        
        return lambda_p


def create_subject_mapping(subjects: np.ndarray) -> Tuple[dict, torch.Tensor]:
    """
    Create subject ID to index mapping.
    
    Args:
        subjects: Array of subject ID strings
        
    Returns:
        mapping: Dict from subject_id -> index
        indices: Tensor of subject indices [n_samples]
    """
    unique_subjects = np.unique(subjects)
    mapping = {s: i for i, s in enumerate(unique_subjects)}
    indices = torch.tensor([mapping[s] for s in subjects], dtype=torch.long)
    return mapping, indices
