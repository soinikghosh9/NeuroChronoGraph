"""
Multi-Objective Loss Functions for NeuroChronoGraph v2.

Combines:
- Classification loss (Focal)
- MMSE regression loss
- Explainability constraints
- Domain invariance loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    """
    
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predictions [batch, n_classes]
            targets: Ground truth [batch]
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for regularization.
    """
    
    def __init__(self,
                 n_classes: int,
                 smoothing: float = 0.1):
        super().__init__()
        
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class LabelSmoothingFocalLoss(nn.Module):
    """
    Combined Label Smoothing + Focal Loss for imbalanced classification.
    
    This combines:
    - Focal Loss: Down-weights easy examples, focuses on hard ones
    - Label Smoothing: Reduces overconfidence on majority class
    
    Best for AD/FTD classification where:
    - Classes are imbalanced (AD > FTD)
    - Model tends to be overconfident on majority class
    """
    
    def __init__(self,
                 n_classes: int = 3,
                 gamma: float = 2.0,
                 smoothing: float = 0.1,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            n_classes: Number of classes
            gamma: Focal loss focusing parameter (higher = more focus on hard examples)
            smoothing: Label smoothing factor (0.1 = 10% uncertainty)
            alpha: Optional class weights [n_classes]
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        
        self.n_classes = n_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined label smoothing focal loss.
        
        Args:
            logits: Predictions [batch, n_classes]
            targets: Ground truth [batch]
            
        Returns:
            Loss value
        """
        # Create smooth labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (self.n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - p_t)^gamma for the true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [batch]
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        # Compute smoothed cross-entropy with focal weighting
        # Sum over classes, then apply focal weight per sample
        ce_per_sample = torch.sum(-smooth_targets * log_probs, dim=-1)  # [batch]
        focal_loss = focal_weight * ce_per_sample
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ExplainabilityLoss(nn.Module):
    """
    Constrain model explanations to match anatomical priors.
    
    For AD: should attend to posterior regions
    For FTD: should attend to frontal regions
    """
    
    def __init__(self,
                 n_channels: int = 19,
                 temperature: float = 1.0):
        super().__init__()
        
        self.temperature = temperature
        
        # Define anatomical priors (which regions are expected for each class)
        # Channel order from config.py:
        # Fp1=0, Fp2=1, F3=2, F4=3, C3=4, C4=5, P3=6, P4=7, O1=8, O2=9,
        # F7=10, F8=11, T3=12, T4=13, T5=14, T6=15, Fz=16, Cz=17, Pz=18
        
        # AD: posterior focus (parietal P3=6, P4=7, Pz=18 + occipital O1=8, O2=9)
        ad_prior = torch.zeros(n_channels)
        ad_prior[[6, 7, 8, 9, 18]] = 1.0  # Parietal-Occipital
        ad_prior = ad_prior / ad_prior.sum()
        
        # FTD: frontal focus (Fp1=0, Fp2=1, F3=2, F4=3, F7=10, F8=11, Fz=16)
        # Plus temporal (T3=12, T4=13, T5=14, T6=15)
        ftd_prior = torch.zeros(n_channels)
        ftd_prior[[0, 1, 2, 3, 10, 11, 16]] = 0.7   # Frontal
        ftd_prior[[12, 13, 14, 15]] = 0.3  # Temporal
        ftd_prior = ftd_prior / ftd_prior.sum()
        
        cn_prior = torch.ones(n_channels) / n_channels  # Uniform
        
        self.register_buffer('priors', torch.stack([ad_prior, ftd_prior, cn_prior]))
        
    def forward(self,
                attention: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute explainability constraint loss.
        
        Args:
            attention: Attention weights over channels [batch, n_channels]
            targets: Class labels [batch]
            
        Returns:
            KL divergence from expected prior
        """
        # Get expected priors for each sample (move to correct device)
        priors = self.priors.to(targets.device)
        expected = priors[targets]  # [batch, n_channels]
        
        # Normalize attention
        attention = F.softmax(attention / self.temperature, dim=-1)
        
        # KL divergence
        kl = F.kl_div(
            torch.log(attention + 1e-8),
            expected,
            reduction='batchmean'
        )
        
        return kl


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss for domain adaptation.
    
    Minimizes distribution distance between different subjects/sites.
    """
    
    def __init__(self,
                 kernel: str = 'rbf',
                 sigma: float = 1.0):
        super().__init__()
        
        self.kernel = kernel
        self.sigma = sigma
        
    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel."""
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        dist = xx + yy.T - 2 * torch.mm(x, y.T)
        return torch.exp(-dist / (2 * self.sigma ** 2))
    
    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between source and target distributions.
        """
        k_ss = self._rbf_kernel(source, source).mean()
        k_tt = self._rbf_kernel(target, target).mean()
        k_st = self._rbf_kernel(source, target).mean()
        
        return k_ss + k_tt - 2 * k_st


class MultiObjectiveLoss(nn.Module):
    """
    Combined multi-objective loss for NeuroChronoGraph v2.
    """
    
    def __init__(self,
                 n_classes: int = 3,
                 focal_gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None,
                 lambda_mmse: float = 0.1,
                 lambda_explain: float = 0.05,
                 lambda_domain: float = 0.01,
                 use_uncertainty: bool = True):
        super().__init__()
        
        self.lambda_mmse = lambda_mmse
        self.lambda_explain = lambda_explain
        self.lambda_domain = lambda_domain
        self.use_uncertainty = use_uncertainty
        
        # Class weights for imbalance (AD=36, FTD=23, CN=29)
        if class_weights is None:
            # Inverse frequency weighting
            class_weights = torch.tensor([1/36, 1/23, 1/29])
            class_weights = class_weights / class_weights.sum()
        
        
        # Use LabelSmoothingFocalLoss for better calibration and to prevent overconfidence
        # This is critical for handling noisy labels or subtle mismatches (e.g. FTD vs AD)
        self.classification_loss = LabelSmoothingFocalLoss(
            n_classes=3, # Not used directly, but good for init
            gamma=focal_gamma,
            smoothing=0.1,
            alpha=class_weights
        )
        # self.label_smoothing = LabelSmoothingLoss(n_classes, smoothing=0.1) # Integrated into above
        self.explain_loss = ExplainabilityLoss(n_channels=19)
        self.mmd_loss = MMDLoss()
        
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.
        
        Args:
            outputs: Model outputs dict
            targets: Target dict with 'class', 'mmse', 'subjects'
            
        Returns:
            Dict with individual losses and total
        """
        losses = {}
        
        # 1. Classification loss
        logits = outputs['logits']
        class_targets = targets['class']
        
        # Combined Label Smoothing + Focal Loss
        losses['classification'] = self.classification_loss(logits, class_targets)
        # smooth = self.label_smoothing(logits, class_targets) # Integrated
        # losses['classification'] = 0.7 * focal + 0.3 * smooth
        
        # 2. MMSE regression loss
        if 'mmse_pred' in outputs and 'mmse' in targets:
            mmse_pred = outputs['mmse_pred']
            mmse_target = targets['mmse']
            
            # Masked loss (only for samples with valid MMSE)
            mask = mmse_target > 0
            if mask.sum() > 0:
                losses['mmse'] = F.mse_loss(mmse_pred[mask], mmse_target[mask])
            else:
                losses['mmse'] = torch.tensor(0.0, device=logits.device)
        
        # 3. Explainability constraint
        if 'learned_adjacency' in outputs:
            # Use node importance from adjacency
            adj = outputs['learned_adjacency']
            node_importance = adj.sum(dim=-1)  # [batch, n_nodes]
            losses['explainability'] = self.explain_loss(node_importance, class_targets)
        
        # 4. Uncertainty loss (evidential) - compute directly without creating new head
        if self.use_uncertainty and 'evidential_alpha' in outputs:
            alpha = outputs['evidential_alpha']
            S = alpha.sum(dim=-1, keepdim=True)
            n_classes = alpha.shape[-1]
            
            # One-hot targets
            targets_oh = F.one_hot(class_targets, n_classes).float()
            
            # Expected log likelihood (Dirichlet-based)
            A = torch.sum(targets_oh * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
            
            # KL divergence from uniform prior
            alpha0 = torch.ones_like(alpha)
            S0 = alpha0.sum(dim=-1, keepdim=True)
            kl = (torch.lgamma(S) - torch.lgamma(S0) - 
                  torch.lgamma(alpha).sum(dim=-1, keepdim=True) + 
                  torch.lgamma(alpha0).sum(dim=-1, keepdim=True))
            kl = kl + torch.sum((alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(S)), dim=-1, keepdim=True)
            
            losses['uncertainty'] = A.mean() + 0.1 * kl.mean()
        
        # 5. Domain invariance (subject-level)
        if 'embeddings' in outputs and 'subjects' in targets:
            embeddings = outputs['embeddings']['combined']
            subjects = targets['subjects']
            
            # Sample two subjects
            unique_subjects = torch.unique(subjects)
            if len(unique_subjects) >= 2:
                s1, s2 = unique_subjects[0], unique_subjects[1]
                losses['domain'] = self.mmd_loss(
                    embeddings[subjects == s1],
                    embeddings[subjects == s2]
                )
            else:
                losses['domain'] = torch.tensor(0.0, device=logits.device)
        
        # Combine losses
        total = losses['classification']
        
        if 'mmse' in losses:
            total = total + self.lambda_mmse * losses['mmse']
        if 'explainability' in losses:
            total = total + self.lambda_explain * losses['explainability']
        if 'domain' in losses:
            total = total + self.lambda_domain * losses['domain']
        if 'uncertainty' in losses:
            total = total + 0.01 * losses['uncertainty']
        
        losses['total'] = total
        
        return losses


class CurriculumScheduler:
    """
    Curriculum learning scheduler for loss weights.
    
    Gradually increases difficulty and constraint weights.
    """
    
    def __init__(self,
                 warmup_epochs: int = 10,
                 total_epochs: int = 100):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
    def get_lambda_explain(self) -> float:
        """Explainability weight (starts low, increases)."""
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        
        progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return min(0.1, progress * 0.1)
    
    def get_lambda_domain(self) -> float:
        """Domain adaptation weight (starts low)."""
        if self.current_epoch < self.warmup_epochs // 2:
            return 0.0
        
        progress = self.current_epoch / self.total_epochs
        return min(0.05, progress * 0.05)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss for Domain Adaptation.
    
    MMD measures the difference between two probability distributions
    by comparing their embeddings in a reproducing kernel Hilbert space (RKHS).
    
    This is used to learn subject-invariant features by minimizing the
    distribution difference between:
    - Different subjects (cross-subject generalization)
    - Different sessions (session adaptation)
    
    Reference: 
    - Gretton et al. (2012) "A Kernel Two-Sample Test"
    - Long et al. (2015) "Learning Transferable Features with Deep Adaptation Networks"
    """
    
    def __init__(self,
                 kernel: str = 'rbf',
                 bandwidth: float = None,
                 multi_scale: bool = True):
        """
        Args:
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            bandwidth: RBF kernel bandwidth (if None, uses median heuristic)
            multi_scale: Use multiple bandwidths for robustness
        """
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.multi_scale = multi_scale
        
    def _rbf_kernel(self, 
                    x: torch.Tensor, 
                    y: torch.Tensor, 
                    bandwidth: float) -> torch.Tensor:
        """Compute RBF (Gaussian) kernel between x and y."""
        # ||x - y||^2
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        dist = xx + yy.t() - 2 * xy
        
        return torch.exp(-dist / (2 * bandwidth ** 2))
    
    def _compute_bandwidth(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute bandwidth using median heuristic."""
        combined = torch.cat([x, y], dim=0)
        n = combined.size(0)
        
        # Pairwise distances
        xx = torch.sum(combined ** 2, dim=1, keepdim=True)
        dist = xx + xx.t() - 2 * torch.mm(combined, combined.t())
        dist = torch.sqrt(torch.clamp(dist, min=1e-10))
        
        # Median of non-zero distances
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        median_dist = torch.median(dist[mask])
        
        return median_dist.item() + 1e-6
    
    def forward(self, 
                source: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between source and target distributions.
        
        Args:
            source: Source domain features [n_source, d]
            target: Target domain features [n_target, d]
            
        Returns:
            MMD loss value (scalar)
        """
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)
        
        # Compute bandwidth
        if self.bandwidth is None:
            bandwidth = self._compute_bandwidth(source, target)
        else:
            bandwidth = self.bandwidth
        
        if self.multi_scale:
            # Use multiple bandwidths for robustness
            bandwidths = [bandwidth * 0.5, bandwidth, bandwidth * 2.0]
            mmd_sum = 0.0
            
            for bw in bandwidths:
                K_ss = self._rbf_kernel(source, source, bw)
                K_tt = self._rbf_kernel(target, target, bw)
                K_st = self._rbf_kernel(source, target, bw)
                
                mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
                mmd_sum += mmd
            
            return mmd_sum / len(bandwidths)
        else:
            K_ss = self._rbf_kernel(source, source, bandwidth)
            K_tt = self._rbf_kernel(target, target, bandwidth)
            K_st = self._rbf_kernel(source, target, bandwidth)
            
            mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
            return mmd


class DomainAdaptationLoss(nn.Module):
    """
    Complete Domain Adaptation Loss combining classification and MMD.
    
    This loss encourages the model to:
    1. Classify correctly (task loss)
    2. Learn subject-invariant features (MMD loss)
    """
    
    def __init__(self,
                 n_classes: int = 3,
                 mmd_weight: float = 0.1,
                 class_weights: torch.Tensor = None):
        super().__init__()
        
        self.classification_loss = LabelSmoothingFocalLoss(
            n_classes=n_classes,
            gamma=2.0,
            smoothing=0.1,
            alpha=class_weights
        )
        self.mmd_loss = MMDLoss(multi_scale=True)
        self.mmd_weight = mmd_weight
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                embeddings: torch.Tensor,
                subject_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits: Classification logits [batch, n_classes]
            targets: Ground truth labels [batch]
            embeddings: Feature embeddings [batch, d]
            subject_ids: Subject IDs for each sample [batch]
            
        Returns:
            Dictionary with 'total', 'classification', 'mmd' losses
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, targets)
        
        # MMD loss between subjects
        unique_subjects = torch.unique(subject_ids)
        
        if len(unique_subjects) >= 2:
            # Compare first half to second half of subjects
            mid = len(unique_subjects) // 2
            source_mask = torch.isin(subject_ids, unique_subjects[:mid])
            target_mask = torch.isin(subject_ids, unique_subjects[mid:])
            
            source_embeddings = embeddings[source_mask]
            target_embeddings = embeddings[target_mask]
            
            mmd = self.mmd_loss(source_embeddings, target_embeddings)
        else:
            mmd = torch.tensor(0.0, device=logits.device)
        
        total = cls_loss + self.mmd_weight * mmd
        
        return {
            'total': total,
            'classification': cls_loss,
            'mmd': mmd
        }

