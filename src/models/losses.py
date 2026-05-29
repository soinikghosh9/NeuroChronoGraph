"""
Hierarchical Loss Module with Curriculum Learning.

Implements masked loss for hierarchical classification:
1. Screening: Healthy vs Impaired (CN vs [MCI, AD, FTD])
2. Staging: MCI vs Dementia (MCI vs [AD, FTD]) - Masked for CN
3. Subtype: AD vs FTD - Masked for CN/MCI

Curriculum Learning Strategy (Extended for better per-phase learning):
- Phase 1 (epochs 1-8): Focus on screening (100% weight)
- Phase 2 (epochs 9-16): Add staging (screening + staging)
- Phase 3 (epochs 17-30): All three tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better calibration."""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        pt = torch.exp(log_pt)
        pt = pt.clamp(min=1e-8, max=1.0 - 1e-8)

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets)
            loss = -at * focal_term * log_pt
        else:
            loss = -1 * focal_term * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HierarchicalLoss(nn.Module):
    """
    Hierarchical Loss with Curriculum Learning.

    Curriculum phases prevent multi-task gradient conflicts:
    - Phase 1: Train screening head alone (epochs 1-8)
    - Phase 2: Add staging head (epochs 9-16)
    - Phase 3: Add subtype head (epochs 17+)

    Extended phase durations allow each head to stabilize before adding the next.
    """

    # Curriculum phase boundaries - EXTENDED for better per-phase learning
    # More epochs per phase allows each head to stabilize before adding the next
    PHASE1_END = 12  # Screening only until epoch 12 (extended for stream-gate stabilization)
    PHASE2_END = 20  # Add staging until epoch 20
    # Phase 3: All tasks from epoch 21+ (14 epochs with 34 total)

    def __init__(self, weights: dict = None, subtype_weights: torch.Tensor = None,
                 staging_weights: torch.Tensor = None, screening_weights: torch.Tensor = None,
                 label_smoothing: float = 0.1, use_curriculum: bool = True):
        super().__init__()
        self.base_weights = weights if weights else {'screening': 1.0, 'staging': 1.0, 'subtype': 1.0}
        self.label_smoothing = label_smoothing
        self.use_curriculum = use_curriculum
        self.current_epoch = 0

        # Screening: CN vs Impaired
        if screening_weights is not None:
            self.ce_screen = FocalLoss(alpha=screening_weights, gamma=1.0, reduction='none')
        else:
            self.ce_screen = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

        # Subtype: AD vs FTD
        self.ce_subtype = FocalLoss(alpha=subtype_weights, gamma=1.5, reduction='none')

        # Staging: MCI vs Dementia
        if staging_weights is not None:
            self.ce_staging = FocalLoss(alpha=staging_weights, gamma=2.0, reduction='none')
        else:
            self.ce_staging = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def get_curriculum_weights(self, epoch: int) -> dict:
        """
        Get loss weights based on curriculum phase.

        Gradual transition prevents sudden changes that cause instability.
        """
        if not self.use_curriculum:
            return self.base_weights

        # Phase 1: Screening only (epochs 1-6)
        if epoch <= self.PHASE1_END:
            return {
                'screening': self.base_weights['screening'],
                'staging': 0.0,
                'subtype': 0.0
            }

        # Phase 2: Screening + Staging (epochs 7-12)
        elif epoch <= self.PHASE2_END:
            # Gradually ramp up staging weight
            progress = (epoch - self.PHASE1_END) / (self.PHASE2_END - self.PHASE1_END)
            staging_weight = self.base_weights['staging'] * progress
            return {
                'screening': self.base_weights['screening'],
                'staging': staging_weight,
                'subtype': 0.0
            }

        # Phase 3: All three tasks (epochs 13+)
        else:
            # Gradually ramp up subtype weight
            ramp_epochs = 4
            progress = min((epoch - self.PHASE2_END) / ramp_epochs, 1.0)
            subtype_weight = self.base_weights['subtype'] * progress
            return {
                'screening': self.base_weights['screening'],
                'staging': self.base_weights['staging'],
                'subtype': subtype_weight
            }

    def forward(self, outputs, targets, epoch: int = None):
        """
        Compute hierarchical loss with curriculum scheduling.

        Args:
            outputs: Dict containing 'logits_screen', 'logits_stage', 'logits_subtype'
            targets: Tensor of shape (batch_size,) with values 0..3
                     0=AD, 1=FTD, 2=CN, 3=MCI
            epoch: Current training epoch (1-indexed). If None, uses set_epoch value.
        """
        device = targets.device

        # Use provided epoch or stored value
        current_epoch = epoch if epoch is not None else self.current_epoch

        # Get curriculum-adjusted weights
        weights = self.get_curriculum_weights(current_epoch)

        # --- 1. Screening Loss ---
        screen_targets = (targets != 2).long()  # 0 if CN, 1 if Impaired
        loss_screen = self.ce_screen(outputs['logits_screen'], screen_targets)
        loss_screen = loss_screen.mean()

        # --- 2. Staging Loss ---
        mask_impaired = (targets != 2)

        if mask_impaired.sum() > 0 and weights['staging'] > 0:
            stage_logits = outputs['logits_stage'][mask_impaired]
            sub_targets = targets[mask_impaired]
            stage_targets = (sub_targets != 3).long()  # 0 if MCI, 1 if Dementia

            loss_stage = self.ce_staging(stage_logits, stage_targets)
            loss_stage = loss_stage.mean()
        else:
            loss_stage = torch.tensor(0.0, device=device)

        # --- 3. Subtype Loss ---
        mask_dementia = (targets == 0) | (targets == 1)

        if mask_dementia.sum() > 0 and weights['subtype'] > 0:
            subtype_logits = outputs['logits_subtype'][mask_dementia]
            subtype_targets = targets[mask_dementia]

            loss_subtype = self.ce_subtype(subtype_logits, subtype_targets)
            loss_subtype = loss_subtype.mean()
        else:
            loss_subtype = torch.tensor(0.0, device=device)

        # Total Loss with curriculum weights
        total_loss = (weights['screening'] * loss_screen +
                      weights['staging'] * loss_stage +
                      weights['subtype'] * loss_subtype)

        return total_loss, {
            'loss_screen': loss_screen.item(),
            'loss_stage': loss_stage.item() if isinstance(loss_stage, torch.Tensor) else loss_stage,
            'loss_subtype': loss_subtype.item() if isinstance(loss_subtype, torch.Tensor) else loss_subtype,
            'weights': weights  # For logging
        }
