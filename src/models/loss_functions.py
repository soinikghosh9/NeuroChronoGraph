"""
Loss Functions for Dementia Classification.

This module re-exports loss functions from v2 for backward compatibility.
For new code, import directly from src.models.v2.losses.
"""

# Re-export from v2 for backward compatibility
from .v2.losses import (
    FocalLoss,
    LabelSmoothingLoss,
    MultiObjectiveLoss,
    MMDLoss,
    ExplainabilityLoss,
    CurriculumScheduler
)

import torch
import torch.nn as nn
from typing import Optional


class CombinedLoss(nn.Module):
    """
    Combined loss function: Focal Loss + Label Smoothing.
    
    Kept for backward compatibility with v1 training scripts.
    """
    
    def __init__(self,
                 n_classes: int = 3,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 smoothing: float = 0.1,
                 focal_weight: float = 0.8):
        super().__init__()
        
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.smoothing = LabelSmoothingLoss(n_classes=n_classes, smoothing=smoothing)
        self.focal_weight = focal_weight
        
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        focal_loss = self.focal(inputs, targets)
        smooth_loss = self.smoothing(inputs, targets)
        
        return self.focal_weight * focal_loss + (1 - self.focal_weight) * smooth_loss


def create_loss_function(loss_type: str = 'focal',
                         class_weights: Optional[list] = None,
                         gamma: float = 2.0,
                         n_classes: int = 3) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: 'focal', 'ce', 'label_smoothing', 'combined', or 'multi_objective'
        class_weights: Class weights list [w_AD, w_FTD, w_CN]
        gamma: Focal loss gamma parameter
        n_classes: Number of classes
        
    Returns:
        Loss function module
    """
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        weights = None
    
    if loss_type == 'focal':
        return FocalLoss(alpha=weights, gamma=gamma)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(n_classes=n_classes)
    elif loss_type == 'combined':
        return CombinedLoss(n_classes=n_classes, alpha=weights, gamma=gamma)
    elif loss_type == 'multi_objective':
        return MultiObjectiveLoss(n_classes=n_classes, class_weights=weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    'FocalLoss',
    'LabelSmoothingLoss', 
    'CombinedLoss',
    'MultiObjectiveLoss',
    'create_loss_function'
]
