"""
Utility Functions and Metrics.

This module provides utility functions for evaluation, visualization,
and reproducibility.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    classification_report
)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None,
                        class_names: List[str] = None) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: Names of classes
        
    Returns:
        Dictionary of metrics
    """
    if class_names is None:
        class_names = ['AD', 'FTD', 'CN']
    
    metrics = {
        # Overall metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        
        # Per-class metrics
        'f1_per_class': f1_score(y_true, y_pred, average=None).tolist(),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        
        # Confusion matrix
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            # Multi-class AUC
            metrics['auc_macro'] = roc_auc_score(y_true, y_prob, 
                                                  multi_class='ovr', average='macro')
            metrics['auc_weighted'] = roc_auc_score(y_true, y_prob,
                                                     multi_class='ovr', average='weighted')
        except:
            metrics['auc_macro'] = None
            metrics['auc_weighted'] = None
    
    # Per-class sensitivity and specificity
    cm = np.array(metrics['confusion_matrix'])
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics[f'{class_name}_sensitivity'] = sensitivity
        metrics[f'{class_name}_specificity'] = specificity
        metrics[f'{class_name}_f1'] = metrics['f1_per_class'][i]
    
    return metrics


def print_classification_report(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 class_names: List[str] = None):
    """Print detailed classification report."""
    if class_names is None:
        class_names = ['AD', 'FTD', 'CN']
    
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"{'':>8} {'Pred AD':>8} {'Pred FTD':>8} {'Pred CN':>8}")
    for i, name in enumerate(class_names):
        print(f"{name:>8} {cm[i,0]:>8} {cm[i,1]:>8} {cm[i,2]:>8}")


def format_confusion_matrix(cm: np.ndarray,
                            class_names: List[str] = None) -> str:
    """Format confusion matrix as string."""
    if class_names is None:
        class_names = ['AD', 'FTD', 'CN']
    
    lines = []
    lines.append(f"{'':>10}" + "".join([f"{name:>10}" for name in class_names]))
    
    for i, name in enumerate(class_names):
        row = f"{name:>10}" + "".join([f"{cm[i,j]:>10}" for j in range(len(class_names))])
        lines.append(row)
    
    return "\n".join(lines)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (for loss or accuracy)
            restore_best: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current validation score
            model: Model to potentially save
            
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            improved = self.best_score is None or score < self.best_score - self.min_delta
        else:
            improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore(self, model: torch.nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class MetricLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self):
        self.history = {}
    
    def log(self, metrics: Dict, step: int = None):
        """Log metrics for a step."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get(self, key: str) -> List:
        """Get history for a metric."""
        return self.history.get(key, [])
    
    def get_best(self, key: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best value and its index."""
        values = self.get(key)
        if not values:
            return None, None
        
        if mode == 'max':
            idx = np.argmax(values)
        else:
            idx = np.argmin(values)
        
        return values[idx], idx
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = {
                    'final': values[-1],
                    'best': max(values) if 'loss' not in key.lower() else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        return summary
