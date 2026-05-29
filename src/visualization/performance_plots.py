"""
Model Performance Visualization Module - Publication Ready.

This module provides comprehensive visualization functions for
model evaluation metrics, confusion matrices, ROC curves, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

# Import publication style
from .style_config import (
    PALETTE_PRIMARY, PALETTE_NEUTRAL, CLASS_COLORS, CLASS_NAMES,
    set_publication_style, add_panel_label, format_axis, despine,
    create_colorbar, FIGURE_SIZES, add_significance_annotation,
    safe_tight_layout
)

set_publication_style()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str] = None,
                          normalize: bool = True,
                          title: str = '',
                          cmap: str = 'Blues',
                          figsize: Tuple[float, float] = None,
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot detailed confusion matrix - Publication Ready.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        normalize: Whether to show percentages
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if figsize is None:
        figsize = FIGURE_SIZES['single_col_square']
    
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        vmax = 100
    else:
        cm_display = cm.astype('float')
        vmax = cm.max()
    
    im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label('Percentage (%)' if normalize else 'Count', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)
    
    # Add cell annotations
    thresh = vmax * 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_display[i, j]
            count = cm[i, j]
            color = 'white' if val > thresh else PALETTE_NEUTRAL['dark_gray']
            
            if normalize:
                text = f'{val:.1f}%\n({count})'
            else:
                text = f'{count}'
            
            ax.text(j, i, text, ha='center', va='center', 
                   color=color, fontsize=10, fontweight='bold')
    
    # Axis formatting
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=11, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    # Add grid lines between cells
    for i in range(n_classes + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
        ax.axvline(i - 0.5, color='white', linewidth=2)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    
    # Add accuracy annotation
    accuracy = np.trace(cm) / cm.sum() * 100
    ax.text(0.5, -0.12, f'Overall Accuracy: {accuracy:.1f}%',
           transform=ax.transAxes, ha='center', fontsize=10,
           fontweight='bold', color=PALETTE_PRIMARY['CN'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_multiclass_roc(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        class_names: List[str] = None,
                        title: str = '',
                        figsize: Tuple[float, float] = None,
                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification - Publication Ready.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if figsize is None:
        figsize = FIGURE_SIZES['single_col_square']
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Compute ROC for each class
    for i, (class_name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        ax.plot(fpr[i], tpr[i], color=color, linewidth=2,
               label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Compute micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    ax.plot(fpr_micro, tpr_micro, color=PALETTE_NEUTRAL['dark_gray'], 
           linewidth=2.5, linestyle='--',
           label=f'Micro-avg (AUC = {roc_auc_micro:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color=PALETTE_NEUTRAL['light_gray'], 
           linewidth=1.5, linestyle=':', label='Chance', alpha=0.8)
    
    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.95)
    ax.set_aspect('equal')
    
    # Light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    despine(ax)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_precision_recall_curves(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  class_names: List[str] = None,
                                  title: str = '',
                                  figsize: Tuple[float, float] = None,
                                  save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot Precision-Recall curves - Publication Ready.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if figsize is None:
        figsize = FIGURE_SIZES['single_col_square']
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (class_name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        
        ax.plot(recall, precision, color=color, linewidth=2,
               label=f'{class_name} (AP = {ap:.3f})')
    
    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.legend(loc='lower left', fontsize=9, frameon=True, framealpha=0.95)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    despine(ax)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_metrics_summary(metrics: Dict[str, float],
                         title: str = '',
                         figsize: Tuple[float, float] = None,
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot summary of classification metrics - Publication Ready.
    
    Args:
        metrics: Dictionary of metric name to value
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['double_col']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Overall metrics
    ax1 = axes[0]
    main_metrics = ['Accuracy', 'Macro F1', 'Weighted F1', 'MCC']
    main_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'mcc']
    values = [metrics.get(k, 0) for k in main_keys]
    
    colors_bar = [PALETTE_PRIMARY['FTD'], PALETTE_PRIMARY['CN'], 
                  '#9B59B6', '#F39C12']
    
    y_pos = np.arange(len(main_metrics))
    bars = ax1.barh(y_pos, values, color=colors_bar, alpha=0.85,
                   edgecolor='white', linewidth=0.5, height=0.7)
    
    ax1.set_xlim(0, 1.05)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(main_metrics, fontsize=10)
    ax1.set_xlabel('Score', fontsize=10)
    ax1.set_title('A  Overall Metrics', fontsize=11, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    despine(ax1)
    
    # Right panel: Per-class F1
    ax2 = axes[1]
    f1_values = metrics.get('f1_per_class', [0, 0, 0])
    if not isinstance(f1_values, list):
        f1_values = list(f1_values)
    
    x_pos = np.arange(len(CLASS_NAMES))
    bars2 = ax2.bar(x_pos, f1_values, color=CLASS_COLORS, alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.7)
    
    for bar, val in zip(bars2, f1_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax2.set_ylabel('F1 Score', fontsize=10)
    ax2.set_title('B  Per-Class F1 Scores', fontsize=11, fontweight='bold')
    
    despine(ax2)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_cross_validation_results(fold_results: List[Dict],
                                   title: str = '',
                                   figsize: Tuple[float, float] = None,
                                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot cross-validation results across folds - Publication Ready.
    
    Args:
        fold_results: List of result dictionaries per fold
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['double_col']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Subject-level results
    ax1 = axes[0]
    n_subjects = len(fold_results)
    correct = [1 if r.get('correct', r.get('pred') == r.get('true')) else 0 
               for r in fold_results]
    groups = [r.get('true', 0) for r in fold_results]
    colors = [CLASS_COLORS[g] for g in groups]
    
    bars = ax1.bar(range(n_subjects), correct, color=colors, alpha=0.8,
                  width=1.0, edgecolor='white', linewidth=0.3)
    
    accuracy = np.mean(correct)
    ax1.axhline(accuracy, color=PALETTE_NEUTRAL['dark_gray'], linestyle='--', 
               linewidth=2, zorder=10)
    ax1.text(n_subjects * 0.98, accuracy + 0.05, f'Accuracy: {accuracy:.1%}',
            ha='right', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Subject Index', fontsize=10)
    ax1.set_ylabel('Correct', fontsize=10)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlim(-0.5, n_subjects - 0.5)
    ax1.set_title('A  Per-Subject Predictions', fontsize=11, fontweight='bold')
    
    # Legend
    legend_elements = [Patch(facecolor=c, edgecolor='white', label=n, alpha=0.8) 
                      for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9,
              frameon=True, framealpha=0.95)
    
    despine(ax1)
    
    # Right panel: Accuracy by group
    ax2 = axes[1]
    group_accuracy = {}
    for group_idx, group_name in enumerate(CLASS_NAMES):
        group_results = [r for r in fold_results if r.get('true', 0) == group_idx]
        if group_results:
            acc = np.mean([1 if r.get('correct', r.get('pred') == r.get('true')) else 0 
                          for r in group_results])
            group_accuracy[group_name] = acc
    
    x_pos = np.arange(len(group_accuracy))
    bars2 = ax2.bar(x_pos, list(group_accuracy.values()), 
                   color=CLASS_COLORS[:len(group_accuracy)], alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.7)
    
    for bar, (name, acc) in zip(bars2, group_accuracy.items()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(group_accuracy.keys()), fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_title('B  Accuracy per Class', fontsize=11, fontweight='bold')
    
    despine(ax2)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_probability_calibration(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  class_names: List[str] = None,
                                  n_bins: int = 10,
                                  title: str = '',
                                  figsize: Tuple[float, float] = None,
                                  save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot probability calibration curves - Publication Ready.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: Class names
        n_bins: Number of calibration bins
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if figsize is None:
        figsize = (10, 3.5)
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    
    for i, (ax, class_name, color) in enumerate(zip(axes, class_names, CLASS_COLORS)):
        # Compute calibration
        prob_true = []
        prob_pred = []
        
        bins = np.linspace(0, 1, n_bins + 1)
        for j in range(n_bins):
            mask = (y_prob[:, i] >= bins[j]) & (y_prob[:, i] < bins[j + 1])
            if mask.sum() > 0:
                prob_true.append(y_true_bin[mask, i].mean())
                prob_pred.append(y_prob[mask, i].mean())
        
        ax.plot([0, 1], [0, 1], linestyle='--', color=PALETTE_NEUTRAL['light_gray'],
               label='Perfect', linewidth=1.5)
        ax.plot(prob_pred, prob_true, 's-', color=color, 
               label=class_name, markersize=8, linewidth=2,
               markeredgecolor='white', markeredgewidth=1)
        
        ax.set_xlabel('Mean Predicted Prob.', fontsize=9)
        ax.set_ylabel('Fraction of Positives', fontsize=9)
        ax.set_title(class_name, fontsize=11, fontweight='bold', color=color)
        ax.legend(loc='lower right', fontsize=8, frameon=True)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        despine(ax)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def create_metrics_table(metrics: Dict[str, float],
                          class_names: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create formatted metrics tables.
    
    Args:
        metrics: Dictionary of metrics
        class_names: Class names
        
    Returns:
        Tuple of (overall_df, per_class_df)
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Overall metrics
    overall = {
        'Metric': ['Accuracy', 'Macro F1', 'Weighted F1', 'MCC'],
        'Value': [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('f1_macro', 0):.4f}",
            f"{metrics.get('f1_weighted', 0):.4f}",
            f"{metrics.get('mcc', 0):.4f}"
        ]
    }
    
    # Per-class metrics
    per_class = {'Class': class_names}
    f1 = metrics.get('f1_per_class', [0, 0, 0])
    per_class['F1'] = [f'{f:.4f}' for f in f1]
    
    for i, name in enumerate(class_names):
        per_class.setdefault('Sensitivity', []).append(
            f"{metrics.get(f'{name}_sensitivity', 0):.4f}")
        per_class.setdefault('Specificity', []).append(
            f"{metrics.get(f'{name}_specificity', 0):.4f}")
    
    return pd.DataFrame(overall), pd.DataFrame(per_class)
