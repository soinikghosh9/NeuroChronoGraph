"""
Statistical Visualization Module.

This module provides visualization functions for statistical analyses,
including significance testing, effect sizes, and group comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import color settings
from .style_config import CLASS_COLORS, CLASS_NAMES, PALETTE_BANDS


def plot_permutation_test(observed: float,
                          null_distribution: np.ndarray,
                          metric_name: str = 'Score',
                          p_value: float = None,
                          title: str = 'Permutation Test',
                          figsize: Tuple[float, float] = (10, 5),
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot permutation test results.
    
    Args:
        observed: Observed metric value
        null_distribution: Null distribution from permutations
        metric_name: Name of the metric
        p_value: P-value (computed if not provided)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot null distribution
    ax.hist(null_distribution, bins=50, color='#3498DB', alpha=0.7, 
           edgecolor='white', label='Null Distribution')
    
    # Plot observed value
    ax.axvline(observed, color='#E74C3C', linewidth=3, linestyle='--',
              label=f'Observed = {observed:.4f}')
    
    # Compute p-value if not provided
    if p_value is None:
        p_value = (np.sum(null_distribution >= observed) + 1) / (len(null_distribution) + 1)
    
    # Add significance annotation
    significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
    
    ax.text(0.95, 0.95, f'p = {p_value:.4f} {significance}',
           transform=ax.transAxes, ha='right', va='top',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bootstrap_ci(point_estimate: float,
                      ci_lower: float,
                      ci_upper: float,
                      bootstrap_distribution: np.ndarray = None,
                      metric_name: str = 'Score',
                      confidence: float = 0.95,
                      title: str = 'Bootstrap Confidence Interval',
                      figsize: Tuple[float, float] = (8, 5),
                      save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bootstrap confidence interval.
    
    Args:
        point_estimate: Point estimate value
        ci_lower: Lower bound of CI
        ci_upper: Upper bound of CI
        bootstrap_distribution: Bootstrap distribution
        metric_name: Name of the metric
        confidence: Confidence level
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if bootstrap_distribution is not None:
        ax.hist(bootstrap_distribution, bins=50, color='#2ECC71', alpha=0.7,
               edgecolor='white', label='Bootstrap Distribution')
    
    # Plot CI bounds
    ax.axvline(ci_lower, color='#9B59B6', linewidth=2, linestyle='--',
              label=f'CI Lower = {ci_lower:.4f}')
    ax.axvline(ci_upper, color='#9B59B6', linewidth=2, linestyle='--',
              label=f'CI Upper = {ci_upper:.4f}')
    ax.axvline(point_estimate, color='#E74C3C', linewidth=3,
              label=f'Point Estimate = {point_estimate:.4f}')
    
    # Shade CI region
    if bootstrap_distribution is not None:
        ymax = ax.get_ylim()[1]
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='#9B59B6')
    
    # Add annotation
    ci_width = ci_upper - ci_lower
    ax.text(0.95, 0.95, f'{confidence*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\nWidth: {ci_width:.4f}',
           transform=ax.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_effect_size_bar(effect_sizes: Dict[str, float],
                          interpretations: Dict[str, str] = None,
                          title: str = 'Effect Sizes',
                          figsize: Tuple[float, float] = (10, 5),
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bar chart of effect sizes.
    
    Args:
        effect_sizes: Dictionary of {metric: effect_size}
        interpretations: Dictionary of {metric: interpretation}
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = list(effect_sizes.keys())
    values = list(effect_sizes.values())
    
    # Color by interpretation
    color_map = {
        'Negligible': '#95A5A6',
        'Small': '#3498DB',
        'Medium': '#F39C12',
        'Large': '#E74C3C',
        'Very Large': '#8E44AD'
    }
    
    if interpretations:
        colors = [color_map.get(interpretations.get(m, 'Medium'), '#3498DB') for m in metrics]
    else:
        colors = ['#3498DB'] * len(metrics)
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add interpretation lines
    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Small (0.1)')
    ax.axhline(0.3, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.3)')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Large (0.5)')
    
    ax.set_ylabel('Effect Size (Cramér\'s V)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_group_comparison_boxplot(data_dict: Dict[str, np.ndarray],
                                   feature_name: str,
                                   title: str = None,
                                   show_significance: bool = True,
                                   figsize: Tuple[float, float] = (8, 6),
                                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot boxplot comparing groups with significance annotations.
    
    Args:
        data_dict: {group_name: values_array}
        feature_name: Name of the feature
        title: Plot title
        show_significance: Whether to show significance stars
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = list(data_dict.keys())
    data = [data_dict[g] for g in groups]
    
    bp = ax.boxplot(data, labels=groups, patch_artist=True)
    
    # Color boxes
    colors = CLASS_COLORS[:len(groups)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add jittered points
    for i, (d, color) in enumerate(zip(data, colors)):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, color=color, s=30, edgecolors='black', linewidths=0.5)
    
    # Add significance annotations
    if show_significance and len(groups) >= 2:
        # Perform pairwise t-tests
        y_max = max([max(d) for d in data])
        y_offset = y_max * 0.05
        
        pairs = [(0, 1), (1, 2), (0, 2)] if len(groups) == 3 else [(0, 1)]
        
        for idx, (i, j) in enumerate(pairs):
            if i < len(data) and j < len(data):
                t_stat, p_val = stats.ttest_ind(data[i], data[j])
                
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'n.s.'
                
                y = y_max + y_offset * (idx + 1)
                ax.plot([i + 1, j + 1], [y, y], 'k-', linewidth=1)
                ax.text((i + j + 2) / 2, y, sig, ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel(feature_name, fontsize=12)
    
    if title is None:
        title = f'{feature_name} by Group'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_feature_comparison(data: Dict[str, Dict[str, np.ndarray]],
                                   feature_names: List[str],
                                   title: str = 'Feature Comparison Across Groups',
                                   figsize: Tuple[float, float] = (14, 6),
                                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot comparison of multiple features across groups.
    
    Args:
        data: {group: {feature: values}} nested dictionary
        feature_names: Features to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, feature_names):
        feature_data = {}
        for group in data.keys():
            if feature in data[group]:
                feature_data[group] = data[group][feature]
        
        if feature_data:
            plot_group_comparison_boxplot(feature_data, feature, 
                                          show_significance=True)
            plt.close()  # Close individual figure
            
            # Recreate on this axis
            groups = list(feature_data.keys())
            values = [feature_data[g] for g in groups]
            
            bp = ax.boxplot(values, labels=groups, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], CLASS_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_significance_summary_table(results: Dict[str, Dict],
                                     title: str = 'Statistical Significance Summary',
                                     figsize: Tuple[float, float] = (10, 4),
                                     save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a formatted table of significance results.
    
    Args:
        results: {metric: {value, p_value, significant, ci_lower, ci_upper}}
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Prepare table data
    columns = ['Metric', 'Value', 'p-value', 'Significance', '95% CI']
    rows = []
    
    for metric, data in results.items():
        sig = '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else ''
        
        ci_str = f"[{data.get('ci_lower', 0):.3f}, {data.get('ci_upper', 0):.3f}]"
        
        rows.append([
            metric.replace('_', ' ').title(),
            f"{data['value']:.4f}",
            f"{data['p_value']:.4f}",
            sig if sig else 'n.s.',
            ci_str
        ])
    
    table = ax.table(cellText=rows, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#3498DB')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style significant rows
    for i, row_data in enumerate(rows):
        if row_data[3] != 'n.s.':
            for j in range(len(columns)):
                table[(i + 1, j)].set_facecolor('#E8F8F5')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_clinical_correlation(clinical_values: np.ndarray,
                               prediction_probs: np.ndarray,
                               clinical_name: str = 'MMSE',
                               class_labels: np.ndarray = None,
                               title: str = None,
                               figsize: Tuple[float, float] = (10, 5),
                               save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot correlation between clinical scores and model predictions.
    
    Args:
        clinical_values: Clinical measure values
        prediction_probs: Model prediction probabilities
        clinical_name: Name of clinical measure
        class_labels: True class labels for coloring
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for i, (ax, class_name, color) in enumerate(zip(axes, CLASS_NAMES, CLASS_COLORS)):
        ax.scatter(clinical_values, prediction_probs[:, i], 
                  c=color, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        
        # Fit regression line
        z = np.polyfit(clinical_values, prediction_probs[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(clinical_values), max(clinical_values), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--')
        
        # Compute correlation
        r, p_val = stats.pearsonr(clinical_values, prediction_probs[:, i])
        
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3f}',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(clinical_name, fontsize=10)
        ax.set_ylabel(f'P({class_name})', fontsize=10)
        ax.set_title(class_name, fontsize=11, fontweight='bold', color=color)
    
    if title is None:
        title = f'Prediction Probability vs {clinical_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
