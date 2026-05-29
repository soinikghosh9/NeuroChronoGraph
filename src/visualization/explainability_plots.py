"""
Explainability Visualization Module.

This module provides visualization functions for interpreting
model decisions, including feature importance, attention maps,
and brain region contributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import color settings
from .style_config import CLASS_COLORS, CLASS_NAMES, PALETTE_BANDS


def plot_node_importance(importance: Dict[str, float],
                         title: str = 'Brain Region Importance',
                         top_n: int = 19,
                         color: str = '#3498DB',
                         figsize: Tuple[float, float] = (10, 6),
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bar chart of node (brain region) importance.
    
    Args:
        importance: Dictionary mapping region names to importance values
        title: Plot title
        top_n: Number of top regions to show
        color: Bar color
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_importance_by_class(class_importance: Dict[str, Dict[str, float]],
                              top_n: int = 10,
                              title: str = 'Region Importance by Class',
                              figsize: Tuple[float, float] = (14, 5),
                              save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot node importance for each class.
    
    Args:
        class_importance: {class_name: {region: importance}} nested dict
        top_n: Number of top regions per class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    classes = list(class_importance.keys())
    n_classes = len(classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    
    if n_classes == 1:
        axes = [axes]
    
    for ax, class_name, color in zip(axes, classes, CLASS_COLORS):
        importance = class_importance[class_name]
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_items:
            continue
            
        names, values = zip(*sorted_items)
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=color, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(class_name, fontsize=12, fontweight='bold', color=color)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_importance_topography(importance: Dict[str, float],
                                ch_names: List[str],
                                title: str = 'Importance Topography',
                                cmap: str = 'hot',
                                figsize: Tuple[float, float] = (8, 8),
                                save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot importance scores as scalp topography.
    
    Args:
        importance: Dictionary mapping channel names to importance
        ch_names: List of all channel names
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from .eeg_plots import plot_topography
    
    # Convert dict to array
    values = np.array([importance.get(ch, 0) for ch in ch_names])
    
    fig = plot_topography(values, ch_names, title=title, cmap=cmap,
                         vmin=0, vmax=max(values) if max(values) > 0 else 1)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_edge_importance(edge_importance: np.ndarray,
                          labels: List[str],
                          title: str = 'Connection Importance',
                          top_n: int = 20,
                          figsize: Tuple[float, float] = (12, 6),
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot most important connections/edges.
    
    Args:
        edge_importance: Edge importance matrix (n_nodes, n_nodes)
        labels: Node labels
        title: Plot title
        top_n: Number of top edges to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Importance matrix
    ax1 = axes[0]
    im = ax1.imshow(edge_importance, cmap='hot', aspect='equal')
    ax1.set_xticks(range(0, len(labels), 3))
    ax1.set_yticks(range(0, len(labels), 3))
    ax1.set_xticklabels([labels[i] for i in range(0, len(labels), 3)], rotation=45, fontsize=8)
    ax1.set_yticklabels([labels[i] for i in range(0, len(labels), 3)], fontsize=8)
    ax1.set_title('Edge Importance Matrix', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # Right: Top edges bar plot
    ax2 = axes[1]
    
    n_nodes = len(labels)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edges.append((labels[i], labels[j], edge_importance[i, j]))
    
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]
    
    edge_names = [f'{e[0]}-{e[1]}' for e in edges_sorted]
    edge_values = [e[2] for e in edges_sorted]
    
    y_pos = np.arange(len(edge_names))
    ax2.barh(y_pos, edge_values, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(edge_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance')
    ax2.set_title(f'Top {top_n} Connections', fontsize=11, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_weights(attention: np.ndarray,
                           labels: List[str],
                           title: str = 'Attention Weights',
                           figsize: Tuple[float, float] = (10, 8),
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot attention weight heatmap.
    
    Args:
        attention: Attention weight matrix (n_queries, n_keys)
        labels: Labels for axes
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(attention, xticklabels=labels, yticklabels=labels,
               cmap='viridis', annot=False, ax=ax, 
               cbar_kws={'label': 'Attention Weight'})
    
    ax.set_xlabel('Key', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_differential_importance(class_importance: Dict[str, Dict[str, float]],
                                         compare_against: str = 'CN',
                                         title: str = 'Differential Importance vs Controls',
                                         figsize: Tuple[float, float] = (12, 8),
                                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot differential importance between disease groups and controls.
    
    Args:
        class_importance: {class: {region: importance}} nested dict
        compare_against: Reference class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    comparison_classes = ['AD', 'FTD']
    
    for ax, comp_class, color in zip(axes, comparison_classes, CLASS_COLORS[:2]):
        if comp_class not in class_importance or compare_against not in class_importance:
            continue
        
        comp_imp = class_importance[comp_class]
        ref_imp = class_importance[compare_against]
        
        # Get all regions
        all_regions = set(comp_imp.keys()) | set(ref_imp.keys())
        
        # Compute difference
        diff = {}
        for region in all_regions:
            diff[region] = comp_imp.get(region, 0) - ref_imp.get(region, 0)
        
        # Sort by absolute difference
        sorted_items = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        names, values = zip(*sorted_items)
        
        colors = [color if v > 0 else '#95A5A6' for v in values]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
        ax.axvline(0, color='black', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(f'Importance Difference ({comp_class} - {compare_against})')
        ax.set_title(f'{comp_class} vs {compare_against}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_contributions(feature_names: List[str],
                               contributions: np.ndarray,
                               predicted_class: str,
                               title: str = 'Feature Contributions',
                               top_n: int = 15,
                               figsize: Tuple[float, float] = (10, 7),
                               save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot feature contributions (SHAP-style waterfall).
    
    Args:
        feature_names: Feature names
        contributions: Contribution values
        predicted_class: Predicted class name
        title: Plot title
        top_n: Number of features to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Sort by absolute contribution
    sorted_idx = np.argsort(np.abs(contributions))[::-1][:top_n]
    
    names = [feature_names[i] for i in sorted_idx]
    values = contributions[sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in values]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Contribution to Prediction', fontsize=12)
    ax.set_title(f'{title}\nPredicted: {predicted_class}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Increases prediction'),
        Patch(facecolor='#3498DB', label='Decreases prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_brain_schematic(node_importance: Dict[str, float],
                          edge_importance: np.ndarray = None,
                          labels: List[str] = None,
                          title: str = 'Brain Region Importance',
                          figsize: Tuple[float, float] = (12, 10),
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot schematic brain diagram with importance overlaid.
    
    Args:
        node_importance: {region: importance} dictionary
        edge_importance: Optional edge importance matrix
        labels: Node labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Brain outline (simplified schematic)
    # Head circle
    head = Circle((0.5, 0.5), 0.45, fill=False, linewidth=3, color='black')
    ax.add_patch(head)
    
    # Nose indicator
    ax.plot([0.5, 0.5], [0.95, 1.0], 'k-', linewidth=2)
    
    # Lobe positions (approximate)
    lobe_positions = {
        'Frontal': (0.5, 0.75),
        'L_Frontal': (0.3, 0.72),
        'R_Frontal': (0.7, 0.72),
        'Temporal': (0.5, 0.5),
        'L_Temporal': (0.15, 0.5),
        'R_Temporal': (0.85, 0.5),
        'Parietal': (0.5, 0.4),
        'L_Parietal': (0.3, 0.35),
        'R_Parietal': (0.7, 0.35),
        'Occipital': (0.5, 0.15),
        'L_Occipital': (0.35, 0.18),
        'R_Occipital': (0.65, 0.18),
    }
    
    # Normalize importance for sizing/coloring
    if node_importance:
        max_imp = max(node_importance.values())
        min_imp = min(node_importance.values())
        imp_range = max_imp - min_imp if max_imp != min_imp else 1
    
    # Plot regions
    cmap = plt.cm.hot
    for region, (x, y) in lobe_positions.items():
        imp = node_importance.get(region, 0)
        norm_imp = (imp - min_imp) / imp_range if imp_range > 0 else 0.5
        
        size = 800 + 1200 * norm_imp
        color = cmap(norm_imp)
        
        ax.scatter(x, y, s=size, c=[color], edgecolors='black', 
                  linewidths=2, zorder=5)
        ax.text(x, y - 0.08, region.replace('_', ' '), ha='center', 
               fontsize=9, fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=min_imp, vmax=max_imp))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Importance', fontsize=10)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
