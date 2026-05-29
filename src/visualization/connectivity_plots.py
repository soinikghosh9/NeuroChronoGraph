"""
Connectivity Visualization Module - Publication Ready.

This module provides functions for visualizing brain connectivity
matrices, networks, and graph structures with publication-quality styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import publication style
from .style_config import (
    PALETTE_PRIMARY, PALETTE_NEUTRAL, CLASS_COLORS, CLASS_NAMES,
    set_publication_style, add_panel_label, format_axis, despine,
    create_colorbar, FIGURE_SIZES
)

set_publication_style()


def plot_connectivity_matrix(matrix: np.ndarray,
                             labels: List[str],
                             title: str = '',
                             cmap: str = 'viridis',
                             vmin: float = 0,
                             vmax: float = None,
                             figsize: Tuple[float, float] = None,
                             annotate: bool = False,
                             save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot connectivity matrix as heatmap - Publication Ready.
    
    Args:
        matrix: Connectivity matrix (n_nodes, n_nodes)
        labels: Node labels
        title: Plot title
        cmap: Colormap
        vmin, vmax: Color limits
        figsize: Figure size
        annotate: Whether to show values in cells
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['single_col_square']
    
    if vmax is None:
        vmax = np.max(matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label('Connectivity (wPLI)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)
    
    # Set ticks and labels
    n = len(labels)
    
    # Reduce labels for large matrices
    if n > 12:
        tick_indices = list(range(0, n, max(1, n // 8)))
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels([labels[i] for i in tick_indices], rotation=45, 
                          ha='right', fontsize=8)
        ax.set_yticklabels([labels[i] for i in tick_indices], fontsize=8)
    else:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    
    # Annotate values for small matrices
    if annotate and n <= 10:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = 'white' if val > (vmax - vmin) / 2 + vmin else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=7)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_multiband_connectivity(matrices: Dict[str, np.ndarray],
                                 labels: List[str],
                                 title: str = None,
                                 figsize: Tuple[float, float] = None,
                                 save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot connectivity matrices for multiple frequency bands - Publication Ready.
    
    Args:
        matrices: Dictionary {band_name: matrix}
        labels: Node labels
        title: Overall title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    bands = list(matrices.keys())
    n_bands = len(bands)
    
    if figsize is None:
        figsize = (3.2 * n_bands + 0.8, 3.2)
    
    fig, axes = plt.subplots(1, n_bands, figsize=figsize)
    
    if n_bands == 1:
        axes = [axes]
    
    # Get global vmin/vmax
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    vmin, vmax = 0, np.percentile(all_vals, 98)
    
    for ax, band in zip(axes, bands):
        im = ax.imshow(matrices[band], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(band.capitalize(), fontsize=10, fontweight='bold')
        
        # Minimal tick labels
        n = len(labels)
        ax.set_xticks([0, n//2, n-1])
        ax.set_yticks([0, n//2, n-1])
        ax.set_xticklabels([labels[0], labels[n//2], labels[n-1]], 
                          rotation=45, fontsize=7)
        ax.set_yticklabels([labels[0], labels[n//2], labels[n-1]], fontsize=7)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('wPLI', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_group_connectivity_comparison(matrices_dict: Dict[str, Dict[str, np.ndarray]],
                                         labels: List[str],
                                         band: str = 'alpha',
                                         title: str = None,
                                         figsize: Tuple[float, float] = None,
                                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Compare connectivity matrices across groups - Publication Ready.
    
    Args:
        matrices_dict: {group: {band: matrix}} nested dictionary
        labels: Node labels
        band: Frequency band to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['double_col']
    
    groups = CLASS_NAMES
    n_groups = len(groups)
    # n_groups columns + 1 for difference plot
    fig, axes = plt.subplots(1, n_groups + 1, figsize=figsize)
    
    # Get global vmin/vmax
    all_vals = []
    for group in groups:
        if group in matrices_dict and band in matrices_dict[group]:
            all_vals.extend(matrices_dict[group][band].flatten())
    vmax = np.percentile(all_vals, 98) if all_vals else 1
    
    # Plot each group
    for i, (group, color) in enumerate(zip(groups, CLASS_COLORS)):
        ax = axes[i]
        if group in matrices_dict and band in matrices_dict[group]:
            im = ax.imshow(matrices_dict[group][band], cmap='viridis', 
                          vmin=0, vmax=vmax)
        
        ax.set_title(group, fontsize=11, fontweight='bold', color=color)
        n = len(labels)
        ax.set_xticks([0, n-1])
        ax.set_yticks([0, n-1])
        ax.set_xticklabels([labels[0], labels[-1]], rotation=45, fontsize=7)
        ax.set_yticklabels([labels[0], labels[-1]], fontsize=7)
    
    # Difference plot (AD - CN)
    diff_ax = axes[-1]
    if 'AD' in matrices_dict and 'CN' in matrices_dict:
        diff = matrices_dict['AD'][band] - matrices_dict['CN'][band]
        diff_max = np.max(np.abs(diff))
        im_diff = diff_ax.imshow(diff, cmap='RdBu_r', 
                                  vmin=-diff_max, vmax=diff_max)
        diff_ax.set_title('AD − CN', fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(im_diff, ax=diff_ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        n = len(labels)
        diff_ax.set_xticks([0, n-1])
        diff_ax.set_yticks([0, n-1])
        diff_ax.set_xticklabels([labels[0], labels[-1]], rotation=45, fontsize=7)
        diff_ax.set_yticklabels([labels[0], labels[-1]], fontsize=7)
    
    if title is None:
        title = f'{band.capitalize()} Band Connectivity'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_brain_network(adjacency: np.ndarray,
                       labels: List[str],
                       threshold: float = 0.3,
                       node_values: np.ndarray = None,
                       title: str = '',
                       layout: str = 'circular',
                       figsize: Tuple[float, float] = None,
                       save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot brain network as graph - Publication Ready.
    
    Args:
        adjacency: Weighted adjacency matrix
        labels: Node labels
        threshold: Threshold for edge visibility
        node_values: Values for node coloring
        title: Plot title
        layout: Graph layout
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['single_col_square']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create graph
    G = nx.Graph()
    n_nodes = len(labels)
    
    for i, label in enumerate(labels):
        G.add_node(i, label=label)
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = adjacency[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
    
    # Get layout
    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(n_nodes))
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)
    
    # Draw edges with varying widths
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if weights:
        edge_norm = Normalize(vmin=threshold, vmax=max(weights))
        edge_colors = plt.cm.Greys(edge_norm(weights))
        edge_widths = [w * 4 for w in weights]
        
        for (u, v), color, width in zip(edges, edge_colors, edge_widths):
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color=color, linewidth=width, alpha=0.6, zorder=1)
    
    # Draw nodes
    if node_values is not None:
        node_colors = node_values
        cmap = 'YlOrRd'
    else:
        node_colors = [G.degree(n) for n in G.nodes()]
        cmap = 'Blues'
    
    node_positions = np.array([pos[i] for i in range(n_nodes)])
    scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1],
                         c=node_colors, cmap=cmap, s=300, zorder=5,
                         edgecolors=PALETTE_NEUTRAL['dark_gray'], linewidths=1.5)
    
    # Draw labels
    for i, label in enumerate(labels):
        ax.annotate(label, pos[i], fontsize=8, ha='center', va='center',
                   fontweight='bold', color='white' if node_colors[i] > np.median(node_colors) else 'black')
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_graph_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                                   metric_names: List[str] = None,
                                   title: str = '',
                                   figsize: Tuple[float, float] = None,
                                   save_path: Optional[Path] = None) -> plt.Figure:
    """
    Compare graph metrics across groups - Publication Ready.
    
    Args:
        metrics_dict: {group: {metric: value}} nested dictionary
        metric_names: Metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['single_col']
    
    groups = ['AD', 'FTD', 'CN']
    
    if metric_names is None:
        first_group = list(metrics_dict.values())[0]
        metric_names = list(first_group.keys())[:6]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    for i, (group, color) in enumerate(zip(groups, CLASS_COLORS)):
        if group in metrics_dict:
            values = [metrics_dict[group].get(m, 0) for m in metric_names]
            bars = ax.bar(x + i * width - width, values, width, 
                         label=group, color=color, alpha=0.85,
                         edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], 
                       rotation=30, ha='right', fontsize=9)
    ax.legend(frameon=True, framealpha=0.95)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    
    despine(ax)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_circular_connectome(adjacency: np.ndarray,
                             labels: List[str],
                             threshold: float = 0.3,
                             title: str = '',
                             cmap: str = 'viridis',
                             figsize: Tuple[float, float] = None,
                             save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot brain connectivity as circular connectome - Publication Ready.
    
    Args:
        adjacency: Weighted adjacency matrix
        labels: Node labels
        threshold: Threshold for edge visibility
        title: Plot title
        cmap: Colormap for edges
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['single_col_square']
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    n_nodes = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    
    # Draw node markers
    ax.scatter(angles, np.ones(n_nodes), s=100, c=PALETTE_NEUTRAL['dark_gray'], 
              zorder=5, edgecolors='white', linewidths=1)
    
    # Add labels
    for angle, label in zip(angles, labels):
        rotation = np.degrees(angle)
        if angle > np.pi/2 and angle < 3*np.pi/2:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        
        ax.text(angle, 1.15, label, ha=ha, va='center', 
               rotation=rotation, fontsize=8, fontweight='medium')
    
    # Draw edges
    edges = []
    weights = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j] > threshold:
                edges.append((i, j))
                weights.append(adjacency[i, j])
    
    if weights:
        norm = Normalize(vmin=threshold, vmax=max(weights))
        cmap_obj = plt.cm.get_cmap(cmap)
        
        for (i, j), w in zip(edges, weights):
            angle_i, angle_j = angles[i], angles[j]
            
            # Create curved edge using Bezier-like curve
            n_points = 50
            t = np.linspace(0, 1, n_points)
            
            # Control point at center
            r_control = 0.3 + 0.2 * (1 - w/max(weights))
            
            # Interpolate angles
            if abs(angle_j - angle_i) > np.pi:
                if angle_j > angle_i:
                    angle_j -= 2 * np.pi
                else:
                    angle_i -= 2 * np.pi
            
            theta = angle_i * (1 - t) + angle_j * t
            r = 1 - 4 * (1 - r_control) * t * (1 - t)
            
            color = cmap_obj(norm(w))
            ax.plot(theta, r, color=color, alpha=0.7, 
                   linewidth=norm(w) * 2.5 + 0.5, zorder=1)
    
    ax.set_ylim(0, 1.3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig
