import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from src.config.config import VISUALIZATION_CONFIG

def setup_plotting_style():
    """Apply publication-ready style settings globally."""
    plt.style.use(VISUALIZATION_CONFIG['style'])
    sns.set_context(VISUALIZATION_CONFIG['context'], font_scale=VISUALIZATION_CONFIG['font_scale'])
    
    # Custom tweaks for readability
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2
    })

def plot_connectivity_circle(adjacency_matrix, node_names, title="Brain Connectivity", output_path=None):
    """
    Plot a circular connectivity graph.
    
    Args:
        adjacency_matrix: [N, N] symmetric matrix of connection strengths (0-1).
        node_names: List of channel names.
        title: Plot title.
        output_path: Path to save the figure (optional).
    """
    try:
        from mne_connectivity.viz import plot_connectivity_circle
    except ImportError:
        print("MNE Connectivity not installed, using simplified heatmap instead.")
        return plot_connectivity_heatmap(adjacency_matrix, node_names, title, output_path)

    # Robust Thresholding for Contrast
    # If variance is low (uniform data), percentile might pick weak connections.
    # Use Mean + 1.5 STD as a safer robust threshold.
    mean, std = np.mean(adjacency_matrix), np.std(adjacency_matrix)
    threshold = max(np.percentile(adjacency_matrix, 90), mean + 1.5 * std)

    # Ensure threshold isn't higher than max (empty plot)
    if threshold >= np.max(adjacency_matrix):
        threshold = np.percentile(adjacency_matrix, 95)

    vmax = np.max(adjacency_matrix)
    # Widen the color scale so edges span the full perceptual range of the
    # colormap instead of collapsing into a tiny high-end slice (previously
    # vmin=threshold, vmax=max made the range ~0.005 wide -> everything looked
    # washed out on white).
    vmin = max(0.0, threshold - 0.5 * (vmax - threshold))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Nature-style muted palette: deep navy -> slate teal -> muted coral.
    # Avoids the carnival-vibrancy of plasma while keeping the low end dark
    # enough to stay visible on white paper.
    from matplotlib.colors import LinearSegmentedColormap
    nature_cmap = LinearSegmentedColormap.from_list(
        'nature_conn',
        [
            (0.00, '#1F2E4A'),  # deep navy (weak edges — still clearly visible)
            (0.35, '#355775'),  # steel blue
            (0.65, '#5E8A8A'),  # slate teal
            (0.85, '#C28A6E'),  # muted terracotta
            (1.00, '#D9A48F'),  # soft coral (strongest)
        ],
        N=256,
    )

    plot_connectivity_circle(
        adjacency_matrix,
        node_names,
        n_lines=50,
        node_angles=None,
        node_colors=None,
        title=f"{title}\n(Thresh={threshold:.2f}, Max={vmax:.2f})",
        ax=ax,
        show=False,
        facecolor='white',
        textcolor='black',
        colormap=nature_cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=2.2,
    )
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
        plt.close()
    else:
        plt.show()

def plot_connectivity_heatmap(adjacency_matrix, node_names, title="Connectivity", output_path=None):
    """
    Fallback: Plot connectivity as a heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(adjacency_matrix, xticklabels=node_names, yticklabels=node_names, 
                cmap=VISUALIZATION_CONFIG['cmap_sequential'], square=True, linewidths=0.5)
    plt.title(title, weight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
        plt.close()
    else:
        plt.show()

def plot_temporal_attention(temporal_weights, time_axis, title="Temporal Attention", output_path=None, color=None):
    """
    Plot temporal attention weights over the trial duration.
    """
    if color is None:
        color = VISUALIZATION_CONFIG['palette']['General']
        
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, temporal_weights, label='Attention Weight', color=color, linewidth=2)
    plt.fill_between(time_axis, temporal_weights, alpha=0.3, color=color)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(title, weight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Attention Weight")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
        plt.close()
    else:
        plt.show()

def plot_clinical_robustness(cv_results_df, output_path=None):
    """
    Plot Sensitivity vs Specificity across folds for AD and FTD.
    """
    plt.figure(figsize=(8, 6))
    
    # AD Robustness
    if 'Sens_AD' in cv_results_df.columns:
        plt.scatter(cv_results_df['fold'], cv_results_df['Sens_AD'], 
                   label='AD Sensitivity', marker='o', s=100, color=VISUALIZATION_CONFIG['palette']['AD'])
    
    # FTD Robustness
    if 'Sens_FTD' in cv_results_df.columns:
        plt.scatter(cv_results_df['fold'], cv_results_df['Sens_FTD'], 
                   label='FTD Sensitivity', marker='^', s=100, color=VISUALIZATION_CONFIG['palette']['FTD'])
        
    plt.title("Clinical Robustness across Cross-Validation Folds", weight='bold')
    plt.xlabel("Fold")
    plt.ylabel("Sensitivity (Recall)")
    plt.ylim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
        plt.close()

def plot_confusion_matrix_enhanced(cm, classes, title="Confusion Matrix", output_path=None, cmap=None):
    """
    Plot an enhanced confusion matrix with percentages.
    """
    if cmap is None:
        cmap = VISUALIZATION_CONFIG['cmap_sequential']

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(8, 7))
    
    # Annotations with counts and percentages
    annot_labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_norm.flatten())]
    annot_labels = np.asarray(annot_labels).reshape(cm.shape)
    
    sns.heatmap(cm_norm, annot=annot_labels, fmt='', cmap=cmap, 
                xticklabels=classes, yticklabels=classes,
                square=True, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Normalized Accuracy'})
                
    plt.title(title, weight='bold', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
        plt.close()

def safe_tight_layout():
    """Execute tight_layout with error handling for complex grids."""
    import warnings as _w
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            plt.tight_layout()
    except ValueError:
        pass
    except Exception as e:
        print(f"Warning: tight_layout failed: {e}")

