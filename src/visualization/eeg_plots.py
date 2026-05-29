"""
EEG Data Visualization Module - Publication Ready.

This module provides functions for visualizing EEG signals,
topographies, and spectral analyses with publication-quality styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
import mne
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import publication style
from .style_config import (
    PALETTE_PRIMARY, PALETTE_BANDS, PALETTE_NEUTRAL,
    CLASS_COLORS, CLASS_NAMES, BAND_COLORS, BAND_NAMES,
    set_publication_style, add_panel_label, format_axis, despine,
    create_colorbar, FIGURE_SIZES, get_figure_size
)

# Ensure style is applied
set_publication_style()


def plot_eeg_segment(data: np.ndarray,
                     ch_names: List[str],
                     sfreq: float = 500,
                     duration: float = 5.0,
                     start_time: float = 0,
                     title: str = None,
                     figsize: Tuple[float, float] = None,
                     save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot EEG segment with all channels - Publication Ready.
    
    Args:
        data: EEG data (n_channels, n_times)
        ch_names: Channel names
        sfreq: Sampling frequency
        duration: Duration to plot in seconds
        start_time: Start time in seconds
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['double_col_tall']
    
    n_channels = data.shape[0]
    start_sample = int(start_time * sfreq)
    end_sample = min(start_sample + int(duration * sfreq), data.shape[1])
    
    time = np.arange(start_sample, end_sample) / sfreq
    segment = data[:, start_sample:end_sample]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize and offset channels for display
    spacing = np.percentile(np.abs(segment), 95) * 2.5
    
    for i, (ch_data, ch_name) in enumerate(zip(segment, ch_names)):
        offset = (n_channels - 1 - i) * spacing
        ax.plot(time, ch_data + offset, color=PALETTE_NEUTRAL['dark_gray'], 
                linewidth=0.6, rasterized=True)
        ax.text(time[0] - 0.15, offset, ch_name, fontsize=8, ha='right', 
                va='center', fontweight='medium')
    
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    
    # Add scale bar
    scale_y = spacing * 0.8
    scale_x_start = time[-1] - 0.5
    ax.plot([scale_x_start, scale_x_start + 0.5], [0, 0], 
            color=PALETTE_NEUTRAL['dark_gray'], linewidth=2)
    ax.text(scale_x_start + 0.25, -spacing * 0.3, '500 ms', ha='center', fontsize=8)
    
    if title:
        ax.set_title(title)
    
    despine(ax, left=True)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_psd_comparison(psd_dict: Dict[str, np.ndarray],
                        freqs: np.ndarray,
                        title: str = None,
                        freq_range: Tuple[float, float] = (0.5, 45),
                        show_bands: bool = True,
                        figsize: Tuple[float, float] = None,
                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot PSD comparison across groups - Publication Ready.
    
    Args:
        psd_dict: Dictionary {group_name: psd_array}
        freqs: Frequency array
        title: Plot title
        freq_range: Frequency range to plot
        show_bands: Whether to shade frequency bands
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['double_col']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    plot_freqs = freqs[freq_mask]
    
    # Left panel: PSD curves
    ax1 = axes[0]
    
    for group, color in zip(['AD', 'FTD', 'CN'], CLASS_COLORS):
        if group in psd_dict:
            psd = psd_dict[group]
            if psd.ndim == 3:
                psd = psd[:, :, freq_mask]
                mean_psd = np.mean(psd, axis=(0, 1))
                std_psd = np.std(np.mean(psd, axis=1), axis=0)
            else:
                mean_psd = psd[freq_mask] if psd.ndim == 1 else np.mean(psd[:, freq_mask], axis=0)
                std_psd = np.zeros_like(mean_psd)
            
            ax1.semilogy(plot_freqs, mean_psd, color=color, label=group, linewidth=2)
            if std_psd.any():
                ax1.fill_between(plot_freqs, 
                                np.maximum(mean_psd - std_psd, 1e-6), 
                                mean_psd + std_psd,
                                color=color, alpha=0.15)
    
    # Add frequency band shading
    if show_bands:
        bands = {'δ': (0.5, 4), 'θ': (4, 8), 'α': (8, 13), 'β': (13, 30), 'γ': (30, 45)}
        ymin, ymax = ax1.get_ylim()
        
        for (band_name, (f1, f2)), color in zip(bands.items(), BAND_COLORS):
            if f2 >= freq_range[0] and f1 <= freq_range[1]:
                ax1.axvspan(max(f1, freq_range[0]), min(f2, freq_range[1]), 
                           alpha=0.08, color=color, zorder=0)
                mid_freq = (max(f1, freq_range[0]) + min(f2, freq_range[1])) / 2
                ax1.text(mid_freq, ymax * 0.6, band_name, ha='center', 
                        fontsize=10, fontweight='bold', alpha=0.7)
    
    format_axis(ax1, xlabel='Frequency (Hz)', ylabel='Power (μV²/Hz)',
                title='A  Power Spectral Density', xlim=freq_range, legend=True)
    ax1.legend(loc='upper right', framealpha=0.95)
    
    # Right panel: Band power bar plot
    ax2 = axes[1]
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    band_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    
    x = np.arange(len(band_names))
    width = 0.25
    
    for i, (group, color) in enumerate(zip(['AD', 'FTD', 'CN'], CLASS_COLORS)):
        if group in psd_dict:
            psd = psd_dict[group]
            band_powers = []
            for f1, f2 in band_ranges:
                mask = (freqs >= f1) & (freqs <= f2)
                if psd.ndim == 3:
                    power = np.mean(psd[:, :, mask])
                else:
                    power = np.mean(psd[..., mask])
                band_powers.append(power)
            
            bars = ax2.bar(x + i * width - width, band_powers, width, 
                          label=group, color=color, alpha=0.85, edgecolor='white',
                          linewidth=0.5)
    
    format_axis(ax2, xlabel='Frequency Band', ylabel='Mean Power (μV²/Hz)',
                title='B  Band Power Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names, fontsize=9)
    ax2.legend(loc='upper right', framealpha=0.95)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_topography(values: np.ndarray,
                    ch_names: List[str],
                    title: str = '',
                    cmap: str = 'RdBu_r',
                    vmin: float = None,
                    vmax: float = None,
                    ax: plt.Axes = None,
                    show_names: bool = True,
                    contours: int = 6,
                    colorbar: bool = True,
                    save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot scalp topography - Publication Ready.
    
    Args:
        values: Values for each channel
        ch_names: Channel names
        title: Plot title
        cmap: Colormap
        vmin, vmax: Color limits
        ax: Existing axes
        show_names: Whether to show channel names
        contours: Number of contour lines
        colorbar: Whether to show colorbar
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Standard 10-20 positions (2D projection)
    layout_10_20 = {
        'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
        'F7': (-0.7, 0.5), 'F3': (-0.35, 0.5), 'Fz': (0, 0.5), 
        'F4': (0.35, 0.5), 'F8': (0.7, 0.5),
        'T3': (-0.9, 0), 'C3': (-0.45, 0), 'Cz': (0, 0), 
        'C4': (0.45, 0), 'T4': (0.9, 0),
        'T5': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5), 
        'P4': (0.35, -0.5), 'T6': (0.7, -0.5),
        'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
    }
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    
    # Get positions for our channels
    positions = []
    valid_values = []
    valid_names = []
    
    for i, ch in enumerate(ch_names):
        if ch in layout_10_20:
            positions.append(layout_10_20[ch])
            valid_values.append(values[i])
            valid_names.append(ch)
    
    positions = np.array(positions)
    valid_values = np.array(valid_values)
    
    # Create interpolation grid
    xi = np.linspace(-1.05, 1.05, 150)
    yi = np.linspace(-1.05, 1.05, 150)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    from scipy.interpolate import griddata
    Zi = griddata(positions, valid_values, (Xi, Yi), method='cubic')
    
    # Mask outside head circle
    mask = Xi**2 + Yi**2 > 1.02
    Zi[mask] = np.nan
    
    # Determine color limits
    if vmin is None:
        vmin = np.nanmin(valid_values)
    if vmax is None:
        vmax = np.nanmax(valid_values)
    
    # Symmetric colormap for diverging
    if 'RdBu' in cmap or 'coolwarm' in cmap or 'diverging' in cmap:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    
    # Plot contour fill
    im = ax.contourf(Xi, Yi, Zi, levels=30, cmap=cmap, vmin=vmin, vmax=vmax,
                     extend='neither')
    
    # Add contour lines
    if contours > 0:
        ax.contour(Xi, Yi, Zi, levels=contours, colors='black', 
                  linewidths=0.3, alpha=0.4)
    
    # Draw head outline
    head = Circle((0, 0), 1, fill=False, linewidth=2.5, 
                  color=PALETTE_NEUTRAL['dark_gray'])
    ax.add_patch(head)
    
    # Draw nose
    nose_x = [0, -0.08, 0, 0.08, 0]
    nose_y = [1, 1.08, 1.18, 1.08, 1]
    ax.plot(nose_x, nose_y, color=PALETTE_NEUTRAL['dark_gray'], linewidth=2)
    
    # Draw ears
    ear_left = Wedge((-1.02, 0), 0.12, 90, 270, width=0.04, 
                     facecolor='none', edgecolor=PALETTE_NEUTRAL['dark_gray'], 
                     linewidth=2)
    ear_right = Wedge((1.02, 0), 0.12, 270, 90, width=0.04,
                      facecolor='none', edgecolor=PALETTE_NEUTRAL['dark_gray'],
                      linewidth=2)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
    
    # Plot electrodes
    ax.scatter(positions[:, 0], positions[:, 1], c='black', s=20, zorder=5,
              edgecolors='white', linewidths=0.5)
    
    if show_names:
        for pos, name in zip(positions, valid_names):
            ax.annotate(name, pos, xytext=(2, 2), textcoords='offset points',
                       fontsize=7, ha='left', fontweight='medium')
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.25, 1.35)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    
    # Colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(0.5)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_multi_topography(data_dict: Dict[str, np.ndarray],
                          ch_names: List[str],
                          title: str = None,
                          cmap: str = 'viridis',
                          figsize: Tuple[float, float] = None,
                          save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot multiple topographies side by side - Publication Ready.
    
    Args:
        data_dict: Dictionary of {name: values} for each topography
        ch_names: Channel names
        title: Overall title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_plots = len(data_dict)
    
    if figsize is None:
        figsize = (4.5 * n_plots, 4)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Get global min/max for consistent coloring
    all_values = np.concatenate([v for v in data_dict.values()])
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    
    # Determine if diverging colormap needed
    if np.min(all_values) < 0:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
        cmap = 'RdBu_r'
    
    for ax, (name, values) in zip(axes, data_dict.items()):
        plot_topography(values, ch_names, title=name, cmap=cmap,
                       vmin=vmin, vmax=vmax, ax=ax, show_names=False,
                       colorbar=False)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.0)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_erp(epochs: 'mne.Epochs',
             channel: str = 'Cz',
             groups: Dict[str, List[int]] = None,
             title: str = None,
             figsize: Tuple[float, float] = None,
             save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot Event-Related Potential - Publication Ready.
    
    Args:
        epochs: MNE Epochs object
        channel: Channel to plot
        groups: Dictionary mapping group names to epoch indices
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if figsize is None:
        figsize = FIGURE_SIZES['single_col']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ch_idx = epochs.ch_names.index(channel)
    times = epochs.times * 1000  # Convert to ms
    
    if groups is None:
        groups = {'All': list(range(len(epochs)))}
    
    for i, (group_name, indices) in enumerate(groups.items()):
        color = CLASS_COLORS[i] if i < len(CLASS_COLORS) else f'C{i}'
        data = epochs.get_data()[indices, ch_idx, :]
        mean = np.mean(data, axis=0) * 1e6  # Convert to µV
        sem = np.std(data, axis=0) / np.sqrt(len(indices)) * 1e6
        
        ax.plot(times, mean, color=color, label=f'{group_name} (n={len(indices)})', 
                linewidth=1.8)
        ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.15)
    
    ax.axhline(0, color=PALETTE_NEUTRAL['light_gray'], linestyle='-', 
               linewidth=0.8, zorder=0)
    ax.axvline(0, color=PALETTE_NEUTRAL['light_gray'], linestyle='-', 
               linewidth=0.8, zorder=0)
    
    format_axis(ax, xlabel='Time (ms)', ylabel='Amplitude (μV)',
                title=title or f'ERP at {channel}', legend=True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig
