"""
Publication-Ready Plot Styling Configuration.

This module provides comprehensive styling for high-impact journal
publications with aesthetic colors, light backgrounds, clear fonts,
and professional formatting.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

# =============================================================================
# PUBLICATION COLOR PALETTES
# =============================================================================

# Primary palette - accessible and print-friendly
PALETTE_PRIMARY = {
    'AD': '#C0392B',       # Deep red (Alzheimer's)
    'FTD': '#2980B9',      # Strong blue (Frontotemporal)
    'CN': '#27AE60',       # Forest green (Control)
    'MCI': '#F39C12',      # Orange (Mild Cognitive Impairment)
}

# Extended palette for multi-class comparisons
PALETTE_EXTENDED = {
    'AD_light': '#E74C3C',
    'AD_dark': '#922B21',
    'FTD_light': '#3498DB',
    'FTD_dark': '#1A5276',
    'CN_light': '#2ECC71',
    'CN_dark': '#1E8449',
    'MCI_light': '#F5B041',
    'MCI_dark': '#D68910',
}

# Frequency band colors - spectral ordering
PALETTE_BANDS = {
    'delta': '#8E44AD',    # Purple
    'theta': '#3498DB',    # Blue
    'alpha': '#27AE60',    # Green
    'beta': '#F39C12',     # Orange
    'gamma': '#E74C3C',    # Red
}

# Neutral colors for backgrounds and annotations
PALETTE_NEUTRAL = {
    'background': '#FFFFFF',
    'grid': '#E5E5E5',
    'text': '#2C3E50',
    'axis': '#34495E',
    'light_gray': '#BDC3C7',
    'medium_gray': '#7F8C8D',
    'dark_gray': '#2C3E50',
}

# Colorblind-safe palette (Okabe-Ito)
PALETTE_COLORBLIND = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#009E73',  # Bluish green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish purple
]

# Sequential colormaps for heatmaps
CMAP_CONNECTIVITY = 'viridis'
CMAP_IMPORTANCE = 'YlOrRd'
CMAP_DIVERGING = 'RdBu_r'

# List exports
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
CLASS_COLORS = [PALETTE_PRIMARY[c] for c in CLASS_NAMES]
BAND_COLORS = list(PALETTE_BANDS.values())
BAND_NAMES = list(PALETTE_BANDS.keys())


# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================

def set_publication_style(context: str = 'paper',
                          font_scale: float = 1.0,
                          style: str = 'ticks'):
    """
    Set comprehensive publication-ready style for all plots.
    
    Args:
        context: 'paper', 'notebook', 'talk', or 'poster'
        font_scale: Font size multiplier
        style: seaborn style ('ticks', 'whitegrid', 'darkgrid', 'white')
    """
    # Reset to defaults first
    plt.rcdefaults()
    
    # Use seaborn styling
    sns.set_theme(context=context, style=style, font_scale=font_scale,
                  palette=CLASS_COLORS)
    
    # Apply custom rcParams for publication quality
    publication_params = {
        # Figure settings
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.autolayout': False,
        'figure.constrained_layout.use': True,
        
        # Savefig settings (high quality for journals)
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.transparent': False,
        
        # Font settings (use standard fonts for compatibility)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 10,
        'font.weight': 'normal',
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': PALETTE_NEUTRAL['axis'],
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.titlepad': 12,
        'axes.labelsize': 11,
        'axes.labelweight': 'normal',
        'axes.labelpad': 8,
        'axes.labelcolor': PALETTE_NEUTRAL['text'],
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.prop_cycle': plt.cycler('color', CLASS_COLORS),
        
        # Tick settings
        'xtick.direction': 'out',
        'xtick.major.size': 5,
        'xtick.major.width': 1.0,
        'xtick.minor.size': 3,
        'xtick.minor.width': 0.5,
        'xtick.labelsize': 10,
        'xtick.color': PALETTE_NEUTRAL['axis'],
        'xtick.top': False,
        'xtick.bottom': True,
        
        'ytick.direction': 'out',
        'ytick.major.size': 5,
        'ytick.major.width': 1.0,
        'ytick.minor.size': 3,
        'ytick.minor.width': 0.5,
        'ytick.labelsize': 10,
        'ytick.color': PALETTE_NEUTRAL['axis'],
        'ytick.left': True,
        'ytick.right': False,
        
        # Grid settings
        'grid.color': PALETTE_NEUTRAL['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.edgecolor': PALETTE_NEUTRAL['light_gray'],
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        'legend.borderpad': 0.5,
        'legend.labelspacing': 0.4,
        'legend.handlelength': 1.5,
        'legend.handleheight': 0.7,
        'legend.columnspacing': 1.0,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.0,
        
        # Patch settings (for bars, boxes)
        'patch.linewidth': 1.0,
        'patch.edgecolor': PALETTE_NEUTRAL['dark_gray'],
        'patch.facecolor': CLASS_COLORS[0],
        
        # Hatch settings
        'hatch.linewidth': 0.5,
        
        # Text settings
        'text.color': PALETTE_NEUTRAL['text'],
        
        # Math text
        'mathtext.fontset': 'dejavusans',
    }
    
    plt.rcParams.update(publication_params)
    
    print("Publication style applied successfully.")


def set_nature_style():
    """Apply Nature journal styling guidelines."""
    set_publication_style(context='paper', font_scale=1.0)
    
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.figsize': (3.5, 2.5),  # Single column width
    })


def set_science_style():
    """Apply Science journal styling guidelines."""
    set_publication_style(context='paper', font_scale=1.0)
    
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.figsize': (3.5, 2.8),
    })


def set_neurology_style():
    """Apply Neurology/JAMA Neurology styling guidelines."""
    set_publication_style(context='paper', font_scale=1.0)
    
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (3.5, 3.0),
    })


# =============================================================================
# FIGURE SIZE PRESETS
# =============================================================================

FIGURE_SIZES = {
    # Single column (Nature, Science)
    'single_col': (3.5, 2.8),
    'single_col_tall': (3.5, 4.0),
    'single_col_square': (3.5, 3.5),
    
    # 1.5 column
    'medium': (5.5, 4.0),
    'medium_wide': (5.5, 3.0),
    'medium_tall': (5.5, 5.5),
    
    # Double column / full width
    'double_col': (7.0, 4.0),
    'double_col_wide': (7.0, 3.0),
    'double_col_tall': (7.0, 6.0),
    'double_col_square': (7.0, 7.0),
    
    # Full page
    'full_page': (7.0, 9.0),
    
    # Presentation sizes
    'presentation': (10, 6),
    'presentation_wide': (12, 5),
}


def get_figure_size(preset: str = 'single_col') -> Tuple[float, float]:
    """Get figure size by preset name."""
    return FIGURE_SIZES.get(preset, FIGURE_SIZES['single_col'])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_panel_label(ax, label: str, x: float = -0.15, y: float = 1.05,
                    fontsize: int = 14, fontweight: str = 'bold'):
    """
    Add panel label (A, B, C, etc.) to axis.
    
    Args:
        ax: Matplotlib axis
        label: Panel label (e.g., 'A', 'B')
        x, y: Position in axis coordinates
        fontsize: Font size
        fontweight: Font weight
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            verticalalignment='top', horizontalalignment='left',
            color=PALETTE_NEUTRAL['text'])


def add_significance_annotation(ax, x1: float, x2: float, y: float,
                                 p_value: float, height: float = 0.02):
    """
    Add significance bracket with stars.
    
    Args:
        ax: Matplotlib axis
        x1, x2: X positions
        y: Y position of bracket
        p_value: P-value for determining stars
        height: Height of bracket arms
    """
    # Determine significance level
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    
    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color=PALETTE_NEUTRAL['dark_gray'], linewidth=1.0)
    
    # Add text
    ax.text((x1 + x2) / 2, y + height, sig_text,
            ha='center', va='bottom', fontsize=9,
            color=PALETTE_NEUTRAL['text'])


def format_axis(ax, xlabel: str = None, ylabel: str = None,
                title: str = None, xlim: tuple = None, ylim: tuple = None,
                legend: bool = True, legend_loc: str = 'best',
                grid: bool = False, grid_axis: str = 'both'):
    """
    Apply consistent formatting to axis.
    
    Args:
        ax: Matplotlib axis
        xlabel, ylabel: Axis labels
        title: Axis title
        xlim, ylim: Axis limits
        legend: Whether to show legend
        legend_loc: Legend location
        grid: Whether to show grid
        grid_axis: Which axis to grid ('x', 'y', 'both')
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, frameon=True)
    
    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.7)
        ax.set_axisbelow(True)


def despine(ax, left: bool = False, bottom: bool = False,
            top: bool = True, right: bool = True):
    """Remove specified spines from axis."""
    sns.despine(ax=ax, left=left, bottom=bottom, top=top, right=right)


def create_colorbar(mappable, ax, label: str = '', orientation: str = 'vertical',
                    fraction: float = 0.046, pad: float = 0.04):
    """Create a consistently styled colorbar."""
    cbar = plt.colorbar(mappable, ax=ax, orientation=orientation,
                        fraction=fraction, pad=pad)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)
    return cbar


# =============================================================================
# CUSTOM COLORMAPS
# =============================================================================

def create_brain_connectivity_cmap():
    """Create custom colormap for brain connectivity."""
    colors = ['#FFFFFF', '#FFF7EC', '#FEE8C8', '#FDD49E',
              '#FDBB84', '#FC8D59', '#EF6548', '#D7301F', '#990000']
    return LinearSegmentedColormap.from_list('brain_connectivity', colors)


def create_diverging_cmap():
    """Create custom diverging colormap (blue-white-red)."""
    colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0',
              '#FFFFFF',
              '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    return LinearSegmentedColormap.from_list('custom_diverging', colors)


def create_importance_cmap():
    """Create custom colormap for importance/attention."""
    colors = ['#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C',
              '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    return LinearSegmentedColormap.from_list('importance', colors)


# Register custom colormaps
# Register custom colormaps
def register_cmap_safe(name, cmap):
    try:
        # Modern Matplotlib (3.5+)
        if hasattr(mpl, 'colormaps'):
            try:
                mpl.colormaps.register(cmap, name=name)
            except ValueError:
                pass # Already registered
        # Older Matplotlib
        elif hasattr(plt.cm, 'register_cmap'):
            try:
                plt.cm.register_cmap(name, cmap)
            except ValueError:
                pass
    except Exception as e:
        print(f"Warning: Could not register colormap {name}: {e}")

register_cmap_safe('brain_connectivity', create_brain_connectivity_cmap())
register_cmap_safe('custom_diverging', create_diverging_cmap())
register_cmap_safe('importance', create_importance_cmap())


def safe_tight_layout(fig=None, **kwargs):
    """
    Apply tight_layout gracefully, handling engine conflicts.
    
    Args:
        fig: Optional figure instance. If None, uses plt.gcf().
        **kwargs: Arguments passed to tight_layout.
    """
    try:
        if fig:
            fig.tight_layout(**kwargs)
        else:
            plt.tight_layout(**kwargs)
    except RuntimeError:
        # Occurs when colorbars conflict with 'constrained' or other layout engines
        pass
    except Exception as e:
        print(f"Warning: tight_layout failed: {e}")


# =============================================================================
# APPLY DEFAULT STYLE ON IMPORT
# =============================================================================

# Apply publication style by default
set_publication_style()
