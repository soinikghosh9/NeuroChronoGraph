#!/usr/bin/env python
"""
Advanced Connectivity Visualization for NeuroChronoGraph.

Creates publication-quality connectivity network visualizations:
- Circular connectome plots with properly labeled nodes
- Edge thickness proportional to connection strength (wPLI)
- Disease-specific connectivity patterns (AD vs CN, FTD vs CN)
- Modern, clean aesthetic suitable for high-impact journals

Based on MNE-Connectivity visualization methods.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
from scipy.signal import coherence
from scipy import stats
import mne

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_PATH = PROJECT_ROOT / "outputs" / "preprocessed"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"

# Nature-style Aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# EEG Channel Configuration (10-20 system)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                 'T3', 'C3', 'Cz', 'C4', 'T4',
                 'T5', 'P3', 'Pz', 'P4', 'T6',
                 'O1', 'O2']

# Region mapping for color coding
REGION_COLORS = {
    'Frontal': '#F08080',     # Light Coral
    'Central': '#DDA0DD',     # Plum
    'Temporal': '#87CEEB',    # Sky Blue
    'Parietal': '#90EE90',    # Light Green
    'Occipital': '#FFD700'    # Gold
}

CHANNEL_REGIONS = {
    'Fp1': 'Frontal', 'Fp2': 'Frontal', 'F7': 'Frontal', 'F3': 'Frontal',
    'Fz': 'Frontal', 'F4': 'Frontal', 'F8': 'Frontal',
    'T3': 'Temporal', 'T4': 'Temporal', 'T5': 'Temporal', 'T6': 'Temporal',
    'C3': 'Central', 'Cz': 'Central', 'C4': 'Central',
    'P3': 'Parietal', 'Pz': 'Parietal', 'P4': 'Parietal',
    'O1': 'Occipital', 'O2': 'Occipital'
}

# Class configuration
# Class configuration
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
CLASS_COLORS = {'AD': '#E64B35', 'FTD': '#4DBBD5', 'CN': '#00A087', 'MCI': '#F39B7F'}


def compute_wpli(epochs_data, sfreq, band=(8, 13)):
    """Compute weighted Phase Lag Index for a frequency band."""
    from scipy.signal import hilbert, butter, filtfilt
    
    n_epochs, n_channels, n_times = epochs_data.shape
    
    # Bandpass filter
    nyq = sfreq / 2
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)
    b, a = butter(4, [low, high], btype='band')
    
    # Compute wPLI matrix
    wpli_matrix = np.zeros((n_channels, n_channels))
    
    for ep in range(min(10, n_epochs)):  # Use first 10 epochs for speed
        # Filter and get phase
        filtered = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            try:
                filtered[ch] = filtfilt(b, a, epochs_data[ep, ch, :])
            except:
                continue
        
        # Hilbert transform for phase
        analytic = hilbert(filtered, axis=1)
        phase = np.angle(analytic)
        
        # Compute cross-spectrum imaginary part
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                phase_diff = phase[i] - phase[j]
                imag_csd = np.sin(phase_diff)
                
                # wPLI formula
                num = np.abs(np.mean(np.abs(imag_csd) * np.sign(imag_csd)))
                den = np.mean(np.abs(imag_csd))
                
                if den > 0:
                    wpli = num / den
                else:
                    wpli = 0
                    
                wpli_matrix[i, j] = wpli
                wpli_matrix[j, i] = wpli
    
    return wpli_matrix


def load_subject_data(pkl_file):
    """Load preprocessed EEG data from pickle file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_group_connectivity(class_name, n_subjects=15):
    """Compute average connectivity matrix for a diagnostic group."""
    print(f"  Computing connectivity for {class_name}...")
    
    # Class mapping
    # Class mapping (Robust)
    class_mapping = {}
    # 1. ds004504 (OpenNeuro)
    # Keys must match filename stems: ds004504_sub-XXX
    for i in range(1, 89):
        sid = f"ds004504_sub-{i:03d}"
        if 1 <= i <= 36: class_mapping[sid] = 'AD'
        elif 37 <= i <= 59: class_mapping[sid] = 'FTD'
        elif 60 <= i <= 88: class_mapping[sid] = 'CN'
        
        # Also add without prefix just in case some files are named differently
        short_sid = f"sub-{i:03d}"
        if 1 <= i <= 36: class_mapping[short_sid] = 'AD' 
        elif 37 <= i <= 59: class_mapping[short_sid] = 'FTD'
        elif 60 <= i <= 88: class_mapping[short_sid] = 'CN'

    # Heuristic mapping for others (covering MCI)
    all_pkls = list(PREPROCESSED_PATH.glob("*_preprocessed.pkl"))
    for pkl in all_pkls:
        sid = pkl.stem.replace('_preprocessed', '')
        if sid in class_mapping: continue
        
        if "Mendeley" in sid:
            if "_AD_" in sid: class_mapping[sid] = 'AD'
            elif "_MCI_" in sid: class_mapping[sid] = 'MCI'
            elif "_normal_" in sid: class_mapping[sid] = 'CN'
        elif "MCIData" in sid:
            if "_AD_" in sid: class_mapping[sid] = 'AD'
            elif "_MCI_" in sid: class_mapping[sid] = 'MCI'
            elif "_CONTROL_" in sid: class_mapping[sid] = 'CN'
        elif "AlzEEG" in sid:
            if "_AD_" in sid: class_mapping[sid] = 'AD'
            elif "_Frontotemporal_" in sid: class_mapping[sid] = 'FTD'
            elif "_Healthy_" in sid: class_mapping[sid] = 'CN'
    
    pkl_files = sorted(list(PREPROCESSED_PATH.glob("*_preprocessed.pkl")))
    
    connectivity_matrices = []
    count = 0
    
    for pkl_file in pkl_files:
        if count >= n_subjects:
            break
            
        subject_id = pkl_file.stem.replace('_preprocessed', '')
        if class_mapping.get(subject_id) != class_name:
            continue
            
        try:
            data = load_subject_data(pkl_file)
            # FIX: data contains 'epochs_data' (numpy), not 'epochs' (MNE object)
            # Robust Loading
            if 'epochs_data' in data:
                # Numpy array format
                epochs_data = data['epochs_data']
                sfreq = data.get('sfreq', 500)
                # Create rudimentary EpochsArray for wPLI computation
                info = mne.create_info(ch_names=data.get('ch_names', ['ch']*epochs_data.shape[1]), 
                                     sfreq=sfreq, ch_types='eeg')
                epochs = mne.EpochsArray(epochs_data, info, verbose=False)
            elif 'epochs' in data:
                # MNE Epochs object
                epochs = data['epochs']
                sfreq = epochs.info['sfreq']
                epochs_data = epochs.get_data() # Extract numpy array for wPLI
            else:
                print(f"      WARNING: No epochs data found for {subject_id}")
                continue
            
            # Check size: [n_epochs, n_ch, n_times]
            if epochs_data.shape[0] < 5: # Skip if too few epochs
                continue
            
            # Compute alpha-band wPLI
            wpli = compute_wpli(epochs_data, sfreq, band=(8, 13))
            
            connectivity_matrices.append(wpli)
            count += 1
            
        except Exception as e:
            print(f"    Error processing {subject_id}: {e}")
            continue
    
    print(f"  {class_name}: Processed {len(connectivity_matrices)} subjects.")
    
    if connectivity_matrices:
        avg_conn = np.nanmean(connectivity_matrices, axis=0)
        print(f"  {class_name} Avg Stats - Mean: {np.nanmean(avg_conn):.6f}, Max: {np.nanmax(avg_conn):.6f}")
        return avg_conn
    else:
        print(f"  WARNING: No data for {class_name}!")
        return np.zeros((19, 19))


def plot_circular_connectome(connectivity_matrix, ch_names, title, output_path, 
                              threshold=0.3, node_size=800):
    """
    Create circular connectome plot with nodes arranged by brain region.
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    
    n_channels = len(ch_names)
    
    # Arrange nodes in a circle, grouped by region
    region_order = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Central']
    ordered_channels = []
    for region in region_order:
        for ch in ch_names:
            if CHANNEL_REGIONS.get(ch) == region:
                ordered_channels.append(ch)
    
    # Add any missing channels
    for ch in ch_names:
        if ch not in ordered_channels:
            ordered_channels.append(ch)
    
    n = len(ordered_channels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Node positions
    node_positions = {}
    for i, ch in enumerate(ordered_channels):
        x = np.cos(angles[i])
        y = np.sin(angles[i])
        node_positions[ch] = (x, y)
    
    # Draw edges (connections above threshold)
    max_conn = np.max(connectivity_matrix)
    min_conn = threshold

    # Perceptually-uniform colormap that stays visible on white.
    # 'plasma' goes deep-purple -> magenta -> orange -> yellow; we restrict to
    # [0.15, 0.95] so even the weakest edge keeps a saturated dark-purple color
    # instead of fading to near-white.
    edge_cmap = plt.cm.plasma
    edge_vmin, edge_vmax = 0.15, 0.95

    # Sort edges so strongest are drawn on top
    edges_to_draw = []
    for i, ch1 in enumerate(ordered_channels):
        for j, ch2 in enumerate(ordered_channels):
            if i >= j:
                continue
            try:
                idx1 = ch_names.index(ch1)
                idx2 = ch_names.index(ch2)
            except ValueError:
                continue
            conn_strength = connectivity_matrix[idx1, idx2]
            if conn_strength > threshold:
                edges_to_draw.append((conn_strength, ch1, ch2))

    edges_to_draw.sort(key=lambda e: e[0])
    for conn_strength, ch1, ch2 in edges_to_draw:
        norm_strength = (conn_strength - min_conn) / (max_conn - min_conn + 1e-6)
        norm_strength = float(np.clip(norm_strength, 0.0, 1.0))

        # Map weight to color (darker/more saturated = stronger)
        color = edge_cmap(edge_vmin + norm_strength * (edge_vmax - edge_vmin))
        # Width scales but has a visible floor
        line_width = 1.2 + norm_strength * 3.8
        # Alpha has a high floor so weakest edges still read on white
        alpha = 0.75 + norm_strength * 0.25

        x1, y1 = node_positions[ch1]
        x2, y2 = node_positions[ch2]

        ax.plot([x1, x2], [y1, y2],
               color=color, alpha=alpha, linewidth=line_width,
               solid_capstyle='round', zorder=1)

    # Edge colorbar so readers can map color -> wPLI strength
    sm = ScalarMappable(
        cmap=edge_cmap,
        norm=Normalize(vmin=min_conn, vmax=max(max_conn, min_conn + 1e-3)),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=25)
    cbar.set_label('wPLI (alpha)', fontsize=10)
    cbar.outline.set_visible(False)
    
    # Draw nodes
    for ch in ordered_channels:
        x, y = node_positions[ch]
        region = CHANNEL_REGIONS.get(ch, 'Central')
        color = REGION_COLORS.get(region, '#808080')
        
        circle = plt.Circle((x, y), 0.08, color=color, ec='white', 
                            linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Label
        label_x = x * 1.15
        label_y = y * 1.15
        ax.text(label_x, label_y, ch, ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='white',
                                       label=region)
                       for region, color in REGION_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def plot_connectivity_comparison(conn_ad, conn_ftd, conn_cn, conn_mci, ch_names, output_path):
    """
    Create 2x2 comparison figure showing:
    - AD connectivity
    - FTD connectivity  
    - AD-CN difference
    - MCI-CN difference
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    matrices = [
        (conn_ad, 'AD Alpha Connectivity'),
        (conn_ftd, 'FTD Alpha Connectivity'),
        (conn_ad - conn_cn, 'AD - CN (Difference)'),
        (conn_mci - conn_cn, 'MCI - CN (Difference)') # Replaced FTD diff with MCI diff for variety
    ]
    
    for ax, (matrix, title) in zip(axes.flat, matrices):
        # Determine colormap
        if 'Difference' in title:
            vmax = np.max(np.abs(matrix))
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(matrix, cmap='hot', vmin=0, vmax=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(ch_names)))
        ax.set_yticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_yticklabels(ch_names, fontsize=7)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Functional Connectivity Analysis (Alpha Band wPLI)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def plot_disease_signature_network(conn_ad, conn_cn, ch_names, output_path):
    """
    Create publication figure showing disease-specific network signatures.
    Shows edges that are significantly different between AD and CN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.3)
    
    # Compute difference
    diff = conn_ad - conn_cn
    
    # Define node positions (circular)
    n = len(ch_names)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    positions = {ch: (np.cos(angles[i]), np.sin(angles[i])) 
                 for i, ch in enumerate(ch_names)}
                 
    # We will use matplotlib colormaps for enhanced colors
    cmap_hypo = plt.cm.Blues
    cmap_hyper = plt.cm.Reds
    
    # --- Left: AD hypoconnectivity ---
    ax = axes[0]
    ax.set_aspect('equal')
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.axis('off')
    ax.set_title('AD Hypoconnectivity\n(Reduced vs Controls)', 
                fontsize=16, fontweight='bold', color='#1f77b4', pad=20)
    
    # Threshold for hypoconnectivity (changed to -0.05 for more visibility)
    hypo_threshold = -0.05
    hypo_edges = [(abs(diff[i, j]), ch_names[i], ch_names[j])
                  for i in range(len(ch_names)) for j in range(i+1, len(ch_names))
                  if diff[i, j] < hypo_threshold]
    hypo_max = max((s for s, _, _ in hypo_edges), default=1.0)
    hypo_min = min((s for s, _, _ in hypo_edges), default=abs(hypo_threshold))
    
    hypo_edges.sort(key=lambda e: e[0])
    for strength, ch1, ch2 in hypo_edges:
        x1, y1 = positions[ch1]
        x2, y2 = positions[ch2]
        
        # Normalize between 0 and 1
        val_range = max(hypo_max - hypo_min, 1e-6)
        norm = float(np.clip((strength - hypo_min) / val_range, 0.0, 1.0))
        
        # Enhanced visibility: more dramatic variation in linewidth and color
        # Intense blue colors, starting from medium blue to dark blue
        color = cmap_hypo(0.4 + 0.6 * norm)
        
        # Prominent thickness and exponential scaling
        line_width = 1.8 + (norm ** 1.5) * 8.0
        alpha = 0.5 + 0.5 * norm
        
        ax.plot([x1, x2], [y1, y2], color=color,
               alpha=alpha, linewidth=line_width,
               solid_capstyle='round', zorder=1)
    
    # Draw nodes for left plot
    for ch in ch_names:
        x, y = positions[ch]
        region = CHANNEL_REGIONS.get(ch, 'Central')
        color = REGION_COLORS.get(region, '#808080')
        circle = plt.Circle((x, y), 0.08, color=color, ec='white', linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Dynamic text alignment
        ha = 'left' if x > 0.1 else ('right' if x < -0.1 else 'center')
        va = 'bottom' if y > 0.1 else ('top' if y < -0.1 else 'center')
        ax.text(x*1.18, y*1.18, ch, ha=ha, va=va, fontsize=11, fontweight='bold')
        
    # --- Right: AD hyperconnectivity ---
    hyper_threshold = 0.05
    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.axis('off')
    ax.set_title(f'AD Hyperconnectivity\n(Increased vs Controls)', 
                fontsize=16, fontweight='bold', color='#d62728', pad=20)
    
    hyper_edges = [(abs(diff[i, j]), ch_names[i], ch_names[j])
                   for i in range(len(ch_names)) for j in range(i+1, len(ch_names))
                   if diff[i, j] > hyper_threshold]
    hyper_max = max((s for s, _, _ in hyper_edges), default=1.0)
    hyper_min = min((s for s, _, _ in hyper_edges), default=hyper_threshold)
    
    hyper_edges.sort(key=lambda e: e[0])
    for strength, ch1, ch2 in hyper_edges:
        x1, y1 = positions[ch1]
        x2, y2 = positions[ch2]
        
        val_range = max(hyper_max - hyper_min, 1e-6)
        norm = float(np.clip((strength - hyper_min) / val_range, 0.0, 1.0))
        
        # Enhanced visibility: Intense red colors
        color = cmap_hyper(0.4 + 0.6 * norm)
        
        line_width = 1.8 + (norm ** 1.5) * 8.0
        alpha = 0.5 + 0.5 * norm
        
        ax.plot([x1, x2], [y1, y2], color=color,
               alpha=alpha, linewidth=line_width,
               solid_capstyle='round', zorder=1)
    
    # Draw nodes for right plot
    for ch in ch_names:
        x, y = positions[ch]
        region = CHANNEL_REGIONS.get(ch, 'Central')
        color = REGION_COLORS.get(region, '#808080')
        circle = plt.Circle((x, y), 0.08, color=color, ec='white', linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        # Dynamic text alignment
        ha = 'left' if x > 0.1 else ('right' if x < -0.1 else 'center')
        va = 'bottom' if y > 0.1 else ('top' if y < -0.1 else 'center')
        ax.text(x*1.18, y*1.18, ch, ha=ha, va=va, fontsize=11, fontweight='bold')
    
    # --- Legends and Colorbars ---
    # Add a custom legend for Brain Regions
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='white', label=region)
                       for region, color in REGION_COLORS.items()]
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.05), ncol=5, frameon=False, fontsize=12)
    
    # Add Colorbars to show strength scales
    # Left Context (Hypo)
    sm_hypo = ScalarMappable(cmap=cmap_hypo, norm=Normalize(vmin=hypo_min, vmax=hypo_max))
    sm_hypo.set_array([])
    cbar_hypo = fig.colorbar(sm_hypo, ax=axes[0], shrink=0.5, pad=0.08, aspect=20, orientation='vertical')
    cbar_hypo.set_label('Decrease Magnitude (wPLI)', fontsize=12)
    cbar_hypo.outline.set_visible(False)
    
    # Right Context (Hyper)
    sm_hyper = ScalarMappable(cmap=cmap_hyper, norm=Normalize(vmin=hyper_min, vmax=hyper_max))
    sm_hyper.set_array([])
    cbar_hyper = fig.colorbar(sm_hyper, ax=axes[1], shrink=0.5, pad=0.08, aspect=20, orientation='vertical')
    cbar_hyper.set_label('Increase Magnitude (wPLI)', fontsize=12)
    cbar_hyper.outline.set_visible(False)
    
    plt.suptitle('Disease-Specific Connectivity Signatures (Alpha Band)', 
                 fontsize=20, fontweight='bold', y=0.95)
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    print("=" * 60)
    print("Advanced Connectivity Visualization")
    print("=" * 60)
    
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Compute group connectivity matrices
    print("\nComputing group connectivity matrices...")
    print("\nComputing group connectivity matrices...")
    conn_ad = compute_group_connectivity('AD', n_subjects=10)
    conn_ftd = compute_group_connectivity('FTD', n_subjects=10)
    conn_cn = compute_group_connectivity('CN', n_subjects=10)
    conn_mci = compute_group_connectivity('MCI', n_subjects=10)
    
    # Use first 19 channels (standard 10-20)
    ch_names = CHANNEL_NAMES[:min(19, conn_ad.shape[0])]
    n_ch = len(ch_names)
    n_ch = len(ch_names)
    conn_ad = conn_ad[:n_ch, :n_ch]
    conn_ftd = conn_ftd[:n_ch, :n_ch]
    conn_cn = conn_cn[:n_ch, :n_ch]
    conn_mci = conn_mci[:n_ch, :n_ch]
    
    print("\nGenerating visualizations...")

    # 0. MNE-style circular connectograms (rainbow perimeter + Bezier edges)
    #    Written to explainability/ so Figure5 picks them up via its primary path.
    try:
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.visualization import plot_connectivity_circle as _mne_circle
        expl_dir = FIGURES_PATH / "explainability"
        expl_dir.mkdir(parents=True, exist_ok=True)
        print("  Creating MNE-style circular connectograms...")
        for name, adj in [('AD', conn_ad), ('FTD', conn_ftd),
                          ('CN', conn_cn), ('MCI', conn_mci)]:
            _mne_circle(adj, ch_names, title=f"{name} Connectivity",
                        output_path=expl_dir / f"connectivity_{name}.png")
    except Exception as e:
        print(f"  MNE-style circle rendering skipped: {e}")

    # 1. Circular connectomes for each group
    print("  Creating circular connectomes...")
    plot_circular_connectome(conn_ad, ch_names, 
                            'AD Alpha-Band Connectivity',
                            FIGURES_PATH / 'connectome_AD.png', 
                            threshold=0.25)
    plot_circular_connectome(conn_ftd, ch_names,
                            'FTD Alpha-Band Connectivity', 
                            FIGURES_PATH / 'connectome_FTD.png',
                            threshold=0.25)
    plot_circular_connectome(conn_cn, ch_names,
                            'CN Alpha-Band Connectivity',
                            FIGURES_PATH / 'connectome_CN.png',
                            threshold=0.25)
    plot_circular_connectome(conn_mci, ch_names,
                            'MCI Alpha-Band Connectivity',
                            FIGURES_PATH / 'connectome_MCI.png',
                            threshold=0.25)
    
    # 2. Matrix comparison figure
    print("  Creating connectivity matrices comparison...")
    plot_connectivity_comparison(conn_ad, conn_ftd, conn_cn, conn_mci, ch_names,
                                FIGURES_PATH / 'connectivity_matrices.png')
    
    # 3. Disease signature network
    print("  Creating disease signature network...")
    plot_disease_signature_network(conn_ad, conn_cn, ch_names,
                                   FIGURES_PATH / 'connectivity_disease_signatures.png')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Figures saved to: {FIGURES_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
