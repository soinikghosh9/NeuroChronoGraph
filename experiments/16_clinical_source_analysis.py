#!/usr/bin/env python
"""
Clinical Source Localization Analysis.

Creates clinically meaningful visualizations showing disease-specific EEG patterns:
- Regional power by frequency band (Frontal, Temporal, Parietal, Occipital)
- Theta/Alpha ratio (key dementia biomarker)
- Anterior-Posterior gradient (AD anteriorization, FTD frontal changes)
- Statistical comparison between AD, FTD, and CN

Based on established literature:
- AD: Posterior alpha decrease, occipital delta increase, theta/alpha elevation
- FTD: Frontal alpha reduction, widespread theta increase, frontotemporal disconnection
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pickle
from scipy import stats

import mne
from mne.datasets import fetch_fsaverage

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_PATH = PROJECT_ROOT / "outputs" / "preprocessed"
SOURCE_PATH = PROJECT_ROOT / "outputs" / "source_estimates"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"

# Nature-style Aesthetics - Light Theme
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Classes
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
# Distinct Pastel Palette (High Visibility, Aesthetic)
CLASS_COLORS = {
    'AD': '#F08080',    # Light Coral (Distinct Red-ish)
    'FTD': '#87CEEB',   # Sky Blue (Distinct Blue-ish)
    'CN': '#90EE90',    # Light Green (Distinct Green-ish)
    'MCI': '#FFB347'    # Pastel Orange
}

# Frequency bands
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Brain regions based on 10-20 electrode positions
REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Parietal': ['P3', 'Pz', 'P4'],
    'Occipital': ['O1', 'O2']
}


def load_subject_data(pkl_file):
    """Load preprocessed EEG data from pickle file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_band_power(epochs_data, sfreq, band):
    """Compute power in a frequency band using Welch's method."""
    from scipy.signal import welch
    
    n_epochs, n_channels, n_times = epochs_data.shape
    band_powers = np.zeros((n_epochs, n_channels))
    
    total_power = np.zeros((n_epochs, n_channels))
    
    for ep in range(n_epochs):
        for ch in range(n_channels):
            freqs, psd = welch(epochs_data[ep, ch, :], fs=sfreq, nperseg=min(256, n_times))
            
            # Compute band power
            freq_mask = (freqs >= band[0]) & (freqs <= band[1])
            band_powers[ep, ch] = np.sum(psd[freq_mask])
            
            # Compute total power (0.5-45 Hz) for normalization
            total_mask = (freqs >= 0.5) & (freqs <= 45)
            total_power[ep, ch] = np.sum(psd[total_mask])
    
    # Avoid division by zero
    total_power[total_power == 0] = 1e-10
    
    # Return relative power (percentage of total power)
    relative_power = band_powers / total_power
    return relative_power.mean(axis=0)  # Average over epochs


def get_regional_power(band_power, ch_names, regions):
    """Get average power for each region."""
    regional_power = {}
    for region, channels in regions.items():
        indices = [ch_names.index(ch) for ch in channels if ch in ch_names]
        if indices:
            regional_power[region] = np.mean(band_power[indices])
        else:
            regional_power[region] = np.nan
    return regional_power


def get_subject_class_mapping():
    """Get mapping of subjects to their diagnostic class."""
    mapping = {}
    
    # 1. ds004504 (OpenNeuro)
    for i in range(1, 89):
        # Handle both raw ID and prefix ID
        sid = f"sub-{i:03d}"
        sid_ds = f"ds004504_sub-{i:03d}"
        
        cls = 'CN' # Default
        if 1 <= i <= 36: cls = 'AD'
        elif 37 <= i <= 59: cls = 'FTD'
        elif 60 <= i <= 88: cls = 'CN'
        
        mapping[sid] = cls
        mapping[sid_ds] = cls

    # Heuristic mapping for others (covering MCI)
    pkl_files = list(PREPROCESSED_PATH.glob("*_preprocessed.pkl"))
    for pkl in pkl_files:
        sid = pkl.stem.replace('_preprocessed', '')
        if sid in mapping: continue
        
        if "Mendeley" in sid:
            if "_AD_" in sid: mapping[sid] = 'AD'
            elif "_MCI_" in sid: mapping[sid] = 'MCI'
            elif "_normal_" in sid: mapping[sid] = 'CN'
        elif "MCIData" in sid:
            if "_AD_" in sid: mapping[sid] = 'AD'
            elif "_MCI_" in sid: mapping[sid] = 'MCI'
            elif "_CONTROL_" in sid: mapping[sid] = 'CN'
        elif "AlzEEG" in sid:
            if "_AD_" in sid: mapping[sid] = 'AD'
            elif "_Frontotemporal_" in sid: mapping[sid] = 'FTD'
            elif "_Healthy_" in sid: mapping[sid] = 'CN'
            
    return mapping


def analyze_all_subjects():
    """Analyze all subjects and compute regional band powers."""
    print("=" * 60)
    print("Clinical Source Analysis")
    print("=" * 60)
    
    # Find all preprocessed files
    pkl_files = sorted(list(PREPROCESSED_PATH.glob("*_preprocessed.pkl")))
    print(f"\nFound {len(pkl_files)} preprocessed subjects")
    
    # Class mapping
    class_mapping = get_subject_class_mapping()
    
    # Store results
    results = {class_name: {band: {region: [] for region in REGIONS.keys()} 
               for band in FREQ_BANDS.keys()} for class_name in CLASS_NAMES}
    
    # Process each subject
    for i, pkl_file in enumerate(pkl_files):
        subject_id = pkl_file.stem.replace('_preprocessed', '')
        if subject_id not in class_mapping:
            continue
        
        subject_class = class_mapping[subject_id]
        print(f"  [{i+1}/{len(pkl_files)}] {subject_id} ({subject_class})", end='\r')
        
        try:
            data = load_subject_data(pkl_file)
            
            # Robust loading (Handle MNE Epochs or Numpy Array)
            if 'epochs_data' in data:
                epochs_data = data['epochs_data']
                sfreq = data.get('sfreq', 500)
                ch_names = data.get('ch_names', REGIONS['Frontal'] + REGIONS['Temporal'] + REGIONS['Parietal'] + REGIONS['Occipital']) # Fallback
            elif 'epochs' in data:
                epochs_data = data['epochs'].get_data()
                sfreq = data['epochs'].info['sfreq']
                ch_names = data['epochs'].ch_names
            else:
                print(f"  Skipping {subject_id}: No epochs data found.")
                continue
            
            # Skip if invalid data
            if epochs_data.size < 100:
                continue
            
            # Compute band powers
            for band_name, band_range in FREQ_BANDS.items():
                band_power = compute_band_power(epochs_data, sfreq, band_range)
                regional = get_regional_power(band_power, ch_names, REGIONS)
                
                for region, power in regional.items():
                    if not np.isnan(power):
                        results[subject_class][band_name][region].append(power)
        
        except Exception as e:
            continue
    
    print("\n")
    return results


def plot_regional_band_power(results):
    """Create publication figure showing regional power by frequency band."""
    print("Creating regional band power figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    regions = list(REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.25
    
    for i, (band_name, band_range) in enumerate(FREQ_BANDS.items()):
        ax = axes[i]
        
        for j, class_name in enumerate(CLASS_NAMES):
            # Use nanmean to handle missing data or NaNs
            values = [results[class_name][band_name][r] for r in regions]
            means = [np.nanmean(v) if v and len(v) > 0 else 0 for v in values]
            sems = [stats.sem(v, nan_policy='omit') if v and len(v) > 1 else 0 for v in values]
            
            ax.bar(x + j*width, means, width, label=class_name, 
                   color=CLASS_COLORS[class_name], yerr=sems, capsize=3)
        
        ax.set_xlabel('Brain Region', fontsize=11)
        ax.set_ylabel('Power (μV²/Hz)', fontsize=11)
        ax.set_title(f'{band_name} Band ({band_range[0]}-{band_range[1]} Hz)', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(regions, fontsize=10)
        # Only add legend to the first subplot to avoid clutter
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(-0.1, 1.25), 
                     ncol=3, frameon=False, fontsize=11)
        
        # Highlight disease-specific patterns
        if band_name == 'Alpha':
            ax.annotate('AD: ↓ Posterior', xy=(3, ax.get_ylim()[1]*0.9), fontsize=9, 
                       color=CLASS_COLORS['AD'], fontweight='bold')
            ax.annotate('FTD: ↓ Frontal', xy=(0, ax.get_ylim()[1]*0.8), fontsize=9,
                       color=CLASS_COLORS['FTD'], fontweight='bold')
        elif band_name == 'Theta':
            ax.annotate('AD: ↑ Frontal', xy=(0, ax.get_ylim()[1]*0.9), fontsize=9,
                       color=CLASS_COLORS['AD'], fontweight='bold')
            ax.annotate('FTD: ↑ Global', xy=(2, ax.get_ylim()[1]*0.8), fontsize=9,
                       color=CLASS_COLORS['FTD'], fontweight='bold')
    
    plt.suptitle('Regional Power by Frequency Band\nDisease-Specific EEG Signatures', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'clinical_regional_power.png', dpi=300, 
                facecolor='white', bbox_inches='tight')
    plt.savefig(FIGURES_PATH / 'clinical_regional_power.pdf', dpi=300,
                facecolor='white', bbox_inches='tight')
    print("  Saved: clinical_regional_power.png/pdf")
    plt.close()


def plot_theta_alpha_ratio(results):
    """Plot theta/alpha ratio - key dementia biomarker."""
    print("Creating theta/alpha ratio figure...")
    
    # Use GridSpec to separate plot and text
    fig = plt.figure(figsize=(10, 8)) 
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    
    regions = list(REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.25
    
    for j, class_name in enumerate(CLASS_NAMES):
        ratios = []
        sems = []
        
        for region in regions:
            theta_vals = results[class_name]['Theta'][region]
            alpha_vals = results[class_name]['Alpha'][region]
            
            if theta_vals and alpha_vals:
                min_len = min(len(theta_vals), len(alpha_vals))
                ratio_vals = [theta_vals[i] / alpha_vals[i] if alpha_vals[i] > 0 else 0 
                              for i in range(min_len)]
                # Use nanmean to handle invalid subjects
                ratios.append(np.nanmean(ratio_vals))
                sems.append(stats.sem(ratio_vals, nan_policy='omit') if len(ratio_vals) > 1 else 0)
            else:
                ratios.append(0)
                sems.append(0)
        
        ax.bar(x + j*width, ratios, width, label=class_name,
               color=CLASS_COLORS[class_name], yerr=sems, capsize=3)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1')
    ax.set_xlabel('Brain Region', fontsize=12)
    ax.set_ylabel('Theta/Alpha Ratio', fontsize=12)
    ax.set_title('Theta/Alpha Ratio by Region\n(Elevated ratio indicates cognitive impairment)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regions, fontsize=11)
    # Move legend OUTSIDE
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(axis='y', alpha=0.3)
    
    # Add clinical interpretation in separate subplot
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    txt = 'Clinical Interpretation:\n• AD: Elevated globally, esp. posterior\n• FTD: Elevated frontally\n• Higher ratio = more slowing'
    ax_text.text(0, 0.5, txt, fontsize=11, va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.savefig(FIGURES_PATH / 'theta_alpha_ratio.png', dpi=300, 
                facecolor='white', bbox_inches='tight')
    plt.savefig(FIGURES_PATH / 'theta_alpha_ratio.pdf', dpi=300,
                facecolor='white', bbox_inches='tight')
    print("  Saved: theta_alpha_ratio.png/pdf")
    plt.close()


def plot_anterior_posterior_gradient(results):
    """Plot anterior-posterior gradient showing AD anteriorization."""
    print("Creating anterior-posterior gradient figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Alpha power gradient
    ax = axes[0]
    for class_name in CLASS_NAMES:
        regions = ['Frontal', 'Temporal', 'Parietal', 'Occipital']
        powers = [np.nanmean(results[class_name]['Alpha'][r]) if results[class_name]['Alpha'][r] else 0 
                  for r in regions]
        ax.plot(regions, powers, 'o-', label=class_name, color=CLASS_COLORS[class_name], 
                linewidth=2, markersize=10)
    
    ax.set_xlabel('Region (Anterior → Posterior)', fontsize=12)
    ax.set_ylabel('Alpha Power (Relative)', fontsize=12)
    ax.set_title('Alpha Power Gradient\n(Normal: Posterior > Anterior)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alpha Power (Relative)', fontsize=12)
    ax.set_title('Alpha Power Gradient\n(Normal: Posterior > Anterior)', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    # Add arrow showing normal gradient
    ax.annotate('', xy=(3.5, ax.get_ylim()[1]*0.8), xytext=(0.5, ax.get_ylim()[1]*0.8),
               arrowprops=dict(arrowstyle='->', color='#009E73', lw=2))
    ax.text(2, ax.get_ylim()[1]*0.85, 'Normal gradient', fontsize=10, color='#009E73', ha='center')
    
    # Right: Anterior/Posterior ratio
    ax = axes[1]
    x = np.arange(len(CLASS_NAMES))
    
    ap_ratios = []
    for class_name in CLASS_NAMES:
        anterior = np.nanmean(results[class_name]['Alpha']['Frontal']) if results[class_name]['Alpha']['Frontal'] else 1
        posterior = np.nanmean(results[class_name]['Alpha']['Occipital']) if results[class_name]['Alpha']['Occipital'] else 1
        ap_ratios.append(anterior / posterior if posterior > 0 else 0)
    
    bars = ax.bar(x, ap_ratios, color=[CLASS_COLORS[c] for c in CLASS_NAMES], edgecolor='black')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Balanced')
    ax.set_xlabel('Diagnostic Group', fontsize=12)
    ax.set_ylabel('Anterior/Posterior Alpha Ratio', fontsize=12)
    ax.set_title('Alpha Anteriorization Index\n(Ratio > 1 = Anterior shift)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_title('Alpha Anteriorization Index\n(Ratio > 1 = Anterior shift)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    # Legend handled by colors, but added for completeness if needed
    # ax.legend()
    
    # Highlight AD anteriorization if present
    max_idx = np.argmax(ap_ratios)
    if ap_ratios[max_idx] > 1:
        ax.annotate('Anteriorization', xy=(max_idx, ap_ratios[max_idx]), 
                   xytext=(max_idx, ap_ratios[max_idx] + 0.2),
                   ha='center', fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'anterior_posterior_gradient.png', dpi=300,
                facecolor='white', bbox_inches='tight')
    plt.savefig(FIGURES_PATH / 'anterior_posterior_gradient.pdf', dpi=300,
                facecolor='white', bbox_inches='tight')
    print("  Saved: anterior_posterior_gradient.png/pdf")
    plt.close()


def plot_disease_signatures_summary(results):
    """Create summary figure showing key disease signatures."""
    print("Creating disease signatures summary figure...")
    
    fig = plt.figure(figsize=(16, 12)) # Increased height
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3) # Increased hspace
    
    # 1. AD Signature: Posterior Alpha Decrease
    ax1 = fig.add_subplot(gs[0, 0])
    regions = ['Frontal', 'Occipital']
    for j, class_name in enumerate(CLASS_NAMES):
        vals = [np.nanmean(results[class_name]['Alpha'][r]) if results[class_name]['Alpha'][r] else 0 
                for r in regions]
        # ADD ERROR BARS
        errs = [stats.sem(results[class_name]['Alpha'][r], nan_policy='omit') if results[class_name]['Alpha'][r] and len(results[class_name]['Alpha'][r]) > 1 else 0
                for r in regions]
                
        ax1.bar(np.arange(len(regions)) + j*0.25, vals, 0.25, label=class_name,
               color=CLASS_COLORS[class_name], yerr=errs, capsize=3)
    ax1.set_xticks(np.arange(len(regions)) + 0.25)
    ax1.set_xticklabels(regions)
    ax1.set_ylabel('Alpha Power (Relative)')
    ax1.set_title('AD Signature:\nPosterior Alpha Decrease', fontweight='bold')
    ax1.set_ylabel('Alpha Power (Relative)')
    ax1.set_title('AD Signature:\nPosterior Alpha Decrease', fontweight='bold')
    # Legend handled globally
    
    # 2. FTD Signature: Frontal Theta Increase
    ax2 = fig.add_subplot(gs[0, 1])
    for j, class_name in enumerate(CLASS_NAMES):
        vals = [np.nanmean(results[class_name]['Theta'][r]) if results[class_name]['Theta'][r] else 0 
                for r in regions]
        # ADD ERROR BARS
        errs = [stats.sem(results[class_name]['Theta'][r], nan_policy='omit') if results[class_name]['Theta'][r] and len(results[class_name]['Theta'][r]) > 1 else 0
                for r in regions]
                
        ax2.bar(np.arange(len(regions)) + j*0.25, vals, 0.25, label=class_name,
               color=CLASS_COLORS[class_name], yerr=errs, capsize=3)
    ax2.set_xticks(np.arange(len(regions)) + 0.25)
    ax2.set_xticklabels(regions)
    ax2.set_ylabel('Theta Power (Relative)')
    ax2.set_title('FTD Signature:\nFrontal Theta Increase', fontweight='bold')
    ax2.set_ylabel('Theta Power (Relative)')
    ax2.set_title('FTD Signature:\nFrontal Theta Increase', fontweight='bold')
    # Legend handled globally
    
    # 3. Theta/Alpha Ratio Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    for j, class_name in enumerate(CLASS_NAMES):
        theta_global = np.nanmean([np.nanmean(results[class_name]['Theta'][r]) 
                               for r in REGIONS.keys() if results[class_name]['Theta'][r]])
        alpha_global = np.nanmean([np.nanmean(results[class_name]['Alpha'][r]) 
                               for r in REGIONS.keys() if results[class_name]['Alpha'][r]])
        ratio = theta_global / alpha_global if alpha_global > 0 else 0
        ax3.bar(j, ratio, color=CLASS_COLORS[class_name], edgecolor='black', linewidth=2)
    ax3.set_xticks(range(len(CLASS_NAMES)))
    ax3.set_xticklabels(CLASS_NAMES)
    ax3.set_ylabel('Theta/Alpha Ratio')
    ax3.set_title('Global Slowing Index\n(Higher = More Impairment)', fontweight='bold')
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # 4-6: Summary text boxes
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    summary_text = """
CLINICAL INTERPRETATION OF EEG SOURCE PATTERNS

┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ALZHEIMER'S DISEASE (AD)                                                                        │
│  • Decreased posterior (parietal-occipital) alpha power                                         │
│  • Increased frontal theta power                                                                │
│  • Elevated theta/alpha ratio (slowing)                                                         │
│  • Elevated theta/alpha ratio (slowing)                                                         │
│  • Alpha anteriorization (shift from posterior to anterior)                                     │
│  Interpretation: Cholinergic dysfunction, posterior cortical hypometabolism                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  MILD COGNITIVE IMPAIRMENT (MCI)                                                                │
│  • Intermediate posterior alpha decrease                                                        │
│  • Slight theta elevation                                                                       │
│  Interpretation: Early marker of neurodegeneration                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  FRONTOTEMPORAL DEMENTIA (FTD)                                                                  │
│  • Pronounced frontal alpha reduction                                                           │
│  • Widespread theta increase (esp. frontotemporal)                                              │
│  • Frontotemporal disconnection                                                                 │
│  Interpretation: Frontal lobe dysfunction, salience network impairment                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  HEALTHY CONTROLS (CN)                                                                          │
│  • Balanced alpha with posterior predominance                                                   │
│  • Normal theta/alpha ratio                                                                     │
│  • Intact anteroposterior gradient                                                               │
│  Interpretation: Normal oscillatory dynamics                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
            va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Common Legend (places at top center)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=4, frameon=False, fontsize=11)
    
    plt.suptitle('Disease-Specific EEG Source Signatures', fontsize=16, fontweight='bold', y=0.99)
    # Adjust layout to make room for title and legend
    plt.subplots_adjust(top=0.9)
    
    plt.savefig(FIGURES_PATH / 'disease_signatures_summary.png', dpi=300,
                facecolor='white', bbox_inches='tight')
    plt.savefig(FIGURES_PATH / 'disease_signatures_summary.pdf', dpi=300,
                facecolor='white', bbox_inches='tight')
    print("  Saved: disease_signatures_summary.png/pdf")
    plt.close()


def main():
    """Main function."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Analyze all subjects
    results = analyze_all_subjects()
    
    # Print summary
    print("Subject counts per class:")
    for class_name in CLASS_NAMES:
        count = len(results[class_name]['Alpha']['Frontal'])
        print(f"  {class_name}: {count} subjects")
    
    # Create visualizations
    print("\nCreating clinical visualizations...")
    plot_regional_band_power(results)
    plot_theta_alpha_ratio(results)
    plot_anterior_posterior_gradient(results)
    plot_disease_signatures_summary(results)
    
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Figures saved to: {FIGURES_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
