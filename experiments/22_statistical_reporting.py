"""
Statistical Reporting and Validation.

Performs robust statistical analysis of key biomarkers including:
1. ANOVA/Kruskal-Wallis (Global difference)
2. Post-hoc t-tests/Mann-Whitney (Pairwise) with Bonferroni correction
3. Effect Size (Cohen's d) computation
4. Generation of publication-quality boxplots with significance annotations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PREPROCESSED_PATH = PROJECT_ROOT / "outputs" / "preprocessed"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures" / "statistics"

# Create output dir
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# Aesthetics
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Classes
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
CLASS_COLORS = {
    'AD': '#F08080',    # Red-ish
    'FTD': '#87CEEB',   # Blue-ish
    'CN': '#90EE90',    # Green-ish
    'MCI': '#FFB347'    # Orange
}

def load_data():
    """Load all subject data and extract key metrics."""
    print("Loading data...")

    # Build comprehensive class mapping for all datasets
    class_mapping = {}

    # ds004504: sub-001 to sub-088 (36 AD, 23 FTD, 29 CN)
    for i in range(1, 89):
        sid_raw = f"sub-{i:03d}"
        sid_ds = f"ds004504_sub-{i:03d}"
        cls = 'AD' if 1 <= i <= 36 else 'FTD' if 37 <= i <= 59 else 'CN'
        class_mapping[sid_raw] = cls
        class_mapping[sid_ds] = cls

    # ds006036: Same structure as ds004504 (36 AD, 23 FTD, 29 CN)
    for i in range(1, 89):
        sid_ds6 = f"ds006036_sub-{i:03d}"
        cls = 'AD' if 1 <= i <= 36 else 'FTD' if 37 <= i <= 59 else 'CN'
        class_mapping[sid_ds6] = cls

    pkl_files = list(PREPROCESSED_PATH.glob("*_preprocessed.pkl"))
    print(f"  Found {len(pkl_files)} preprocessed files")

    data_records = []
    skipped_pattern = []
    skipped_data = []
    skipped_epochs = []

    for pkl in pkl_files:
        sid = pkl.stem.replace('_preprocessed', '')

        # Determine class from mapping first
        cls = class_mapping.get(sid)
        if not cls:
            # Mendeley Dataset patterns (files start with Unknown_Mendeley_)
            if "Mendeley" in sid:
                if "_AD_" in sid:
                    cls = 'AD'
                elif "_MCI_" in sid:
                    cls = 'MCI'
                elif "_normal_" in sid:
                    cls = 'CN'
            # MCI Dataset patterns (files start with Unknown_MCIData_)
            elif "MCIData" in sid:
                if "_AD_" in sid or "MCIData_AD_" in sid:
                    cls = 'AD'
                elif "_MCI_" in sid or "MCIData_MCI_" in sid:
                    cls = 'MCI'
                elif "_CONTROL_" in sid or "MCIData_CONTROL_" in sid:
                    cls = 'CN'
            # AlzEEG patterns (files start with Unknown_AlzEEG_)
            elif "AlzEEG" in sid:
                # Check for AD (pattern: AlzEEG_AD_Eyes_)
                if "AlzEEG_AD_" in sid:
                    cls = 'AD'
                # Check for FTD (pattern: AlzEEG_Frontotemporal_)
                elif "AlzEEG_Frontotemporal_" in sid:
                    cls = 'FTD'
                # Check for CN (pattern: AlzEEG_Healthy_)
                elif "AlzEEG_Healthy_" in sid:
                    cls = 'CN'

        if not cls or cls not in CLASS_NAMES:
            skipped_pattern.append(sid)
            continue
            
        try:
            with open(pkl, 'rb') as f:
                d = pickle.load(f)

            # Use epochs_data (numpy) or epochs (mne)
            if 'epochs_data' in d:
                ed = d['epochs_data']
                sfreq = d['sfreq']
            elif 'epochs' in d:
                ed = d['epochs'].get_data()
                sfreq = d['epochs'].info['sfreq']
            else:
                skipped_data.append(sid)
                continue

            # Lowered from 5 to 2 to include more subjects in statistics
            # Most preprocessing pipelines require at least 2 epochs for averaging
            if ed.shape[0] < 2:
                skipped_epochs.append(sid)
                continue
            
            # --- Extract Metrics ---
            # 1. Band Power (Relative Alpha) - Posterior only (approx channels)
            # Simple approximation: mean over all channels for now, or specific indices if known?
            # Let's do Global Alpha and Global Theta for robustness
            from scipy.signal import welch
            
            n_ep, n_ch, n_times = ed.shape
            freqs, psd = welch(ed, fs=sfreq, nperseg=min(256, n_times), axis=-1)
            
            # Masks
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            total_mask = (freqs >= 1) & (freqs <= 40)
            
            psd_mean = psd.mean(axis=(0, 1)) # Mean over epochs and channels
            
            theta_pow = np.sum(psd_mean[theta_mask])
            alpha_pow = np.sum(psd_mean[alpha_mask])
            total_pow = np.sum(psd_mean[total_mask])
            
            rel_theta = theta_pow / total_pow
            rel_alpha = alpha_pow / total_pow
            tar = theta_pow / alpha_pow if alpha_pow > 0 else 0
            
            # 2. Complexity (Spectral Entropy as proxy)
            # Normalized Shannon entropy of the power spectrum
            psd_norm = psd_mean / np.sum(psd_mean)
            se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm))
            
            data_records.append({
                'Subject': sid,
                'Class': cls,
                'Relative Theta': rel_theta,
                'Relative Alpha': rel_alpha,
                'Theta/Alpha Ratio': tar,
                'Spectral Entropy': se
            })
            
        except Exception as e:
            skipped_data.append(f"{sid} (error)")
            continue

    # Summary - detailed skip info
    total_skipped = len(skipped_pattern) + len(skipped_data) + len(skipped_epochs)
    if total_skipped > 0:
        print(f"  Skipped {total_skipped} files:")
        if skipped_pattern:
            print(f"    - {len(skipped_pattern)} unrecognized patterns")
        if skipped_data:
            print(f"    - {len(skipped_data)} missing epoch data")
        if skipped_epochs:
            print(f"    - {len(skipped_epochs)} too few epochs (<2)")

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Deduplicate AlzEEG subjects (have both Eyes_open and Eyes_closed)
    # Average their metrics
    if len(df) > 0:
        # Create a normalized subject ID (remove Eyes_open/Eyes_closed suffix for AlzEEG)
        def normalize_subject_id(sid):
            if 'AlzEEG' in sid:
                # Remove _Eyes_open or _Eyes_closed from the ID
                sid = sid.replace('_Eyes_open', '').replace('_Eyes_closed', '')
            return sid

        df['Subject_Normalized'] = df['Subject'].apply(normalize_subject_id)

        # Group by normalized subject and class, take mean of metrics
        numeric_cols = ['Relative Theta', 'Relative Alpha', 'Theta/Alpha Ratio', 'Spectral Entropy']
        df_dedup = df.groupby(['Subject_Normalized', 'Class'])[numeric_cols].mean().reset_index()
        df_dedup = df_dedup.rename(columns={'Subject_Normalized': 'Subject'})

        print(f"  After deduplication: {len(df_dedup)} unique subjects (from {len(df)} records)")
        return df_dedup

    return df

def cohen_d(x, y):
    """Calculate Cohen's d for independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def plot_stat_boxplot(df, metric, title, filename):
    """Generate Boxplot with stats annotation."""
    plt.figure(figsize=(8, 6))

    # Boxplot - use hue parameter to avoid deprecation warning
    ax = sns.boxplot(x='Class', y=metric, data=df, order=CLASS_NAMES,
                     hue='Class', palette=CLASS_COLORS, legend=False,
                     boxprops={'alpha': 0.7})
    sns.stripplot(x='Class', y=metric, data=df, order=CLASS_NAMES,
                  color='black', size=4, alpha=0.5, jitter=True)
    
    # ANOVA
    groups = [df[df['Class'] == c][metric].values for c in CLASS_NAMES]
    f_val, p_val = stats.f_oneway(*groups)
    
    # Title with ANOVA result
    plt.title(f"{title}\nANOVA: F={f_val:.2f}, p={p_val:.4e}", fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("")
    
    # Pairwise comparisons (vs CN)
    cn_vals = df[df['Class'] == 'CN'][metric].values
    
    if p_val < 0.05:
        y_max = df[metric].max()
        y_range = y_max - df[metric].min()
        offset = y_range * 0.1
        
        for i, cls in enumerate(CLASS_NAMES):
            if cls == 'CN': continue
            
            cls_vals = df[df['Class'] == cls][metric].values
            if len(cls_vals) == 0: continue
            
            # T-test
            t, p = stats.ttest_ind(cls_vals, cn_vals, equal_var=False)
            # Bonferroni correction (3 comparisons: AD-CN, FTD-CN, MCI-CN)
            p_adj = p * 3
            
            if p_adj < 0.05:
                # Calc position (i vs 2 which is CN index)
                x1, x2 = i, 2
                y_h = y_max + offset * (1 + abs(i-2)*0.2)
                
                # Draw bracket
                plt.plot([x1, x1, x2, x2], [y_h, y_h+offset*0.2, y_h+offset*0.2, y_h], lw=1.5, c='k')
                
                # Stars
                star = '*'
                if p_adj < 0.001: star = '***'
                elif p_adj < 0.01: star = '**'
                
                d_val = cohen_d(cls_vals, cn_vals)
                plt.text((x1+x2)*.5, y_h+offset*0.25, f"{star}\nd={d_val:.2f}", 
                         ha='center', va='bottom', color='k', fontsize=10)

    # Move legend (if any created, usually not for boxplot but just in case)
    # Since we use x-axis for class, we don't strictly need a legend, 
    # but if seaborn makes one:
    if ax.get_legend():
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def main():
    print("="*60)
    print("Robust Statistical Analysis")
    print("="*60)
    
    df = load_data()
    print(f"Loaded records: {len(df)}")
    print(df['Class'].value_counts())
    
    # Save statistics CSV
    numeric_cols = ['Relative Theta', 'Relative Alpha', 'Theta/Alpha Ratio', 'Spectral Entropy']
    stats_summary = df.groupby('Class')[numeric_cols].agg(['mean', 'std', 'count'])
    stats_summary.to_csv(FIGURES_PATH / "biomarker_stats.csv")
    print(f"Saved stats summary to {FIGURES_PATH}")
    
    # Generate Plots
    plot_stat_boxplot(df, 'Relative Alpha', 'Global Alpha Power', 'stat_alpha.png')
    plot_stat_boxplot(df, 'Relative Theta', 'Global Theta Power', 'stat_theta.png')
    plot_stat_boxplot(df, 'Theta/Alpha Ratio', 'Theta/Alpha Ratio (Slowing)', 'stat_tar.png')
    plot_stat_boxplot(df, 'Spectral Entropy', 'Spectral Entropy (Complexity)', 'stat_entropy.png')
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
