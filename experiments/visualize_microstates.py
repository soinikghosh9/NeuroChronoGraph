"""
Publication-Quality Microstate Analysis Visualization.

Generates:
1. Global microstate topographies (MS-A/B/C/D).
2. Per-subject parameters (Coverage, Duration, Occurrence).
3. Group comparisons (AD vs FTD vs CN vs MCI).
4. Statistical annotations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import seaborn as sns
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset_factory import DatasetFactory
from src.data.preprocessor import create_epochs
from src.features.microstates import MicrostateAnalyzer
from src.config.config import FIGURES_PATH, CHANNEL_NAMES, VISUALIZATION_CONFIG
from src.utils.visualization import setup_plotting_style

# Apply global style
setup_plotting_style()

# Color scheme for groups (4 classes) from Config
GROUP_COLORS = VISUALIZATION_CONFIG['palette']
MS_LABELS = ['MS-A\n(Left-Right)', 'MS-B\n(Occipital)', 'MS-C\n(Salience)', 'MS-D\n(Attention)']

def get_group_name(label_idx):
    """Map integer label to string."""
    # 0=AD, 1=FTD, 2=CN, 3=MCI
    mapping = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
    return mapping.get(label_idx, 'Unknown')

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def run_microstate_analysis():
    print("=" * 70)
    print("PUBLICATION-QUALITY MICROSTATE ANALYSIS (Unified)")
    print("=" * 70)
    
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Factory and Loaders
    print("\n[1/5] Initializing datasets...")
    factory = DatasetFactory()
    datasets = {
        'ds004504': project_root / "datasets" / "openneuro_ds004504",
        'ds006036': project_root / "datasets" / "ds006036",
        'Alz_EEG': project_root / "datasets" / "Alz_EEG_data",
        'Mendeley': project_root / "datasets" / "Mendeley Dataset",
        'MCI_Dataset': project_root / "datasets" / "mci dataset"
    }
    for name, path in datasets.items():
        if path.exists():
            try:
                factory.add_dataset(name, path)
            except Exception as e:
                print(f"Failed to add {name}: {e}")
    
    # 2. Iterate and Load Raw Data
    all_epochs_list = []
    group_epochs = {'AD': [], 'FTD': [], 'CN': [], 'MCI': []}
    
    print("\n[2/5] Loading and epoching data (this may take a while)...")
    
    # We loop through loaders directly
    for loader in factory.datasets:
        print(f"  Processing {loader.__class__.__name__}...")
        ids = loader.get_subject_ids()
        
        # Limit for speed if needed, but per request "all experiments"
        # We'll try to process all.
        
        for sid in ids:
            try:
                info = loader.get_subject_info(sid)
                grp_name = get_group_name(info['label'])
                
                if grp_name not in group_epochs:
                    continue
                    
                raw = loader.load_raw(sid)
                # Filter to std 1-40Hz for microstates
                raw.filter(1.0, 40.0, verbose=False)
                
                # Create standard epochs
                epochs = create_epochs(raw, duration=2.0, overlap=0.0, reject=None, flat=None)
                
                if len(epochs) > 0:
                    # Resample to common 100Hz for speed/consistency
                    epochs.resample(100.0, verbose=False)
                    # Strict crop to avoid shape mismatch (e.g. 201 vs 200 samples)
                    # 2.0s * 100Hz = 200 samples
                    if epochs.times[-1] >= 1.99:
                         epochs.crop(tmin=0, tmax=1.99, include_tmax=False)
                    
                    # Pick common channels (19) or intersect
                    epochs.pick(CHANNEL_NAMES, verbose=False)
                    
                    all_epochs_list.append(epochs)
                    group_epochs[grp_name].append((sid, epochs))
                    
            except Exception as e:
                print(f"    Skipping {sid}: {e}")
                pass

    if not all_epochs_list:
        print("ERROR: No data loaded.")
        return

    # Counts
    for g, lst in group_epochs.items():
        print(f"  {g}: {len(lst)} subjects")
        
    # ---------------------------------------------------------
    # Fit Global Microstate Model
    # ---------------------------------------------------------
    print("\n[3/5] Fitting global microstate model (Concatenated)...")
    
    # Subsample epochs for global map fitting to avoid OOM
    all_epochs = mne.concatenate_epochs(all_epochs_list, verbose=False)
    info = all_epochs.info
    
    np.random.seed(42)
    max_fit = min(2000, len(all_epochs))
    indices = np.random.choice(len(all_epochs), max_fit, replace=False)
    fit_epochs = all_epochs[sorted(indices)]
    
    analyzer = MicrostateAnalyzer(n_states=4, random_state=42, verbose=False)
    
    try:
        analyzer.fit(fit_epochs, n_runs=5) # Reduced runs for speed
        centers = analyzer.cluster_centers
        print(f"  Fitted 4 microstate classes successfully")
    except Exception as e:
        print(f"  FIT ERROR: {e}")
        return

    # ---------------------------------------------------------
    # Compute Parameters
    # ---------------------------------------------------------
    print("\n[4/5] Computing subject-level parameters...")
    results = []
    
    for group_name, subj_list in group_epochs.items():
        if not subj_list: continue
        print(f"  Analyzing {group_name}...")
        for sub_id, epochs in subj_list:
            try:
                labels = analyzer.segment(epochs)
                params = analyzer.compute_parameters()
                
                cov = params['coverage']
                dur = params['duration']
                occ = params['occurrence']
                
                for i in range(4):
                    results.append({
                        'Subject': sub_id,
                        'Group': group_name,
                        'Microstate': f'MS-{chr(65+i)}',
                        'MS_Index': i,
                        'Coverage': cov[i] * 100,
                        'Duration_ms': dur[i] * 1000,
                        'Occurrence': occ[i]
                    })
            except:
                pass

    df = pd.DataFrame(results)
    if df.empty:
        print("Error: No params computed.")
        return

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    print("\n[5/5] Generating Plots...")
    
    # Plot 1: Topographies
    fig = plt.figure(figsize=(16, 12))
    
    # Top: Maps
    for i in range(4):
        ax = fig.add_subplot(4, 4, i+1)
        mne.viz.plot_topomap(centers[i], info, axes=ax, show=False, contours=6, cmap='RdBu_r')
        ax.set_title(MS_LABELS[i], weight='bold')

    # Rows: Coverage, Duration, Occurrence
    metrics = [('Coverage', '%'), ('Duration_ms', 'ms'), ('Occurrence', 'Hz')]
    
    for r_idx, (metric, unit) in enumerate(metrics):
        ax = fig.add_subplot(4, 1, r_idx+2)
        
        # Stats Aggregation
        stats_df = df.groupby(['Group', 'MS_Index'])[metric].agg(['mean', 'std', 'count']).reset_index()
        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        
        x = np.arange(4)
        width = 0.2
        GROUPS = ['CN', 'MCI', 'AD', 'FTD'] # Ordered
        
        for g_idx, grp_name in enumerate(GROUPS):
            if grp_name not in stats_df['Group'].values: continue
            
            grp_data = stats_df[stats_df['Group'] == grp_name].sort_values('MS_Index')
            ax.bar(x + g_idx*width, grp_data['mean'], width, 
                   yerr=grp_data['se'], label=grp_name, color=GROUP_COLORS[grp_name],
                   capsize=3, edgecolor='k', linewidth=0.5)
        
        ax.set_ylabel(f'{metric} ({unit})')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(['MS-A', 'MS-B', 'MS-C', 'MS-D'])
        if r_idx == 0:
            # Place legend outside plot area to prevent overlap
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'Microstate {metric} by Group')

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Make room for legend
    plt.savefig(FIGURES_PATH / "microstates_4class_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: microstates_4class_analysis.png")
    plt.close()

if __name__ == "__main__":
    print("Starting Main Execution...", flush=True)
    run_microstate_analysis()
    print("Main Execution Completed.", flush=True)
