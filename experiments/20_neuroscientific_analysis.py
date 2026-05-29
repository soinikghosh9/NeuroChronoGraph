"""
Neuroscientific Analysis & Visualization Script (Hierarchical + Flat).

Generates:
1. Flat 4-Class Confusion Matrix (CN, MCI, AD, FTD) via Chain Rule.
2. Hierarchical Confusion Matrices (Screening, Staging, Subtyping).
3. Connectivity Heatmaps (Adjacency & Microstates).
4. Band Coupling Analysis (Cross-Band Attention).
5. Temporal Attention Profiles.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import mne

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Publication-Ready Plot Style definitions moved to src.config.config
from src.config.config import (
    DATA_ROOT, DEVICE, CHANNEL_NAMES, VISUALIZATION_CONFIG
)
from src.data.dataset_factory import DatasetFactory
from src.models.v2.neuro_chrono_graph_v2 import create_neuro_chrono_graph_v2
from src.utils.evaluation import plot_confusion_matrix
from src.utils.visualization import setup_plotting_style
from src.visualization.style_config import safe_tight_layout

# Apply global style
setup_plotting_style()

def load_data_and_model(fold=0):
    """Load model and validation data for a specific fold."""
    # ... (unchanged) ...
    # 1. Load Data
    factory = DatasetFactory()
    datasets = {
        'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
        'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
        'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
        'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
        'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset"  # New MCI data source
    }
    for name, path in datasets.items():
        if path.exists():
            factory.add_dataset(name, path)
        else:
            print(f"Skipping missing dataset: {path}")

    dataset, groups, labels = factory.create_torch_datasets(config={'n_times': 2000})
    
    # ---------------------------------------------------------
    # OPTIMIZED DATA SELECTION
    # user reported "stuck" on full dataset.
    # We will use ALL MCI/FTD (rare) and subsample AD/CN (abundant).
    # ---------------------------------------------------------
    print("Selecting balanced subset for analysis...")
    from torch.utils.data import Subset
    import numpy as np
    
    # logical indices (labels are tensors in dataset, but we have 'labels' list from factory return)
    # dataset.labels is often a tensor, factory return might be list/array.
    # factory.create_torch_datasets returns (dataset, groups, labels)
    lbls = np.array(labels) 
    
    indices = np.arange(len(lbls))
    
    idx_ad = indices[lbls == 0]
    idx_ftd = indices[lbls == 1]
    idx_cn = indices[lbls == 2]
    idx_mci = indices[lbls == 3]
    
    print(f"  Found: AD={len(idx_ad)}, FTD={len(idx_ftd)}, CN={len(idx_cn)}, MCI={len(idx_mci)}")
    
    # Subsample AD and CN to speed up (keep max 50)
    np.random.seed(42)
    if len(idx_ad) > 50: idx_ad = np.random.choice(idx_ad, 50, replace=False)
    if len(idx_cn) > 50: idx_cn = np.random.choice(idx_cn, 50, replace=False)
    
    # Keep ALL MCI and FTD
    final_indices = np.concatenate([idx_ad, idx_ftd, idx_cn, idx_mci])
    print(f"  Analysis Subset: {len(final_indices)} subjects (AD={len(idx_ad)}, FTD={len(idx_ftd)}, CN={len(idx_cn)}, MCI={len(idx_mci)})")
    
    val_ds = Subset(dataset, final_indices)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False) # Smaller batch for stability
    
    # 2. Load Model
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"hierarchical_model_fold{fold}.pt"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "hierarchical_model_fold0.pt"
        if not ckpt_path.exists():
             pass
    
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        # Inspect checkpoint to detect whether it was trained with stream-fusion;
        # if so we must instantiate the model with matching feature_streams.
        state = torch.load(ckpt_path, map_location=DEVICE)
        has_streams = any(k.startswith('stream_fusion.') for k in state.keys())
        cfg = {
            'n_classes': 3,
            'hidden_dim': 128,
            'dropout': 0.5,
        }
        if has_streams:
            cfg.update({
                'feature_streams': ['spectral', 'connectivity', 'complexity', 'microstate'],
                'stream_dim': 64,
            })
        model = create_neuro_chrono_graph_v2(cfg).to(DEVICE)
        model.load_state_dict(state, strict=False)
        model.eval()
    else:
        print("WARNING: No checkpoint found! Skipping model-based analysis part.")
        model = None
    
    return model, val_loader

# ==================================================================================
# MICROSTATE ANALYSIS EXTENSION
# ==================================================================================
from src.features.microstates import MicrostateAnalyzer
from src.data.preprocessor import create_epochs

def run_microstate_analysis_internal(output_dir):
    print("\n" + "=" * 70)
    print("RUNNING MICROSTATE ANALYSIS (Integrated)")
    print("=" * 70)
    
    # 1. Initialize Factory (Fresh to ensure raw access)
    factory = DatasetFactory()
    datasets = {
        'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
        'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
        'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
        'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
        'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset"
    }
    for name, path in datasets.items():
        if path.exists():
            try:
                factory.add_dataset(name, path)
            except Exception as e:
                print(f"Failed to add {name}: {e}")

    # 2. Load Raw
    all_epochs_list = []
    group_epochs = {'AD': [], 'FTD': [], 'CN': [], 'MCI': []}
    
    print("Loading raw data for microstates (Subsampled for speed)...")
    
    # NEW: Smart limits for microstates too
    limits = {'AD': 30, 'CN': 30, 'FTD': 100, 'MCI': 100} 
    counts = {'AD': 0, 'CN': 0, 'FTD': 0, 'MCI': 0}
    
    for loader in factory.datasets:
        ids = loader.get_subject_ids()
        for sid in ids:
            try:
                info = loader.get_subject_info(sid)
                label_idx = info['label']
                mapping = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
                grp_name = mapping.get(label_idx, 'Unknown')
                
                if grp_name not in group_epochs: continue
                if counts[grp_name] >= limits[grp_name]: continue
                
                # Check MCI specifically
                if grp_name == 'MCI':
                     print(f"  Processing MCI subject: {sid}...")

                raw = loader.load_raw(sid)
                raw.filter(1.0, 40.0, verbose=False)
                epochs = create_epochs(raw, duration=2.0, overlap=0.0, reject=None, flat=None)
                
                if len(epochs) > 0:
                    epochs.resample(100.0, verbose=False)
                    epochs.pick(CHANNEL_NAMES, verbose=False)
                    
                    # Fix: Force strict 200 samples via numpy slicing
                    # crop() can be floating-point sensitive.
                    data = epochs.get_data()
                    # data shape: [n_epochs, n_channels, n_times]
                    if data.shape[2] >= 200:
                        data = data[:, :, :200]
                        # Re-create EpochsArray to ensure metadata matches data shape
                        info = epochs.info
                        events = epochs.events
                        event_id = epochs.event_id
                        epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0, verbose=False)
                    else:
                        print(f"  Warning: Epochs too short ({data.shape[2]}), skipping.")
                        continue

                    all_epochs_list.append(epochs)
                    group_epochs[grp_name].append((sid, epochs))
                    counts[grp_name] += 1
                    
            except Exception:
                pass

    if not all_epochs_list:
        print("No epochs loaded for microstates.")
        return

    # Counts
    for g, lst in group_epochs.items():
        print(f"Microstate Data: {g}: {len(lst)} subjects")

    # 3. Fit Global
    print("Fitting global microstates...")
    all_epochs = mne.concatenate_epochs(all_epochs_list, verbose=False)
    analyzer = MicrostateAnalyzer(n_states=4, random_state=42, verbose=False)
    
    # Subsample for fit
    np.random.seed(42)
    max_fit = min(1000, len(all_epochs))
    indices = np.random.choice(len(all_epochs), max_fit, replace=False)
    analyzer.fit(all_epochs[sorted(indices)], n_runs=5)
    
    # 4. Compute Params & Plot
    print("Computing metrics & Plotting...")
    results = []
    for group_name, subj_list in group_epochs.items():
        if not subj_list: continue
        for sub_id, epochs in subj_list:
            try:
                params = analyzer.segment(epochs)
                params = analyzer.compute_parameters()
                
                for i in range(4):
                    results.append({
                        'Subject': sub_id,
                        'Group': group_name,
                        'Microstate': f'MS-{chr(65+i)}',
                        'MS_Index': i,
                        'Coverage': params['coverage'][i] * 100,
                        'Duration_ms': params['duration'][i] * 1000,
                        'Occurrence': params['occurrence'][i]
                    })
            except: pass
            
    df = pd.DataFrame(results)
    if df.empty: return

    # Plot
    fig = plt.figure(figsize=(16, 12))
    MS_LABELS = ['MS-A', 'MS-B', 'MS-C', 'MS-D']
    centers = analyzer.cluster_centers
    info = all_epochs.info
    GROUP_COLORS = VISUALIZATION_CONFIG['palette']
    
    for i in range(4):
        ax = fig.add_subplot(4, 4, i+1)
        mne.viz.plot_topomap(centers[i], info, axes=ax, show=False, contours=6, cmap='RdBu_r')
        ax.set_title(MS_LABELS[i])

    metrics = [('Coverage', '%'), ('Duration_ms', 'ms'), ('Occurrence', 'Hz')]
    for r_idx, (metric, unit) in enumerate(metrics):
        ax = fig.add_subplot(4, 1, r_idx+2)
        stats_df = df.groupby(['Group', 'MS_Index'])[metric].agg(['mean', 'std', 'count']).reset_index()
        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        
        x = np.arange(4)
        width = 0.2
        GROUPS = ['CN', 'MCI', 'AD', 'FTD']
        
        for g_idx, grp_name in enumerate(GROUPS):
            if grp_name not in stats_df['Group'].values: continue
            grp_data = stats_df[stats_df['Group'] == grp_name].sort_values('MS_Index')
            ax.bar(x + g_idx*width, grp_data['mean'], width, yerr=grp_data['se'], 
                   label=grp_name, color=GROUP_COLORS[grp_name])
        
        ax.set_ylabel(metric)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(MS_LABELS)
        if r_idx == 0:
            # Place legend outside plot area to prevent overlap
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Note: Primary microstate plot generated by visualize_microstates.py (step 23)
    # Skip saving duplicate here - the main one is better quality
    plt.close()
    print("  Microstate analysis complete (main plot in figures/microstates_4class_analysis.png)")


def calculate_flat_probabilities(probs_screen, probs_stage, probs_subtype):
    """
    Apply Chain Rule to convert Hierarchical Probs to Flat 4-Class Probs.
    """
    p_healthy = probs_screen[:, 0]
    p_impaired = probs_screen[:, 1]
    
    p_mci_given_imp = probs_stage[:, 0]
    p_dem_given_imp = probs_stage[:, 1]
    
    p_ad_given_dem = probs_subtype[:, 0]
    p_ftd_given_dem = probs_subtype[:, 1]
    
    # Chain Rule
    P_CN = p_healthy
    P_MCI = p_impaired * p_mci_given_imp
    P_AD = p_impaired * p_dem_given_imp * p_ad_given_dem
    P_FTD = p_impaired * p_dem_given_imp * p_ftd_given_dem
    
    # Stack: 0=AD, 1=FTD, 2=CN, 3=MCI (Matching dataset labels)
    flat_probs = np.stack([P_AD, P_FTD, P_CN, P_MCI], axis=1)
    return flat_probs


def plot_hierarchical_decision_explainability(results_json_path, output_dir):
    """
    Generate a clinically interpretable visualization of the hierarchical decision-making process.

    Shows:
    1. Three-stage decision tree structure (Screening → Staging → Subtyping)
    2. Mean probability flow through each stage
    3. Class-specific decision paths
    4. Clinical interpretation at each node
    """
    import json
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.lines import Line2D

    # Load results
    try:
        with open(results_json_path, 'r') as f:
            results = json.load(f)
    except:
        print("  Could not load results.json for hierarchical decision visualization")
        return

    # Get holdout results for visualization
    holdout = results.get('holdout', {})
    if not holdout:
        print("  No holdout results available")
        return

    subject_results = holdout.get('subject_results', [])
    if not subject_results:
        print("  No subject results available")
        return

    # Compute stage-wise statistics from subject results
    # Group by true class
    class_stats = {
        'AD': {'screen_imp': [], 'stage_dem': [], 'sub_ad': [], 'probs': []},
        'FTD': {'screen_imp': [], 'stage_dem': [], 'sub_ftd': [], 'probs': []},
        'CN': {'screen_cn': [], 'probs': []},
        'MCI': {'screen_imp': [], 'stage_mci': [], 'probs': []}
    }
    class_map = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}

    for s in subject_results:
        true_class = class_map[s['true']]
        probs = s['probs']  # [AD, FTD, CN, MCI]
        class_stats[true_class]['probs'].append(probs)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Colors
    colors = {'AD': '#E64B35', 'FTD': '#4DBBD5', 'CN': '#00A087', 'MCI': '#3C5488'}

    # --- Panel A: Feature Importance (Discriminative Power) ---
    ax1 = fig.add_subplot(2, 2, 1)

    # Calculate feature importance based on how well each probability dimension
    # discriminates between classes (using Fisher's discriminant ratio)
    prob_features = ['P(AD)', 'P(FTD)', 'P(CN)', 'P(MCI)']
    class_order = ['AD', 'FTD', 'CN', 'MCI']

    # Collect probabilities by class
    probs_by_class = {cls: [] for cls in class_order}
    for s in subject_results:
        true_class = class_map[s['true']]
        probs_by_class[true_class].append(s['probs'])

    # Calculate discriminative power for each probability feature
    # Using between-class variance / within-class variance (Fisher ratio)
    feature_importance = []
    for feat_idx in range(4):
        # Get all values for this feature across all classes
        all_values = []
        class_means = []
        within_var = 0
        total_n = 0

        for cls in class_order:
            if probs_by_class[cls]:
                values = [p[feat_idx] for p in probs_by_class[cls]]
                all_values.extend(values)
                class_means.append(np.mean(values))
                within_var += np.var(values) * len(values)
                total_n += len(values)

        if total_n > 0:
            within_var /= total_n
            global_mean = np.mean(all_values)
            between_var = np.var(class_means) if len(class_means) > 1 else 0

            # Fisher ratio (avoid division by zero)
            fisher_ratio = between_var / (within_var + 1e-8)
            feature_importance.append(fisher_ratio)
        else:
            feature_importance.append(0)

    # Normalize to sum to 1 for interpretability
    total_importance = sum(feature_importance)
    if total_importance > 0:
        feature_importance = [f / total_importance for f in feature_importance]

    # Create horizontal bar chart
    y_pos = np.arange(len(prob_features))
    bar_colors = [colors['AD'], colors['FTD'], colors['CN'], colors['MCI']]
    bars = ax1.barh(y_pos, feature_importance, color=bar_colors, edgecolor='black', height=0.6)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(prob_features, fontsize=12)
    ax1.set_xlabel('Relative Discriminative Power', fontsize=12)
    ax1.set_title('A. Feature Importance (Fisher Discriminant Ratio)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max(feature_importance) * 1.2 if feature_importance else 1)

    # Add value labels
    for bar, val in zip(bars, feature_importance):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

    ax1.grid(axis='x', alpha=0.3)

    # --- Panel B: Class-wise Probability Distribution ---
    ax2 = fig.add_subplot(2, 2, 2)

    # Calculate mean probabilities for each class
    class_order = ['CN', 'MCI', 'AD', 'FTD']
    mean_probs = {}
    for cls in class_order:
        if class_stats[cls]['probs']:
            probs_arr = np.array(class_stats[cls]['probs'])
            mean_probs[cls] = probs_arr.mean(axis=0)  # [AD, FTD, CN, MCI]
        else:
            mean_probs[cls] = np.zeros(4)

    x = np.arange(4)
    width = 0.2
    prob_labels = ['P(AD)', 'P(FTD)', 'P(CN)', 'P(MCI)']

    for i, cls in enumerate(class_order):
        ax2.bar(x + i*width, mean_probs[cls], width, label=f'True {cls}',
                color=colors[cls], alpha=0.8, edgecolor='black')

    ax2.set_xticks(x + 1.5*width)
    ax2.set_xticklabels(prob_labels, fontsize=11)
    ax2.set_ylabel('Mean Predicted Probability', fontsize=12)
    ax2.set_title('B. Mean Predicted Probabilities by True Class', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax2.grid(axis='y', alpha=0.3)

    # --- Panel C: Stage-wise Performance Analysis ---
    ax3 = fig.add_subplot(2, 2, 3)

    # Calculate stage-wise performance from subject results
    screening_correct, screening_total = 0, 0
    staging_correct, staging_total = 0, 0
    subtyping_correct, subtyping_total = 0, 0

    for s in subject_results:
        true_class = s['true']  # 0=AD, 1=FTD, 2=CN, 3=MCI
        pred_class = s['pred']
        probs = s['probs']  # [AD, FTD, CN, MCI]

        # Screening: CN (2) vs Impaired (0,1,3)
        true_impaired = true_class != 2
        pred_impaired = pred_class != 2
        screening_total += 1
        if true_impaired == pred_impaired:
            screening_correct += 1

        # Staging: Only for impaired subjects (MCI=3 vs Dementia=0,1)
        if true_class != 2:  # True impaired
            staging_total += 1
            true_dementia = true_class in [0, 1]
            pred_dementia = pred_class in [0, 1]
            if true_dementia == pred_dementia:
                staging_correct += 1

            # Subtyping: Only for dementia subjects (AD=0 vs FTD=1)
            if true_class in [0, 1]:  # True dementia
                subtyping_total += 1
                if true_class == pred_class:
                    subtyping_correct += 1

    # Calculate accuracies
    screen_acc = screening_correct / screening_total * 100 if screening_total > 0 else 0
    stage_acc = staging_correct / staging_total * 100 if staging_total > 0 else 0
    subtype_acc = subtyping_correct / subtyping_total * 100 if subtyping_total > 0 else 0

    # Create bar chart
    stages = ['Screening\n(CN vs Impaired)', 'Staging\n(MCI vs Dementia)', 'Subtyping\n(AD vs FTD)']
    accuracies = [screen_acc, stage_acc, subtype_acc]
    stage_colors = ['#90EE90', '#87CEEB', '#DDA0DD']

    bars = ax3.bar(stages, accuracies, color=stage_colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('C. Stage-wise Classification Performance', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance Level')

    # Add value labels on bars
    for bar, acc, n in zip(bars, accuracies, [screening_total, staging_total, subtyping_total]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- Panel D: Decision Confidence Distribution ---
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate confidence (max probability) for correct vs incorrect predictions
    correct_confidences = []
    incorrect_confidences = []
    class_confidences = {cls: [] for cls in ['AD', 'FTD', 'CN', 'MCI']}

    for s in subject_results:
        true_class = s['true']
        pred_class = s['pred']
        probs = s['probs']  # [AD, FTD, CN, MCI]
        confidence = max(probs)  # Max probability as confidence
        true_class_name = class_map[true_class]

        class_confidences[true_class_name].append(confidence)

        if true_class == pred_class:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)

    # Create violin plot for confidence distribution
    data_to_plot = []
    labels = []
    plot_colors = []

    if correct_confidences:
        data_to_plot.append(correct_confidences)
        labels.append(f'Correct\n(n={len(correct_confidences)})')
        plot_colors.append('#4CAF50')

    if incorrect_confidences:
        data_to_plot.append(incorrect_confidences)
        labels.append(f'Incorrect\n(n={len(incorrect_confidences)})')
        plot_colors.append('#F44336')

    if data_to_plot:
        parts = ax4.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmeans=True, showmedians=True)

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)

        # Style mean/median lines
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)

        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, fontsize=11)
        ax4.set_ylabel('Prediction Confidence (Max Probability)', fontsize=12)
        ax4.set_title('D. Decision Confidence Distribution', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1.05)
        ax4.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
        ax4.grid(axis='y', alpha=0.3)

        # Add mean annotations
        if correct_confidences:
            mean_corr = np.mean(correct_confidences)
            ax4.text(0, mean_corr + 0.05, f'Mean: {mean_corr:.2f}', ha='center', fontsize=9)
        if incorrect_confidences:
            mean_incorr = np.mean(incorrect_confidences)
            ax4.text(1 if len(data_to_plot) > 1 else 0, mean_incorr + 0.05, f'Mean: {mean_incorr:.2f}', ha='center', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No prediction data available', ha='center', va='center', fontsize=12)
        ax4.axis('off')

    # Use subplots_adjust instead of tight_layout to avoid colorbar conflict
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, wspace=0.3, hspace=0.35)

    # Save
    out_file = output_dir / "hierarchical_decision_explainability.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_file}")


# Import high-quality visualization modules
from src.visualization.manuscript_figures import generate_all_manuscript_figures

def run_analysis(model, loader, output_dir):
    """Run full analysis pipeline."""
    
    metrics_by_class = {0: {}, 1: {}, 2: {}, 3: {}} # AD, FTD, CN, MCI
    for k in metrics_by_class:
        metrics_by_class[k] = {'adj': [], 'band': [], 'temp': [], 'att_corr': []}

    all_targets = []
    
    # Prob lists
    list_screen, list_stage, list_subtype = [], [], []
    
    print("Running Inference & Explanation Extraction...")
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            meta = batch['metadata'].to(DEVICE)
            # band_features dict: {'delta': [B, T, N], ...} (assuming shape from loader)
            band_features = {k: v.to(DEVICE) for k, v in batch['band_features'].items()}
            clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}
            
            # Forward with Embeddings
            outputs = model(x, band_features=band_features, clinical_data=clinical, return_embeddings=True)
            expl = model.get_explainability(outputs)
            
            # Collect Probs
            list_screen.append(outputs['probs_screen'].cpu().numpy())
            list_stage.append(outputs['probs_stage'].cpu().numpy())
            list_subtype.append(outputs['probs_subtype'].cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Collect Explanation Metrics
            adj = expl['adjacency'].cpu().numpy() if expl.get('adjacency') is not None else None
            band_coup_mat = expl['band_coupling'].cpu().numpy() if expl.get('band_coupling') is not None else None
            temp_weights = expl['temporal_weights'].cpu().numpy() if expl.get('temporal_weights') is not None else None 
            # temp_weights shape: [Batch, Time]
            
            # Compute Correlation between Attention and Band Power
            # We want to know: "When attention is high, is Delta high?"
            # band_features values are [Batch, Time, Nodes] (or similar, need to average nodes)
            # transform to [Batch, Time] per band
            
            labels = y.cpu().numpy()
            
            # Process batch for "Attention Drivers"
            if temp_weights is not None:
                batch_size, n_time = temp_weights.shape
                band_names_ordered = ['delta', 'theta', 'alpha', 'beta', 'gamma']

                for b_idx in range(batch_size):
                    lbl_int = int(labels[b_idx])
                    if lbl_int not in metrics_by_class:
                        continue

                    # Get attention series for this sample
                    att_series = temp_weights[b_idx]  # [Time]

                    # Compute correlation for each band
                    sample_corrs = []
                    for bname in band_names_ordered:
                        # Handle potential key casing mismatch (Delta vs delta)
                        keys_lower = {k.lower(): k for k in band_features.keys()}
                        actual_key = keys_lower.get(bname.lower())

                        if actual_key:
                            # band_features shape: [Batch, Channels, Time]
                            feat = band_features[actual_key][b_idx]  # [Channels, Time]

                            # Average over channels to get temporal profile
                            if feat.dim() == 2:
                                # [Channels, Time] -> mean over dim 0 -> [Time]
                                feat_profile = feat.mean(dim=0).cpu().numpy()
                            else:
                                feat_profile = feat.cpu().numpy()

                            # Resample to match attention length if needed
                            if len(feat_profile) != len(att_series):
                                # Simple resampling using linear interpolation
                                from scipy.interpolate import interp1d
                                x_orig = np.linspace(0, 1, len(feat_profile))
                                x_new = np.linspace(0, 1, len(att_series))
                                f = interp1d(x_orig, feat_profile, kind='linear', fill_value='extrapolate')
                                feat_profile = f(x_new)

                            # Pearson Correlation
                            if np.std(att_series) > 1e-9 and np.std(feat_profile) > 1e-9:
                                corr = np.corrcoef(att_series, feat_profile)[0, 1]
                                if np.isnan(corr):
                                    corr = 0.0
                            else:
                                corr = 0.0
                            sample_corrs.append(corr)
                        else:
                            sample_corrs.append(0.0)

                    metrics_by_class[lbl_int]['att_corr'].append(sample_corrs)

            
            for i, lbl in enumerate(labels):
                lbl_int = int(lbl)
                if lbl_int in metrics_by_class:
                    if adj is not None: metrics_by_class[lbl_int]['adj'].append(adj[i])
                    if band_coup_mat is not None: metrics_by_class[lbl_int]['band'].append(band_coup_mat[i])
                    if temp_weights is not None: metrics_by_class[lbl_int]['temp'].append(temp_weights[i])

    # ---------------------------------------------------------
    # 1. Prediction & Evaluation Data Prep
    # ---------------------------------------------------------
    probs_screen = np.concatenate(list_screen)
    probs_stage = np.concatenate(list_stage)
    probs_subtype = np.concatenate(list_subtype)
    
    flat_probs = calculate_flat_probabilities(probs_screen, probs_stage, probs_subtype)
    y_pred_flat = np.argmax(flat_probs, axis=1)
    y_true = np.concatenate(all_targets)
    
    # Prepare Result Dict for Figure 4
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    results_dict = {
        'y_true': y_true,
        'y_pred': y_pred_flat,
        'y_prob': flat_probs,
        'accuracy': accuracy_score(y_true, y_pred_flat),
        'f1_macro': f1_score(y_true, y_pred_flat, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred_flat, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred_flat)
    }

    # Note: Confusion matrix plots are generated by 09_generate_publication_plots.py
    # Removed duplicate cm_flat_4class.png from here to avoid redundancy
    classes = ['AD', 'FTD', 'CN', 'MCI']  # 0,1,2,3
                          
    # ---------------------------------------------------------
    # 2. Metric Averaging for Manuscript Figures
    # ---------------------------------------------------------
    avg_metrics = {}
    connectivity_data = {}
    
    # Map class indices to names for manuscript figures
    idx_to_name = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
    
    # Accumulate Node Importance (Proxy: Node Strength from Adjacency)
    importance_data = {'AD': {}, 'FTD': {}, 'CN': {}, 'MCI': {}} # Included MCI
    
    for cls_idx, metrics in metrics_by_class.items():
        name = idx_to_name.get(cls_idx)
        avg_data = {}
        
        # Adjacency
        if metrics['adj']:
            mean_adj = np.stack(metrics['adj']).mean(axis=0)
            avg_data['adj'] = mean_adj
            connectivity_data[name] = mean_adj
            
            # Compute Node Importance (Degree/Strength)
            node_strength = mean_adj.sum(axis=1) # [Nodes]
            # Normalize to 0-1
            node_strength = (node_strength - node_strength.min()) / (node_strength.max() - node_strength.min() + 1e-6)
            
            if name in importance_data:
                importance_data[name] = {CHANNEL_NAMES[i]: val for i, val in enumerate(node_strength)}
            
        else:
            avg_data['adj'] = None
        
        # Band Coupling
        if metrics['band']:
             avg_data['band'] = np.stack(metrics['band']).mean(axis=0)
        else:
             avg_data['band'] = None
             
        # Temporal
        if metrics['temp']:
             avg_data['temp'] = np.stack(metrics['temp']).mean(axis=0)
        else:
             avg_data['temp'] = None
             
        # Attention Correlations
        if metrics['att_corr']:
            # List of [5] arrays -> Stack -> Mean axis 0
            avg_data['att_corr'] = np.stack(metrics['att_corr']).mean(axis=0) # [5]
        else:
            avg_data['att_corr'] = np.zeros(5)
            
        avg_metrics[cls_idx] = avg_data

    # ---------------------------------------------------------
    # 3. Generate Manuscript Figures (Active Call)
    # ---------------------------------------------------------
    print("\n>>> Generating High-Quality Manuscript Figures...")
    try:
        # Generate all standard figures
        # Note: We pass results_dict which drives Figure 4.
        figs = generate_all_manuscript_figures(output_dir / "manuscript", results=results_dict)
        
        # Manually trigger specific generators with real data overrides
        from src.visualization.manuscript_figures import (
            generate_figure_3_connectivity, generate_figure_5_explainability
        )
        
        # Figure 3: Real Connectivity
        if connectivity_data:
            generate_figure_3_connectivity(connectivity_data=connectivity_data, 
                                          save_path=output_dir / "manuscript" / "figure_3_connectivity_real.png")
                                              
        # Figure 5: Real Explainability
        # Check if importance data is populated
        if any(importance_data.values()):
            generate_figure_5_explainability(importance_data=importance_data,
                                            save_path=output_dir / "manuscript" / "figure_5_explainability_real.png")
        
    except Exception as e:
        print(f"Error generating manuscript figures: {e}")
        import traceback
        traceback.print_exc()

    return avg_metrics

def plot_neuroscientific_figures(metrics, output_dir):
    """Plot Connectivity, Band Coupling, and Attention Drivers."""
    from src.utils.visualization import plot_connectivity_circle, safe_tight_layout
    
    labels = CHANNEL_NAMES if len(CHANNEL_NAMES) == 19 else [f"Ch{i}" for i in range(19)]
    classes = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
    colors = VISUALIZATION_CONFIG['palette']
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    # 1. Connectivity Circles
    for cls_idx, name in classes.items():
        adj = metrics[cls_idx]['adj']
        if adj is not None:
             plot_connectivity_circle(adj, labels, title=f"{name} Connectivity", 
                                    output_path=output_dir / f"connectivity_{name}.png")

    # 2. Band Coupling Heatmaps
    for cls_idx, name in classes.items():
        band_coup = metrics[cls_idx]['band'] # [5, 5]
        if band_coup is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(band_coup, annot=True, fmt='.2f', cmap=VISUALIZATION_CONFIG['cmap_sequential'],
                        xticklabels=band_names, yticklabels=band_names)
            plt.title(f"{name} Band Coupling")
            safe_tight_layout()
            plt.savefig(output_dir / f"band_coupling_{name}.png")
            plt.close()

    # 3. Temporal Attention Drivers (Bar Chart)
    # Why does the model pay attention? -> Correlation with bands
    plt.figure(figsize=(10, 6))

    # Prepare data for grouped bar chart
    bar_width = 0.2
    x = np.arange(len(band_names))

    # Debug: Print correlation values for each class
    print("\n  Attention-Band Correlations by Class:")
    has_valid_data = False

    for i, (cls_idx, name) in enumerate(classes.items()):
        corrs = metrics[cls_idx]['att_corr']  # [5]

        # Check if we have valid data
        if corrs is not None and len(corrs) == 5:
            if np.any(np.abs(corrs) > 1e-6):
                has_valid_data = True
            print(f"    {name}: {[f'{c:.4f}' for c in corrs]}")

            # Plot bars
            plt.bar(x + i * bar_width, corrs, width=bar_width, label=name,
                    color=colors[name], edgecolor='black')
        else:
            print(f"    {name}: No valid correlation data (shape: {corrs.shape if hasattr(corrs, 'shape') else 'N/A'})")

    if not has_valid_data:
        print("  WARNING: All attention correlations are zero or near-zero!")
        print("  This may indicate temporal_weights are not being returned by the model.")

    plt.xlabel("Frequency Band")
    plt.ylabel("Pearson Correlation with Attention Weight")
    plt.title("What Drives Model Attention?\n(Correlation between Temporal Attention & Band Power)")
    plt.xticks(x + bar_width * 1.5, band_names)
    plt.legend(title="Class")
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.8)

    safe_tight_layout()
    plt.savefig(output_dir / "attention_drivers_by_band.png", dpi=300)
    print(f"Saved: attention_drivers_by_band.png")
    plt.close()

def main():
    print("Starting Detailed Neuroscientific Analysis...")
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model, loader = load_data_and_model(fold=0)
        
        # 1. Run Main Model Analysis (Explainability)
        if model:
            metrics = run_analysis(model, loader, output_dir)
            plot_neuroscientific_figures(metrics, output_dir)
        else:
            print("Skipping model-based analysis (no checkpoint found).")

        # 2. Run Microstate Analysis (internal - no duplicate save)
        run_microstate_analysis_internal(output_dir)

        # 3. Generate Hierarchical Decision Explainability Visualization
        print("\n>>> Generating Hierarchical Decision Explainability...")
        results_json = PROJECT_ROOT / "outputs" / "results" / "v3_holdout_results" / "results.json"
        plot_hierarchical_decision_explainability(results_json, output_dir)
        
        print("\n>>> Analysis Complete.")
        print(f"Figures saved to: {output_dir}")
        
        try:
            from experiments.visualize_microstates import run_microstate_analysis
            run_microstate_analysis()
        except:
            # Fallback for path issues
            import experiments.visualize_microstates as ms_viz
            ms_viz.run_microstate_analysis()

        # 5. Source Localization Integration
        print("\n>>> Starting Source Localization Analysis...")
        try:
             # A. Generate Source Estimates
             print("   > Generating source estimates (using eLORETA)...")
             import runpy
             # Run 13_class_average_sources.py to generate .stc files
             # Note: Script is in archive folder
             source_script = PROJECT_ROOT / "experiments" / "archive" / "13_class_average_sources.py"
             if source_script.exists():
                 runpy.run_path(str(source_script))
             else:
                 print(f"   Warning: Source script not found at {source_script}")

             # B. Visualize Source Estimates
             print("   > Visualizing source estimates...")
             # Run 11_visualize_source_estimates.py to create plots from .stc files
             viz_script = PROJECT_ROOT / "experiments" / "archive" / "11_visualize_source_estimates.py"
             if viz_script.exists():
                 runpy.run_path(str(viz_script))
             else:
                 print(f"   Warning: Visualization script not found at {viz_script}")
             
             # C. Raw Signal Inspection
             print("\n   > Visualizing Raw Inspection Segments...")
             runpy.run_path(str(PROJECT_ROOT / "experiments" / "21_visualize_raw_segments.py"))
             
        except Exception as e:
            print(f"Source/Raw Visualization skipped/failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
