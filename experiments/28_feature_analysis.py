"""
Feature Importance and Class-Specific Discriminative Feature Analysis.

Performs:
1. Feature Domain Ablation: Measures accuracy drop when zeroing out each
   feature domain (raw EEG, delta, theta, alpha, beta, gamma, clinical)
2. Band Power Feature Extraction: Computes per-class spectral features
   (relative band powers, theta/alpha ratio, individual alpha peak)
3. Class-Specific Discriminative Features: Statistical tests (Kruskal-Wallis,
   Mann-Whitney U) identifying which features distinguish each class
4. Feature Correlation Analysis: Inter-feature correlations by class
5. Publication-Ready Plots: Feature importance bar chart, class-specific
   radar plots, band power heatmap, discriminative feature ranking

Saves results to outputs/results/feature_analysis.json
Saves figures to outputs/figures/publication/
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats
from scipy.signal import welch
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
from src.utils.reproducibility import DEFAULT_SEED, set_global_seed
SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)

from src.config.config import DEVICE, DATASET_CONFIG, DATALOADER_CONFIG, CHANNEL_NAMES
from src.data.dataset_factory import DatasetFactory
from src.models.v2.neuro_chrono_graph_v2 import create_neuro_chrono_graph_v2

CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
BAND_RANGES = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
    'beta': (13, 30), 'gamma': (30, 45)
}

COLORS = {'AD': '#D55E00', 'FTD': '#0072B2', 'CN': '#009E73', 'MCI': '#E69F00'}
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "publication"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'figure.dpi': 300, 'savefig.dpi': 300,
})


# ============================================================
# 1. Feature Domain Ablation
# ============================================================

def run_feature_ablation(model, dataloader, device):
    """
    Ablate each feature domain and measure accuracy drop.
    Domains: raw_eeg, delta, theta, alpha, beta, gamma, clinical_metadata.
    """
    print("\n[1] Feature Domain Ablation...")

    def evaluate_with_ablation(model, loader, ablate_domain=None):
        """
        Evaluate model with one feature domain zeroed out.

        Args:
            ablate_domain: feature domain to zero out, or None for baseline.
                           Use 'standard_chain' to replace geometric mean with
                           plain multiplicative chain rule (tests calibration contribution).
            chain_rule: 'geometric' (default, depth-normalised) or 'standard' (plain product).
        """
        model.eval()
        all_preds, all_true = [], []

        use_standard_chain = (ablate_domain == 'standard_chain')

        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                y = batch['label']
                meta = batch['metadata'].to(device)
                band_features = {k: v.to(device) for k, v in batch['band_features'].items()}
                clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}

                # Apply feature ablation
                if ablate_domain == 'raw_eeg':
                    x = torch.zeros_like(x)
                elif ablate_domain in BAND_NAMES:
                    for k in band_features:
                        if k.lower() == ablate_domain:
                            band_features[k] = torch.zeros_like(band_features[k])
                elif ablate_domain == 'all_bands':
                    band_features = {k: torch.zeros_like(v) for k, v in band_features.items()}
                elif ablate_domain == 'source_features':
                    # eLORETA source-power features: these occupy the last n_source_features
                    # columns of each band_feature tensor if source localization is enabled.
                    # Zero the source-derived channels (channels 0-4 map to lobar ROIs when
                    # eLORETA is active; sensor channels follow).  If source features are not
                    # present the ablation will have zero effect, which is itself informative.
                    for k in band_features:
                        bf = band_features[k]
                        if bf.shape[-1] > 19:  # more columns than sensor channels
                            bf = bf.clone()
                            bf[..., 19:] = 0.0  # zero source-derived columns
                            band_features[k] = bf
                elif ablate_domain == 'clinical':
                    clinical = {
                        'mmse': torch.zeros_like(meta[:, 2]),
                        'age': torch.zeros_like(meta[:, 0]),
                        'sex': torch.zeros_like(meta[:, 1])
                    }

                out = model(x, band_features=band_features, clinical_data=clinical)

                # Extract stage probabilities
                p_h = out['probs_screen'][:, 0].cpu().numpy()
                p_i = out['probs_screen'][:, 1].cpu().numpy()
                p_m = out['probs_stage'][:, 0].cpu().numpy()
                p_d = out['probs_stage'][:, 1].cpu().numpy()
                p_a = out['probs_subtype'][:, 0].cpu().numpy()
                p_f = out['probs_subtype'][:, 1].cpu().numpy()

                eps = 1e-8
                if use_standard_chain:
                    # Standard multiplicative chain rule (no depth normalisation)
                    P_CN  = p_h
                    P_MCI = p_i * p_m + eps
                    P_AD  = p_i * p_d * p_a + eps
                    P_FTD = p_i * p_d * p_f + eps
                else:
                    # Geometric mean chain rule (depth-normalised)
                    P_CN  = p_h
                    P_MCI = np.power(p_i * p_m + eps, 1 / 2)
                    P_AD  = np.power(p_i * p_d * p_a + eps, 1 / 3)
                    P_FTD = np.power(p_i * p_d * p_f + eps, 1 / 3)

                probs = np.stack([P_AD, P_FTD, P_CN, P_MCI], axis=1)
                probs = probs / (probs.sum(axis=1, keepdims=True) + eps)
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_true.extend(y.numpy())

        return accuracy_score(all_true, all_preds)

    # Baseline (no ablation, geometric mean)
    baseline_acc = evaluate_with_ablation(model, dataloader, ablate_domain=None)
    print(f"  Baseline accuracy: {baseline_acc:.1%}")

    # Ablate each domain (feature domains + chain-rule calibration)
    domains = [
        'raw_eeg', 'delta', 'theta', 'alpha', 'beta', 'gamma',
        'all_bands', 'clinical', 'source_features', 'standard_chain'
    ]
    domain_labels = [
        'Raw EEG Signal', 'Delta (0.5-4 Hz)', 'Theta (4-8 Hz)',
        'Alpha (8-13 Hz)', 'Beta (13-30 Hz)', 'Gamma (30-45 Hz)',
        'All Band Features', 'Clinical Metadata',
        'Source Features (eLORETA columns)',
        'Standard Chain Rule (no geometric mean)'
    ]

    ablation_results = {}
    for domain, label in zip(domains, domain_labels):
        abl_acc = evaluate_with_ablation(model, dataloader, ablate_domain=domain)
        drop = baseline_acc - abl_acc
        ablation_results[domain] = {
            'label': label,
            'accuracy': float(abl_acc),
            'drop': float(drop),
            'relative_importance': float(drop / baseline_acc * 100)
        }
        print(f"  -{label}: Acc={abl_acc:.1%}, Drop={drop:+.1%} ({drop/baseline_acc*100:.1f}%)")

    return baseline_acc, ablation_results


# ============================================================
# 2. Extract Spectral Features Per Subject
# ============================================================

def extract_spectral_features(dataloader, sfreq=256.0):
    """Extract spectral features from raw EEG for each sample."""
    print("\n[2] Extracting spectral features per class...")

    features_by_class = defaultdict(list)

    for batch in dataloader:
        x = batch['x'].numpy()  # [B, C, T]
        labels = batch['label'].numpy()

        for i in range(len(labels)):
            signal = x[i]  # [C, T]
            label = int(labels[i])

            # Compute PSD via Welch
            freqs, psd = welch(signal, fs=sfreq, nperseg=min(256, signal.shape[1]),
                              noverlap=128, axis=1)

            # Total power (0.5-45 Hz)
            total_mask = (freqs >= 0.5) & (freqs <= 45)
            total_power = psd[:, total_mask].sum(axis=1).mean()

            feat = {}
            for band_name, (flo, fhi) in BAND_RANGES.items():
                mask = (freqs >= flo) & (freqs <= fhi)
                band_power = psd[:, mask].sum(axis=1).mean()
                feat[f'{band_name}_rel_power'] = band_power / (total_power + 1e-10)

                # Regional breakdown (Frontal, Temporal, Parietal, Occipital)
                frontal_chs = [j for j, ch in enumerate(CHANNEL_NAMES) if ch.startswith('F')]
                temporal_chs = [j for j, ch in enumerate(CHANNEL_NAMES) if ch.startswith('T')]
                parietal_chs = [j for j, ch in enumerate(CHANNEL_NAMES) if ch.startswith('P')]
                occipital_chs = [j for j, ch in enumerate(CHANNEL_NAMES) if ch.startswith('O')]

                for region, ch_idx in [('frontal', frontal_chs), ('temporal', temporal_chs),
                                        ('parietal', parietal_chs), ('occipital', occipital_chs)]:
                    if ch_idx:
                        reg_power = psd[ch_idx][:, mask].sum(axis=1).mean()
                        reg_total = psd[ch_idx][:, total_mask].sum(axis=1).mean()
                        feat[f'{band_name}_{region}'] = reg_power / (reg_total + 1e-10)

            # Derived features
            feat['theta_alpha_ratio'] = feat['theta_rel_power'] / (feat['alpha_rel_power'] + 1e-10)
            feat['delta_alpha_ratio'] = feat['delta_rel_power'] / (feat['alpha_rel_power'] + 1e-10)
            feat['alpha_beta_ratio'] = feat['alpha_rel_power'] / (feat['beta_rel_power'] + 1e-10)

            # Individual alpha peak frequency
            alpha_mask = (freqs >= 6) & (freqs <= 14)
            mean_psd_alpha = psd[:, alpha_mask].mean(axis=0)
            alpha_freqs = freqs[alpha_mask]
            feat['alpha_peak_freq'] = float(alpha_freqs[np.argmax(mean_psd_alpha)])

            # Spectral entropy (normalized)
            psd_norm = psd[:, total_mask] / (psd[:, total_mask].sum(axis=1, keepdims=True) + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1).mean()
            feat['spectral_entropy'] = float(spectral_entropy)

            # Spectral edge frequency (95% power)
            cumsum = np.cumsum(psd[:, total_mask].mean(axis=0))
            threshold = 0.95 * cumsum[-1]
            sef_idx = np.searchsorted(cumsum, threshold)
            feat['spectral_edge_freq_95'] = float(freqs[total_mask][min(sef_idx, len(freqs[total_mask])-1)])

            features_by_class[label].append(feat)

    return features_by_class


# ============================================================
# 3. Statistical Tests for Class-Specific Features
# ============================================================

def compute_discriminative_features(features_by_class):
    """Identify features that best discriminate each class from others."""
    print("\n[3] Computing class-specific discriminative features...")

    feature_names = list(features_by_class[0][0].keys()) if features_by_class[0] else []
    results = {}

    # Overall: Kruskal-Wallis across all 4 classes
    kw_results = {}
    for feat_name in feature_names:
        groups = []
        for cls in range(4):
            if cls in features_by_class:
                vals = [f[feat_name] for f in features_by_class[cls]]
                groups.append(vals)
        if len(groups) >= 2:
            try:
                stat, p = stats.kruskal(*groups)
                kw_results[feat_name] = {'H': float(stat), 'p': float(p)}
            except:
                kw_results[feat_name] = {'H': 0, 'p': 1.0}
    results['kruskal_wallis'] = kw_results

    # Per-class: Which features distinguish this class from all others?
    class_discriminative = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx not in features_by_class or not features_by_class[cls_idx]:
            continue

        feat_scores = []
        for feat_name in feature_names:
            cls_vals = [f[feat_name] for f in features_by_class[cls_idx]]
            other_vals = []
            for other_cls in range(4):
                if other_cls != cls_idx and other_cls in features_by_class:
                    other_vals.extend([f[feat_name] for f in features_by_class[other_cls]])

            if len(cls_vals) > 2 and len(other_vals) > 2:
                try:
                    stat, p = stats.mannwhitneyu(cls_vals, other_vals, alternative='two-sided')
                    # Cohen's d
                    d = (np.mean(cls_vals) - np.mean(other_vals)) / \
                        (np.sqrt((np.var(cls_vals) + np.var(other_vals)) / 2) + 1e-10)
                    feat_scores.append({
                        'feature': feat_name,
                        'U': float(stat),
                        'p_value': float(p),
                        'cohens_d': float(d),
                        'class_mean': float(np.mean(cls_vals)),
                        'other_mean': float(np.mean(other_vals)),
                        'direction': 'elevated' if np.mean(cls_vals) > np.mean(other_vals) else 'reduced'
                    })
                except:
                    pass

        # Sort by absolute Cohen's d
        feat_scores.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
        class_discriminative[cls_name] = feat_scores[:15]  # Top 15

    results['class_discriminative'] = class_discriminative
    return results


# ============================================================
# 4. Visualization
# ============================================================

def plot_feature_ablation(baseline_acc, ablation_results, save_path):
    """Bar chart of feature domain importance via ablation."""
    print("  Plotting feature ablation results...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by drop (importance)
    sorted_items = sorted(ablation_results.items(), key=lambda x: x[1]['drop'], reverse=True)
    labels = [v['label'] for _, v in sorted_items]
    drops = [v['drop'] * 100 for _, v in sorted_items]

    colors_bar = ['#e74c3c' if d > 3 else '#f39c12' if d > 1 else '#95a5a6' for d in drops]

    bars = ax.barh(range(len(labels)), drops, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Accuracy Drop (%)', fontsize=13)
    ax.set_title('Feature Domain Importance (Ablation Study)', fontsize=14)
    ax.invert_yaxis()

    for bar, d in zip(bars, drops):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{d:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_class_specific_features(class_disc, save_path):
    """Bar chart of top discriminative features per class."""
    print("  Plotting class-specific discriminative features...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (cls_name, feats) in enumerate(class_disc.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        top_n = min(10, len(feats))
        if top_n == 0:
            ax.text(0.5, 0.5, f'No discriminative features\nfor {cls_name}',
                   ha='center', va='center', fontsize=12)
            continue

        top_feats = feats[:top_n]
        names = [f['feature'].replace('_', ' ').title()[:25] for f in top_feats]
        d_values = [f['cohens_d'] for f in top_feats]
        directions = [f['direction'] for f in top_feats]

        colors_d = [COLORS[cls_name] if d == 'elevated' else '#95a5a6' for d in directions]

        bars = ax.barh(range(len(names)), [abs(d) for d in d_values],
                      color=colors_d, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("|Cohen's d|", fontsize=11)
        ax.set_title(f'{cls_name} vs Others', fontsize=13, fontweight='bold',
                    color=COLORS[cls_name])
        ax.invert_yaxis()

        # Significance markers
        for i, f in enumerate(top_feats):
            sig = '***' if f['p_value'] < 0.001 else '**' if f['p_value'] < 0.01 else '*' if f['p_value'] < 0.05 else 'ns'
            direction_arrow = '↑' if f['direction'] == 'elevated' else '↓'
            ax.text(abs(f['cohens_d']) + 0.05, i,
                   f"{sig} {direction_arrow}", va='center', fontsize=9)

        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Class-Specific Discriminative Features (Mann-Whitney U)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_band_power_heatmap(features_by_class, save_path):
    """Heatmap of relative band power across classes and regions."""
    print("  Plotting band power heatmap...")

    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    regions = ['frontal', 'temporal', 'parietal', 'occipital']
    classes = ['AD', 'FTD', 'CN', 'MCI']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for cls_idx, (cls_name, ax) in enumerate(zip(classes, axes)):
        if cls_idx not in features_by_class or not features_by_class[cls_idx]:
            continue

        matrix = np.zeros((len(bands), len(regions)))
        for b_idx, band in enumerate(bands):
            for r_idx, region in enumerate(regions):
                key = f'{band}_{region}'
                vals = [f[key] for f in features_by_class[cls_idx] if key in f]
                matrix[b_idx, r_idx] = np.mean(vals) if vals else 0

        sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f',
                   xticklabels=[r.title() for r in regions],
                   yticklabels=[b.title() for b in bands],
                   cbar_kws={'label': 'Rel. Power'},
                   vmin=0, vmax=0.5)
        ax.set_title(f'{cls_name}', fontsize=14, fontweight='bold', color=COLORS[cls_name])

    plt.suptitle('Regional Band Power Distribution by Diagnostic Class', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_feature_set_comparison(features_by_class, save_path):
    """Radar/spider plot comparing feature profiles across classes."""
    print("  Plotting feature set comparison radar plot...")

    # Select key features for the radar
    key_features = [
        'delta_rel_power', 'theta_rel_power', 'alpha_rel_power',
        'beta_rel_power', 'gamma_rel_power', 'theta_alpha_ratio',
        'alpha_peak_freq', 'spectral_entropy', 'spectral_edge_freq_95'
    ]
    labels = [
        'Delta Power', 'Theta Power', 'Alpha Power',
        'Beta Power', 'Gamma Power', 'θ/α Ratio',
        'Alpha Peak (Hz)', 'Spectral Entropy', 'SEF95 (Hz)'
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
    angles += angles[:1]

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx not in features_by_class or not features_by_class[cls_idx]:
            continue

        values = []
        for feat in key_features:
            vals = [f[feat] for f in features_by_class[cls_idx] if feat in f]
            values.append(np.mean(vals) if vals else 0)

        # Normalize to 0-1 for radar
        all_vals = []
        for feat in key_features:
            for c in range(4):
                if c in features_by_class:
                    all_vals.extend([f[feat] for f in features_by_class[c] if feat in f])
        global_min = min(all_vals) if all_vals else 0
        global_max = max(all_vals) if all_vals else 1
        norm_values = [(v - global_min) / (global_max - global_min + 1e-10) for v in values]
        norm_values += norm_values[:1]

        ax.plot(angles, norm_values, 'o-', linewidth=2, label=cls_name,
               color=COLORS[cls_name], alpha=0.8)
        ax.fill(angles, norm_values, alpha=0.1, color=COLORS[cls_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Spectral Feature Profiles by Diagnostic Class', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# 5. ADDITIONAL SCIENTIFIC VISUALIZATIONS
# ============================================================

def plot_violin_distributions(features_by_class, save_path):
    """
    Violin + strip plots showing full distribution of key spectral features
    per diagnostic class. More informative than bar charts as they reveal
    distribution shape, overlap, and outliers.
    """
    print("  Plotting violin distribution plots...")

    key_features = [
        ('theta_alpha_ratio', 'Theta/Alpha Ratio'),
        ('alpha_rel_power', 'Alpha Relative Power'),
        ('delta_rel_power', 'Delta Relative Power'),
        ('alpha_peak_freq', 'Alpha Peak Frequency (Hz)'),
        ('spectral_entropy', 'Spectral Entropy'),
        ('spectral_edge_freq_95', 'Spectral Edge Frequency 95% (Hz)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (feat_key, feat_label) in enumerate(key_features):
        ax = axes[idx]

        # Build dataframe for seaborn
        plot_data = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            if cls_idx not in features_by_class:
                continue
            for f in features_by_class[cls_idx]:
                if feat_key in f:
                    plot_data.append({'Class': cls_name, 'Value': f[feat_key]})

        if not plot_data:
            continue

        df = pd.DataFrame(plot_data)
        palette = [COLORS[c] for c in CLASS_NAMES if c in df['Class'].values]

        # Violin with inner box
        parts = sns.violinplot(x='Class', y='Value', hue='Class', data=df, ax=ax,
                              palette=palette, inner='box', cut=0,
                              alpha=0.7, linewidth=1.2, legend=False,
                              order=[c for c in CLASS_NAMES if c in df['Class'].values])

        # Overlay strip (individual points)
        sns.stripplot(x='Class', y='Value', data=df, ax=ax,
                     color='black', alpha=0.15, size=2, jitter=True,
                     order=[c for c in CLASS_NAMES if c in df['Class'].values])

        ax.set_title(feat_label, fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(feat_label.split('(')[0].strip(), fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add significance brackets (AD vs CN)
        classes_present = df['Class'].unique()
        if 'AD' in classes_present and 'CN' in classes_present:
            ad_vals = df[df['Class'] == 'AD']['Value'].values
            cn_vals = df[df['Class'] == 'CN']['Value'].values
            try:
                _, p = stats.mannwhitneyu(ad_vals, cn_vals)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                y_max = df['Value'].max()
                y_range = df['Value'].max() - df['Value'].min()
                # Find positions of AD and CN
                order = [c for c in CLASS_NAMES if c in classes_present]
                ad_pos = order.index('AD')
                cn_pos = order.index('CN')
                ax.plot([ad_pos, ad_pos, cn_pos, cn_pos],
                       [y_max + 0.05*y_range, y_max + 0.08*y_range,
                        y_max + 0.08*y_range, y_max + 0.05*y_range],
                       'k-', linewidth=1)
                ax.text((ad_pos + cn_pos)/2, y_max + 0.09*y_range, sig,
                       ha='center', fontsize=11, fontweight='bold')
            except:
                pass

    plt.suptitle('Spectral Feature Distributions Across Diagnostic Classes',
                fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_cohens_d_heatmap(features_by_class, disc_results, save_path):
    """
    Cohen's d effect size heatmap: features (rows) × classes (columns).
    Provides an at-a-glance overview of which features discriminate each class.
    Red = elevated, Blue = reduced relative to other classes.
    """
    print("  Plotting Cohen's d effect size heatmap...")

    # Select top features across all classes (union of top features)
    all_top_features = set()
    for cls_name, feats in disc_results['class_discriminative'].items():
        for f in feats[:8]:
            all_top_features.add(f['feature'])

    feature_list = sorted(all_top_features)
    if not feature_list:
        print("    Warning: No discriminative features to plot")
        return

    # Build effect size matrix
    d_matrix = np.zeros((len(feature_list), len(CLASS_NAMES)))
    p_matrix = np.ones((len(feature_list), len(CLASS_NAMES)))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_name not in disc_results['class_discriminative']:
            continue
        feat_dict = {f['feature']: f for f in disc_results['class_discriminative'][cls_name]}
        for feat_idx, feat_name in enumerate(feature_list):
            if feat_name in feat_dict:
                d_matrix[feat_idx, cls_idx] = feat_dict[feat_name]['cohens_d']
                p_matrix[feat_idx, cls_idx] = feat_dict[feat_name]['p_value']

    # Create annotation with significance stars
    annot = np.empty_like(d_matrix, dtype=object)
    for i in range(d_matrix.shape[0]):
        for j in range(d_matrix.shape[1]):
            d_val = d_matrix[i, j]
            p_val = p_matrix[i, j]
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            annot[i, j] = f'{d_val:+.2f}{stars}'

    # Pretty feature names
    pretty_names = [f.replace('_', ' ').title()[:30] for f in feature_list]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_list) * 0.5)))

    # Diverging colormap: red=elevated, blue=reduced
    vmax = max(abs(d_matrix.min()), abs(d_matrix.max()), 1.0)
    sns.heatmap(d_matrix, ax=ax, cmap='RdBu_r', center=0,
               annot=annot, fmt='', vmin=-vmax, vmax=vmax,
               xticklabels=CLASS_NAMES, yticklabels=pretty_names,
               cbar_kws={'label': "Cohen's d\n(+elevated / −reduced)"},
               linewidths=0.5, linecolor='white')

    ax.set_title("Effect Size Matrix: Feature Discriminative Power per Class",
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Diagnostic Class', fontsize=12)
    ax.set_ylabel('Spectral Feature', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_topographic_maps(features_by_class, save_path):
    """
    Topographic scalp maps of band power per class.
    The gold-standard EEG visualization showing spatial distribution
    of spectral power across the scalp.
    """
    print("  Plotting topographic scalp maps...")

    try:
        import mne
    except ImportError:
        print("    Warning: MNE not available, skipping topographic maps")
        return

    # Create MNE info with standard 10-20 montage
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        # Filter to channels we actually have
        available_chs = [ch for ch in CHANNEL_NAMES if ch in montage.ch_names]
        if len(available_chs) < 5:
            print(f"    Warning: Only {len(available_chs)} channels match montage, skipping")
            return

        info = mne.create_info(ch_names=available_chs, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
    except Exception as e:
        print(f"    Warning: Could not create montage: {e}")
        return

    bands_to_plot = ['theta', 'alpha', 'beta']
    classes = ['AD', 'FTD', 'CN', 'MCI']

    fig, axes = plt.subplots(len(classes), len(bands_to_plot),
                            figsize=(4 * len(bands_to_plot), 4 * len(classes)))

    ch_indices = [CHANNEL_NAMES.index(ch) for ch in available_chs if ch in CHANNEL_NAMES]

    for row_idx, (cls_name, cls_idx) in enumerate(zip(classes, range(4))):
        if cls_idx not in features_by_class or not features_by_class[cls_idx]:
            for col_idx in range(len(bands_to_plot)):
                axes[row_idx, col_idx].set_visible(False)
            continue

        for col_idx, band in enumerate(bands_to_plot):
            ax = axes[row_idx, col_idx]

            # Get per-channel band power for each available channel
            ch_powers = []
            for ch_pos, ch_name in enumerate(available_chs):
                ch_global_idx = CHANNEL_NAMES.index(ch_name)
                key = f'{band}_{_get_region(ch_name)}'
                vals = [f.get(key, 0) for f in features_by_class[cls_idx]]
                ch_powers.append(np.mean(vals) if vals else 0)

            ch_powers = np.array(ch_powers)

            try:
                mne.viz.plot_topomap(ch_powers, info, axes=ax, show=False,
                                   cmap='YlOrRd', contours=4, sensors=True,
                                   names=available_chs if len(available_chs) <= 19 else None)
            except Exception:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')

            if row_idx == 0:
                ax.set_title(f'{band.title()} ({BAND_RANGES[band][0]}-{BAND_RANGES[band][1]} Hz)',
                           fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(cls_name, fontsize=14, fontweight='bold',
                            color=COLORS[cls_name], rotation=0, labelpad=40)

    plt.suptitle('Topographic Distribution of Band Power by Diagnostic Class',
                fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def _get_region(ch_name):
    """Map channel name to brain region."""
    if ch_name.startswith('F'):
        return 'frontal'
    elif ch_name.startswith('T'):
        return 'temporal'
    elif ch_name.startswith('P') or ch_name.startswith('C'):
        return 'parietal'
    elif ch_name.startswith('O'):
        return 'occipital'
    return 'frontal'


def plot_combined_feature_figure(baseline_acc, ablation_results, features_by_class,
                                 disc_results, save_path):
    """
    Combined 6-panel publication figure (Figure X in manuscript).
    Composites the most important feature analysis plots into one figure.

    Panels:
      A: Feature domain ablation (horizontal bar)
      B: Radar plot of spectral profiles per class
      C: Band power heatmap (CN vs AD difference)
      D: Violin plots of top 2 discriminative features
      E: Cohen's d heatmap (compact)
      F: Regional alpha power comparison
    """
    print("  Plotting combined publication figure...")

    fig = plt.figure(figsize=(20, 14))

    # --- Panel A: Feature Domain Ablation ---
    ax_a = fig.add_subplot(2, 3, 1)
    sorted_items = sorted(ablation_results.items(), key=lambda x: x[1]['drop'], reverse=True)
    labels_a = [v['label'].replace(' (', '\n(') for _, v in sorted_items]
    drops = [v['drop'] * 100 for _, v in sorted_items]
    colors_bar = ['#e74c3c' if d > 3 else '#f39c12' if d > 1 else '#95a5a6' for d in drops]

    bars = ax_a.barh(range(len(labels_a)), drops, color=colors_bar,
                    edgecolor='black', linewidth=0.5, height=0.7)
    ax_a.set_yticks(range(len(labels_a)))
    ax_a.set_yticklabels(labels_a, fontsize=9)
    ax_a.set_xlabel('Accuracy Drop (%)', fontsize=11)
    ax_a.set_title('A. Feature Domain Ablation', fontsize=13, fontweight='bold')
    ax_a.invert_yaxis()
    for bar, d in zip(bars, drops):
        ax_a.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                 f'{d:.1f}%', va='center', fontsize=8, fontweight='bold')
    ax_a.grid(axis='x', alpha=0.3)

    # --- Panel B: Radar Plot ---
    ax_b = fig.add_subplot(2, 3, 2, polar=True)
    key_features = [
        'delta_rel_power', 'theta_rel_power', 'alpha_rel_power',
        'beta_rel_power', 'gamma_rel_power', 'theta_alpha_ratio'
    ]
    radar_labels = ['δ Power', 'θ Power', 'α Power', 'β Power', 'γ Power', 'θ/α Ratio']
    angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
    angles += angles[:1]

    all_vals_radar = []
    for feat in key_features:
        for c in range(4):
            if c in features_by_class:
                all_vals_radar.extend([f[feat] for f in features_by_class[c] if feat in f])
    g_min = min(all_vals_radar) if all_vals_radar else 0
    g_max = max(all_vals_radar) if all_vals_radar else 1

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx not in features_by_class:
            continue
        values = []
        for feat in key_features:
            vals = [f[feat] for f in features_by_class[cls_idx] if feat in f]
            values.append(np.mean(vals) if vals else 0)
        norm = [(v - g_min) / (g_max - g_min + 1e-10) for v in values]
        norm += norm[:1]
        ax_b.plot(angles, norm, 'o-', linewidth=2, label=cls_name,
                 color=COLORS[cls_name], markersize=4)
        ax_b.fill(angles, norm, alpha=0.08, color=COLORS[cls_name])

    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(radar_labels, fontsize=9)
    ax_b.set_title('B. Spectral Profiles', fontsize=13, fontweight='bold', pad=15)
    ax_b.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=9)

    # --- Panel C: AD vs CN Difference Heatmap ---
    ax_c = fig.add_subplot(2, 3, 3)
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    regions = ['frontal', 'temporal', 'parietal', 'occipital']

    if 0 in features_by_class and 2 in features_by_class:
        diff_matrix = np.zeros((len(bands), len(regions)))
        for b_idx, band in enumerate(bands):
            for r_idx, region in enumerate(regions):
                key = f'{band}_{region}'
                ad_vals = [f[key] for f in features_by_class[0] if key in f]
                cn_vals = [f[key] for f in features_by_class[2] if key in f]
                diff_matrix[b_idx, r_idx] = np.mean(ad_vals) - np.mean(cn_vals) if ad_vals and cn_vals else 0

        sns.heatmap(diff_matrix, ax=ax_c, cmap='RdBu_r', center=0,
                   annot=True, fmt='+.3f',
                   xticklabels=[r.title() for r in regions],
                   yticklabels=[b.title() for b in bands],
                   cbar_kws={'label': 'AD − CN', 'shrink': 0.8},
                   linewidths=0.5)
        ax_c.set_title('C. AD vs CN Power Difference', fontsize=13, fontweight='bold')

    # --- Panel D: Violin of theta/alpha ratio + alpha power ---
    ax_d = fig.add_subplot(2, 3, 4)
    plot_data_d = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx not in features_by_class:
            continue
        for f in features_by_class[cls_idx]:
            if 'theta_alpha_ratio' in f:
                plot_data_d.append({'Class': cls_name, 'Value': f['theta_alpha_ratio']})
    if plot_data_d:
        df_d = pd.DataFrame(plot_data_d)
        order_d = [c for c in CLASS_NAMES if c in df_d['Class'].values]
        palette_d = [COLORS[c] for c in order_d]
        sns.violinplot(x='Class', y='Value', hue='Class', data=df_d, ax=ax_d,
                      palette=palette_d, inner='quartile', cut=0,
                      order=order_d, alpha=0.7, legend=False)
        sns.stripplot(x='Class', y='Value', data=df_d, ax=ax_d,
                     color='black', alpha=0.1, size=1.5, jitter=True,
                     order=order_d)
        ax_d.set_title('D. Theta/Alpha Ratio Distribution', fontsize=13, fontweight='bold')
        ax_d.set_xlabel('')
        ax_d.set_ylabel('Theta/Alpha Ratio', fontsize=11)
        ax_d.grid(axis='y', alpha=0.3)

    # --- Panel E: Compact Cohen's d for top features ---
    ax_e = fig.add_subplot(2, 3, 5)
    top_feats_all = set()
    for cls_name, feats in disc_results['class_discriminative'].items():
        for f in feats[:5]:
            top_feats_all.add(f['feature'])
    feat_list_e = sorted(top_feats_all)[:10]  # Max 10 features

    if feat_list_e:
        d_mat = np.zeros((len(feat_list_e), len(CLASS_NAMES)))
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            if cls_name not in disc_results['class_discriminative']:
                continue
            feat_dict = {f['feature']: f for f in disc_results['class_discriminative'][cls_name]}
            for fi, fn in enumerate(feat_list_e):
                if fn in feat_dict:
                    d_mat[fi, cls_idx] = feat_dict[fn]['cohens_d']

        pretty = [f.replace('_', ' ').title()[:20] for f in feat_list_e]
        vmax_e = max(abs(d_mat.min()), abs(d_mat.max()), 0.5)
        sns.heatmap(d_mat, ax=ax_e, cmap='RdBu_r', center=0,
                   annot=True, fmt='+.1f', vmin=-vmax_e, vmax=vmax_e,
                   xticklabels=CLASS_NAMES, yticklabels=pretty,
                   cbar_kws={'label': "Cohen's d", 'shrink': 0.8},
                   linewidths=0.5, linecolor='white')
        ax_e.set_title("E. Effect Size Matrix", fontsize=13, fontweight='bold')
        ax_e.set_xlabel('')

    # --- Panel F: Alpha power violin ---
    ax_f = fig.add_subplot(2, 3, 6)
    plot_data_f = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if cls_idx not in features_by_class:
            continue
        for f in features_by_class[cls_idx]:
            if 'alpha_rel_power' in f:
                plot_data_f.append({'Class': cls_name, 'Value': f['alpha_rel_power']})
    if plot_data_f:
        df_f = pd.DataFrame(plot_data_f)
        order_f = [c for c in CLASS_NAMES if c in df_f['Class'].values]
        palette_f = [COLORS[c] for c in order_f]
        sns.violinplot(x='Class', y='Value', hue='Class', data=df_f, ax=ax_f,
                      palette=palette_f, inner='quartile', cut=0,
                      order=order_f, alpha=0.7, legend=False)
        sns.stripplot(x='Class', y='Value', data=df_f, ax=ax_f,
                     color='black', alpha=0.1, size=1.5, jitter=True,
                     order=order_f)
        ax_f.set_title('F. Alpha Relative Power Distribution', fontsize=13, fontweight='bold')
        ax_f.set_xlabel('')
        ax_f.set_ylabel('Alpha Relative Power', fontsize=11)
        ax_f.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Importance and Class-Specific Spectral Analysis',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path} (+ .pdf)")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("FEATURE IMPORTANCE & CLASS-SPECIFIC ANALYSIS")
    print("=" * 70)

    results = {}

    # Load data
    print("\n[0] Loading data and model...")
    factory = DatasetFactory()
    dataset_paths = {
        'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
        'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
        'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
        'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
        'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset",
    }
    for name, path in dataset_paths.items():
        if path.exists():
            factory.add_dataset(name, path)

    dataset, groups, labels = factory.create_torch_datasets(config=DATASET_CONFIG)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, **DATALOADER_CONFIG)

    # Load best model checkpoint
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    ckpt_path = ckpt_dir / "hierarchical_model_fold0.pt"
    if not ckpt_path.exists():
        # Try to find any checkpoint
        ckpts = list(ckpt_dir.glob("hierarchical_model_fold*.pt"))
        if ckpts:
            ckpt_path = ckpts[0]
        else:
            print("ERROR: No model checkpoint found!")
            print("  Please run 18_train_hierarchical.py first.")
            return

    print(f"  Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=DEVICE)
    has_streams = any(k.startswith('stream_fusion.') for k in state.keys())
    cfg = {'n_classes': 3, 'hidden_dim': 128, 'dropout': 0.4}
    if has_streams:
        cfg.update({
            'feature_streams': ['spectral', 'connectivity', 'complexity', 'microstate'],
            'stream_dim': 64,
        })
    model = create_neuro_chrono_graph_v2(cfg).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 1. Feature Domain Ablation
    baseline_acc, ablation_results = run_feature_ablation(model, dataloader, DEVICE)
    results['baseline_accuracy'] = float(baseline_acc)
    results['ablation'] = ablation_results

    # 2. Extract Spectral Features
    features_by_class = extract_spectral_features(dataloader, sfreq=256.0)
    for cls in features_by_class:
        n = len(features_by_class[cls])
        print(f"  {CLASS_NAMES[cls]}: {n} samples")

    # 3. Discriminative Features
    disc_results = compute_discriminative_features(features_by_class)
    results['discriminative_features'] = disc_results

    # Print top features per class
    print("\n" + "=" * 70)
    print("TOP DISCRIMINATIVE FEATURES PER CLASS")
    print("=" * 70)
    for cls_name, feats in disc_results['class_discriminative'].items():
        print(f"\n  {cls_name}:")
        for f in feats[:5]:
            sig = '***' if f['p_value'] < 0.001 else '**' if f['p_value'] < 0.01 else '*' if f['p_value'] < 0.05 else 'ns'
            print(f"    {f['feature']:<30} d={f['cohens_d']:+.2f} ({f['direction']}) {sig}")

    # 4. Significant features summary (Kruskal-Wallis)
    kw = disc_results['kruskal_wallis']
    sig_features = [(name, v) for name, v in kw.items() if v['p'] < 0.05]
    sig_features.sort(key=lambda x: x[1]['H'], reverse=True)
    results['n_significant_features'] = len(sig_features)
    results['n_total_features'] = len(kw)
    print(f"\n  Significant features (KW p<0.05): {len(sig_features)}/{len(kw)}")

    # 5. Generate ALL publication plots
    print("\n[4] Generating publication plots...")

    # Original 4 plots
    plot_feature_ablation(baseline_acc, ablation_results,
                         FIG_DIR / "feature_domain_ablation.png")
    plot_class_specific_features(disc_results['class_discriminative'],
                                FIG_DIR / "class_discriminative_features.png")
    plot_band_power_heatmap(features_by_class,
                           FIG_DIR / "band_power_heatmap.png")
    plot_feature_set_comparison(features_by_class,
                               FIG_DIR / "feature_radar_comparison.png")

    # NEW: 4 additional scientific visualizations
    plot_violin_distributions(features_by_class,
                             FIG_DIR / "feature_violin_distributions.png")
    plot_cohens_d_heatmap(features_by_class, disc_results,
                         FIG_DIR / "cohens_d_effect_heatmap.png")
    plot_topographic_maps(features_by_class,
                         FIG_DIR / "topographic_band_power.png")
    plot_combined_feature_figure(baseline_acc, ablation_results, features_by_class,
                                disc_results,
                                FIG_DIR / "Figure_feature_analysis.png")

    # 6. Build summary table for manuscript
    summary_table = []
    for cls_name, feats in disc_results['class_discriminative'].items():
        for f in feats[:5]:
            summary_table.append({
                'Class': cls_name,
                'Feature': f['feature'],
                'Cohens_d': f['cohens_d'],
                'Direction': f['direction'],
                'p_value': f['p_value'],
                'Class_Mean': f['class_mean'],
                'Other_Mean': f['other_mean'],
            })

    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(RESULTS_DIR / "feature_discriminative_summary.csv", index=False)

    # Save
    with open(RESULTS_DIR / "feature_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {RESULTS_DIR / 'feature_analysis.json'}")
    print(f"  Summary table saved to {RESULTS_DIR / 'feature_discriminative_summary.csv'}")
    print(f"  Figures saved to {FIG_DIR}")

    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n  Total figures generated: 8")
    print("  - feature_domain_ablation.png")
    print("  - class_discriminative_features.png")
    print("  - band_power_heatmap.png")
    print("  - feature_radar_comparison.png")
    print("  - feature_violin_distributions.png")
    print("  - cohens_d_effect_heatmap.png")
    print("  - topographic_band_power.png")
    print("  - Figure_feature_analysis.png/.pdf (combined 6-panel)")


if __name__ == '__main__':
    main()

