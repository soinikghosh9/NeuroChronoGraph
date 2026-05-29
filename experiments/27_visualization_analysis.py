"""
Visualization Analysis for Publication.

Generates publication-quality figures:
1. t-SNE/UMAP embeddings colored by class and uncertainty
2. Calibration diagram (reliability plot + ECE)
3. Precision-Recall curves per class
4. Per-fold box plots for CV stability
5. ROC curves per class

Saves all figures to outputs/figures/publication/
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, auc, confusion_matrix, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
from src.utils.reproducibility import DEFAULT_SEED, set_global_seed
SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)

from src.config.config import VISUALIZATION_CONFIG

CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
COLORS = {
    'AD': '#D55E00', 'FTD': '#0072B2', 'CN': '#009E73', 'MCI': '#E69F00'
}
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "publication"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif',
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def load_holdout_data():
    """Load holdout predictions from results JSON."""
    holdout_path = RESULTS_DIR / "v3_holdout_results" / "results.json"
    if not holdout_path.exists():
        print(f"ERROR: {holdout_path} not found!")
        return None
    
    with open(holdout_path) as f:
        data = json.load(f)
    
    holdout = data.get('holdout', {})
    sr = holdout.get('subject_results', [])
    
    if not sr:
        print("ERROR: No subject results found!")
        return None
    
    return {
        'y_true': np.array([r['true'] for r in sr]),
        'y_pred': np.array([r['pred'] for r in sr]),
        'probs': np.array([r['probs'] for r in sr]),
        'subjects': [r['subject'] for r in sr],
    }


# ============================================================
# 1. t-SNE Embedding Visualization
# ============================================================

def plot_tsne_embeddings(data, save_path):
    """Generate t-SNE plot of probability space colored by class."""
    print("  Generating t-SNE embedding plot...")
    
    probs = data['probs']
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    # t-SNE on probability vectors
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    embeddings = tsne.fit_transform(probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: colored by true class
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == cls_idx
        axes[0].scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=COLORS[cls_name], s=20, alpha=0.6, label=cls_name,
                       edgecolors='white', linewidths=0.3)
    axes[0].set_title('True Class Labels')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend(framealpha=0.9)
    
    # Right: colored by prediction correctness
    correct = y_true == y_pred
    axes[1].scatter(embeddings[correct, 0], embeddings[correct, 1],
                   c='#2ecc71', s=20, alpha=0.5, label='Correct', 
                   edgecolors='white', linewidths=0.3)
    axes[1].scatter(embeddings[~correct, 0], embeddings[~correct, 1],
                   c='#e74c3c', s=30, alpha=0.7, label='Misclassified',
                   marker='x', linewidths=1.5)
    axes[1].set_title('Classification Accuracy')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend(framealpha=0.9)
    
    plt.suptitle('t-SNE Visualization of Hold-out Predictions', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# 2. Calibration Diagram
# ============================================================

def plot_calibration_diagram(data, save_path):
    """Generate calibration (reliability) diagram with ECE."""
    print("  Generating calibration diagram...")
    
    probs = data['probs']
    y_true = data['y_true']
    n_bins = 10
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Per-class calibration
    eces = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_binary = (y_true == cls_idx).astype(int)
        y_scores = probs[:, cls_idx]
        
        prob_true, prob_pred = calibration_curve(y_binary, y_scores, n_bins=n_bins, strategy='uniform')
        
        axes[0].plot(prob_pred, prob_true, 'o-', color=COLORS[cls_name],
                    label=cls_name, linewidth=2, markersize=5)
        
        # Compute ECE for this class
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (y_scores >= bin_boundaries[i]) & (y_scores < bin_boundaries[i+1])
            if mask.sum() > 0:
                avg_conf = y_scores[mask].mean()
                avg_acc = y_binary[mask].mean()
                ece += mask.sum() * abs(avg_conf - avg_acc)
        ece /= len(y_scores)
        eces.append(ece)
    
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curves (Reliability Diagram)')
    axes[0].legend(loc='upper left')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_aspect('equal')
    
    # ECE bar chart
    bars = axes[1].bar(CLASS_NAMES, eces, color=[COLORS[c] for c in CLASS_NAMES], alpha=0.8)
    overall_ece = np.mean(eces)
    axes[1].axhline(y=overall_ece, color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean ECE = {overall_ece:.3f}')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Expected Calibration Error (ECE)')
    axes[1].set_title('Per-Class Calibration Error')
    axes[1].legend()
    
    # Add values on bars
    for bar, ece in zip(bars, eces):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{ece:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Calibration Analysis', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")
    
    return {'per_class_ece': dict(zip(CLASS_NAMES, eces)), 'mean_ece': overall_ece}


# ============================================================
# 3. Precision-Recall Curves
# ============================================================

def plot_precision_recall_curves(data, save_path):
    """Generate precision-recall curves per class."""
    print("  Generating precision-recall curves...")
    
    probs = data['probs']
    y_true = data['y_true']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_binary = (y_true == cls_idx).astype(int)
        y_scores = probs[:, cls_idx]
        
        precision, recall, _ = precision_recall_curve(y_binary, y_scores)
        ap = average_precision_score(y_binary, y_scores)
        
        ax.plot(recall, precision, color=COLORS[cls_name], linewidth=2,
               label=f'{cls_name} (AP={ap:.3f})')
    
    # Baseline (random)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        prevalence = (y_true == cls_idx).mean()
        ax.axhline(y=prevalence, color=COLORS[cls_name], linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# 4. ROC Curves Per Class
# ============================================================

def plot_roc_curves(data, save_path):
    """Generate ROC curves per class."""
    print("  Generating ROC curves...")
    
    probs = data['probs']
    y_true = data['y_true']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_binary = (y_true == cls_idx).astype(int)
        y_scores = probs[:, cls_idx]
        
        fpr, tpr, _ = roc_curve(y_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=COLORS[cls_name], linewidth=2,
               label=f'{cls_name} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# 5. Per-Fold Box Plots
# ============================================================

def plot_fold_performance(save_path):
    """Generate per-fold performance box plots."""
    print("  Generating per-fold performance plots...")
    
    cv_path = RESULTS_DIR / "cv_results.csv"
    if not cv_path.exists():
        print(f"    cv_results.csv not found, skipping")
        return
    
    df = pd.read_csv(cv_path)
    
    metrics = {
        'Screening\nAccuracy': 'acc_screen',
        'Screening\nκ': 'cohens_kappa_screen',
        'Staging\nBACC': None,  # Need to compute
        'Subtyping\nBACC': 'Balanced_Acc_Subtype',
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    colors_fold = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, (label, col) in enumerate(metrics.items()):
        ax = axes[i]
        
        if col and col in df.columns:
            values = df[col].values
        elif 'bacc_stage' in df.columns:
            values = df['bacc_stage'].values
        elif 'acc_stage' in df.columns:
            values = df['acc_stage'].values
        else:
            values = np.zeros(len(df))
        
        # Bar plot with individual fold values
        bars = ax.bar(range(1, len(values)+1), values, color=colors_fold[:len(values)], alpha=0.8)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.5)
        ax.fill_between([0.5, len(values)+0.5], mean_val-std_val, mean_val+std_val,
                       alpha=0.1, color='red')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(label)
        ax.set_title(f'μ={mean_val:.3f} ± σ={std_val:.3f}', fontsize=10)
        ax.set_xticks(range(1, len(values)+1))
        ax.set_ylim([max(0, min(values) - 0.1), min(1, max(values) + 0.1)])
    
    plt.suptitle('Cross-Validation Stability (Per-Fold Performance)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("PUBLICATION VISUALIZATION ANALYSIS")
    print("=" * 70)
    
    data = load_holdout_data()
    if data is None:
        return
    
    print(f"\nLoaded {len(data['y_true'])} holdout predictions")
    
    # 1. t-SNE
    plot_tsne_embeddings(data, FIG_DIR / "tsne_embeddings.png")
    
    # 2. Calibration
    ece_results = plot_calibration_diagram(data, FIG_DIR / "calibration_diagram.png")
    
    # 3. Precision-Recall
    plot_precision_recall_curves(data, FIG_DIR / "precision_recall_curves.png")
    
    # 4. ROC curves
    plot_roc_curves(data, FIG_DIR / "roc_curves_per_class.png")
    
    # 5. Per-fold stability
    plot_fold_performance(FIG_DIR / "fold_performance.png")
    
    # Save ECE results
    if ece_results:
        with open(RESULTS_DIR / "calibration_results.json", 'w') as f:
            json.dump(ece_results, f, indent=2)
    
    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
