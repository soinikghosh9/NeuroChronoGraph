#!/usr/bin/env python
"""
Generate Publication-Ready Visualizations from Hold-Out Results.

This script creates all required plots for publication from the 
v3_holdout_results/results.json file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_PATH = Path(__file__).parent.parent / "outputs" / "results" / "v3_holdout_results"
FIGURES_PATH = Path(__file__).parent.parent / "outputs" / "figures"

# Publication style
PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}

# Colors (colorblind-friendly)
COLORS = {
    'AD': '#E64B35',   # Red
    'FTD': '#4DBBD5',  # Cyan  
    'CN': '#00A087',   # Green
    'MCI': '#3C5488',  # Dark Blue
}
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']


def get_metric(data, metric):
    """Helper to get metric with fallback for aggregated_* naming."""
    if metric in data:
        return data[metric]
    elif f'aggregated_{metric}' in data:
        return data[f'aggregated_{metric}']
    return 0


def load_results():
    """Load results from JSON."""
    results_file = RESULTS_PATH / "results.json"
    if not results_file.exists():
        print(f"ERROR: Results not found at {results_file}.")
        print("Please run training (step 18) first to generate results.")
        return None

    with open(results_file) as f:
        return json.load(f)


def plot_confusion_matrices(results):
    """Create confusion matrices for both development and holdout sets."""
    plt.rcParams.update(PUBLICATION_STYLE)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (set_name, ax) in enumerate(zip(['development', 'holdout'], axes)):
        if set_name not in results:
            continue
            
        cm = np.array(results[set_name]['confusion_matrix'])
        
        # Row-normalized (percentage)
        cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        
        # Create annotation with count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
        
        # Plot
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar_kws={'label': 'Count'}, linewidths=0.5, vmin=0)
        
        # Calculate accuracy
        acc = get_metric(results[set_name], 'accuracy') * 100
        n_total = cm.sum()
        n_correct = np.trace(cm)
        
        title = 'Development (LOSO CV)' if set_name == 'development' else 'Hold-Out (Unbiased)'
        ax.set_title(f'{title}\nAccuracy: {acc:.1f}% ({n_correct}/{n_total})', fontsize=13)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'confusion_matrices.png', dpi=300, facecolor='white')
    plt.savefig(FIGURES_PATH / 'confusion_matrices.pdf', dpi=300, facecolor='white')
    plt.close()
    print("  Saved: confusion_matrices.png/pdf")


def plot_roc_curves(results):
    """Plot ROC curves for development and holdout sets."""
    plt.rcParams.update(PUBLICATION_STYLE)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (set_name, ax) in enumerate(zip(['development', 'holdout'], axes)):
        if set_name not in results or 'subject_results' not in results[set_name]:
            continue
        
        subject_results = results[set_name]['subject_results']
        y_true = np.array([s['true'] for s in subject_results])
        y_probs = np.array([s['probs'] for s in subject_results])
        
        # Binarize labels
        y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
        n_classes = 4
        
        # Compute ROC curve for each class
        colors = [COLORS['AD'], COLORS['FTD'], COLORS['CN'], COLORS['MCI']]
        
        for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{cls} (AUC = {roc_auc:.3f})')
        
        # Macro-average
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        mean_auc = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
                label=f'Macro-avg (AUC = {mean_auc:.3f})')
        
        # Reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        title = 'Development (LOSO CV)' if set_name == 'development' else 'Hold-Out (Unbiased)'
        ax.set_title(f'{title}', fontsize=13)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'roc_curves.png', dpi=300, facecolor='white')
    plt.savefig(FIGURES_PATH / 'roc_curves.pdf', dpi=300, facecolor='white')
    plt.close()
    print("  Saved: roc_curves.png/pdf")


def plot_class_sensitivity_comparison(results):
    """Bar chart comparing class-specific sensitivity between dev and holdout."""
    plt.rcParams.update(PUBLICATION_STYLE)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    # Development sensitivities
    dev_sens = [results['development']['per_class'][cls]['sensitivity'] * 100 
                for cls in CLASS_NAMES]
    
    # Holdout sensitivities
    hold_sens = [results['holdout']['per_class'][cls]['sensitivity'] * 100 
                 for cls in CLASS_NAMES]
    
    bars1 = ax.bar(x - width/2, dev_sens, width, label='Development (LOSO)', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, hold_sens, width, label='Hold-Out (Unbiased)',
                   color='coral', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, dev_sens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10)
    for bar, val in zip(bars2, hold_sens):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10)
    
    ax.set_ylabel('Sensitivity (%)')
    ax.set_title('Class-Specific Sensitivity: Development vs Hold-Out')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.set_ylim(0, 115)
    ax.axhline(y=25.0, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'sensitivity_comparison.png', dpi=300, facecolor='white')
    plt.savefig(FIGURES_PATH / 'sensitivity_comparison.pdf', dpi=300, facecolor='white')
    plt.close()
    print("  Saved: sensitivity_comparison.png/pdf")


def plot_performance_metrics_summary(results):
    """Create summary plot of all performance metrics."""
    plt.rcParams.update(PUBLICATION_STYLE)

    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'cohens_kappa']
    metric_labels = ['Accuracy', 'Balanced\nAccuracy', 'F1 (Macro)', "Cohen's\nKappa"]

    dev_vals = [get_metric(results['development'], m) for m in metrics]
    hold_vals = [get_metric(results['holdout'], m) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, dev_vals, width, label='Development (LOSO)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, hold_vals, width, label='Hold-Out (Unbiased)',
                   color='coral', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars1, dev_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    for bar, val in zip(bars2, hold_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

    # Highlight gap
    dev_acc = get_metric(results['development'], 'accuracy')
    hold_acc = get_metric(results['holdout'], 'accuracy')
    gap = dev_acc - hold_acc
    ax.annotate(f'Gap: {gap*100:.1f}%', xy=(0, (dev_vals[0] + hold_vals[0])/2),
                fontsize=11, color='red', ha='center')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'performance_summary.png', dpi=300, facecolor='white')
    plt.savefig(FIGURES_PATH / 'performance_summary.pdf', dpi=300, facecolor='white')
    plt.close()
    print("  Saved: performance_summary.png/pdf")


def plot_subject_predictions(results, set_name='holdout'):
    """Plot individual subject predictions with confidence, grouped by class."""
    plt.rcParams.update(PUBLICATION_STYLE)

    subject_results = results[set_name]['subject_results']

    # Group by true class for better visualization
    class_groups = {0: [], 1: [], 2: [], 3: []}  # AD, FTD, CN, MCI
    for s in subject_results:
        class_groups[s['true']].append(s)

    # Sort within each class by confidence
    for cls in class_groups:
        class_groups[cls].sort(key=lambda x: max(x['probs']), reverse=True)

    # Flatten while keeping class order
    ordered_results = []
    class_boundaries = [0]
    for cls in [0, 1, 2, 3]:  # AD, FTD, CN, MCI
        ordered_results.extend(class_groups[cls])
        class_boundaries.append(len(ordered_results))

    subjects = [s['subject'] for s in ordered_results]
    true_labels = [s['true'] for s in ordered_results]
    correct = [s['correct'] for s in ordered_results]
    confidences = [max(s['probs']) for s in ordered_results]

    # Create figure with sufficient width
    n_samples = len(subjects)
    fig_width = max(16, n_samples * 0.15)  # Dynamic width based on sample count
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Color by correctness with class-specific shading
    class_colors_correct = {'AD': '#66bb6a', 'FTD': '#4fc3f7', 'CN': '#81c784', 'MCI': '#64b5f6'}
    class_colors_wrong = {'AD': '#ef5350', 'FTD': '#ff7043', 'CN': '#e57373', 'MCI': '#f06292'}

    colors = []
    for i, (c, t) in enumerate(zip(correct, true_labels)):
        cls_name = CLASS_NAMES[t]
        colors.append(class_colors_correct[cls_name] if c else class_colors_wrong[cls_name])

    bars = ax.bar(range(n_samples), confidences, color=colors, alpha=0.85, width=0.8)

    # Add class separation lines and labels
    for i, boundary in enumerate(class_boundaries[1:-1]):
        ax.axvline(x=boundary - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    # Add class labels at the bottom
    for i, cls in enumerate(CLASS_NAMES):
        start = class_boundaries[i]
        end = class_boundaries[i + 1]
        mid = (start + end) / 2 - 0.5
        n_cls = end - start
        n_correct_cls = sum(correct[start:end])
        ax.text(mid, -0.08, f'{cls}\n({n_correct_cls}/{n_cls})',
                ha='center', va='top', fontsize=11, fontweight='bold',
                transform=ax.get_xaxis_transform())

    n_correct = sum(correct)
    ax.axhline(y=np.mean(confidences), color='navy', linestyle='--', linewidth=2,
               label=f'Mean Confidence: {np.mean(confidences):.3f}')

    title_set = 'Hold-Out Test Set' if set_name == 'holdout' else 'Development Set'
    ax.set_title(f'{title_set}: Subject Classification\n'
                 f'Total: {n_correct}/{n_samples} Correct ({n_correct/n_samples*100:.1f}%) | '
                 f'Green=Correct, Red=Incorrect', fontsize=13)
    ax.set_xlabel('Samples (Grouped by True Class)', fontsize=12)
    ax.set_ylabel('Prediction Confidence', fontsize=12)

    # Remove individual x-tick labels (too many), show only class boundaries
    ax.set_xticks([])
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Chance')

    # Add legend for class colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#66bb6a', label='Correct'),
        Patch(facecolor='#ef5350', label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for class labels
    plt.savefig(FIGURES_PATH / f'subject_predictions_{set_name}.png', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: subject_predictions_{set_name}.png")


def plot_training_dynamics(results):
    """Plot training dynamics: loss and accuracy curves over epochs."""
    plt.rcParams.update(PUBLICATION_STYLE)

    # Check if training history is available
    if 'training_history' not in results:
        print("  SKIPPED: No training history found in results")
        return

    history = results['training_history']
    n_folds = len(history)

    if n_folds == 0:
        print("  SKIPPED: Empty training history")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors for folds
    fold_colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

    # Panel A: Training Loss
    ax1 = axes[0, 0]
    for fold_idx, fold_hist in enumerate(history):
        epochs = range(1, len(fold_hist['train_loss']) + 1)
        ax1.plot(epochs, fold_hist['train_loss'], color=fold_colors[fold_idx],
                 alpha=0.7, linewidth=1.5, label=f'Fold {fold_idx+1}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('A. Training Loss per Fold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel B: Training Accuracy (Screen)
    ax2 = axes[0, 1]
    for fold_idx, fold_hist in enumerate(history):
        epochs = range(1, len(fold_hist['train_acc']) + 1)
        ax2.plot(epochs, fold_hist['train_acc'], color=fold_colors[fold_idx],
                 alpha=0.7, linewidth=1.5, label=f'Fold {fold_idx+1}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (Screen)')
    ax2.set_title('B. Training Accuracy per Fold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel C: Validation Accuracy (Subtype Balanced)
    ax3 = axes[1, 0]
    for fold_idx, fold_hist in enumerate(history):
        epochs = range(1, len(fold_hist['val_bacc_subtype']) + 1)
        ax3.plot(epochs, fold_hist['val_bacc_subtype'], color=fold_colors[fold_idx],
                 alpha=0.7, linewidth=1.5, label=f'Fold {fold_idx+1}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Balanced Accuracy (Subtype)')
    ax3.set_title('C. Validation Subtype Balanced Accuracy')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax3.grid(True, alpha=0.3)

    # Panel D: Learning Rate Schedule
    ax4 = axes[1, 1]
    for fold_idx, fold_hist in enumerate(history):
        epochs = range(1, len(fold_hist['learning_rate']) + 1)
        ax4.plot(epochs, fold_hist['learning_rate'], color=fold_colors[fold_idx],
                 alpha=0.7, linewidth=1.5, label=f'Fold {fold_idx+1}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('D. Learning Rate Schedule')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'training_dynamics.png', dpi=300, facecolor='white')
    plt.savefig(FIGURES_PATH / 'training_dynamics.pdf', dpi=300, facecolor='white')
    plt.close()
    print("  Saved: training_dynamics.png/pdf")


def plot_combined_summary(results):
    """Create a combined 4-panel publication figure."""
    plt.rcParams.update(PUBLICATION_STYLE)

    fig = plt.figure(figsize=(16, 14))

    # Panel A: Development Confusion Matrix
    ax1 = fig.add_subplot(2, 2, 1)
    cm_dev = np.array(results['development']['confusion_matrix'])
    cm_pct = cm_dev.astype('float') / cm_dev.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f'{cm_dev[i,j]}\n({cm_pct[i,j]:.1f}%)'
                       for j in range(4)] for i in range(4)])
    sns.heatmap(cm_dev, annot=annot, fmt='', cmap='Blues', ax=ax1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    acc_dev = get_metric(results['development'], 'accuracy') * 100
    ax1.set_title(f'A. Development Set (LOSO)\nAccuracy: {acc_dev:.1f}%', fontsize=13)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Panel B: Holdout Confusion Matrix
    ax2 = fig.add_subplot(2, 2, 2)
    cm_hold = np.array(results['holdout']['confusion_matrix'])
    cm_pct = cm_hold.astype('float') / cm_hold.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f'{cm_hold[i,j]}\n({cm_pct[i,j]:.1f}%)' 
                       for j in range(4)] for i in range(4)])
    sns.heatmap(cm_hold, annot=annot, fmt='', cmap='Oranges', ax=ax2,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, linewidths=0.5)
    acc_hold = get_metric(results['holdout'], 'accuracy') * 100
    ax2.set_title(f'B. Hold-Out Test Set (Unbiased)\nAccuracy: {acc_hold:.1f}%', fontsize=13)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Panel C: ROC Curves (Holdout)
    ax3 = fig.add_subplot(2, 2, 3)
    subject_results = results['holdout']['subject_results']
    y_true = np.array([s['true'] for s in subject_results])
    y_probs = np.array([s['probs'] for s in subject_results])
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    
    colors = [COLORS['AD'], COLORS['FTD'], COLORS['CN'], COLORS['MCI']]
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')
    
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax3.set_title('C. ROC Curves (Hold-Out Set)', fontsize=13)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc='lower right')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.02])
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Sensitivity Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    dev_sens = [results['development']['per_class'][cls]['sensitivity'] * 100 
                for cls in CLASS_NAMES]
    hold_sens = [results['holdout']['per_class'][cls]['sensitivity'] * 100 
                 for cls in CLASS_NAMES]
    
    bars1 = ax4.bar(x - width/2, dev_sens, width, label='Development', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, hold_sens, width, label='Hold-Out', color='coral', alpha=0.8)
    
    for bar, val in zip(bars1 + bars2, dev_sens + hold_sens):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', fontsize=9)
    
    ax4.set_title('D. Class-Specific Sensitivity', fontsize=13)
    ax4.set_ylabel('Sensitivity (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(CLASS_NAMES)
    ax4.legend()
    ax4.set_ylim(0, 115)
    ax4.axhline(y=25.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'Figure2_classification_performance.png', dpi=300, facecolor='white')
    plt.savefig(FIGURES_PATH / 'Figure2_classification_performance.pdf', dpi=300, facecolor='white')
    plt.close()
    print("  Saved: Figure2_classification_performance.png/pdf")


def plot_performance_highlights(results):
    """Create a prominent multi-panel figure highlighting the BEST metrics.

    Panel A: Window vs Subject-level accuracy comparison
    Panel B: Hierarchical stage performance (Screening/Staging/Subtyping)
    Panel C: Per-class AUC bars with threshold markers
    Panel D: Bootstrap CI forest plot for key metrics
    """
    plt.rcParams.update(PUBLICATION_STYLE)
    fig = plt.figure(figsize=(16, 12))

    # ---- Load supplementary data ----
    stats_path = Path(__file__).parent.parent / "outputs" / "results" / "statistical_analysis.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    holdout = results.get('holdout', {})
    sr = holdout.get('subject_results', [])
    y_true = np.array([r['true'] for r in sr]) if sr else np.array([])
    y_pred = np.array([r['pred'] for r in sr]) if sr else np.array([])
    y_probs = np.array([r['probs'] for r in sr]) if sr else np.array([])

    # Window-level accuracy
    window_acc = holdout.get('accuracy', 0) * 100

    # Subject-level accuracy from stats
    subj_data = stats.get('subject_level', {})
    subj_acc = subj_data.get('accuracy', 0) * 100 if subj_data else 0
    subj_n = subj_data.get('n_subjects', 0) if subj_data else 0

    # ========== Panel A: Window vs Subject Accuracy ==========
    ax_a = fig.add_subplot(2, 2, 1)
    categories = ['Window-Level\n(per-epoch)', 'Subject-Level\n(majority vote)']
    values = [window_acc, subj_acc]
    bar_colors = ['#5B8FB9', '#2E7D32']
    bars = ax_a.bar(categories, values, color=bar_colors, width=0.55,
                    edgecolor='white', linewidth=2, zorder=3)

    # Value annotations
    for bar, val in zip(bars, values):
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 fontsize=20, fontweight='bold', color=bar.get_facecolor())

    # Improvement arrow
    if subj_acc > window_acc:
        delta = subj_acc - window_acc
        ax_a.annotate(f'+{delta:.1f}%',
                     xy=(1, subj_acc - 2), xytext=(0.5, (window_acc + subj_acc)/2),
                     fontsize=14, fontweight='bold', color='#D32F2F',
                     arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2),
                     ha='center', va='center')

    ax_a.set_ylim(0, 110)
    ax_a.set_ylabel('Accuracy (%)', fontsize=13)
    ax_a.set_title('A. Window vs Subject-Level Accuracy', fontsize=14, fontweight='bold')
    ax_a.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax_a.legend(loc='upper left', fontsize=9)
    ax_a.grid(axis='y', alpha=0.3, zorder=0)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # ========== Panel B: Hierarchical Stage Performance ==========
    ax_b = fig.add_subplot(2, 2, 2)

    # Read hierarchical metrics from development CV statistics
    cv_stats = results.get('development', {}).get('cv_statistics', {})
    screen_acc = cv_stats.get('acc_screen', {}).get('mean', 0) * 100
    stage_bacc = cv_stats.get('bacc_stage', {}).get('mean', 0) * 100
    subtype_bacc = cv_stats.get('Balanced_Acc_Subtype', {}).get('mean', 0) * 100

    stages = ['Screening\n(CN vs Impaired)', 'Staging\n(MCI vs Dementia)', 'Subtyping\n(AD vs FTD)']
    stage_vals = [screen_acc, stage_bacc, subtype_bacc]
    stage_colors = ['#1565C0', '#7B1FA2', '#E65100']

    bars_b = ax_b.barh(stages, stage_vals, color=stage_colors, height=0.5,
                       edgecolor='white', linewidth=2, zorder=3)

    for bar, val in zip(bars_b, stage_vals):
        if val > 0:
            ax_b.text(val + 1, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', ha='left', va='center',
                     fontsize=14, fontweight='bold')

    ax_b.set_xlim(0, 110)
    ax_b.set_xlabel('Performance (%)', fontsize=13)
    ax_b.set_title('B. Hierarchical Stage Performance', fontsize=14, fontweight='bold')
    ax_b.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax_b.grid(axis='x', alpha=0.3, zorder=0)
    ax_b.invert_yaxis()
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # ========== Panel C: Per-Class AUC ==========
    ax_c = fig.add_subplot(2, 2, 3)

    if len(y_probs) > 0 and y_probs.shape[1] == 4:
        y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
        class_aucs = []
        for i in range(4):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            class_aucs.append(auc(fpr, tpr))

        colors_list = [COLORS[c] for c in CLASS_NAMES]
        bars_c = ax_c.bar(CLASS_NAMES, class_aucs, color=colors_list, width=0.55,
                         edgecolor='white', linewidth=2, zorder=3)

        for bar, val in zip(bars_c, class_aucs):
            ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=13, fontweight='bold')

        ax_c.set_ylim(0.85, 1.02)
        ax_c.axhline(y=0.90, color='orange', linestyle='--', alpha=0.6, label='Excellent (0.90)')
        ax_c.axhline(y=0.95, color='green', linestyle='--', alpha=0.6, label='Outstanding (0.95)')
        ax_c.legend(loc='lower right', fontsize=9)
    else:
        ax_c.text(0.5, 0.5, 'AUC data unavailable', ha='center', va='center', fontsize=14)

    ax_c.set_ylabel('AUC (one-vs-rest)', fontsize=13)
    ax_c.set_title('C. Per-Class AUC (Hold-Out)', fontsize=14, fontweight='bold')
    ax_c.grid(axis='y', alpha=0.3, zorder=0)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # ========== Panel D: Bootstrap CI Forest Plot ==========
    ax_d = fig.add_subplot(2, 2, 4)

    bootstrap = stats.get('bootstrap_ci', {})
    metrics_ci = []
    metric_labels = []
    metric_points = []

    for metric_key, label in [('accuracy', 'Accuracy'),
                               ('f1_macro', 'Macro F1'),
                               ('balanced_accuracy', 'Balanced Acc.'),
                               ('cohens_kappa', "Cohen's κ")]:
        ci_data = bootstrap.get(metric_key, {})
        if ci_data:
            mean_val = ci_data.get('point', ci_data.get('mean', 0))
            lo = ci_data.get('lower', ci_data.get('ci_lower', mean_val))
            hi = ci_data.get('upper', ci_data.get('ci_upper', mean_val))
            metrics_ci.append((lo, hi))
            metric_labels.append(label)
            metric_points.append(mean_val)

    if metrics_ci:
        y_pos = np.arange(len(metric_labels))
        ci_colors = ['#1565C0', '#2E7D32', '#7B1FA2', '#E65100']

        for i, (label, point, (lo, hi)) in enumerate(zip(metric_labels, metric_points, metrics_ci)):
            color = ci_colors[i % len(ci_colors)]
            ax_d.plot([lo, hi], [i, i], color=color, linewidth=3, solid_capstyle='round', zorder=3)
            ax_d.plot(point, i, 'o', color=color, markersize=10, zorder=4,
                     markeredgecolor='white', markeredgewidth=2)
            ax_d.text(hi + 0.005, i, f'{point:.3f} [{lo:.3f}, {hi:.3f}]',
                     va='center', fontsize=10, fontweight='bold')

        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels(metric_labels, fontsize=12)
        ax_d.set_xlabel('Value', fontsize=13)
        ax_d.set_xlim(0.7, 1.0)
        ax_d.invert_yaxis()
    else:
        ax_d.text(0.5, 0.5, 'Bootstrap CI data unavailable\nRun Phase 6 first',
                 ha='center', va='center', fontsize=12)

    ax_d.set_title('D. Bootstrap 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax_d.grid(axis='x', alpha=0.3, zorder=0)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # ---- Suptitle ----
    plt.suptitle('NeuroChronoGraph: Performance Highlights',
                fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = FIGURES_PATH / 'Figure3_performance_highlights.png'
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name} (+ .pdf)")


def main():
    """Generate all publication visualizations."""
    print("=" * 60)
    print("Generating Publication-Ready Visualizations")
    print("=" * 60)

    # Create output directory
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\nLoading results...")
    results = load_results()

    if results is None:
        print("ERROR: No results file found. Please run training first (step 18).")
        print("Skipping plot generation.")
        return

    # Handle key name differences between development and holdout
    dev_acc = results['development'].get('accuracy', results['development'].get('aggregated_accuracy', 0))
    hold_acc = results['holdout'].get('accuracy', 0)
    print(f"  Development: {results['development']['n_subjects']} subjects, "
          f"Accuracy: {dev_acc*100:.1f}%")
    print(f"  Hold-Out: {results['holdout']['n_subjects']} subjects, "
          f"Accuracy: {hold_acc*100:.1f}%")
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("\n1. Confusion Matrices")
    plot_confusion_matrices(results)
    
    print("\n2. ROC Curves")
    plot_roc_curves(results)
    
    print("\n3. Sensitivity Comparison")
    plot_class_sensitivity_comparison(results)
    
    print("\n4. Performance Metrics Summary")
    plot_performance_metrics_summary(results)
    
    print("\n5. Subject Predictions")
    plot_subject_predictions(results, 'holdout')

    print("\n6. Training Dynamics")
    plot_training_dynamics(results)

    print("\n7. Combined Publication Figure")
    plot_combined_summary(results)

    print("\n8. Performance Highlights")
    plot_performance_highlights(results)
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {FIGURES_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
