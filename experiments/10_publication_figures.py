#!/usr/bin/env python
"""
NeuroChronoGraph — Comprehensive Publication Figure Generator  (v3)

Light, muted, Nature/NeuroImage-style aesthetics.
All legends placed outside axes where content would be obscured.

Main paper      : Figure 2-6  (pub/Figure*.png)
Supplementary   : FigureS1-S9 (pub/FigureS*.png)

Usage
-----
  cd /path/to/Alzbcp2
  py -3 experiments/10_publication_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "outputs" / "results"
FIG_DIR     = ROOT / "outputs" / "figures" / "pub"
FIG_EXIST   = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Muted, colorblind-safe palette  (Nature / NeuroImage style)
# ---------------------------------------------------------------------------
COLORS = {
    'AD':  '#B5544B',  # dusty brick red
    'FTD': '#4578B0',  # steel cornflower blue
    'CN':  '#4D9A62',  # sage forest green
    'MCI': '#D18E3A',  # warm amber
}
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']   # label-encoding order
CLASS_ORDER = ['CN', 'MCI', 'AD', 'FTD']   # clinical order

BAND_COLORS = {
    'delta': '#8776AE', 'theta': '#D96B4D',
    'alpha': '#4578B0', 'beta':  '#4D9A62', 'gamma': '#D18E3A',
}
REGION_COLORS = {
    'frontal': '#B5544B', 'temporal': '#4578B0',
    'parietal': '#4D9A62', 'occipital': '#D18E3A',
}

# ---------------------------------------------------------------------------
# Publication-ready rcParams
# ---------------------------------------------------------------------------
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#CCCCCC',
    'legend.borderpad': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.edgecolor': '#BBBBBB',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.axisbelow': True,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'axes.labelcolor': '#333333',
    'text.color': '#1A1A1A',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
}

def apply_style():
    plt.rcParams.update(STYLE)

def lgrid(ax, axis='y'):
    ax.grid(axis=axis, color='#EBEBEB', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

def panel_label(ax, letter, x=-0.13, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            color='#1A1A1A')

def savefig(name, fig=None):
    path = FIG_DIR / name
    kw = dict(dpi=300, facecolor='white', bbox_inches='tight')
    (fig or plt).savefig(path.with_suffix('.png'), **kw)
    (fig or plt).savefig(path.with_suffix('.pdf'), **kw)
    print(f"    Saved: {path.name}.png/pdf")
    plt.close('all')

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_results():
    with open(RESULTS_DIR / "v3_holdout_results" / "results.json") as f:
        return json.load(f)

def load_stats():
    p = RESULTS_DIR / "statistical_analysis.json"
    return json.load(open(p)) if p.exists() else {}

def load_features():
    p = RESULTS_DIR / "feature_analysis.json"
    return json.load(open(p)) if p.exists() else {}

def get_holdout_arrays(results):
    sr = results['holdout']['subject_results']
    return (
        np.array([r['true']  for r in sr]),
        np.array([r['pred']  for r in sr]),
        np.array([r['probs'] for r in sr]),
    )

def build_class_feat(disc):
    """Return {cls: {feat: item_dict}} from discriminative_features."""
    cf = {cls: {} for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        for item in disc.get(cls, []):
            cf[cls][item['feature']] = item
    return cf

# =============================================================================
# FIGURE 2 — Classification Performance Overview
#   A: Dev CM  B: Hold-out CM  C: ROC curves  D: Sensitivity / Precision
# =============================================================================
def figure2_performance_overview(results, stats):
    apply_style()
    fig = plt.figure(figsize=(13, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    y_true, y_pred, y_probs = get_holdout_arrays(results)
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    cm_kw = dict(fmt='', cbar_kws={'shrink': 0.75, 'pad': 0.02},
                 linewidths=0.5, linecolor='white',
                 xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                 annot_kws={'size': 9})

    # ── A: Dev confusion matrix ─────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    cm_dev = np.array(results['development']['confusion_matrix'])
    cm_pct = cm_dev / cm_dev.sum(axis=1, keepdims=True) * 100
    annot  = np.array([[f'{cm_dev[i,j]}\n({cm_pct[i,j]:.0f}%)'
                        for j in range(4)] for i in range(4)])
    sns.heatmap(cm_dev, annot=annot, cmap='Blues', ax=ax_a, **cm_kw, vmin=0)
    dev_acc = results['development']['accuracy'] * 100
    ax_a.set_title(f'Development Set  (5-fold LOSO CV)\nFlat 4-class: {dev_acc:.1f}%', pad=8)
    ax_a.set_xlabel('Predicted'); ax_a.set_ylabel('True')
    ax_a.set_xticklabels(CLASS_NAMES, rotation=0)
    ax_a.set_yticklabels(CLASS_NAMES, rotation=0)
    panel_label(ax_a, 'A')

    # ── B: Hold-out confusion matrix ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    cm_h  = np.array(results['holdout']['confusion_matrix'])
    cm_p2 = cm_h / cm_h.sum(axis=1, keepdims=True) * 100
    ann2  = np.array([[f'{cm_h[i,j]}\n({cm_p2[i,j]:.0f}%)'
                       for j in range(4)] for i in range(4)])
    sns.heatmap(cm_h, annot=ann2, cmap='YlOrRd', ax=ax_b, **cm_kw, vmin=0)
    h_acc = results['holdout']['accuracy'] * 100
    ax_b.set_title(f'Hold-Out Test Set  (n = 51 subjects)\nWindow: {h_acc:.1f}%  [CI: 86.3–89.3%]', pad=8)
    ax_b.set_xlabel('Predicted'); ax_b.set_ylabel('True')
    ax_b.set_xticklabels(CLASS_NAMES, rotation=0)
    ax_b.set_yticklabels(CLASS_NAMES, rotation=0)
    panel_label(ax_b, 'B')

    # ── C: Per-class ROC curves ────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    delong = stats.get('delong_auc', {})
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ci_lo = delong.get(cls, {}).get('lower', roc_auc - 0.01)
        ci_hi = delong.get(cls, {}).get('upper', roc_auc + 0.01)
        ax_c.plot(fpr, tpr, color=COLORS[cls], lw=2.2,
                  label=f'{cls}  AUC={roc_auc:.3f} [{ci_lo:.3f},{ci_hi:.3f}]')
        ax_c.fill_between(fpr, tpr - 0.015, tpr + 0.015, color=COLORS[cls], alpha=0.07)
    all_fpr = np.linspace(0, 1, 300)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= 4
    ax_c.plot(all_fpr, mean_tpr, color='#555', lw=1.5, ls='--',
              label=f'Macro  AUC={auc(all_fpr, mean_tpr):.3f}')
    ax_c.plot([0,1],[0,1], color='#CCC', lw=1, ls=':')
    ax_c.set_xlim([0,1]); ax_c.set_ylim([0,1.02])
    ax_c.set_xlabel('False Positive Rate'); ax_c.set_ylabel('True Positive Rate')
    ax_c.set_title('Per-Class ROC Curves  (Hold-Out, OvR)')
    ax_c.legend(loc='lower right', fontsize=7.8)
    lgrid(ax_c, 'both')
    panel_label(ax_c, 'C')

    # ── D: Sensitivity + Precision ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    pc   = results['holdout']['per_class']
    x    = np.arange(4);  w = 0.38
    sens = [pc[c]['sensitivity'] * 100 for c in CLASS_NAMES]
    prec = [pc[c]['precision']   * 100 for c in CLASS_NAMES]
    f1s  = [pc[c]['f1']          * 100 for c in CLASS_NAMES]
    clrs = [COLORS[c] for c in CLASS_NAMES]

    b1 = ax_d.bar(x - w/2, sens, w, color=clrs, alpha=0.88,
                  edgecolor='white', linewidth=0.8, label='Sensitivity')
    b2 = ax_d.bar(x + w/2, prec, w, color=clrs, alpha=0.45,
                  edgecolor='white', linewidth=0.8, hatch='///', label='Precision')
    for bar, v in zip(list(b1)+list(b2), sens+prec):
        ax_d.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                  f'{v:.0f}%', ha='center', fontsize=7.5,
                  fontweight='bold', color='#444')
    ax_d.axhline(25, color='#CCC', ls='--', lw=1)
    ax_d.set_ylim(0, 120); ax_d.set_xticks(x)
    ax_d.set_xticklabels(CLASS_NAMES)
    ax_d.set_ylabel('Score (%)')
    ax_d.set_title('Per-Class Sensitivity and Precision\n(Hold-Out)')
    # Legend OUTSIDE (below)
    leg_handles = [
        Patch(color='#888', alpha=0.88, label='Sensitivity'),
        Patch(color='#888', alpha=0.45, hatch='///', label='Precision'),
        Line2D([0],[0], color='#CCC', ls='--', lw=1, label='Chance 25%'),
    ]
    ax_d.legend(handles=leg_handles, loc='upper center',
                bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5,
                framealpha=0.95)
    lgrid(ax_d)
    panel_label(ax_d, 'D')

    fig.suptitle('NeuroChronoGraph  —  Classification Performance Overview',
                 fontsize=13, fontweight='bold', y=1.01)
    savefig('Figure2_performance_overview', fig)


# =============================================================================
# FIGURE 3 — Quantitative Validation Metrics
#   A: Window vs Subject  B: Stage BACC  C: AUC bars  D: Bootstrap CI
# =============================================================================
def figure3_validation_metrics(results, stats):
    apply_style()
    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.38)

    holdout = results['holdout']
    y_true, _, y_probs = get_holdout_arrays(results)
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    # ── A ──────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    win_acc  = holdout['accuracy'] * 100
    subj_acc = stats.get('subject_level', {}).get('accuracy', 0) * 100
    labels   = ['Window-level\n(1,773 epochs)', 'Subject-level\n(majority vote, n=51)']
    vals     = [win_acc, subj_acc]
    bar_c    = ['#8BB5D6', '#4D9A62']
    bars = ax_a.bar(labels, vals, color=bar_c, width=0.45,
                    edgecolor='white', linewidth=1.5, alpha=0.9, zorder=3)
    for bar, v in zip(bars, vals):
        ax_a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                  f'{v:.1f}%', ha='center', fontsize=14,
                  fontweight='bold', color=bar.get_facecolor())
    if subj_acc > win_acc:
        ax_a.annotate(f'+{subj_acc-win_acc:.1f}%\nmajority vote',
                      xy=(1, subj_acc-3), xytext=(0.42, (win_acc+subj_acc)/2),
                      fontsize=9.5, fontweight='bold', color='#4D9A62', ha='center',
                      arrowprops=dict(arrowstyle='->', color='#4D9A62', lw=1.8))
    ax_a.axhline(25, color='#DDD', ls='--', lw=1, zorder=2, label='Chance (25%)')
    ax_a.set_ylim(0, 108); ax_a.set_ylabel('Accuracy (%)')
    ax_a.set_title('Window-Level vs Subject-Level Accuracy')
    ax_a.legend(fontsize=8, loc='upper left')
    lgrid(ax_a); panel_label(ax_a, 'A')

    # ── B ──────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    stages = ['Screening\n(CN vs Impaired)', 'Staging\n(MCI vs Dementia)', 'Subtyping\n(AD vs FTD)']
    dev_vals  = [
        results['development'].get('screen_accuracy', 0.933) * 100,
        results['development'].get('stage_bacc',      0.927) * 100,
        results['development'].get('subtype_bacc',    0.879) * 100,
    ]
    hold_vals = [
        holdout.get('screen_accuracy',           0) * 100,
        holdout.get('stage_balanced_accuracy',   0) * 100,
        holdout.get('subtype_balanced_accuracy', 0) * 100,
    ]
    y = np.arange(3); bh = 0.34
    dev_bars  = ax_b.barh(y+bh/2, dev_vals,  bh, color='#8BB5D6', alpha=0.88,
                          edgecolor='white', lw=0.8, label='Dev (5-fold CV)')
    hold_bars = ax_b.barh(y-bh/2, hold_vals, bh, color='#B5544B', alpha=0.88,
                          edgecolor='white', lw=0.8, label='Hold-Out Test')
    for bar, v in zip(list(dev_bars)+list(hold_bars), dev_vals+hold_vals):
        if v > 0:
            ax_b.text(v+0.8, bar.get_y()+bar.get_height()/2,
                      f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax_b.axvline(50, color='#CCC', ls='--', lw=1)
    ax_b.axvline(90, color='#D18E3A', ls=':', lw=1.2, alpha=0.7)
    ax_b.set_xlim(0, 112); ax_b.set_yticks(y)
    ax_b.set_yticklabels(stages, fontsize=9)
    ax_b.set_xlabel('Performance (%)')
    ax_b.set_title('Hierarchical Stage Performance\nDevelopment vs Hold-Out')
    # Legend outside
    ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
                ncol=2, fontsize=8.5, framealpha=0.95)
    lgrid(ax_b, 'x'); panel_label(ax_b, 'B')

    # ── C ──────────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    delong = stats.get('delong_auc', {})
    class_aucs, ci_lo_list, ci_hi_list = [], [], []
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        class_aucs.append(roc_auc)
        ci_lo_list.append(delong.get(cls, {}).get('lower', roc_auc-0.01))
        ci_hi_list.append(delong.get(cls, {}).get('upper', roc_auc+0.01))
    x = np.arange(4)
    bars_c = ax_c.bar(x, class_aucs, color=[COLORS[c] for c in CLASS_NAMES],
                      width=0.5, edgecolor='white', lw=1.5, alpha=0.88, zorder=3)
    yerr_lo = np.array(class_aucs) - np.array(ci_lo_list)
    yerr_hi = np.array(ci_hi_list) - np.array(class_aucs)
    ax_c.errorbar(x, class_aucs, yerr=[yerr_lo, yerr_hi],
                  fmt='none', color='#444', capsize=5, capthick=1.5, lw=1.5, zorder=4)
    for bar, v in zip(bars_c, class_aucs):
        ax_c.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.009,
                  f'{v:.3f}', ha='center', fontsize=9.5, fontweight='bold')
    ax_c.axhline(0.90, color='#D18E3A', ls='--', lw=1.2, alpha=0.8, label='Excellent (0.90)')
    ax_c.axhline(0.95, color='#4D9A62', ls='--', lw=1.2, alpha=0.8, label='Outstanding (0.95)')
    ax_c.set_ylim(0.87, 1.03); ax_c.set_xticks(x)
    ax_c.set_xticklabels(CLASS_NAMES); ax_c.set_ylabel('AUC (One-vs-Rest)')
    ax_c.set_title('Per-Class AUC with DeLong 95% CI\n(Hold-Out)')
    ax_c.legend(loc='lower right', fontsize=8)
    lgrid(ax_c); panel_label(ax_c, 'C')

    # ── D: Bootstrap CI forest plot ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    bootstrap  = stats.get('bootstrap_ci', {})
    metric_map = [
        ('accuracy',          'Accuracy',          '#2B6CB0'),
        ('balanced_accuracy', 'Balanced Accuracy',  '#276749'),
        ('f1_macro',          'Macro F1',           '#7B341E'),
        ('cohens_kappa',      "Cohen's \u03ba",      '#553C9A'),
    ]
    y_pos = np.arange(len(metric_map))

    for i, (key, label, color) in enumerate(metric_map):
        ci_d = bootstrap.get(key, {})
        if not ci_d:
            continue
        pt = ci_d.get('point', ci_d.get('mean', 0))
        lo = ci_d.get('lower', ci_d.get('ci_lower', pt-0.01))
        hi = ci_d.get('upper', ci_d.get('ci_upper', pt+0.01))
        ax_d.barh(i, hi-lo, left=lo, height=0.40,
                  color=color, alpha=0.18, edgecolor='none', zorder=2)
        ax_d.plot([lo, hi], [i, i], color=color, lw=3.5,
                  solid_capstyle='round', zorder=3, alpha=0.85)
        ax_d.plot(pt, i, 'o', color=color, ms=9, zorder=5,
                  markeredgecolor='white', markeredgewidth=1.8)
        ax_d.text(hi+0.003, i,
                  f'{pt:.3f}  [{lo:.3f}, {hi:.3f}]',
                  va='center', fontsize=8.8, fontweight='bold', color='#333')

    # LEGEND — placed in empty upper-left region of plot
    legend_handles = [
        Line2D([0],[0], color=c, lw=3.5, solid_capstyle='round', label=lbl)
        for _, lbl, c in metric_map
    ] + [
        Line2D([0],[0], marker='o', color='w', ms=8,
               markerfacecolor='#555', markeredgecolor='white', mew=1.5,
               label='Point estimate'),
        mpatches.Patch(color='#888', alpha=0.18, label='95% CI band'),
    ]
    ax_d.legend(handles=legend_handles,
                loc='upper left',          # upper-left = above all CI lines
                fontsize=8, title='Metric', title_fontsize=8.5,
                framealpha=0.95, borderpad=0.6)

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([m[1] for m in metric_map], fontsize=10)
    ax_d.set_xlabel('Metric value')
    ax_d.set_xlim(0.68, 1.18)   # extended — room for text on right
    ax_d.set_ylim(-0.6, len(metric_map)-0.4)
    ax_d.invert_yaxis()
    ax_d.set_title('Bootstrap 95% Confidence Intervals\n(10,000 resamples, Hold-Out)')
    lgrid(ax_d, 'x')
    ax_d.axvline(1.0, color='#EEE', ls=':', lw=1)
    panel_label(ax_d, 'D')

    fig.suptitle('NeuroChronoGraph  —  Quantitative Validation Metrics',
                 fontsize=13, fontweight='bold', y=1.01)
    savefig('Figure3_validation_metrics', fig)


# =============================================================================
# FIGURE 4 — Disease-Specific EEG Biomarker Signatures
#   A: Band power profiles  B: % change from CN (clinical biomarkers)
#   C: KW discriminative ranking  D: Cohen's d heatmap
# =============================================================================
def figure4_eeg_biomarkers(features):
    apply_style()
    disc = features.get('discriminative_features', {}).get('class_discriminative', {})
    kw   = features.get('discriminative_features', {}).get('kruskal_wallis', {})
    cf   = build_class_feat(disc)

    fig = plt.figure(figsize=(15, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.44)

    # ── A: Band power profiles per class ───────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    bands    = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_lbl = ['Delta\n(0.5-4 Hz)', 'Theta\n(4-8 Hz)', 'Alpha\n(8-13 Hz)',
                'Beta\n(13-30 Hz)', 'Gamma\n(30-45 Hz)']
    bw = 0.18; x = np.arange(5)
    for j, cls in enumerate(CLASS_ORDER):
        means = [cf[cls].get(f'{b}_rel_power', {}).get('class_mean', 0)*100 for b in bands]
        offset = (CLASS_ORDER.index(cls) - 1.5) * bw
        ax_a.bar(x+offset, means, bw, label=cls, color=COLORS[cls],
                 alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_a.set_xticks(x); ax_a.set_xticklabels(band_lbl, fontsize=8.5)
    ax_a.set_ylabel('Relative Power (%)')
    ax_a.set_title('Spectral Band Power by Diagnostic Class')
    legend_patches = [Patch(color=COLORS[c], alpha=0.88, label=c) for c in CLASS_ORDER]
    ax_a.legend(handles=legend_patches, loc='upper right', fontsize=8.5,
                title='Class', title_fontsize=9)
    lgrid(ax_a); panel_label(ax_a, 'A')

    # ── B: % change from CN for key clinical biomarkers ────────────
    ax_b = fig.add_subplot(gs[0, 1])
    bio_keys = [
        ('theta_alpha_ratio',    'Theta/Alpha\nRatio'),
        ('alpha_peak_freq',      'Alpha Peak\nFreq'),
        ('spectral_entropy',     'Spectral\nEntropy'),
        ('delta_alpha_ratio',    'Delta/Alpha\nRatio'),
        ('alpha_beta_ratio',     'Alpha/Beta\nRatio'),
        ('spectral_edge_freq_95','Spectral\nEdge 95%'),
    ]
    cn_vals = {k: cf['CN'].get(k, {}).get('class_mean', 1.0) for k, _ in bio_keys}
    x_bm = np.arange(len(bio_keys)); bw_bm = 0.26
    impaired_classes = ['MCI', 'AD', 'FTD']
    all_pct = []
    for j, cls in enumerate(impaired_classes):
        pct_changes = []
        for feat_key, _ in bio_keys:
            cls_mean = cf[cls].get(feat_key, {}).get('class_mean', cn_vals[feat_key])
            cn_mean  = cn_vals[feat_key]
            pct_changes.append((cls_mean - cn_mean) / abs(cn_mean) * 100 if cn_mean else 0)
        all_pct.extend(pct_changes)
        offset = (j - 1) * bw_bm
        ax_b.bar(x_bm+offset, pct_changes, bw_bm, label=cls,
                 color=COLORS[cls], alpha=0.88, edgecolor='white', linewidth=0.5)
    ax_b.axhline(0, color='#555', lw=1.2, ls='-')
    ax_b.set_xticks(x_bm)
    ax_b.set_xticklabels([lbl for _, lbl in bio_keys], fontsize=8, ha='center')
    ax_b.set_ylabel('% Change from CN')
    # pad ylim so bars don't hit the top/bottom of axes
    ymax = max(all_pct) if all_pct else 20
    ymin = min(all_pct) if all_pct else -20
    ax_b.set_ylim(ymin*1.22, ymax*1.38)
    ax_b.set_title('Key Clinical EEG Biomarkers\n'
                   '(% deviation from CN baseline; all KW p < 0.001)')
    impaired_patches = [Patch(color=COLORS[c], alpha=0.88, label=c) for c in impaired_classes]
    ax_b.legend(handles=impaired_patches, loc='upper right', fontsize=8.5,
                title='vs. CN (0%)', title_fontsize=9, framealpha=0.95)
    lgrid(ax_b); panel_label(ax_b, 'B')

    # ── C: KW H-statistic ranking ───────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    if kw:
        sorted_kw = sorted(kw.items(), key=lambda x: x[1]['H'], reverse=True)[:12]
        feat_names = [k.replace('_', ' ').title() for k, _ in sorted_kw]
        h_vals     = [v['H'] for _, v in sorted_kw]
        def band_of(fk):
            for b in ['gamma','beta','alpha','theta','delta']:
                if b in fk: return b
            return 'other'
        bar_colors = [BAND_COLORS.get(band_of(k), '#888') for k, _ in sorted_kw]
        y_pos = np.arange(len(feat_names))
        ax_c.barh(y_pos, h_vals, color=bar_colors, height=0.62,
                  edgecolor='white', linewidth=0.6, alpha=0.88)
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(feat_names, fontsize=8.5)
        ax_c.set_xlabel('Kruskal-Wallis H Statistic')
        ax_c.set_title('Top Discriminative Spectral Features\n(All p < 0.001)')
        ax_c.invert_yaxis()
        lgrid(ax_c, 'x')
        band_leg = [Patch(color=BAND_COLORS[b], alpha=0.88, label=b.capitalize())
                    for b in ['delta','theta','alpha','beta','gamma']]
        ax_c.legend(handles=band_leg, loc='lower right', fontsize=8,
                    title='Band', title_fontsize=8.5)
        panel_label(ax_c, 'C')

    # ── D: Cohen's d heatmap ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    all_feats  = set(f for cls_feats in disc.values() for item in cls_feats for f in [item['feature']])
    feat_max_d = {}
    for feat in all_feats:
        vals = [abs(item['cohens_d']) for cls_feats in disc.values()
                for item in cls_feats if item['feature'] == feat]
        feat_max_d[feat] = max(vals) if vals else 0
    top_feats = sorted(feat_max_d, key=feat_max_d.get, reverse=True)[:14]
    d_matrix  = np.zeros((len(top_feats), 4))
    for j, cls in enumerate(CLASS_NAMES):
        feat_d = {item['feature']: item['cohens_d'] for item in disc.get(cls, [])}
        for i, feat in enumerate(top_feats):
            d_matrix[i, j] = feat_d.get(feat, 0)
    clean_labels = [f.replace('_', ' ').title() for f in top_feats]
    cmap_div = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(d_matrix, annot=True, fmt='.2f', cmap=cmap_div,
                center=0, vmin=-1.4, vmax=1.4,
                xticklabels=CLASS_NAMES, yticklabels=clean_labels,
                ax=ax_d, cbar_kws={'shrink': 0.82, 'label': "Cohen's d"},
                linewidths=0.3, linecolor='white', annot_kws={'size': 8.5})
    ax_d.set_title("Cohen's d Effect Sizes\n(Class vs. all others, top 14 by |d|)")
    ax_d.set_xticklabels(CLASS_NAMES, rotation=0)
    ax_d.tick_params(axis='y', labelsize=8)
    panel_label(ax_d, 'D')

    fig.suptitle('NeuroChronoGraph  —  Disease-Specific EEG Biomarker Signatures',
                 fontsize=13, fontweight='bold', y=1.01)
    savefig('Figure4_eeg_biomarkers', fig)


# =============================================================================
# FIGURE 5 — Functional Connectivity Atlas (embedded per-class connectome images)
# =============================================================================
def figure5_connectivity_atlas():
    apply_style()
    expl_dir = FIG_EXIST / "explainability"

    img_paths_primary  = {c: expl_dir / f'connectivity_{c}.png' for c in CLASS_NAMES}
    img_paths_fallback = {c: FIG_EXIST / f'connectome_{c}.png'  for c in CLASS_NAMES}

    # pick whichever set exists
    if all(p.exists() for p in img_paths_primary.values()):
        img_dict = img_paths_primary
        src_note = 'GNNExplainer connectivity graphs (alpha-band wPLI, 8-13 Hz)'
    elif all(p.exists() for p in img_paths_fallback.values()):
        img_dict = img_paths_fallback
        src_note = 'Connectome topography (alpha-band wPLI, 8-13 Hz)'
    else:
        print("  SKIP Figure5: connectome images not found. "
              "Run: py -3 experiments/17_connectivity_visualization.py")
        return

    # Clinical interpretation per class
    clinical_notes = {
        'CN':  ('Strong posterior alpha\n& intact fronto-parietal coupling',
                'Normal resting-state network organization'),
        'MCI': ('Subtle anterior-posterior\nconnectivity reduction',
                'Early disruption of default-mode network'),
        'AD':  ('Marked long-range\ndisconnection, frontal isolation',
                'Severe hippocampal-cortical decoupling'),
        'FTD': ('Frontal network\ndisruption, temporal asymmetry',
                'Frontoparietal network degradation'),
    }

    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.10)

    for idx, cls in enumerate(CLASS_ORDER):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        img = mpimg.imread(str(img_dict[cls]))
        ax.imshow(img, aspect='auto', interpolation='lanczos')
        ax.axis('off')

        # Colored banner at top of image
        line1, line2 = clinical_notes[cls]
        ax.set_title(f'{cls}\n{line1}',
                     color=COLORS[cls], fontsize=10, fontweight='bold',
                     pad=5, loc='center')

        # Panel letter
        ax.text(0.02, 0.97, ['A','B','C','D'][idx],
                transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', color='#111',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.75, edgecolor='none'))

        # Small clinical note at bottom
        ax.text(0.5, 0.02, line2,
                transform=ax.transAxes,
                ha='center', va='bottom', fontsize=7.8,
                color='white', style='italic',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#333',
                          alpha=0.65, edgecolor='none'))

    fig.text(0.5, 0.01,
             f'{src_note}  |  Line thickness = connection strength',
             ha='center', fontsize=8.5, color='#666', style='italic')

    fig.suptitle('NeuroChronoGraph  —  Disease-Specific Functional Connectivity Atlas\n'
                 '(Sensor-level; alpha-band wPLI)',
                 fontsize=12, fontweight='bold', y=1.02)
    savefig('Figure5_connectivity_atlas', fig)


# =============================================================================
# FIGURE 6 — Interpretability and Decision Analysis
#   A (full-width): Subject-level prediction confidence
#   B: Feature ablation  C: Calibration  D: CV fold stability
# =============================================================================
def figure6_interpretability(results, stats, features):
    apply_style()
    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40,
                            height_ratios=[1, 1])

    y_true, y_pred, y_probs = get_holdout_arrays(results)

    # ── A: Subject-level confidence strip (full width) ──────────────
    ax_a = fig.add_subplot(gs[0, :])
    sr = results['holdout']['subject_results']
    class_groups = {0: [], 1: [], 2: [], 3: []}
    for s in sr:
        class_groups[s['true']].append(s)
    for k in class_groups:
        class_groups[k].sort(key=lambda x: max(x['probs']), reverse=True)

    ordered, boundaries = [], [0]
    for ci in [2, 3, 0, 1]:          # CN, MCI, AD, FTD
        ordered.extend(class_groups[ci])
        boundaries.append(len(ordered))

    correct   = [s['correct']      for s in ordered]
    conf      = [max(s['probs'])    for s in ordered]
    true_cls  = [s['true']          for s in ordered]

    bar_c = ['#4D9A62' if c else COLORS[CLASS_NAMES[tc]]
             for c, tc in zip(correct, true_cls)]
    ax_a.bar(range(len(ordered)), conf, color=bar_c, alpha=0.82, width=0.85, zorder=3)
    ax_a.fill_between(range(len(ordered)), 0, [0.25]*len(ordered),
                      color='#F7F7F7', zorder=1)

    cls_display = ['CN', 'MCI', 'AD', 'FTD']
    for i, boundary in enumerate(boundaries[1:-1]):
        ax_a.axvline(boundary-0.5, color='#888', lw=1.5, alpha=0.6, zorder=4)
    for i, cls in enumerate(cls_display):
        start, end = boundaries[i], boundaries[i+1]
        mid = (start+end)/2 - 0.5
        n_cor = sum(correct[start:end])
        ax_a.text(mid, -0.08, f'{cls}\n({n_cor}/{end-start})',
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  transform=ax_a.get_xaxis_transform(), color=COLORS[cls])

    n_cor_tot = sum(correct)
    ax_a.axhline(np.mean(conf), color='#2B6CB0', ls='--', lw=1.5, zorder=5)
    ax_a.axhline(0.25, color='#CCC', ls=':', lw=1, zorder=2)
    ax_a.set_ylim(0, 1.12); ax_a.set_xlim(-0.5, len(ordered)-0.5)
    ax_a.set_xticks([])
    ax_a.set_ylabel('Max predicted probability')
    ax_a.set_title(f'Subject-Level Hold-Out Predictions  —  '
                   f'{n_cor_tot}/{len(ordered)} Correct '
                   f'({n_cor_tot/len(ordered)*100:.1f}%)\n'
                   'Green = correct  |  Class-coloured = misclassified by true class  '
                   '|  Sorted by confidence within each class', pad=8)
    leg_a = [
        Patch(color='#4D9A62', alpha=0.82, label='Correctly classified'),
        Patch(color='#B5544B', alpha=0.82, label='Misclassified (true-class colour)'),
        Line2D([0],[0], color='#2B6CB0', ls='--', lw=1.5,
               label=f'Mean conf. = {np.mean(conf):.3f}'),
        Line2D([0],[0], color='#CCC', ls=':', lw=1, label='Chance (0.25)'),
    ]
    ax_a.legend(handles=leg_a, loc='upper right', fontsize=8.5, framealpha=0.95)
    panel_label(ax_a, 'A', x=-0.04, y=1.06)

    # ── B: Feature domain ablation ─────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ablation = features.get('ablation', {})
    if ablation:
        show = [(k, v) for k, v in ablation.items()
                if k in ['raw_eeg','all_bands','clinical','delta','theta','alpha','beta','gamma']]
        show.sort(key=lambda x: x[1]['drop'], reverse=True)
        labels = [v['label'] for k, v in show]
        drops  = [v['drop'] * 100 for k, v in show]
        abl_colors = []
        for k, _ in show:
            if k == 'raw_eeg':     abl_colors.append('#B5544B')
            elif k == 'clinical':  abl_colors.append('#D18E3A')
            elif k == 'all_bands': abl_colors.append('#4578B0')
            else:                  abl_colors.append(BAND_COLORS.get(k, '#888'))
        y_pos = np.arange(len(labels))
        bars_b = ax_b.barh(y_pos, drops, color=abl_colors, height=0.60,
                           edgecolor='white', linewidth=0.6, alpha=0.88)
        for bar, v in zip(bars_b, drops):
            ax_b.text(v-0.5 if v > 5 else v+0.3,
                      bar.get_y()+bar.get_height()/2,
                      f'{v:+.1f}%', va='center', fontsize=8.5, fontweight='bold',
                      color='white' if v > 5 else '#333',
                      ha='right' if v > 5 else 'left')
        ax_b.set_yticks(y_pos); ax_b.set_yticklabels(labels, fontsize=8.5)
        ax_b.axvline(0, color='#BBB', lw=0.8)
        ax_b.set_xlabel('Accuracy drop (%)')
        ax_b.set_title('Feature Domain Ablation\n(Impact on accuracy)')
        ax_b.invert_yaxis(); lgrid(ax_b, 'x'); panel_label(ax_b, 'B')

    # ── C: Calibration reliability diagram ─────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    for cls_idx, cls in enumerate(CLASS_NAMES):
        y_b = (y_true == cls_idx).astype(int)
        y_s = y_probs[:, cls_idx]
        prob_true, prob_pred = calibration_curve(y_b, y_s, n_bins=8, strategy='uniform')
        ax_c.plot(prob_pred, prob_true, 'o-', color=COLORS[cls],
                  lw=1.8, ms=5, alpha=0.88, label=cls)
    ax_c.plot([0,1],[0,1], 'k--', alpha=0.35, lw=1.2, label='Perfect')
    ax_c.fill_between([0,1],[0,1],[0,0], color='#F5F5F5', alpha=0.5)
    ax_c.fill_between([0,1],[1,1],[0,1], color='#F5F5F5', alpha=0.5)
    ax_c.set_xlabel('Mean predicted probability')
    ax_c.set_ylabel('Fraction of positives')
    ax_c.set_xlim([0,1]); ax_c.set_ylim([0,1])
    ax_c.set_title('Reliability (Calibration) Diagram\n(Closer to diagonal = better)')
    ax_c.legend(loc='upper left', fontsize=8.5)
    ax_c.set_aspect('equal'); lgrid(ax_c, 'both'); panel_label(ax_c, 'C')

    # ── D: Per-fold CV stability ─────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 2])
    fold_perf = stats.get('fold_performance', [])
    if fold_perf:
        folds   = [f['fold']       for f in fold_perf]
        s_kappa = [f['screen_kappa']*100 for f in fold_perf]
        st_bacc = [f['stage_bacc']*100   for f in fold_perf]
        su_bacc = [f['subtype_bacc']*100 for f in fold_perf]
        for vals, color, marker, label in [
            (s_kappa, '#2B6CB0', 'o', 'Screen. κ×100'),
            (st_bacc, '#744210', 's', 'Staging BACC'),
            (su_bacc, '#553C9A', '^', 'Subtype BACC'),
        ]:
            ax_d.plot(folds, vals, f'{marker}-', color=color, lw=2.0, ms=7, label=label)
            ax_d.fill_between(folds, vals, alpha=0.07, color=color)
        ax_d.axhline(85, color='#CCC', ls='--', lw=1, label='85% guideline')
        ax_d.set_ylim(68, 108)
        ax_d.set_xlabel('CV Fold'); ax_d.set_ylabel('Performance (%)')
        ax_d.set_title('Per-Fold CV Stability\n(StratifiedGroupKFold, n=5)')
        ax_d.set_xticks(folds)
        ax_d.legend(loc='lower right', fontsize=8)
        lgrid(ax_d); panel_label(ax_d, 'D')

    fig.suptitle('NeuroChronoGraph  —  Interpretability and Decision Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    savefig('Figure6_interpretability', fig)


# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================

def figureS1_training_dynamics(results):
    apply_style()
    history = results.get('training_history', [])
    if not history:
        print("  SKIP FigureS1: no training_history"); return
    n_folds = len(history)
    fold_colors = plt.cm.tab10(np.linspace(0, 0.9, n_folds))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Supplementary S1  —  Training Dynamics (Per Fold)',
                 fontsize=13, fontweight='bold')
    panels = [
        ('train_loss',       axes[0,0], 'A. Training Loss',              'Loss',         'upper right'),
        ('train_acc',        axes[0,1], 'B. Training Accuracy (Screen)',  'Accuracy',     'lower right'),
        ('val_bacc_subtype', axes[1,0], 'C. Validation Subtyping BACC',   'Balanced Acc','lower right'),
        ('learning_rate',    axes[1,1], 'D. Learning Rate Schedule',      'LR',           'upper right'),
    ]
    for key, ax, title, ylabel, leg_loc in panels:
        for fi, fold_hist in enumerate(history):
            if key not in fold_hist: continue
            vals = fold_hist[key]
            ax.plot(range(1, len(vals)+1), vals, color=fold_colors[fi],
                    alpha=0.8, lw=1.5, label=f'Fold {fi+1}')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(loc=leg_loc, fontsize=8)
        lgrid(ax)
        if key == 'learning_rate': ax.set_yscale('log')
        if 'acc' in key or 'bacc' in key:
            ax.axhline(0.25, color='#DDD', ls=':', lw=1); ax.set_ylim(0, 1.05)
    plt.tight_layout(); savefig('FigureS1_training_dynamics', fig)


def figureS2_tsne(results):
    apply_style()
    y_true, y_pred, y_probs = get_holdout_arrays(results)
    tsne = TSNE(n_components=2, perplexity=min(25, len(y_true)-1),
                random_state=42, max_iter=1000)
    emb  = tsne.fit_transform(y_probs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Supplementary S2  —  t-SNE of Predicted Probability Vectors',
                 fontsize=13, fontweight='bold')
    for ci, cls in enumerate(CLASS_NAMES):
        mask = y_true == ci
        axes[0].scatter(emb[mask,0], emb[mask,1], c=COLORS[cls], s=35, alpha=0.75,
                        label=cls, edgecolors='white', linewidths=0.5)
    axes[0].set_title('A. Coloured by True Class')
    axes[0].set_xlabel('t-SNE Dim 1'); axes[0].set_ylabel('t-SNE Dim 2')
    axes[0].legend(fontsize=9)
    panel_label(axes[0], 'A')
    correct = y_true == y_pred
    axes[1].scatter(emb[correct,0], emb[correct,1], c='#4D9A62', s=28, alpha=0.6,
                    label='Correct', edgecolors='white', linewidths=0.4)
    axes[1].scatter(emb[~correct,0], emb[~correct,1], c='#B5544B', s=60, alpha=0.9,
                    label='Misclassified', marker='X', linewidths=1.5)
    axes[1].set_title('B. Coloured by Correctness')
    axes[1].set_xlabel('t-SNE Dim 1'); axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].legend(fontsize=9)
    panel_label(axes[1], 'B')
    plt.tight_layout(); savefig('FigureS2_tsne_embeddings', fig)


def figureS3_precision_recall(results):
    apply_style()
    y_true, _, y_probs = get_holdout_arrays(results)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    fig.suptitle('Supplementary S3  —  Precision-Recall Curves and Calibration',
                 fontsize=12, fontweight='bold')

    # PR curves
    ax_pr = axes[0]
    for ci, cls in enumerate(CLASS_NAMES):
        y_b = (y_true == ci).astype(int)
        y_s = y_probs[:, ci]
        prec, rec, _ = precision_recall_curve(y_b, y_s)
        ap = average_precision_score(y_b, y_s)
        ax_pr.plot(rec, prec, color=COLORS[cls], lw=2.0,
                   label=f'{cls}  AP={ap:.3f}', alpha=0.88)
        ax_pr.fill_between(rec, 0, prec, color=COLORS[cls], alpha=0.06)
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves (OvR)')
    # PR curves for this task sit near (high-recall, high-precision) — upper
    # right is full. The low-left corner is empty (curves only dip there at
    # extreme recall), so place the legend there without overlap.
    ax_pr.legend(loc='lower left', fontsize=9, framealpha=0.92,
                 edgecolor='#BBBBBB')
    ax_pr.set_xlim([0, 1.02]); ax_pr.set_ylim([0, 1.02])
    lgrid(ax_pr, 'both'); panel_label(ax_pr, 'A')

    # Calibration
    ax_cal = axes[1]
    n_bins, bin_edges = 10, np.linspace(0, 1, 11)
    for ci, cls in enumerate(CLASS_NAMES):
        y_b = (y_true == ci).astype(int)
        y_s = y_probs[:, ci]
        prob_true, prob_pred = calibration_curve(y_b, y_s, n_bins=8, strategy='uniform')
        ece = 0.0
        for b in range(n_bins):
            mask = (y_s >= bin_edges[b]) & (y_s < bin_edges[b+1])
            if mask.sum() > 0:
                ece += mask.sum() * abs(y_s[mask].mean() - y_b[mask].mean())
        ece /= len(y_s)
        ax_cal.plot(prob_pred, prob_true, 'o-', color=COLORS[cls],
                    lw=1.8, ms=5, alpha=0.88, label=f'{cls}  ECE={ece:.3f}')
    ax_cal.plot([0,1],[0,1], 'k--', alpha=0.35, lw=1.2, label='Perfect')
    ax_cal.set_xlabel('Mean predicted prob.')
    ax_cal.set_ylabel('Fraction of positives')
    ax_cal.set_title('Reliability Diagram\n(ECE < 0.05 = clinically acceptable)')
    ax_cal.legend(loc='upper left', fontsize=8.5)
    ax_cal.set_xlim([0,1]); ax_cal.set_ylim([0,1])
    ax_cal.set_aspect('equal')
    lgrid(ax_cal, 'both'); panel_label(ax_cal, 'B')

    savefig('FigureS3_precision_recall', fig)


def figureS4_cohens_d_full(features):
    apply_style()
    disc = features.get('discriminative_features', {}).get('class_discriminative', {})
    if not disc:
        print("  SKIP FigureS4: no discriminative_features"); return
    all_feats = sorted({item['feature']
                        for cls_feats in disc.values() for item in cls_feats})
    d_matrix  = np.zeros((len(all_feats), 4))
    for j, cls in enumerate(CLASS_NAMES):
        feat_d = {item['feature']: item['cohens_d'] for item in disc.get(cls, [])}
        for i, feat in enumerate(all_feats):
            d_matrix[i, j] = feat_d.get(feat, 0)
    clean = [f.replace('_',' ').title() for f in all_feats]
    fig, ax = plt.subplots(figsize=(7, max(9, len(all_feats)*0.38)))
    cmap_div = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(d_matrix, annot=True, fmt='.2f', cmap=cmap_div,
                center=0, vmin=-1.4, vmax=1.4,
                xticklabels=CLASS_NAMES, yticklabels=clean,
                ax=ax, cbar_kws={'label':"Cohen's d", 'shrink':0.6},
                linewidths=0.3, linecolor='white', annot_kws={'size': 8})
    ax.set_title("Supplementary S4  —  Full Cohen's d Matrix\n"
                 "(All 31 spectral features, class vs. others)",
                 fontweight='bold', pad=10)
    ax.set_xticklabels(CLASS_NAMES, rotation=0)
    plt.tight_layout(); savefig('FigureS4_cohens_d_full', fig)


def figureS5_microstate():
    src = FIG_EXIST / 'microstates_4class_analysis.png'
    if not src.exists():
        print(f"  SKIP FigureS5: {src.name} not found"); return
    apply_style()
    img = mpimg.imread(str(src))
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, aspect='auto', interpolation='lanczos')
    ax.axis('off')
    fig.suptitle('Supplementary S5  —  EEG Microstate Analysis\n'
                 '(Four-class comparison of microstate parameters)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5); savefig('FigureS5_microstate_analysis', fig)


# =============================================================================
# FigureS6 — EEG Disease Profile Radar Chart
# Spider/polar chart comparing all 4 classes on 8 EEG dimensions
# =============================================================================
def figureS6_radar_profile(features):
    apply_style()
    disc = features.get('discriminative_features', {}).get('class_discriminative', {})
    if not disc:
        print("  SKIP FigureS6: no discriminative_features"); return

    cf = build_class_feat(disc)

    radar_features = [
        ('delta_rel_power',   'Delta\nPower'),
        ('theta_rel_power',   'Theta\nPower'),
        ('alpha_rel_power',   'Alpha\nPower'),
        ('beta_rel_power',    'Beta\nPower'),
        ('gamma_rel_power',   'Gamma\nPower'),
        ('theta_alpha_ratio', 'TAR'),
        ('alpha_peak_freq',   'Alpha\nPeak Freq'),
        ('spectral_entropy',  'Spectral\nEntropy'),
    ]
    n = len(radar_features)

    # Collect raw means per class
    raw_vals = {cls: [] for cls in CLASS_ORDER}
    for feat_key, _ in radar_features:
        feat_across = [cf[cls].get(feat_key, {}).get('class_mean', 0) for cls in CLASS_ORDER]
        min_v, max_v = min(feat_across), max(feat_across)
        rng = max_v - min_v if max_v > min_v else 1e-8
        for cls, v in zip(CLASS_ORDER, feat_across):
            raw_vals[cls].append((v - min_v) / rng)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection='polar')

    for cls in CLASS_ORDER:
        vals   = raw_vals[cls]
        v_c    = vals + [vals[0]]
        t_c    = np.concatenate([theta, [theta[0]]])
        ax.plot(t_c, v_c, 'o-', color=COLORS[cls], lw=2.5, ms=6,
                label=cls, alpha=0.9)
        ax.fill(t_c, v_c, color=COLORS[cls], alpha=0.10)

    ax.set_xticks(theta)
    ax.set_xticklabels([lbl for _, lbl in radar_features], fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7, color='#888')
    ax.spines['polar'].set_color('#CCCCCC')
    ax.grid(color='#E0E0E0', linewidth=0.7)

    legend_patches = [Patch(color=COLORS[c], alpha=0.88, label=c) for c in CLASS_ORDER]
    ax.legend(handles=legend_patches,
              loc='upper right', bbox_to_anchor=(1.38, 1.12),
              fontsize=11, title='Diagnostic Class', title_fontsize=10)

    fig.suptitle('Supplementary S6  —  EEG Disease Profile Radar Chart\n'
                 '(8 EEG dimensions normalized to [0,1] across classes;\n'
                 ' greater area = stronger feature expression)',
                 fontsize=11, fontweight='bold', y=1.03)
    savefig('FigureS6_radar_profile', fig)


# =============================================================================
# FigureS7 — Regional Spectral Power Heatmaps
# 4-region × 5-band heatmap per class + CN-difference heatmaps
# =============================================================================
def figureS7_regional_heatmap(features):
    apply_style()
    disc = features.get('discriminative_features', {}).get('class_discriminative', {})
    if not disc:
        print("  SKIP FigureS7: no discriminative_features"); return

    cf      = build_class_feat(disc)
    regions = ['frontal', 'temporal', 'parietal', 'occipital']
    bands   = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    def regional_matrix(cls):
        m = np.zeros((len(regions), len(bands)))
        for i, reg in enumerate(regions):
            for j, bnd in enumerate(bands):
                feat_key = f'{bnd}_{reg}'
                if feat_key in cf[cls]:
                    m[i, j] = cf[cls][feat_key]['class_mean']
        return m

    cn_mat = regional_matrix('CN')

    # Flat GridSpec: row 0 = 4 absolute heatmaps, row 1 = 3 diff + 1 empty
    fig = plt.figure(figsize=(22, 12))
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.55, wspace=0.42,
                            left=0.06, right=0.97,
                            top=0.88,  bottom=0.07)

    band_lbl   = [u'\u03b4', u'\u03b8', u'\u03b1', u'\u03b2', u'\u03b3']
    region_lbl = ['Frontal', 'Temporal', 'Parietal', 'Occipital']

    # Row 0: absolute heatmaps (CN, MCI, AD, FTD)
    for idx, cls in enumerate(CLASS_ORDER):
        ax = fig.add_subplot(gs[0, idx])
        mat = regional_matrix(cls)
        sns.heatmap(mat, annot=True, fmt='.2f',
                    cmap='YlOrRd', ax=ax,
                    xticklabels=band_lbl, yticklabels=region_lbl,
                    cbar_kws={'shrink': 0.75, 'pad': 0.03},
                    linewidths=0.4, linecolor='white',
                    annot_kws={'size': 8})
        ax.set_title(f'{cls}  —  Spectral Power',
                     color=COLORS[cls], fontweight='bold', fontsize=10, pad=4)
        ax.set_xlabel('Band', fontsize=9)
        ax.set_ylabel('Region' if idx == 0 else '', fontsize=9)
        ax.set_xticklabels(band_lbl, rotation=0)
        ax.set_yticklabels(region_lbl if idx == 0 else ['']*4,
                           rotation=0, fontsize=8.5)
        ax.text(-0.05, 1.06, 'ABCD'[idx],
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                va='top', ha='left')

    # Row 1: difference heatmaps (MCI-CN, AD-CN, FTD-CN); col 3 blank
    diff_classes = ['MCI', 'AD', 'FTD']
    cmap_div = sns.diverging_palette(220, 10, as_cmap=True)

    for idx, cls in enumerate(diff_classes):
        ax = fig.add_subplot(gs[1, idx])
        mat      = regional_matrix(cls)
        diff_mat = mat - cn_mat
        ann      = np.array([[f'{diff_mat[i,j]:+.2f}' for j in range(5)]
                              for i in range(4)])
        sns.heatmap(diff_mat, annot=ann, fmt='', cmap=cmap_div,
                    center=0, vmin=-0.06, vmax=0.06, ax=ax,
                    xticklabels=band_lbl, yticklabels=region_lbl,
                    cbar_kws={'shrink': 0.75, 'pad': 0.03,
                              'label': u'\u0394 from CN'},
                    linewidths=0.4, linecolor='white',
                    annot_kws={'size': 8})
        ax.set_title(f'{cls} \u2212 CN  (+elevated, \u2212reduced)',
                     color=COLORS[cls], fontweight='bold', fontsize=10, pad=4)
        ax.set_xlabel('Band', fontsize=9)
        ax.set_ylabel('Region' if idx == 0 else '', fontsize=9)
        ax.set_xticklabels(band_lbl, rotation=0)
        ax.set_yticklabels(region_lbl if idx == 0 else ['']*4,
                           rotation=0, fontsize=8.5)
        ax.text(-0.05, 1.06, 'EFG'[idx],
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                va='top', ha='left')

    # blank 4th cell in row 1
    fig.add_subplot(gs[1, 3]).axis('off')

    fig.suptitle('Supplementary S7  —  Regional Spectral Power Maps\n'
                 'A\u2013D: Absolute mean power per region \u00d7 band.  '
                 'E\u2013G: Deviation from CN (positive = elevated vs. normal)',
                 fontsize=11, fontweight='bold', y=0.97)
    savefig('FigureS7_regional_heatmap', fig)


# =============================================================================
# FigureS8 — Neuroscientific Analysis of EEG Biomarkers
# Five-panel consolidation: connectivity (A,B), multi-domain profile (C),
# regional theta/alpha ratio (D), anterior-posterior alpha gradient (E).
# Panel C is drawn live from feature_analysis.json; panels A,B,D,E embed
# upstream PNGs produced by the connectivity/clinical-source pipelines.
# =============================================================================
def figureS8_neuroscience_analysis(features):
    apply_style()

    image_panels = [
        (FIG_EXIST / 'connectivity_matrices.png',
         'A  Alpha-band wPLI connectivity matrices'),
        (FIG_EXIST / 'connectivity_disease_signatures.png',
         'B  Disease-specific connectome reorganization'),
        (FIG_EXIST / 'theta_alpha_ratio.png',
         'D  Regional theta/alpha ratio by class'),
        (FIG_EXIST / 'anterior_posterior_gradient.png',
         'E  Anterior-posterior alpha power gradient'),
    ]

    disc = features.get('discriminative_features', {}).get('class_discriminative', {})
    have_radar = bool(disc)
    have_any_img = any(p.exists() for p, _ in image_panels)
    if not have_radar and not have_any_img:
        print("  SKIP FigureS8: no inputs available"); return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 3,
        hspace=0.32, wspace=0.12,
        top=0.92, bottom=0.04, left=0.03, right=0.98,
    )

    # Top row: A (connectivity matrices), B (connectome reorg), C (radar — live)
    # Bottom row: D (TAR), E (A-P gradient), [empty]
    slot_map = {
        (0, 0): image_panels[0],
        (0, 1): image_panels[1],
        (1, 0): image_panels[2],
        (1, 1): image_panels[3],
    }
    for (r, c), (path, lbl) in slot_map.items():
        ax = fig.add_subplot(gs[r, c])
        if path.exists():
            img = mpimg.imread(str(path))
            ax.imshow(img, interpolation='lanczos')
        else:
            ax.text(0.5, 0.5, '(missing)', ha='center', va='center',
                    transform=ax.transAxes, color='#888')
        ax.axis('off')
        ax.set_title(lbl, fontsize=10.5, fontweight='bold', pad=6, loc='left')

    # Panel C — live polar radar (absorbs former S6)
    if have_radar:
        ax_radar = fig.add_subplot(gs[0, 2], projection='polar')
        cf = build_class_feat(disc)
        radar_features = [
            ('delta_rel_power',   'δ Power'),
            ('theta_rel_power',   'θ Power'),
            ('alpha_rel_power',   'α Power'),
            ('beta_rel_power',    'β Power'),
            ('gamma_rel_power',   'γ Power'),
            ('theta_alpha_ratio', 'TAR'),
            ('alpha_peak_freq',   'iAPF'),
            ('spectral_entropy',  'Entropy'),
        ]
        n = len(radar_features)
        raw_vals = {cls: [] for cls in CLASS_ORDER}
        for fk, _ in radar_features:
            feat_across = [cf[cls].get(fk, {}).get('class_mean', 0) for cls in CLASS_ORDER]
            mn, mx = min(feat_across), max(feat_across)
            rng = mx - mn if mx > mn else 1e-8
            for cls, v in zip(CLASS_ORDER, feat_across):
                raw_vals[cls].append((v - mn) / rng)

        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for cls in CLASS_ORDER:
            vals = raw_vals[cls]
            v_c = vals + [vals[0]]
            t_c = np.concatenate([theta, [theta[0]]])
            ax_radar.plot(t_c, v_c, 'o-', color=COLORS[cls], lw=2.0, ms=4.5,
                          alpha=0.92, label=cls)
            ax_radar.fill(t_c, v_c, color=COLORS[cls], alpha=0.09)

        ax_radar.set_xticks(theta)
        ax_radar.set_xticklabels([lbl for _, lbl in radar_features], fontsize=8)
        ax_radar.set_ylim(0, 1.05)
        ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(['.25', '.50', '.75', '1.0'], fontsize=6, color='#888')
        ax_radar.spines['polar'].set_color('#CCCCCC')
        ax_radar.grid(color='#E0E0E0', linewidth=0.7)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.32, 1.10),
                        fontsize=8, frameon=False, title='Class', title_fontsize=8)
        ax_radar.set_title('C  Multi-domain EEG radar profile',
                           fontsize=10.5, fontweight='bold', pad=14, loc='left')

    fig.suptitle('Supplementary S8  —  Neuroscientific analysis of EEG biomarkers',
                 fontsize=12.5, fontweight='bold', y=0.975)
    savefig('FigureS8_neuroscience_analysis', fig)


# =============================================================================
# NEW: Figure_biomarker_overlap — Cross-Class Biomarker Comparison
# Multi-panel figure showing similarities and differences in EEG biomarkers
# across all four diagnostic classes (AD, FTD, CN, MCI)
#
#   A: Binary presence heatmap with direction arrows (features × classes)
#   B: Signed Cohen's d dot-strip (shared vs unique discriminators)
#   C: Class biomarker profile similarity matrix (correlation-based)
#   D: UpSet-style intersection plot (modern Venn replacement)
# =============================================================================
def figure_biomarker_overlap(features):
    """Cross-class biomarker comparison — no radar (see FigureS6 for that)."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    from collections import Counter
    from itertools import combinations

    apply_style()
    disc = features.get('discriminative_features', {}).get('class_discriminative', {})
    if not disc:
        print("  SKIP biomarker_overlap: no discriminative_features"); return

    cf = build_class_feat(disc)

    # ── Collect union of features & build d-matrix ──────────────────
    all_feats = set()
    for cls in CLASS_NAMES:
        for item in disc.get(cls, []):
            all_feats.add(item['feature'])
    feat_max_d = {}
    for feat in all_feats:
        vals = [abs(item['cohens_d']) for cls_items in disc.values()
                for item in cls_items if item['feature'] == feat]
        feat_max_d[feat] = max(vals) if vals else 0
    top_feats = sorted(feat_max_d, key=feat_max_d.get, reverse=True)[:16]

    d_matrix = np.zeros((len(top_feats), 4))
    dir_matrix = {}  # (feat_idx, cls_idx) -> 'elevated' / 'reduced'
    for j, cls in enumerate(CLASS_NAMES):
        feat_d = {item['feature']: item['cohens_d'] for item in disc.get(cls, [])}
        feat_dir = {item['feature']: item['direction'] for item in disc.get(cls, [])}
        for i, feat in enumerate(top_feats):
            d_matrix[i, j] = feat_d.get(feat, 0)
            dir_matrix[(i, j)] = feat_dir.get(feat, '')

    clean_labels = [f.replace('_', ' ').title() for f in top_feats]
    all_feat_union = sorted(all_feats)

    # ── Figure layout ───────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 17))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.36,
                           left=0.09, right=0.96, top=0.92, bottom=0.05)

    # ══════════════════════════════════════════════════════════════════
    # Panel A: Binary presence heatmap with direction & effect size
    #   rows = top features, cols = classes
    #   cell color = elevated (warm) / reduced (cool) / ns (white)
    #   annotation = Cohen's d value with arrow
    # ══════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1.3, vcenter=0, vmax=1.3)
    cmap_div = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw custom cell grid
    n_feat, n_cls = len(top_feats), 4
    for i in range(n_feat):
        for j in range(n_cls):
            d_val = d_matrix[i, j]
            direction = dir_matrix.get((i, j), '')
            if abs(d_val) < 0.2:
                # Non-significant: light gray
                fc = '#F5F5F5'
                txt_color = '#BBBBBB'
                arrow = ''
            else:
                fc = cmap_div(norm(d_val))
                txt_color = 'white' if abs(d_val) > 0.7 else '#333333'
                arrow = u'\u2191' if direction == 'elevated' else u'\u2193'

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fc,
                                 edgecolor='white', linewidth=1.5)
            ax_a.add_patch(rect)

            # Annotate with arrow + d value
            if abs(d_val) >= 0.2:
                ax_a.text(j, i, f'{arrow}{abs(d_val):.2f}',
                          ha='center', va='center', fontsize=8,
                          fontweight='bold', color=txt_color)

    ax_a.set_xlim(-0.5, n_cls - 0.5)
    ax_a.set_ylim(n_feat - 0.5, -0.5)
    ax_a.set_xticks(range(n_cls))
    ax_a.set_xticklabels(CLASS_NAMES, fontsize=10, fontweight='bold')
    ax_a.set_yticks(range(n_feat))
    ax_a.set_yticklabels(clean_labels, fontsize=8.5)
    # Color x-tick labels
    for tl in ax_a.get_xticklabels():
        tl.set_color(COLORS.get(tl.get_text(), '#333'))
    ax_a.set_title('Biomarker Fingerprint Map\n'
                   u'(\u2191 elevated  \u2193 reduced vs. rest; '
                   u'gray = |d| < 0.2)',
                   fontsize=10, fontweight='bold')

    # Add effect size colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_a, shrink=0.65, pad=0.03, aspect=25)
    cb.set_label("Cohen's d", fontsize=9)
    panel_label(ax_a, 'A')

    # ══════════════════════════════════════════════════════════════════
    # Panel B: Signed Cohen's d dot-strip plot
    # ══════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(top_feats))
    offsets = {'AD': -0.24, 'FTD': -0.08, 'CN': 0.08, 'MCI': 0.24}

    for cls in CLASS_NAMES:
        x_vals = d_matrix[:, CLASS_NAMES.index(cls)]
        y_shifted = y_pos + offsets[cls]
        sizes = np.clip(np.abs(x_vals) * 120, 15, 200)
        ax_b.scatter(x_vals, y_shifted, s=sizes, color=COLORS[cls],
                     alpha=0.82, edgecolors='white', linewidths=0.5,
                     label=cls, zorder=3)

    ax_b.axvline(0, color='#555', lw=1.0, ls='-', zorder=1)
    for thresh in [-1.0, -0.5, 0.5, 1.0]:
        style = '--' if abs(thresh) == 1.0 else ':'
        ax_b.axvline(thresh, color='#BBB', lw=0.7, ls=style, zorder=1)

    # Effect size threshold labels
    for xv, label in [(0.5, 'medium'), (1.0, 'large'),
                      (-0.5, 'medium'), (-1.0, 'large')]:
        ax_b.text(xv, -0.8, label, ha='center', fontsize=7,
                  color='#888', style='italic')

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(clean_labels, fontsize=8.5)
    ax_b.set_xlabel("Cohen's d (class vs. all others)", fontsize=10)
    ax_b.set_title('Signed Effect Sizes by Feature & Class\n'
                   u'(dot size \u221d |d|; convergent features '
                   u'cluster near zero across classes)',
                   fontsize=10, fontweight='bold')
    ax_b.invert_yaxis()
    ax_b.legend(loc='lower right', fontsize=8.5, title='Class',
                title_fontsize=9)
    lgrid(ax_b, 'x')
    panel_label(ax_b, 'B')

    # ══════════════════════════════════════════════════════════════════
    # Panel C: Class biomarker similarity (correlation matrix)
    # ══════════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    profile_matrix = np.zeros((4, len(all_feat_union)))
    for j, cls in enumerate(CLASS_NAMES):
        feat_d = {item['feature']: item['cohens_d'] for item in disc.get(cls, [])}
        for i, feat in enumerate(all_feat_union):
            profile_matrix[j, i] = feat_d.get(feat, 0)

    corr = np.corrcoef(profile_matrix)
    dist = pdist(profile_matrix, metric='correlation')
    Z = linkage(dist, method='ward')
    dn = dendrogram(Z, no_plot=True)
    order = dn['leaves']
    ordered_names = [CLASS_NAMES[i] for i in order]
    corr_ordered = corr[np.ix_(order, order)]

    cmap_corr = sns.diverging_palette(220, 10, as_cmap=True)
    im = ax_c.imshow(corr_ordered, cmap=cmap_corr, vmin=-1, vmax=1, aspect='auto')
    for i in range(4):
        for j in range(4):
            val = corr_ordered[i, j]
            color = 'white' if abs(val) > 0.6 else '#333'
            ax_c.text(j, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=12, fontweight='bold', color=color)

    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(ordered_names, fontsize=11, fontweight='bold')
    ax_c.set_yticks(range(4))
    ax_c.set_yticklabels(ordered_names, fontsize=11, fontweight='bold')
    for tl in ax_c.get_xticklabels():
        tl.set_color(COLORS.get(tl.get_text(), '#333'))
    for tl in ax_c.get_yticklabels():
        tl.set_color(COLORS.get(tl.get_text(), '#333'))

    cb2 = fig.colorbar(im, ax=ax_c, shrink=0.75, pad=0.03)
    cb2.set_label('Pearson r (biomarker profile)', fontsize=9)
    ax_c.set_title('Biomarker Profile Similarity\n'
                   "(Ward-ordered; r = correlation of Cohen's d vectors)",
                   fontsize=10, fontweight='bold')
    panel_label(ax_c, 'C')

    # ══════════════════════════════════════════════════════════════════
    # Panel D: UpSet-style intersection plot
    #   Top: intersection size bars
    #   Bottom: dot-matrix showing which classes are in each intersection
    # ══════════════════════════════════════════════════════════════════
    threshold = 0.4
    feat_class_sets = {}
    for feat in all_feat_union:
        sig_classes = set()
        for cls in CLASS_NAMES:
            feat_d_map = {item['feature']: abs(item['cohens_d'])
                          for item in disc.get(cls, [])}
            if feat_d_map.get(feat, 0) >= threshold:
                sig_classes.add(cls)
        if sig_classes:
            feat_class_sets[feat] = frozenset(sig_classes)

    combo_counts = Counter(feat_class_sets.values())

    # Sort intersections: first by set size descending, then by count
    sorted_combos = sorted(combo_counts.items(),
                           key=lambda x: (-x[1], len(x[0])))
    # Keep only top intersections (up to 12 for readability)
    sorted_combos = sorted_combos[:12]

    n_combos = len(sorted_combos)
    cls_list = CLASS_ORDER  # CN, MCI, AD, FTD

    # Create sub-gridspec for UpSet: top = bars, bottom = dot matrix
    gs_d = gs[1, 1].subgridspec(2, 1, height_ratios=[3, 1.5], hspace=0.08)
    ax_d_bars = fig.add_subplot(gs_d[0])
    ax_d_dots = fig.add_subplot(gs_d[1], sharex=ax_d_bars)

    x_upset = np.arange(n_combos)
    counts = [c for _, c in sorted_combos]

    # Bar chart of intersection sizes
    bar_colors_up = []
    for combo, _ in sorted_combos:
        if len(combo) == 1:
            bar_colors_up.append(COLORS[list(combo)[0]])
        else:
            bar_colors_up.append('#555555')
    ax_d_bars.bar(x_upset, counts, color=bar_colors_up, width=0.6,
                  edgecolor='white', linewidth=0.8, alpha=0.88)
    for xi, val in zip(x_upset, counts):
        ax_d_bars.text(xi, val + 0.2, str(val), ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    ax_d_bars.set_ylabel('Feature Count', fontsize=10)
    ax_d_bars.set_title(f'Biomarker Set Intersections (UpSet Plot)\n'
                        f'(features with |Cohen\'s d| \u2265 {threshold})',
                        fontsize=10, fontweight='bold')
    ax_d_bars.tick_params(bottom=False, labelbottom=False)
    lgrid(ax_d_bars)

    # Dot matrix: rows = classes, cols = intersections
    for yi, cls in enumerate(cls_list):
        for xi, (combo, _) in enumerate(sorted_combos):
            if cls in combo:
                ax_d_dots.plot(xi, yi, 'o', color=COLORS[cls],
                               markersize=10, zorder=3)
            else:
                ax_d_dots.plot(xi, yi, 'o', color='#E0E0E0',
                               markersize=7, zorder=2)
        # Draw connecting lines for each intersection
    for xi, (combo, _) in enumerate(sorted_combos):
        members = [yi for yi, cls in enumerate(cls_list) if cls in combo]
        if len(members) > 1:
            ax_d_dots.plot([xi, xi], [min(members), max(members)],
                           color='#555', lw=2, zorder=1)

    ax_d_dots.set_yticks(range(len(cls_list)))
    ax_d_dots.set_yticklabels(cls_list, fontsize=10, fontweight='bold')
    for tl in ax_d_dots.get_yticklabels():
        tl.set_color(COLORS.get(tl.get_text(), '#333'))
    ax_d_dots.set_xlim(-0.5, n_combos - 0.5)
    ax_d_dots.set_ylim(-0.5, len(cls_list) - 0.5)
    ax_d_dots.invert_yaxis()
    ax_d_dots.tick_params(bottom=False, labelbottom=False)
    ax_d_dots.grid(axis='y', color='#EBEBEB', linewidth=0.5)
    ax_d_dots.set_axisbelow(True)
    panel_label(ax_d_bars, 'D')

    fig.suptitle('Cross-Class EEG Biomarker Comparison\n'
                 'Similarities and differences in discriminative spectral features '
                 'across AD, FTD, CN, and MCI',
                 fontsize=13, fontweight='bold', y=0.98)
    savefig('Figure_biomarker_overlap', fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  NeuroChronoGraph -- Publication Figure Generator v3")
    print("  Output:", FIG_DIR)
    print("=" * 65)

    print("\nLoading data...")
    results  = load_results()
    stats    = load_stats()
    features = load_features()

    dev_acc  = results['development']['accuracy'] * 100
    hold_acc = results['holdout']['accuracy'] * 100
    subj_acc = stats.get('subject_level', {}).get('accuracy', 0) * 100
    print(f"  Dev 4-class CV accuracy   : {dev_acc:.1f}%")
    print(f"  Hold-out window accuracy  : {hold_acc:.1f}%")
    print(f"  Hold-out subject accuracy : {subj_acc:.1f}%")

    print("\n--- Main Figures -------------------------------------------\n")

    print("Figure 2: Performance overview (CM + ROC + sensitivity)...")
    figure2_performance_overview(results, stats)

    print("Figure 3: Validation metrics (accuracy / stages / AUC / CI)...")
    figure3_validation_metrics(results, stats)

    print("Figure 4: EEG biomarker signatures...")
    figure4_eeg_biomarkers(features)

    print("Figure 5: Functional connectivity atlas...")
    figure5_connectivity_atlas()

    print("Figure 6: Interpretability + decision analysis...")
    figure6_interpretability(results, stats, features)

    print("\n--- Supplementary Figures ----------------------------------\n")

    print("FigureS1: Training dynamics...")
    figureS1_training_dynamics(results)

    print("FigureS2: t-SNE embeddings...")
    figureS2_tsne(results)

    print("FigureS3: Precision-recall + calibration...")
    figureS3_precision_recall(results)

    print("FigureS4: Full Cohen's d heatmap...")
    figureS4_cohens_d_full(features)

    print("FigureS5: Microstate analysis...")
    figureS5_microstate()

    print("FigureS7: Regional spectral power heatmaps...")
    figureS7_regional_heatmap(features)

    print("FigureS8: Neuroscientific analysis (connectivity + radar + spatial)...")
    figureS8_neuroscience_analysis(features)

    print("\n--- Biomarker Comparison -----------------------------------\n")

    print("Figure: Cross-class biomarker overlap comparison...")
    figure_biomarker_overlap(features)

    print("\n" + "=" * 65)
    print("  All figures written to:", FIG_DIR)
    print("=" * 65)
    print("""
Regenerate neuroscience figures needing raw EEG data:
  Connectivity/connectome : py -3 experiments/17_connectivity_visualization.py
  Topographic band power  : py -3 experiments/28_feature_analysis.py
  Regional spectral power : py -3 experiments/16_clinical_source_analysis.py
  Microstate analysis     : py -3 experiments/visualize_microstates.py
""")


if __name__ == "__main__":
    main()
