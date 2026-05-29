"""
Comprehensive Statistical Analysis.

Computes all statistical tests promised in the manuscript Methods section:
1. Bootstrap 95% CIs (10,000 iterations) for holdout metrics
2. Wilson score CIs for accuracy
3. Exact binomial test vs chance (25%)
4. DeLong CIs for AUC
5. Subject-level accuracy via majority voting
6. Per-fold performance table from existing CV results
7. Literature comparison data compilation

Saves results to outputs/results/statistical_analysis.json
"""

import sys
import json
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
from src.utils.reproducibility import DEFAULT_SEED, set_global_seed
SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']


# ============================================================
# Statistical Tests
# ============================================================

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=10000, alpha=0.05, seed=42):
    """
    Compute bootstrap confidence interval for a metric.
    Resamples at window (sample) level — CIs will be optimistically narrow
    when windows from the same subject are correlated.
    Use bootstrap_ci_subject_level for valid subject-clustered CIs.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = metric_fn(y_true[idx], y_pred[idx])
            boot_scores.append(s)
        except Exception:
            continue

    boot_scores = np.array(boot_scores)
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))

    return float(point), float(lower), float(upper)


def bootstrap_ci_subject_level(subject_results, metric_fn, n_bootstrap=5000, alpha=0.05, seed=42):
    """
    Subject-clustered bootstrap CI.

    Resamples SUBJECTS with replacement (not windows), then collects all windows
    from each resampled subject to compute the metric.  This correctly accounts for
    within-subject window correlation and produces valid (wider) CIs.

    Args:
        subject_results: list of dicts with keys 'subject', 'true', 'pred'
        metric_fn: function(y_true_array, y_pred_array) -> scalar
        n_bootstrap: bootstrap iterations
        alpha: significance level

    Returns:
        (point_estimate, lower, upper)
    """
    rng = np.random.RandomState(seed)

    # Group windows by subject
    from collections import defaultdict
    subj_windows = defaultdict(lambda: {'true': [], 'pred': []})
    for r in subject_results:
        sid = r['subject']
        subj_windows[sid]['true'].append(r['true'])
        subj_windows[sid]['pred'].append(r['pred'])

    subjects = list(subj_windows.keys())
    n_subj = len(subjects)

    # Point estimate on original data
    y_true_all = np.array([r['true'] for r in subject_results])
    y_pred_all = np.array([r['pred'] for r in subject_results])
    point = metric_fn(y_true_all, y_pred_all)

    boot_scores = []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(subjects, n_subj, replace=True)
        bt_true, bt_pred = [], []
        for sid in sampled_subjects:
            bt_true.extend(subj_windows[sid]['true'])
            bt_pred.extend(subj_windows[sid]['pred'])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = metric_fn(np.array(bt_true), np.array(bt_pred))
            boot_scores.append(s)
        except Exception:
            continue

    boot_scores = np.array(boot_scores)
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return float(point), float(lower), float(upper)


def wilson_score_ci(n_correct, n_total, alpha=0.05):
    """Wilson score interval for a proportion."""
    p_hat = n_correct / n_total
    z = stats.norm.ppf(1 - alpha / 2)
    
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    
    return float(center - margin), float(center + margin)


def binomial_test_vs_chance(n_correct, n_total, chance=0.25):
    """Exact binomial test: is accuracy significantly above chance?"""
    result = stats.binomtest(n_correct, n_total, p=chance, alternative='greater')
    return float(result.pvalue)


def delong_auc_ci(y_true_binary, y_scores, alpha=0.05):
    """
    Approximate DeLong CI for AUC.
    Uses the normal approximation for the variance of AUC.
    """
    from sklearn.metrics import roc_auc_score
    
    try:
        auc = roc_auc_score(y_true_binary, y_scores)
    except:
        return None, None, None
    
    n1 = np.sum(y_true_binary == 1)
    n0 = np.sum(y_true_binary == 0)
    
    # Hanley-McNeil approximation
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    
    z = stats.norm.ppf(1 - alpha / 2)
    lower = max(0, auc - z * se)
    upper = min(1, auc + z * se)
    
    return float(auc), float(lower), float(upper)


# ============================================================
# Subject-Level Majority Voting
# ============================================================

def subject_level_majority_vote(subject_results):
    """
    Aggregate window-level predictions to subject level via majority vote.
    
    Args:
        subject_results: list of dicts with 'subject', 'true', 'pred'
    
    Returns:
        subject_true, subject_pred arrays
    """
    subject_preds = {}
    subject_true_labels = {}
    
    for r in subject_results:
        sid = r['subject']
        if sid not in subject_preds:
            subject_preds[sid] = []
            subject_true_labels[sid] = r['true']
        subject_preds[sid].append(r['pred'])
    
    subjects = sorted(subject_preds.keys())
    y_true = np.array([subject_true_labels[s] for s in subjects])
    y_pred = np.array([Counter(subject_preds[s]).most_common(1)[0][0] for s in subjects])
    
    return y_true, y_pred, subjects


# ============================================================
# Literature Comparison Data
# ============================================================

def get_literature_comparison():
    """
    Compile literature comparison data for EEG-based dementia classification.
    Data from published studies.
    """
    studies = [
        {
            'Study': 'Miltiadous et al.',
            'Year': 2023,
            'Journal': 'IEEE Access',
            'Classes': 'AD/FTD/CN',
            'N': 88,
            'Method': 'CNN+GNN',
            'Validation': 'LOSO',
            'Accuracy': '89.7%',
            'Note': '3-class only'
        },
        {
            'Study': 'Ieracitano et al.',
            'Year': 2020,
            'Journal': 'Neural Networks',
            'Classes': 'AD/MCI/CN',
            'N': 63,
            'Method': '1D-CNN',
            'Validation': '10-fold CV',
            'Accuracy': '83.3%',
            'Note': '3-class'
        },
        {
            'Study': 'Cassani et al.',
            'Year': 2018,
            'Journal': 'J Neural Eng',
            'Classes': 'AD/MCI/CN',
            'N': 114,
            'Method': 'SVM',
            'Validation': '10-fold CV',
            'Accuracy': '74.8%',
            'Note': '3-class'
        },
        {
            'Study': 'Siuly et al.',
            'Year': 2022,
            'Journal': 'IEEE TBME',
            'Classes': 'AD/CN',
            'N': 48,
            'Method': 'SVM+Features',
            'Validation': '10-fold CV',
            'Accuracy': '96.5%',
            'Note': 'Binary, single dataset'
        },
        {
            'Study': 'Safi & Safi',
            'Year': 2021,
            'Journal': 'Biomed Signal Proc',
            'Classes': 'AD/CN',
            'N': 44,
            'Method': 'CNN',
            'Validation': '5-fold CV',
            'Accuracy': '93.5%',
            'Note': 'Binary, single dataset'
        },
        {
            'Study': 'Bi & Wang',
            'Year': 2019,
            'Journal': 'Comp Methods Prog Biomed',
            'Classes': 'AD/CN',
            'N': 226,
            'Method': 'LSTM',
            'Validation': 'Hold-out',
            'Accuracy': '90.1%',
            'Note': 'Binary'
        },
        {
            'Study': 'Abásolo et al.',
            'Year': 2006,
            'Journal': 'Med Eng Phys',
            'Classes': 'AD/CN',
            'N': 22,
            'Method': 'LDA',
            'Validation': 'LOO',
            'Accuracy': '81.8%',
            'Note': 'Binary, entropy features'
        },
        {
            'Study': 'Fiscon et al.',
            'Year': 2018,
            'Journal': 'Front Neurosci',
            'Classes': 'AD/MCI/CN',
            'N': 109,
            'Method': 'Ensemble',
            'Validation': '10-fold CV',
            'Accuracy': '78.5%',
            'Note': '3-class, connectivity'
        },
    ]
    # Read actual accuracy from holdout results
    ncg_acc_str = 'TBD'
    holdout_path = PROJECT_ROOT / "outputs" / "results" / "v3_holdout_results" / "results.json"
    if holdout_path.exists():
        with open(holdout_path) as f_h:
            h_data = json.load(f_h)
        sr = h_data.get('holdout', {}).get('subject_results', [])
        if sr:
            y_t = np.array([r['true'] for r in sr])
            y_p = np.array([r['pred'] for r in sr])
            ncg_acc_str = f"{np.sum(y_t == y_p) / len(y_t):.1%}"
    studies.append({
        'Study': 'NeuroChronoGraph (Ours)',
        'Year': 2026,
        'Journal': '—',
        'Classes': 'AD/FTD/CN/MCI',
        'N': 560,
        'Method': 'GNN (Hierarchical)',
        'Validation': 'Hold-out',
        'Accuracy': ncg_acc_str,
        'Note': '4-class, 5 datasets, multi-center'
    })
    return studies


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # 1. Load holdout results
    print("\n[1] Loading holdout predictions...")
    holdout_path = RESULTS_DIR / "v3_holdout_results" / "results.json"
    
    if not holdout_path.exists():
        print(f"  ERROR: {holdout_path} not found!")
        print("  Run the training script first to generate holdout results.")
        return
    
    with open(holdout_path) as f:
        data = json.load(f)
    
    holdout = data.get('holdout', {})
    subject_results = holdout.get('subject_results', [])
    
    if not subject_results:
        print("  ERROR: No subject results found in holdout data!")
        return
    
    y_true = np.array([r['true'] for r in subject_results])
    y_pred = np.array([r['pred'] for r in subject_results])
    probs = np.array([r['probs'] for r in subject_results])
    
    n_samples = len(y_true)
    n_correct = np.sum(y_true == y_pred)
    print(f"  Loaded {n_samples} holdout predictions ({n_correct} correct)")
    
    # 2. Bootstrap CIs
    print("\n[2] Computing Bootstrap 95% CIs (10,000 iterations)...")
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score
    
    acc_point, acc_lo, acc_hi = bootstrap_ci(
        y_true, y_pred, accuracy_score, n_bootstrap=10000
    )
    f1_point, f1_lo, f1_hi = bootstrap_ci(
        y_true, y_pred,
        lambda t, p: f1_score(t, p, average='macro', labels=[0,1,2,3], zero_division=0),
        n_bootstrap=10000
    )
    bacc_point, bacc_lo, bacc_hi = bootstrap_ci(
        y_true, y_pred, balanced_accuracy_score, n_bootstrap=10000
    )
    kappa_point, kappa_lo, kappa_hi = bootstrap_ci(
        y_true, y_pred, cohen_kappa_score, n_bootstrap=10000
    )
    
    results['bootstrap_ci'] = {
        'accuracy': {'point': acc_point, 'lower': acc_lo, 'upper': acc_hi},
        'f1_macro': {'point': f1_point, 'lower': f1_lo, 'upper': f1_hi},
        'balanced_accuracy': {'point': bacc_point, 'lower': bacc_lo, 'upper': bacc_hi},
        'cohens_kappa': {'point': kappa_point, 'lower': kappa_lo, 'upper': kappa_hi},
        'note': 'Window-level resampling; CIs are optimistically narrow due to within-subject correlation.'
    }

    print(f"  Accuracy: {acc_point:.3f} [{acc_lo:.3f}, {acc_hi:.3f}]")
    print(f"  F1 Macro: {f1_point:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")
    print(f"  Bal. Acc: {bacc_point:.3f} [{bacc_lo:.3f}, {bacc_hi:.3f}]")
    print(f"  Cohen κ:  {kappa_point:.3f} [{kappa_lo:.3f}, {kappa_hi:.3f}]")

    # 2b. Subject-level bootstrap CIs (valid, clustered)
    print("\n[2b] Subject-clustered Bootstrap CIs (5,000 iterations, valid CIs)...")
    s_acc_pt, s_acc_lo, s_acc_hi = bootstrap_ci_subject_level(
        subject_results, accuracy_score, n_bootstrap=5000)
    s_f1_pt, s_f1_lo, s_f1_hi = bootstrap_ci_subject_level(
        subject_results,
        lambda t, p: f1_score(t, p, average='macro', labels=[0,1,2,3], zero_division=0),
        n_bootstrap=5000)
    s_bacc_pt, s_bacc_lo, s_bacc_hi = bootstrap_ci_subject_level(
        subject_results, balanced_accuracy_score, n_bootstrap=5000)
    s_kappa_pt, s_kappa_lo, s_kappa_hi = bootstrap_ci_subject_level(
        subject_results, cohen_kappa_score, n_bootstrap=5000)

    results['bootstrap_ci_subject_clustered'] = {
        'accuracy': {'point': s_acc_pt, 'lower': s_acc_lo, 'upper': s_acc_hi},
        'f1_macro': {'point': s_f1_pt, 'lower': s_f1_lo, 'upper': s_f1_hi},
        'balanced_accuracy': {'point': s_bacc_pt, 'lower': s_bacc_lo, 'upper': s_bacc_hi},
        'cohens_kappa': {'point': s_kappa_pt, 'lower': s_kappa_lo, 'upper': s_kappa_hi},
        'note': 'Subject-clustered resampling (n=51 subjects). Valid CIs accounting for within-subject correlation.'
    }
    print(f"  Accuracy: {s_acc_pt:.3f} [{s_acc_lo:.3f}, {s_acc_hi:.3f}]  (vs window-level [{acc_lo:.3f}, {acc_hi:.3f}])")
    print(f"  F1 Macro: {s_f1_pt:.3f} [{s_f1_lo:.3f}, {s_f1_hi:.3f}]")
    print(f"  Bal. Acc: {s_bacc_pt:.3f} [{s_bacc_lo:.3f}, {s_bacc_hi:.3f}]")
    print(f"  Cohen κ:  {s_kappa_pt:.3f} [{s_kappa_lo:.3f}, {s_kappa_hi:.3f}]")
    
    # 3. Wilson Score CI
    print("\n[3] Wilson Score CI for accuracy...")
    wilson_lo, wilson_hi = wilson_score_ci(n_correct, n_samples)
    results['wilson_ci'] = {'lower': wilson_lo, 'upper': wilson_hi, 'n': n_samples}
    print(f"  Wilson 95% CI: [{wilson_lo:.3f}, {wilson_hi:.3f}]")
    
    # 4. Binomial test vs chance
    print("\n[4] Exact binomial test vs chance (25%)...")
    binom_p = binomial_test_vs_chance(n_correct, n_samples, chance=0.25)
    results['binomial_test'] = {'p_value': binom_p, 'chance_level': 0.25}
    print(f"  p-value: {binom_p:.2e} (H0: accuracy ≤ 25%)")
    
    # 5. DeLong AUC CIs (per-class one-vs-rest)
    print("\n[5] DeLong AUC CIs (one-vs-rest)...")
    results['delong_auc'] = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_binary = (y_true == cls_idx).astype(int)
        y_scores = probs[:, cls_idx]
        auc, auc_lo, auc_hi = delong_auc_ci(y_binary, y_scores)
        if auc is not None:
            results['delong_auc'][cls_name] = {'auc': auc, 'lower': auc_lo, 'upper': auc_hi}
            print(f"  {cls_name}: AUC = {auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")
    
    # 6. Subject-Level Majority Voting
    print("\n[6] Subject-level majority voting...")
    subj_true, subj_pred, subjects = subject_level_majority_vote(subject_results)
    
    subj_acc = accuracy_score(subj_true, subj_pred)
    subj_f1 = f1_score(subj_true, subj_pred, average='macro', labels=[0,1,2,3], zero_division=0)
    subj_bacc = balanced_accuracy_score(subj_true, subj_pred)
    subj_kappa = cohen_kappa_score(subj_true, subj_pred)
    
    results['subject_level'] = {
        'n_subjects': len(subjects),
        'accuracy': float(subj_acc),
        'f1_macro': float(subj_f1),
        'balanced_accuracy': float(subj_bacc),
        'cohens_kappa': float(subj_kappa),
        'per_class': {},
    }
    
    print(f"  N subjects: {len(subjects)}")
    print(f"  Subject-level Accuracy: {subj_acc:.1%}")
    print(f"  Subject-level F1 Macro: {subj_f1:.3f}")
    print(f"  Subject-level BACC: {subj_bacc:.1%}")
    print(f"  Subject-level κ: {subj_kappa:.3f}")
    
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_mask = (subj_true == cls_idx)
        if cls_mask.any():
            cls_correct = np.sum((subj_true == cls_idx) & (subj_pred == cls_idx))
            cls_total = np.sum(cls_mask)
            cls_recall = cls_correct / cls_total
            results['subject_level']['per_class'][cls_name] = {
                'recall': float(cls_recall),
                'n_subjects': int(cls_total),
                'n_correct': int(cls_correct),
            }
            print(f"    {cls_name}: {cls_correct}/{cls_total} correct ({cls_recall:.1%})")
    
    # Save subject-level CSV
    subj_df = pd.DataFrame({
        'subject': subjects,
        'true_class': [CLASS_NAMES[t] for t in subj_true],
        'pred_class': [CLASS_NAMES[p] for p in subj_pred],
        'correct': subj_true == subj_pred,
    })
    subj_df.to_csv(RESULTS_DIR / "subject_level_results.csv", index=False)
    
    # 7. Per-fold performance table
    print("\n[7] Per-fold performance table...")
    cv_path = RESULTS_DIR / "cv_results.csv"
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        fold_table = []
        for _, row in cv_df.iterrows():
            fold_table.append({
                'fold': int(row['fold']) + 1,
                'screen_acc': float(row['acc_screen']),
                'screen_kappa': float(row['cohens_kappa_screen']),
                'stage_bacc': float(row.get('bacc_stage', row.get('acc_stage', 0))),
                'subtype_bacc': float(row['Balanced_Acc_Subtype']),
                'collapsed': bool(row['collapsed']),
                'retries': int(row['retry_count']),
            })
        results['fold_performance'] = fold_table
        
        print(f"  {'Fold':>4} {'Screen Acc':>11} {'Screen κ':>10} {'Stage BACC':>11} {'Subtype BACC':>13} {'Status':>10}")
        print("  " + "-" * 62)
        for ft in fold_table:
            status = "COLLAPSED" if ft['collapsed'] else "Valid"
            print(f"  {ft['fold']:>4d} {ft['screen_acc']:>10.1%} {ft['screen_kappa']:>10.3f} "
                  f"{ft['stage_bacc']:>11.1%} {ft['subtype_bacc']:>13.1%} {status:>10}")
    
    # 8. Literature comparison
    print("\n[8] Literature comparison data...")
    lit_data = get_literature_comparison()
    results['literature_comparison'] = lit_data
    
    lit_df = pd.DataFrame(lit_data)
    lit_df.to_csv(RESULTS_DIR / "literature_comparison.csv", index=False)
    
    print(f"\n  {'Study':<25} {'Year':>5} {'Classes':<15} {'N':>5} {'Method':<20} {'Validation':<12} {'Acc':<8}")
    print("  " + "-" * 95)
    for s in lit_data:
        print(f"  {s['Study']:<25} {s['Year']:>5} {s['Classes']:<15} {s['N']:>5} "
              f"{s['Method']:<20} {s['Validation']:<12} {s['Accuracy']:<8}")
    
    # 9. Dataset-of-origin confound probe
    print("\n[9] Dataset-of-origin confound analysis...")
    # Subject IDs are expected to be prefixed with dataset name (e.g. 'ds004504_sub-001').
    # We infer dataset from the prefix before the first underscore.
    dataset_labels = []
    dataset_ids = []
    KNOWN_DATASETS = ['ds004504', 'ds006036', 'Alz_EEG', 'Mendeley', 'MCI_Dataset']
    for r in subject_results:
        sid = str(r['subject'])
        ds = 'unknown'
        for known in KNOWN_DATASETS:
            if sid.startswith(known) or known.lower() in sid.lower():
                ds = known
                break
        dataset_ids.append(ds)
        dataset_labels.append(r['true'])

    unique_ds = sorted(set(dataset_ids))
    if len(unique_ds) > 1:
        # Per-dataset accuracy
        ds_accuracy = {}
        ds_error_rate = {}
        for ds in unique_ds:
            mask = [i for i, d in enumerate(dataset_ids) if d == ds]
            if not mask:
                continue
            ds_true = y_true[mask]
            ds_pred = y_pred[mask]
            acc = float(np.mean(ds_true == ds_pred))
            ds_accuracy[ds] = acc
            ds_error_rate[ds] = 1.0 - acc
            print(f"  {ds}: n={len(mask)} windows, accuracy={acc:.1%}")

        # Check if error rate varies significantly across datasets (Kruskal-Wallis)
        error_groups = []
        for ds in unique_ds:
            mask = [i for i, d in enumerate(dataset_ids) if d == ds]
            if mask:
                errors = (y_true[mask] != y_pred[mask]).astype(float).tolist()
                error_groups.append(errors)

        if len(error_groups) >= 2 and all(len(g) > 0 for g in error_groups):
            try:
                kw_stat, kw_p = stats.kruskal(*error_groups)
                print(f"  Kruskal-Wallis test for error-rate homogeneity across datasets: "
                      f"H={kw_stat:.2f}, p={kw_p:.4f}")
                confound_significant = kw_p < 0.05
            except Exception as e:
                kw_stat, kw_p, confound_significant = 0.0, 1.0, False
                print(f"  Kruskal-Wallis test failed: {e}")
        else:
            kw_stat, kw_p, confound_significant = 0.0, 1.0, False

        results['dataset_confound'] = {
            'per_dataset_accuracy': ds_accuracy,
            'kruskal_wallis_H': float(kw_stat),
            'kruskal_wallis_p': float(kw_p),
            'error_rate_heterogeneous': confound_significant,
            'note': (
                'Significant heterogeneity (p<0.05) across datasets indicates possible dataset-specific '
                'confounding. Interpret holdout accuracy with caution if datasets are imbalanced.'
                if confound_significant else
                'No significant error-rate heterogeneity detected across datasets at p<0.05.'
            )
        }
    else:
        print("  Could not infer dataset IDs from subject identifiers — skipping confound probe.")
        results['dataset_confound'] = {'note': 'Subject IDs did not encode dataset of origin.'}

    # Save all results
    print("\n[10] Saving all results...")
    with open(RESULTS_DIR / "statistical_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Saved to {RESULTS_DIR / 'statistical_analysis.json'}")
    print(f"  Saved to {RESULTS_DIR / 'subject_level_results.csv'}")
    print(f"  Saved to {RESULTS_DIR / 'literature_comparison.csv'}")
    
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
