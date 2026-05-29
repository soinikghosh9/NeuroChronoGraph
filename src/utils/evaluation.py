
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, 
    roc_curve, auc, precision_recall_curve, accuracy_score, recall_score, precision_score
)
from scipy.stats import sem, t
from pathlib import Path

# Publication-Ready Plot Style (Light Theme, Colorblind Friendly)
# Using Set2 or Colorblind palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

CN_COLOR = "#0072B2"  # Blue
IMP_COLOR = "#D55E00" # Vermillion
MCI_COLOR = "#E69F00" # Orange
DEM_COLOR = "#CC79A7" # Reddish Purple
AD_COLOR = "#009E73"  # Bluish Green
FTD_COLOR = "#F0E442" # Yellow

def calculate_confidence_interval(metric_values, confidence=0.95):
    """
    Calculate Mean and CI for a list of metric values (e.g. from CV folds or Bootstrap).
    Returns (mean, margin_of_error)
    """
    n = len(metric_values)
    if n < 2:
        return np.mean(metric_values), 0.0
        
    m = np.mean(metric_values)
    std_err = sem(metric_values)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return m, h

def calculate_clinical_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate comprehensive clinical metrics: 
    Sens, Spec, PPV, NPV, F1, Acc, AUC.
    Assumes binary classification: 0=Negative (Healthy/MCI/AD), 1=Positive (Impaired/Dementia/FTD)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp / (tp + fn + 1e-10) # Sensitivity / Recall
    spec = tn / (tn + fp + 1e-10) # Specificity
    ppv = tp / (tp + fp + 1e-10)  # Precision / Positive Predictive Value
    npv = tn / (tn + fn + 1e-10)  # Negative Predictive Value
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (ppv * sens) / (ppv + sens + 1e-10)
    
    lr_plus = sens / (1 - spec + 1e-10)
    lr_minus = (1 - sens) / (spec + 1e-10)
    
    metrics = {
        'Accuracy': acc,
        'Sensitivity': sens,
        'Specificity': spec,
        'PPV': ppv,
        'NPV': npv,
        'F1_Score': f1,
        'LR+': lr_plus,
        'LR-': lr_minus
    }
    
    if y_probs is not None:
        try:
            metrics['AUC'] = float(auc(*roc_curve(y_true, y_probs)[:2]))
        except:
            metrics['AUC'] = 0.5
            
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path, cmap='Blues'):
    """Generates and saves a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(7, 6))
    
    labels = [f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(cm.shape)
    
    sns.heatmap(cm_norm, annot=labels, fmt='', cmap=cmap, 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14, "weight": "bold"}, cbar=False, square=True)
    
    plt.title(title, fontsize=16, pad=20, weight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_probs, title, output_path, color=None):
    """Generates and saves a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color=color if color else '#333333', lw=3, 
             label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='#999999', lw=2, linestyle='--')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, weight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, weight='bold')
    plt.title(title, fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, alpha=0.2, linestyle='-')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration_curve(y_true, y_probs, title, output_path, color=None):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color=color if color else '#333333', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#999999', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.title(title, fontsize=16, weight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(targets, probs_screen, probs_stage, probs_subtype, output_dir: Path, prefix=""):
    """
    Generates full suite of plots and detailed clinical metrics (CSV).
    Returns a dictionary containing all calculated metrics for downstream use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    results = {}
    
    # --- 1. SCREENING (CN vs Impaired) ---
    # CN=2 (Neg), Others=0/1/3 (Pos)
    y_screen_true = (targets != 2).astype(int)
    y_screen_pred = np.argmax(probs_screen, axis=1)
    
    plot_confusion_matrix(y_screen_true, y_screen_pred, ['CN', 'Impaired'], 
                          "Screening: Healthy vs Impaired", figures_dir / f"{prefix}cm_screening.png", cmap='Blues')
    plot_roc_curve(y_screen_true, probs_screen[:, 1], 
                   "Screening ROC", figures_dir / f"{prefix}roc_screening.png", color=IMP_COLOR)
    plot_calibration_curve(y_screen_true, probs_screen[:, 1],
                           "Screening Calibration", figures_dir / f"{prefix}cal_screening.png", color=IMP_COLOR)
                   
    m_scr = calculate_clinical_metrics(y_screen_true, y_screen_pred, probs_screen[:, 1])
    dfs.append(pd.DataFrame([m_scr], index=['Screening (CN vs Impaired)']))
    
    # Map to keys expected by main script
    results['acc_screen'] = m_scr['Accuracy']
    from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
    results['cohens_kappa_screen'] = cohen_kappa_score(y_screen_true, y_screen_pred)
    results['screen_auc'] = m_scr.get('AUC', 0.5)
    
    # --- 2. STAGING (MCI vs Dementia) ---
    mask_imp = (targets != 2)
    if mask_imp.sum() > 0:
        # MCI(3)->0 (Neg), AD(0)/FTD(1)->1 (Pos)
        y_stage_true = (targets[mask_imp] != 3).astype(int)
        logits_stage = probs_stage[mask_imp]
        y_stage_pred = np.argmax(logits_stage, axis=1)
        
        plot_confusion_matrix(y_stage_true, y_stage_pred, ['MCI', 'Dementia'], 
                              "Staging: MCI vs Dementia", figures_dir / f"{prefix}cm_staging.png", cmap='Oranges')
        plot_roc_curve(y_stage_true, logits_stage[:, 1],
                       "Staging ROC", figures_dir / f"{prefix}roc_staging.png", color=DEM_COLOR)
        plot_calibration_curve(y_stage_true, logits_stage[:, 1],
                               "Staging Calibration", figures_dir / f"{prefix}cal_staging.png", color=DEM_COLOR)
                               
        m_stg = calculate_clinical_metrics(y_stage_true, y_stage_pred, logits_stage[:, 1])
        dfs.append(pd.DataFrame([m_stg], index=['Staging (MCI vs Dementia)']))
        
        results['acc_stage'] = m_stg['Accuracy']
        results['bacc_stage'] = balanced_accuracy_score(y_stage_true, y_stage_pred)
        results['stage_auc'] = m_stg.get('AUC', 0.5)
    else:
        results['acc_stage'] = 0.0
        results['bacc_stage'] = 0.0
        results['stage_auc'] = 0.0
        
    # --- 3. SUBTYPING (AD vs FTD) ---
    # We define AD=0 (Neg), FTD=1 (Pos) usually, but clinical question is detecting FTD.
    mask_dem = np.isin(targets, [0, 1])
    if mask_dem.sum() > 0:
        y_sub_true = targets[mask_dem] # 0=AD, 1=FTD
        logits_sub = probs_subtype[mask_dem]
        y_sub_pred = np.argmax(logits_sub, axis=1)
        
        plot_confusion_matrix(y_sub_true, y_sub_pred, ['AD', 'FTD'], 
                              "Subtyping: AD vs FTD", figures_dir / f"{prefix}cm_subtyping.png", cmap='Reds')
        plot_roc_curve(y_sub_true, logits_sub[:, 1],
                       "Subtyping ROC (FTD Positive)", figures_dir / f"{prefix}roc_subtyping.png", color=FTD_COLOR)
        plot_calibration_curve(y_sub_true, logits_sub[:, 1],
                               "Subtyping Calibration", figures_dir / f"{prefix}cal_subtyping.png", color=FTD_COLOR)
        
        m_sub = calculate_clinical_metrics(y_sub_true, y_sub_pred, logits_sub[:, 1])
        dfs.append(pd.DataFrame([m_sub], index=['Subtyping (AD vs FTD)']))

        results['acc_subtype'] = m_sub['Accuracy']
        # For Subtyping Balanced Accuracy, we calculate avg recall of AD and FTD
        recall_ad = recall_score(y_sub_true, y_sub_pred, pos_label=0)
        recall_ftd = recall_score(y_sub_true, y_sub_pred, pos_label=1)
        results['Balanced_Acc_Subtype'] = (recall_ad + recall_ftd) / 2
        results['subtype_auc'] = m_sub.get('AUC', 0.5)
    else:
        results['acc_subtype'] = 0.0
        results['Balanced_Acc_Subtype'] = 0.0
        results['subtype_auc'] = 0.0
        
    # Combine and Save
    full_df = pd.concat(dfs)
    full_df.to_csv(output_dir / f"{prefix}clinical_metrics_detailed.csv")

    # Return dict for immediate logging
    return results


def calculate_subject_level_metrics(y_true, y_pred, subject_ids):
    """
    Calculate subject-level metrics using majority voting.

    This is more clinically relevant than window-level metrics because
    in practice we diagnose subjects, not individual EEG windows.

    Args:
        y_true: Array of true labels (window-level)
        y_pred: Array of predicted labels (window-level)
        subject_ids: Array of subject IDs for each window

    Returns:
        Dict with subject-level metrics
    """
    from collections import Counter

    # Group predictions by subject
    subject_preds = {}
    subject_true = {}

    for pred, true, sid in zip(y_pred, y_true, subject_ids):
        if sid not in subject_preds:
            subject_preds[sid] = []
            subject_true[sid] = true  # Assume constant label per subject
        subject_preds[sid].append(pred)

    # Majority vote for each subject
    final_preds = {}
    for sid, preds in subject_preds.items():
        counter = Counter(preds)
        final_preds[sid] = counter.most_common(1)[0][0]

    # Calculate subject-level accuracy
    subjects = list(final_preds.keys())
    y_subj_true = np.array([subject_true[s] for s in subjects])
    y_subj_pred = np.array([final_preds[s] for s in subjects])

    # Metrics
    accuracy = accuracy_score(y_subj_true, y_subj_pred)

    # Per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(
        y_subj_true, y_subj_pred,
        target_names=['AD', 'FTD', 'CN', 'MCI'],
        output_dict=True,
        zero_division=0
    )

    return {
        'subject_level_accuracy': accuracy,
        'n_subjects': len(subjects),
        'per_class_report': report,
        'subject_predictions': dict(zip(subjects, y_subj_pred.tolist())),
        'subject_true_labels': dict(zip(subjects, y_subj_true.tolist()))
    }
