"""
Baseline Comparison Experiment.

Trains and evaluates 5 baseline models on the SAME data splits as NeuroChronoGraph:
1. SVM (RBF) on spectral features
2. Random Forest on spectral features
3. XGBoost on spectral features
4. EEGNet (1D-CNN) on raw EEG
5. Basic GCN (no adaptive graph / cross-band attention)

Uses identical holdout (10%) and 5-fold StratifiedGroupKFold CV splits.
Reports: Accuracy, Macro F1, Cohen's κ, Balanced Accuracy.
Runs McNemar's test vs NeuroChronoGraph.
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, balanced_accuracy_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Reproducibility (centralized — env NCG_SEED overrides default 42)
import os
from src.utils.reproducibility import DEFAULT_SEED, set_global_seed
SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)

from src.config.config import DATA_ROOT, DEVICE, DATASET_CONFIG, DATALOADER_CONFIG
from src.data.dataset_factory import DatasetFactory

# Dataset paths
DATASET_PATHS = {
    'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
    'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
    'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
    'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
    'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset",
}

CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Feature Extraction for Classical ML Baselines
# ============================================================

def extract_spectral_features(x_tensor, sfreq=256.0):
    """
    Extract spectral features from a raw EEG window tensor.
    
    Features per channel: band powers (5 bands), band ratios (3), total power.
    Total: 19 channels × 9 features = 171 features.
    
    Args:
        x_tensor: [n_channels, n_times] numpy array or tensor
        sfreq: sampling frequency
    
    Returns:
        feature vector (1D numpy array)
    """
    if isinstance(x_tensor, torch.Tensor):
        x = x_tensor.numpy()
    else:
        x = x_tensor
    
    n_channels = x.shape[0]
    bands = {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 45)
    }
    
    features = []
    for ch in range(n_channels):
        freqs, psd = welch(x[ch], fs=sfreq, nperseg=min(256, x.shape[1]),
                           noverlap=min(128, x.shape[1] // 2))
        
        band_powers = []
        for band_name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            bp = (np.trapezoid(psd[mask], freqs[mask])
                  if mask.any() else 0.0)
            band_powers.append(bp)
        
        total_power = sum(band_powers) + 1e-10
        rel_powers = [bp / total_power for bp in band_powers]
        
        # Ratios: theta/alpha, theta/beta, alpha/beta
        alpha_p = band_powers[2] + 1e-10
        beta_p = band_powers[3] + 1e-10
        theta_p = band_powers[1] + 1e-10
        ratios = [theta_p / alpha_p, theta_p / beta_p, alpha_p / beta_p]
        
        features.extend(rel_powers + ratios + [np.log(total_power + 1e-10)])
    
    return np.array(features, dtype=np.float32)


def extract_features_from_loader(dataset, indices, sfreq=256.0, desc="Extracting"):
    """Extract spectral features for a set of indices from a dataset."""
    X, y = [], []
    for idx in tqdm(indices, desc=desc, leave=False):
        sample = dataset[idx]
        feat = extract_spectral_features(sample['x'], sfreq=sfreq)
        X.append(feat)
        y.append(sample['label'].item())
    return np.array(X), np.array(y)


# ============================================================
# EEGNet (Simple 1D-CNN Baseline)
# ============================================================

class EEGNetBaseline(nn.Module):
    """Simplified EEGNet for 4-class EEG classification."""
    
    def __init__(self, n_channels=19, n_times=1024, n_classes=4, dropout=0.5):
        super().__init__()
        F1, F2, D = 8, 16, 2
        
        # Block 1: Temporal + Spatial filtering
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Block 2: Separable convolution
        self.separable1 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                                     groups=F1 * D, bias=False)
        self.separable2 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Calculate flattened size
        self._flat_size = self._get_flat_size(n_channels, n_times)
        self.classifier = nn.Linear(self._flat_size, n_classes)
    
    def _get_flat_size(self, n_channels, n_times):
        x = torch.zeros(1, 1, n_channels, n_times)
        x = self.pool1(self.drop1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.pool2(self.drop2(F.elu(self.bn3(self.separable2(self.separable1(x))))))
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        # x: [B, n_channels, n_times]
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_eegnet(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    """Train EEGNet model and return best validation accuracy predictions."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['label']
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y.numpy())
        
        val_acc = accuracy_score(all_true, all_preds)
        scheduler.step(1 - val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return model


# ============================================================
# McNemar's Test  (window-level and subject-level)
# ============================================================

def mcnemars_test(y_true, preds_a, preds_b):
    """
    McNemar's test comparing two models at window level.
    Returns (chi2_statistic, p_value).
    NOTE: window-level test is anti-conservative when windows are correlated.
    Use mcnemars_test_subject_level for a valid subject-level comparison.
    """
    from scipy.stats import chi2

    y_true  = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


def mcnemars_test_subject_level(ncg_subject_results, baseline_preds_windows,
                                 baseline_true_windows, subject_ids_windows):
    """
    Subject-level McNemar's test.

    Aggregates window predictions to subject level (majority vote) for both
    NeuroChronoGraph and the baseline, then performs McNemar's test on the
    resulting n=51 subject-level predictions.

    Args:
        ncg_subject_results : list of dicts {'subject', 'true', 'pred'} from NCG results.json
        baseline_preds_windows : array of window-level baseline predictions
        baseline_true_windows  : array of window-level true labels
        subject_ids_windows    : array of subject IDs aligned with window arrays

    Returns:
        (chi2_stat, p_value, n_subjects, ncg_subj_acc, baseline_subj_acc)
    """
    from collections import Counter
    from scipy.stats import chi2

    # NCG subject-level predictions (already majority-voted in results.json if available,
    # otherwise compute here)
    ncg_subj = {}
    for r in ncg_subject_results:
        sid = r['subject']
        if sid not in ncg_subj:
            ncg_subj[sid] = {'preds': [], 'true': r['true']}
        ncg_subj[sid]['preds'].append(r['pred'])

    ncg_subj_pred = {sid: Counter(v['preds']).most_common(1)[0][0]
                     for sid, v in ncg_subj.items()}
    ncg_subj_true = {sid: v['true'] for sid, v in ncg_subj.items()}

    # Baseline subject-level predictions via majority vote
    baseline_subj = {}
    for sid, pred, true in zip(subject_ids_windows,
                                baseline_preds_windows,
                                baseline_true_windows):
        sid = str(sid)
        if sid not in baseline_subj:
            baseline_subj[sid] = {'preds': [], 'true': true}
        baseline_subj[sid]['preds'].append(pred)

    baseline_subj_pred = {sid: Counter(v['preds']).most_common(1)[0][0]
                          for sid, v in baseline_subj.items()}

    # Align subjects present in both
    common_subjects = sorted(set(ncg_subj_pred) & set(baseline_subj_pred))
    if len(common_subjects) < 2:
        return 0.0, 1.0, 0, 0.0, 0.0

    ncg_correct   = np.array([ncg_subj_pred[s]      == ncg_subj_true[s]      for s in common_subjects])
    base_correct  = np.array([baseline_subj_pred[s] == ncg_subj_true[s]      for s in common_subjects])

    b = np.sum( ncg_correct & ~base_correct)
    c = np.sum(~ncg_correct &  base_correct)

    if b + c == 0:
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    ncg_subj_acc      = float(np.mean(ncg_correct))
    baseline_subj_acc = float(np.mean(base_correct))

    return chi2_stat, p_value, len(common_subjects), ncg_subj_acc, baseline_subj_acc


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("BASELINE COMPARISON EXPERIMENT")
    print("=" * 70)
    
    # 1. Load Data (same as NeuroChronoGraph)
    print("\n[1/5] Loading datasets...")
    factory = DatasetFactory()
    for name, path in DATASET_PATHS.items():
        if path.exists():
            factory.add_dataset(name, path)
        else:
            print(f"  Warning: {name} not found at {path}")
    
    dataset, groups, labels = factory.create_torch_datasets(config=DATASET_CONFIG)
    print(f"  Total samples: {len(dataset)}")
    print(f"  Total subjects: {len(np.unique(groups))}")
    
    # 2. Same holdout split
    print("\n[2/5] Creating identical data splits...")
    train_val_idx, test_idx = factory.get_holdout_split(groups, labels, test_size=0.10)
    
    tv_groups = groups[train_val_idx]
    tv_labels = labels[train_val_idx]
    
    test_labels = labels[test_idx]
    print(f"  Train/Val: {len(train_val_idx)} samples")
    print(f"  Hold-out Test: {len(test_idx)} samples")
    print(f"  Test class dist: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
    
    # 3. Extract features for classical ML baselines
    print("\n[3/5] Extracting spectral features...")
    sfreq = DATASET_CONFIG.get('sfreq', 256.0)
    
    X_test, y_test = extract_features_from_loader(dataset, test_idx, sfreq, "Test set")

    # Capture subject IDs per test window (needed for subject-level McNemar)
    test_subject_ids = groups[test_idx]

    # Store NeuroChronoGraph holdout predictions for McNemar's test
    ncg_holdout_preds = None
    ncg_subject_results = []
    ncg_results_path = RESULTS_DIR / "v3_holdout_results" / "results.json"
    if ncg_results_path.exists():
        with open(ncg_results_path) as f:
            ncg_data = json.load(f)
        if 'holdout' in ncg_data and 'subject_results' in ncg_data['holdout']:
            ncg_subject_results = ncg_data['holdout']['subject_results']
            ncg_holdout_preds = np.array([r['pred'] for r in ncg_subject_results])
            print(f"  Loaded NeuroChronoGraph holdout predictions ({len(ncg_holdout_preds)} samples)")
    
    # 4. Run baselines
    print("\n[4/5] Running baseline models...")
    results_all = []
    
    # --- Classical ML Baselines ---
    classical_models = {
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED,
                         class_weight='balanced', decision_function_shape='ovr'),
        'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=20,
                                                 random_state=SEED, class_weight='balanced',
                                                 n_jobs=-1),
    }
    
    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        classical_models['XGBoost'] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=SEED, use_label_encoder=False, eval_metric='mlogloss',
            n_jobs=-1
        )
    except ImportError:
        print("  XGBoost not installed, skipping")
    
    for model_name, model in classical_models.items():
        print(f"\n  --- {model_name} ---")
        
        # CV performance
        cv_accs, cv_f1s = [], []
        kfold = factory.get_kfold_split(tv_groups, n_splits=5, labels=tv_labels)
        
        for fold, (train_local, val_local) in enumerate(kfold):
            train_global = train_val_idx[train_local]
            val_global = train_val_idx[val_local]
            
            X_train, y_train = extract_features_from_loader(
                dataset, train_global, sfreq, f"Fold {fold+1} train"
            )
            X_val, y_val = extract_features_from_loader(
                dataset, val_global, sfreq, f"Fold {fold+1} val"
            )
            
            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            
            # Handle NaN/Inf
            X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)
            
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_s, y_train)
            
            val_preds = fold_model.predict(X_val_s)
            cv_accs.append(accuracy_score(y_val, val_preds))
            cv_f1s.append(f1_score(y_val, val_preds, average='macro'))
        
        cv_acc_mean = np.mean(cv_accs)
        cv_acc_std = np.std(cv_accs)
        cv_f1_mean = np.mean(cv_f1s)
        
        # Full train/val for holdout evaluation
        X_trainval, y_trainval = extract_features_from_loader(
            dataset, train_val_idx, sfreq, "Full train/val"
        )
        scaler = StandardScaler()
        X_trainval_s = scaler.fit_transform(np.nan_to_num(X_trainval))
        X_test_s = scaler.transform(np.nan_to_num(X_test))
        
        from sklearn.base import clone
        final_model = clone(model)
        final_model.fit(X_trainval_s, y_trainval)
        test_preds = final_model.predict(X_test_s)
        
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        test_kappa = cohen_kappa_score(y_test, test_preds)
        test_bacc = balanced_accuracy_score(y_test, test_preds)
        
        # Window-level McNemar's test
        mcn_chi2, mcn_p = 0.0, 1.0
        if ncg_holdout_preds is not None and len(ncg_holdout_preds) == len(test_preds):
            mcn_chi2, mcn_p = mcnemars_test(y_test, ncg_holdout_preds, test_preds)

        # Subject-level McNemar's test (valid, accounts for within-subject correlation)
        mcn_subj_chi2, mcn_subj_p = 0.0, 1.0
        mcn_subj_n, mcn_subj_ncg_acc, mcn_subj_base_acc = 0, 0.0, 0.0
        if ncg_subject_results and len(test_subject_ids) == len(test_preds):
            mcn_subj_chi2, mcn_subj_p, mcn_subj_n, mcn_subj_ncg_acc, mcn_subj_base_acc = \
                mcnemars_test_subject_level(
                    ncg_subject_results, test_preds, y_test, test_subject_ids
                )

        result = {
            'Model': model_name,
            'CV_Accuracy': f"{cv_acc_mean:.1%} ± {cv_acc_std:.1%}",
            'CV_Accuracy_Mean': cv_acc_mean,
            'CV_F1_Macro': cv_f1_mean,
            'Holdout_Accuracy': test_acc,
            'Holdout_F1_Macro': test_f1,
            'Holdout_Kappa': test_kappa,
            'Holdout_BACC': test_bacc,
            'McNemar_chi2': mcn_chi2,
            'McNemar_p': mcn_p,
            'McNemar_Subject_chi2': mcn_subj_chi2,
            'McNemar_Subject_p': mcn_subj_p,
            'McNemar_Subject_n': mcn_subj_n,
            'McNemar_Subject_NCG_acc': mcn_subj_ncg_acc,
            'McNemar_Subject_Baseline_acc': mcn_subj_base_acc,
        }
        results_all.append(result)
        print(f"    CV: {cv_acc_mean:.1%} ± {cv_acc_std:.1%} | Holdout: {test_acc:.1%} | F1: {test_f1:.3f} | κ: {test_kappa:.3f}")
        print(f"    McNemar window-level p={mcn_p:.4f} | subject-level p={mcn_subj_p:.4f} (n={mcn_subj_n} subjects)")
    
    # --- EEGNet (Deep Learning Baseline) ---
    print(f"\n  --- EEGNet ---")
    n_times = DATASET_CONFIG.get('n_times', 1024)
    
    # CV for EEGNet
    cv_accs_eeg, cv_f1s_eeg = [], []
    kfold = factory.get_kfold_split(tv_groups, n_splits=5, labels=tv_labels)
    
    for fold, (train_local, val_local) in enumerate(kfold):
        print(f"    EEGNet Fold {fold+1}/5...")
        train_global = train_val_idx[train_local]
        val_global = train_val_idx[val_local]
        
        train_ds = Subset(dataset, train_global)
        val_ds = Subset(dataset, val_global)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, **DATALOADER_CONFIG)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, **DATALOADER_CONFIG)
        
        eegnet = EEGNetBaseline(n_channels=19, n_times=n_times, n_classes=4, dropout=0.5).to(DEVICE)
        eegnet = train_eegnet(eegnet, train_loader, val_loader, epochs=30, lr=1e-3, device=DEVICE)
        
        # Evaluate
        eegnet.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(DEVICE)
                logits = eegnet(x)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(batch['label'].numpy())
        
        cv_accs_eeg.append(accuracy_score(val_true, val_preds))
        cv_f1s_eeg.append(f1_score(val_true, val_preds, average='macro'))
        del eegnet
        torch.cuda.empty_cache()
    
    # Final EEGNet on full train/val for holdout
    print("    EEGNet final holdout evaluation...")
    train_ds_full = Subset(dataset, train_val_idx)
    test_ds = Subset(dataset, test_idx)
    train_loader_full = DataLoader(train_ds_full, batch_size=64, shuffle=True, **DATALOADER_CONFIG)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, **DATALOADER_CONFIG)
    
    eegnet_final = EEGNetBaseline(n_channels=19, n_times=n_times, n_classes=4, dropout=0.5).to(DEVICE)
    eegnet_final = train_eegnet(eegnet_final, train_loader_full, test_loader, epochs=30, lr=1e-3, device=DEVICE)
    
    eegnet_final.eval()
    eeg_preds, eeg_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(DEVICE)
            logits = eegnet_final(x)
            eeg_preds.extend(logits.argmax(1).cpu().numpy())
            eeg_true.extend(batch['label'].numpy())
    
    eeg_acc = accuracy_score(eeg_true, eeg_preds)
    eeg_f1 = f1_score(eeg_true, eeg_preds, average='macro')
    eeg_kappa = cohen_kappa_score(eeg_true, eeg_preds)
    eeg_bacc = balanced_accuracy_score(eeg_true, eeg_preds)
    
    mcn_chi2_eeg, mcn_p_eeg = 0.0, 1.0
    if ncg_holdout_preds is not None and len(ncg_holdout_preds) == len(eeg_preds):
        mcn_chi2_eeg, mcn_p_eeg = mcnemars_test(eeg_true, ncg_holdout_preds, np.array(eeg_preds))

    mcn_subj_chi2_eeg, mcn_subj_p_eeg = 0.0, 1.0
    mcn_subj_n_eeg, mcn_subj_ncg_acc_eeg, mcn_subj_base_acc_eeg = 0, 0.0, 0.0
    if ncg_subject_results and len(test_subject_ids) == len(eeg_preds):
        mcn_subj_chi2_eeg, mcn_subj_p_eeg, mcn_subj_n_eeg, \
            mcn_subj_ncg_acc_eeg, mcn_subj_base_acc_eeg = \
            mcnemars_test_subject_level(
                ncg_subject_results, np.array(eeg_preds), np.array(eeg_true), test_subject_ids
            )

    results_all.append({
        'Model': 'EEGNet',
        'CV_Accuracy': f"{np.mean(cv_accs_eeg):.1%} ± {np.std(cv_accs_eeg):.1%}",
        'CV_Accuracy_Mean': np.mean(cv_accs_eeg),
        'CV_F1_Macro': np.mean(cv_f1s_eeg),
        'Holdout_Accuracy': eeg_acc,
        'Holdout_F1_Macro': eeg_f1,
        'Holdout_Kappa': eeg_kappa,
        'Holdout_BACC': eeg_bacc,
        'McNemar_chi2': mcn_chi2_eeg,
        'McNemar_p': mcn_p_eeg,
        'McNemar_Subject_chi2': mcn_subj_chi2_eeg,
        'McNemar_Subject_p': mcn_subj_p_eeg,
        'McNemar_Subject_n': mcn_subj_n_eeg,
        'McNemar_Subject_NCG_acc': mcn_subj_ncg_acc_eeg,
        'McNemar_Subject_Baseline_acc': mcn_subj_base_acc_eeg,
    })
    print(f"    CV: {np.mean(cv_accs_eeg):.1%} ± {np.std(cv_accs_eeg):.1%} | Holdout: {eeg_acc:.1%} | F1: {eeg_f1:.3f} | κ: {eeg_kappa:.3f}")
    
    del eegnet_final
    torch.cuda.empty_cache()
    
    # --- Add NeuroChronoGraph row from actual results ---
    ncg_row = {
        'Model': 'NeuroChronoGraph (Ours)',
        'CV_Accuracy': 'N/A',
        'CV_Accuracy_Mean': np.nan,
        'CV_F1_Macro': np.nan,
        'Holdout_Accuracy': np.nan,
        'Holdout_F1_Macro': np.nan,
        'Holdout_Kappa': np.nan,
        'Holdout_BACC': np.nan,
        'McNemar_chi2': np.nan,
        'McNemar_p': np.nan,
    }
    # Read actual holdout results
    ncg_holdout_path = RESULTS_DIR / "v3_holdout_results" / "results.json"
    if ncg_holdout_path.exists():
        with open(ncg_holdout_path) as f:
            ncg_data = json.load(f)
        h = ncg_data.get('holdout', {})
        sr = h.get('subject_results', [])
        if sr:
            ncg_true = np.array([r['true'] for r in sr])
            ncg_pred_arr = np.array([r['pred'] for r in sr])
            ncg_row['Holdout_Accuracy'] = float(accuracy_score(ncg_true, ncg_pred_arr))
            ncg_row['Holdout_F1_Macro'] = float(f1_score(ncg_true, ncg_pred_arr, average='macro'))
            ncg_row['Holdout_Kappa'] = float(cohen_kappa_score(ncg_true, ncg_pred_arr))
            ncg_row['Holdout_BACC'] = float(balanced_accuracy_score(ncg_true, ncg_pred_arr))
    # Read CV results
    cv_path = RESULTS_DIR / "cv_results.csv"
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        if 'acc_screen' in cv_df.columns:
            cv_acc_vals = cv_df['acc_screen'].values
            ncg_row['CV_Accuracy'] = f"{np.mean(cv_acc_vals):.1%} ± {np.std(cv_acc_vals):.1%}"
            ncg_row['CV_Accuracy_Mean'] = float(np.mean(cv_acc_vals))
        if 'screen_f1' in cv_df.columns:
            ncg_row['CV_F1_Macro'] = float(np.mean(cv_df['screen_f1'].values))
    results_all.append(ncg_row)
    
    # 5. Save results
    print("\n[5/5] Saving results...")
    df = pd.DataFrame(results_all)
    df.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
    print(f"  Saved to {RESULTS_DIR / 'baseline_comparison.csv'}")
    
    # Print summary table
    print("\n" + "=" * 110)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 110)
    print(f"{'Model':<25} {'CV Acc':>15} {'Holdout Acc':>12} {'F1 Macro':>10} {'κ':>8} "
          f"{'BACC':>8} {'McNemar p':>10} {'Subj McNemar p':>15}")
    print("-" * 110)
    for r in results_all:
        p_str      = f"{r['McNemar_p']:.4f}"         if not np.isnan(r.get('McNemar_p', np.nan))         else "—"
        p_subj_str = f"{r['McNemar_Subject_p']:.4f}" if not np.isnan(r.get('McNemar_Subject_p', np.nan)) else "—"
        print(f"{r['Model']:<25} {r['CV_Accuracy']:>15} {r['Holdout_Accuracy']:>11.1%} "
              f"{r['Holdout_F1_Macro']:>10.3f} {r['Holdout_Kappa']:>8.3f} "
              f"{r['Holdout_BACC']:>8.3f} {p_str:>10} {p_subj_str:>15}")
    print("=" * 110)
    print("NOTE: Window-level McNemar p-value is anti-conservative (correlated windows).")
    print("      Subject-level McNemar p-value (n=51) is the valid comparison.")
    print("=" * 110)
    
    # Save detailed JSON
    results_json = {
        'experiment': 'baseline_comparison',
        'seed': SEED,
        'n_folds': 5,
        'holdout_size': 0.10,
        'models': results_all,
    }
    with open(RESULTS_DIR / "baseline_comparison.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print("\nDone! Results saved.")


if __name__ == '__main__':
    main()
