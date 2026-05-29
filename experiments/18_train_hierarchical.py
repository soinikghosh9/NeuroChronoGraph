"""
Hierarchical Training Script.

Orchestrates Multi-Dataset Training with Hierarchical Heads.
Pipeline:
1. Load Datasets (OpenNeuro EC+EO, AlzEEG, Mendeley).
2. GroupKFold Cross-Validation.
3. Train NeuroChronoGraphV2 (Hierarchical).
4. Evaluate Screening/Staging/Subtype performance.
"""

import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, cohen_kappa_score, balanced_accuracy_score

# ================== REPRODUCIBILITY ==================
# Centralized: ``src.utils.reproducibility`` seeds every RNG we know about,
# pins cuBLAS workspace, and toggles cuDNN determinism. The env var
# ``NCG_SEED`` (set by run_all.py) overrides the default so the whole pipeline
# uses one consistent seed across phases.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.append(str(_PROJECT_ROOT))
from src.utils.reproducibility import (
    DEFAULT_SEED, set_global_seed, worker_init_fn as _ncg_worker_init_fn,
    make_dataloader_generator,
)

SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)
DATALOADER_GEN = make_dataloader_generator(SEED)

def set_seed(seed: int = SEED):
    """Backwards-compatible wrapper around :func:`set_global_seed`."""
    set_global_seed(seed)

def worker_init_fn(worker_id):
    _ncg_worker_init_fn(worker_id, base_seed=SEED)

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import (
    DATA_ROOT, DEVICE, TRAINING_CONFIG, DATALOADER_CONFIG, DATASET_CONFIG
)
from src.data.augmentation import calculate_class_weights
from src.data.dataset_factory import DatasetFactory
from src.models.v2.neuro_chrono_graph_v2 import create_neuro_chrono_graph_v2

# --- Feature-stream configuration ---
# DEFAULT (production): feature streams are DISABLED. The ablation in
# Section "Ablation Study" of the paper showed that the FeatureStreamFusion
# module (482K params, hand-crafted spectral/connectivity/complexity/microstate
# vectors) reduced CV accuracy by 3.8 pp and hold-out accuracy by 4.9 pp,
# while doubling fold-level variance. Streams duplicate signal the backbone
# already extracts via its conv + GAT pipeline.
# Set NCG_USE_STREAMS=1 to opt back in (e.g. for ablation reproducibility);
# outputs are then suffixed with `_streams` to keep the artefacts separate.
USE_STREAMS = os.environ.get("NCG_USE_STREAMS", "").strip() in ("1", "true", "True", "yes")
FEATURE_STREAMS = ['spectral', 'connectivity', 'complexity', 'microstate'] if USE_STREAMS else None
RUN_SUFFIX = "_streams" if USE_STREAMS else ""
if USE_STREAMS:
    print(">>> STREAMS MODE: feature_streams=ALL (NCG_USE_STREAMS=1)")
    print(f">>> Outputs will be suffixed with '{RUN_SUFFIX}'")


def _streams_to_device(batch, device):
    """Move per-window biomarker streams to device, return None if absent."""
    if 'feature_streams' not in batch:
        return None
    return {k: v.to(device) for k, v in batch['feature_streams'].items()}
from src.models.losses import HierarchicalLoss
from src.utils.evaluation import generate_detailed_report

# Define Datasets to Load
DATASET_PATHS = {
    'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
    'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
    'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
    'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
    'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset"  # New MCI data source
}

# ================== EARLY STOPPING ==================
class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[Early Stopping] No improvement for {self.patience} epochs. Best: {self.best_score:.4f} at epoch {self.best_epoch+1}")
                return True
        return False

def main():
    print("Initializing Hierarchical Training Pipeline...")
    
    # 1. Load Data
    factory = DatasetFactory()
    for name, path in DATASET_PATHS.items():
        if path.exists():
            factory.add_dataset(name, path)
        else:
            print(f"Warning: Dataset {name} not found at {path}")
            
    # Create Combined Dataset with anti-overfitting settings.
    # In ablation mode, disable stream computation in the dataloader so we
    # don't pay the per-window biomarker cost when streams aren't consumed.
    _ds_cfg = dict(DATASET_CONFIG)
    if NO_STREAMS:
        _ds_cfg['compute_feature_streams'] = False
    dataset, groups, labels = factory.create_torch_datasets(config=_ds_cfg)
    print(f"Total Samples: {len(dataset)}")
    print(f"Total Subjects: {len(np.unique(groups))}")

    # Print overall class distribution
    print("\nOverall Class Distribution:")
    for cls_idx, cls_name in enumerate(['AD', 'FTD', 'CN', 'MCI']):
        count = (labels == cls_idx).sum()
        pct = count / len(labels) * 100
        print(f"  {cls_name}: {count} ({pct:.1f}%)")
    
    # 2. Hold-out Test Split (10%)
    print("\nCreating Hold-out Test Set (10%)...")
    train_val_idx, test_idx = factory.get_holdout_split(groups, labels, test_size=0.10)
    
    # Create Test Dataset
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    # Reduced batch size to prevent OOM on 8GB GPU (Increased slightly for speed)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn, **DATALOADER_CONFIG)
    
    # Report Split
    train_val_subjs = np.unique(groups[train_val_idx])
    test_subjs = np.unique(groups[test_idx])
    print(f"Train/Val Subjects: {len(train_val_subjs)}")
    print(f"Test Hold-out Subjects: {len(test_subjs)}")
    
    # Verify Leakage
    overlap = np.intersect1d(train_val_subjs, test_subjs)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL DATA LEAKAGE: Subjects in both Train/Val and Test: {overlap}")
    print("Test Set Leakage Check: PASS")

    # Check Class Distribution in Test Set
    test_labels = labels[test_idx]
    print("Test Set Class Distribution:")
    for cls_idx, cls_name in enumerate(['AD', 'FTD', 'CN', 'MCI']):
        count = (test_labels == cls_idx).sum()
        print(f"  {cls_name}: {count} samples")

    # 3. Cross-Validation Loop (on train_val_idx)
    # Re-extract groups/labels for the subset to feed KFold correctly?
    # GroupKFold usually expects alignment with input.
    # We can pass the subset indices to KFold, but GroupKFold needs the GROUPS array to match the indices.
    # Simpler: Split the *indices* of train_val_idx using GroupKFold.

    # Get groups and labels for the train_val subset
    tv_groups = groups[train_val_idx]
    tv_labels = labels[train_val_idx]

    # Get KFold splits on these indices relative to the subset
    # BUT we need original indices for Subset()
    # So: split local indices (0..N_tv) then map back to train_val_idx[local]

    # Use STRATIFIED GroupKFold to prevent fold collapse on minority classes
    kfold = factory.get_kfold_split(tv_groups, n_splits=5, labels=tv_labels)

    results = []
    dev_subject_results = []
    training_histories = []  # Collect training dynamics from each fold

    # Track best model for testing
    best_val_score = 0.0
    best_model_state = None

    # Per-fold interpretability tracking
    fold_alpha_values = []       # wPLI-prior blending scalar per fold
    fold_band_coupling = []      # Cross-band coupling matrix per fold (5×5)
    
    for fold, (train_local, val_local) in enumerate(kfold):
        print(f"\n===== Fold {fold+1}/5 =====")
        
        # Map local KeyFold indices back to global dataset indices
        train_idx = train_val_idx[train_local]
        val_idx = train_val_idx[val_local]
        
        # Stats & Leakage Check
        train_subjs = np.unique(groups[train_idx])
        val_subjs = np.unique(groups[val_idx])
        print(f"  Train: {len(train_idx)} samples ({len(train_subjs)} subjects)")
        print(f"  Val:   {len(val_idx)} samples ({len(val_subjs)} subjects)")
        
        if np.intersect1d(train_subjs, val_subjs).size > 0:
            raise ValueError("Leakage detected in Fold!")
            
        # Class Distribution Check with Minimum Validation
        print("  Class Distribution:")
        min_samples_per_class = {'train': 10, 'val': 5}  # Minimum required samples
        for split_name, split_idx in [("Train", train_idx), ("Val", val_idx)]:
            s_labels = labels[split_idx]
            counts = [f"{cls}:{(s_labels==i).sum()}" for i, cls in enumerate(['AD', 'FTD', 'CN', 'MCI'])]
            print(f"    {split_name}: {', '.join(counts)}")

            # Warn if any class has too few samples
            min_req = min_samples_per_class[split_name.lower()]
            for i, cls in enumerate(['AD', 'FTD', 'CN', 'MCI']):
                count = (s_labels == i).sum()
                if count < min_req:
                    print(f"    ⚠️  WARNING: {split_name} has only {count} {cls} samples (min: {min_req})")

        # Subset
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        
        # Calculate Sample Weights for WeightedRandomSampler
        y_train = labels[train_idx]
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        
        
        sample_weights = class_weights[y_train]
        sample_weights = torch.from_numpy(sample_weights).float()

        # ========== FOLD COLLAPSE RECOVERY MECHANISM ==========
        # Retry training if collapse is detected (max 3 retries with different seeds)
        MAX_FOLD_RETRIES = 3
        fold_collapsed = True  # Start as collapsed, set to False on success
        fold_retry_count = 0
        final_history = None
        final_metrics = None

        while fold_collapsed and fold_retry_count < MAX_FOLD_RETRIES:
            retry_seed = SEED + fold + (fold_retry_count * 1000)  # Different seed per retry
            if fold_retry_count > 0:
                print(f"\n  🔄 RETRY {fold_retry_count}/{MAX_FOLD_RETRIES} for Fold {fold+1} (seed: {retry_seed})")

            # Use generator for reproducibility
            g = torch.Generator()
            g.manual_seed(retry_seed)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True, generator=g)

            # Reduced batch sizes for stability (Increased slightly for speed)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, sampler=sampler, worker_init_fn=worker_init_fn, **DATALOADER_CONFIG)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn, **DATALOADER_CONFIG)

            # ============ SCREENING CLASS WEIGHTS ============
            # CN vs Impaired: CN is minority (~4500 vs ~9000)
            y_train = labels[train_idx]
            screen_mapped = (y_train != 2).astype(int)  # 0=CN, 1=Impaired
            screen_counts = np.bincount(screen_mapped, minlength=2)
            print(f"    Screening Counts (Train): CN={screen_counts[0]}, Impaired={screen_counts[1]}")

            # Screening weights with CAPPED imbalance ratio
            screen_weights = 1.0 / (screen_counts + 1e-6)
            screen_weights[0] *= 1.5  # Moderate CN boost
            screen_weights = screen_weights / screen_weights.sum()
            # CAP: Maximum weight ratio of 4:1 to prevent extreme imbalance
            max_screen_ratio = 4.0
            if screen_weights[0] / screen_weights[1] > max_screen_ratio:
                screen_weights[0] = max_screen_ratio / (max_screen_ratio + 1)
                screen_weights[1] = 1.0 / (max_screen_ratio + 1)
            elif screen_weights[1] / screen_weights[0] > max_screen_ratio:
                screen_weights[1] = max_screen_ratio / (max_screen_ratio + 1)
                screen_weights[0] = 1.0 / (max_screen_ratio + 1)
            screen_weights = torch.tensor(screen_weights, dtype=torch.float32).to(DEVICE)
            print(f"    Screening Weights: CN={screen_weights[0]:.3f}, Impaired={screen_weights[1]:.3f}")

            # ============ SUBTYPE CLASS WEIGHTS (AD vs FTD) ============
            # Issue: AD sensitivity 51.4% is low, 29% AD misclassified as FTD
            # Root cause: FTD is minority so gets higher weight, but AD detection is critical
            # Fix: Increase AD boost to improve AD sensitivity
            y_sub = y_train[np.isin(y_train, [0, 1])]
            sub_counts = np.bincount(y_sub, minlength=2)
            sub_weights = 1.0 / (sub_counts + 1e-6)
            # INCREASED AD boost from 1.3x to 2.0x to improve AD sensitivity
            # Clinical rationale: AD misdiagnosis as FTD leads to wrong treatment
            sub_weights[0] *= 2.0  # Strong AD boost for better sensitivity
            sub_weights = sub_weights / sub_weights.sum()
            # CAP: Maximum weight ratio of 3:1 for subtype
            max_sub_ratio = 3.0
            if sub_weights[0] / sub_weights[1] > max_sub_ratio:
                sub_weights[0] = max_sub_ratio / (max_sub_ratio + 1)
                sub_weights[1] = 1.0 / (max_sub_ratio + 1)
            elif sub_weights[1] / sub_weights[0] > max_sub_ratio:
                sub_weights[1] = max_sub_ratio / (max_sub_ratio + 1)
                sub_weights[0] = 1.0 / (max_sub_ratio + 1)
            sub_weights = torch.tensor(sub_weights, dtype=torch.float32).to(DEVICE)
            print(f"    Subtype Weights: AD={sub_weights[0]:.3f}, FTD={sub_weights[1]:.3f}")

            # ============ STAGING CLASS WEIGHTS (MCI vs Dementia) ============
            # MCI=3, Dementia=[0,1]
            # Our Staging Head sees: 0 (if label=3), 1 (if label=0,1)
            is_impaired = y_train != 2
            imp_labels = y_train[is_impaired]

            # 0=MCI (label 3), 1=Dem (label 0,1)
            stage_mapped = np.where(imp_labels == 3, 0, 1)
            stage_counts = np.bincount(stage_mapped, minlength=2)
            print(f"    Staging Counts (Train): MCI={stage_counts[0]}, Dem={stage_counts[1]}")

            # Staging weights with CAPPED imbalance ratio
            # Previous: MCI=0.94, Dem=0.06 was too extreme (16:1 ratio!)
            # This caused MCI overfit (99% sens, 41% PPV) and AD→MCI confusion
            stage_weights = 1.0 / (stage_counts + 1e-6)
            stage_weights[0] *= 1.5  # Moderate MCI boost (reduced from 2.0)
            stage_weights = stage_weights / stage_weights.sum()
            # CAP: Maximum weight ratio of 4:1 to prevent extreme imbalance
            max_ratio = 4.0
            if stage_weights[0] / stage_weights[1] > max_ratio:
                stage_weights[0] = max_ratio / (max_ratio + 1)
                stage_weights[1] = 1.0 / (max_ratio + 1)
            stage_weights = torch.tensor(stage_weights, dtype=torch.float32).to(DEVICE)
            print(f"    Staging Weights: MCI={stage_weights[0]:.3f}, Dem={stage_weights[1]:.3f}")

            # Reset ONLY torch seed for consistent model weight initialization
            # Don't reset numpy/random as it affects the already-seeded sampler
            torch.manual_seed(retry_seed)
            torch.cuda.manual_seed(retry_seed)

            # Initialize V2 hierarchical model.
            # When FEATURE_STREAMS is None (ablation mode), the FeatureStreamFusion
            # module is omitted -- ~480K fewer params and no stream conditioning.
            _model_cfg = {
                'n_classes': 3,
                'n_times': DATASET_CONFIG['n_times'],
                'hidden_dim': 128,
                'dropout': 0.4,
            }
            if FEATURE_STREAMS is not None:
                _model_cfg['feature_streams'] = FEATURE_STREAMS
                _model_cfg['stream_dim'] = 64
            model = create_neuro_chrono_graph_v2(_model_cfg).to(DEVICE)

            # Optimizer with moderate weight decay - single param group.
            # NOTE: Earlier attempt at 0.3x stream LR + p=0.2 stream-dropout
            # over-regularized: stream gates barely moved from init and the
            # backbone had less info than the no-stream baseline (CV 75.8%
            # vs 78.3% prev). Reverted to uniform LR; stream_dropout reduced
            # to 0.1 for milder noise injection.
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,
                weight_decay=0.02
            )

            # Learning Rate Scheduler - OneCycleLR.
            # Curriculum extended: Phase 1 (screen-only) 8 -> 12 epochs so the
            # stream gates have time to settle before staging/subtype losses
            # turn on; total budget 34 epochs.
            steps_per_epoch = len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=3e-5,
                epochs=34,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.20,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100
            )

            # BALANCED Loss Weights - Equal importance for all heads
            # Previous issue: staging=5.0 >> screening=1.0 caused multi-head interference
            # New approach: Equal base weights, let class weights handle imbalance
            loss_weights = {
                'screening': 2.0,  # Equal importance
                'staging': 2.0,    # REDUCED from 4-5 to prevent AD→MCI confusion
                'subtype': 2.0     # Equal importance
            }
            print(f"    Loss Weights: screen={loss_weights['screening']:.2f}, "
                  f"stage={loss_weights['staging']:.2f}, subtype={loss_weights['subtype']:.2f}")

            # All class weights enabled for proper balance across all heads
            criterion = HierarchicalLoss(
                weights=loss_weights,
                screening_weights=screen_weights,  # NEW: Prevent CN under-detection
                subtype_weights=sub_weights,       # AD/FTD balance
                staging_weights=stage_weights      # MCI/Dementia balance (MCI boosted)
            ).to(DEVICE)

            # Early Stopping - INCREASED patience for curriculum learning
            # Curriculum needs more epochs (6 screen + 6 stage + subtype)
            early_stopper = EarlyStopping(patience=8, mode='max')

            # Training Loop - 34 epochs with extended Phase-1 warmup
            # Phase 1 (1-12): screening only, Phase 2 (13-20): +staging, Phase 3 (21-34): +subtype
            # Longer Phase 1 gives stream gates room to settle before staging starts.
            history = train_model(
                model, train_loader, val_loader, optimizer, criterion,
                scheduler=scheduler, early_stopper=early_stopper, epochs=34
            )

            # ========== CHECK FOR COLLAPSE ==========
            if history.get('collapsed', False):
                fold_retry_count += 1
                print(f"  ⚠️ Fold {fold+1} collapsed at epoch {history.get('collapse_epoch', '?')}")
                if fold_retry_count < MAX_FOLD_RETRIES:
                    print(f"     Will retry with different seed...")
                    torch.cuda.empty_cache()  # Clear GPU memory before retry
                    continue
                else:
                    print(f"  ❌ Fold {fold+1} collapsed after {MAX_FOLD_RETRIES} retries. Marking as collapsed.")
                    final_history = history
                    final_history['retry_count'] = fold_retry_count
                    break
            else:
                # Success - no collapse
                fold_collapsed = False
                final_history = history
                final_history['retry_count'] = fold_retry_count
                print(f"  ✓ Fold {fold+1} completed successfully" + (f" (after {fold_retry_count} retries)" if fold_retry_count > 0 else ""))

        # ========== END RETRY LOOP ==========

        # Store collapse status in history
        if final_history is None:
            final_history = {'collapsed': True, 'retry_count': fold_retry_count}

        # Evaluation (even for collapsed folds, to track metrics)
        metrics = evaluate_hierarchical(model, val_loader)
        metrics['fold'] = fold
        metrics['collapsed'] = final_history.get('collapsed', False)
        metrics['retry_count'] = fold_retry_count
        results.append(metrics)

        # --- Per-fold interpretability logging ---
        # Alpha: wPLI-prior blending weight (fixed buffer, already sigmoid-squashed)
        try:
            alpha_val = float(model.adaptive_graph.alpha.item())
        except AttributeError:
            alpha_val = float('nan')
        fold_alpha_values.append(alpha_val)
        metrics['alpha_value'] = alpha_val

        # Cross-band coupling matrix (5×5, averaged over the hold-out batch)
        try:
            model.eval()
            coupling_sum = None
            coupling_n = 0
            with torch.no_grad():
                for _cb_batch in val_loader:
                    _x = _cb_batch['x'].to(DEVICE)
                    _meta = _cb_batch['metadata'].to(DEVICE)
                    _bf = {k: v.to(DEVICE) for k, v in _cb_batch['band_features'].items()}
                    _clin = {'mmse': _meta[:, 2], 'age': _meta[:, 0], 'sex': _meta[:, 1]}
                    _streams = _streams_to_device(_cb_batch, DEVICE)
                    _out = model(_x, band_features=_bf, clinical_data=_clin,
                                 feature_streams=_streams)
                    bc = _out.get('band_coupling')
                    if bc is not None:
                        bc_mean = bc.mean(dim=0).cpu().numpy()  # [5,5]
                        coupling_sum = bc_mean if coupling_sum is None else coupling_sum + bc_mean
                        coupling_n += 1
                    break  # one batch is sufficient for coupling diagnostics
            if coupling_sum is not None and coupling_n > 0:
                coupling_avg = (coupling_sum / coupling_n).tolist()
            else:
                coupling_avg = None
        except Exception:
            coupling_avg = None
        fold_band_coupling.append(coupling_avg)
        metrics['band_coupling_matrix'] = coupling_avg
        print(f"  Fold {fold+1} alpha={alpha_val:.3f}"
              + (" [collapsed]" if metrics['collapsed'] else ""))

        # Save training history for this fold
        training_histories.append(final_history)

        # Checkpointing (suffix preserves any pre-existing baseline checkpoints)
        ckpt_path = PROJECT_ROOT / "outputs" / f"checkpoints{RUN_SUFFIX}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / f"hierarchical_model_fold{fold}.pt")
        
        
        # --- Collect Validation Predictions for JSON (Development Results) ---
        model.eval()
        val_subjs_list = groups[val_idx]
        curr_v_idx = 0
        
        with torch.no_grad():
            for batch in val_loader:
                vx = batch['x'].to(DEVICE)
                vy = batch['label'].to(DEVICE)
                vmeta = batch['metadata'].to(DEVICE)
                vband = {k: v.to(DEVICE) for k, v in batch['band_features'].items()}
                vclin = {'mmse': vmeta[:, 2], 'age': vmeta[:, 0], 'sex': vmeta[:, 1]}
                vstreams = _streams_to_device(batch, DEVICE)

                vout = model(vx, band_features=vband, clinical_data=vclin,
                             feature_streams=vstreams)
                
                # Flat Probs with improved calculation
                vp_h = vout['probs_screen'][:, 0].cpu().numpy()
                vp_i = vout['probs_screen'][:, 1].cpu().numpy()
                vp_m = vout['probs_stage'][:, 0].cpu().numpy()
                vp_d = vout['probs_stage'][:, 1].cpu().numpy()
                vp_a = vout['probs_subtype'][:, 0].cpu().numpy()
                vp_f = vout['probs_subtype'][:, 1].cpu().numpy()

                # GEOMETRIC MEAN CHAIN RULE - compensates for depth-based probability dilution
                eps = 1e-8

                # Depth 1: CN
                VP_CN = vp_h

                # Depth 2: MCI -> take sqrt
                VP_MCI = np.power(vp_i * vp_m + eps, 1/2)

                # Depth 3: AD -> take cube root
                VP_AD = np.power(vp_i * vp_d * vp_a + eps, 1/3)

                # Depth 3: FTD -> take cube root
                VP_FTD = np.power(vp_i * vp_d * vp_f + eps, 1/3)

                # Normalize to sum to 1
                v_probs_raw = np.stack([VP_AD, VP_FTD, VP_CN, VP_MCI], axis=1)
                v_probs = v_probs_raw / (v_probs_raw.sum(axis=1, keepdims=True) + eps)
                v_preds = np.argmax(v_probs, axis=1)
                
                batch_n = vx.size(0)
                v_subjs = val_subjs_list[curr_v_idx : curr_v_idx + batch_n]
                curr_v_idx += batch_n
                
                for i in range(batch_n):
                    dev_subject_results.append({
                        'subject': str(v_subjs[i]),
                        'true': int(vy[i].item()),
                        'pred': int(v_preds[i]),
                        'probs': v_probs[i].tolist(),
                        'correct': bool(v_preds[i] == vy[i].item()),
                        'fold': fold
                    })

        print(f"Fold {fold+1} Screen F1: {metrics['screen_f1']:.4f}, Stage F1: {metrics['stage_f1']:.4f}, Subtype F1: {metrics['subtype_f1']:.4f}")

        # Detect fold collapse - warn if any head shows random-level performance
        screen_kappa = metrics.get('cohens_kappa_screen', 0)
        stage_bacc = metrics.get('bacc_stage', 0)
        subtype_bacc = metrics.get('Balanced_Acc_Subtype', 0)

        fold_collapsed = False
        if screen_kappa < 0.3:
            print(f"  ⚠️ WARNING: Fold {fold+1} SCREENING HEAD may have collapsed (kappa={screen_kappa:.3f})")
            fold_collapsed = True
        if stage_bacc < 0.55:
            print(f"  ⚠️ WARNING: Fold {fold+1} STAGING HEAD may have collapsed (bacc={stage_bacc:.3f})")
            fold_collapsed = True
        if subtype_bacc < 0.55:
            print(f"  ⚠️ WARNING: Fold {fold+1} SUBTYPE HEAD may have collapsed (bacc={subtype_bacc:.3f})")
            fold_collapsed = True

        # Calculate combined score (use 0.01 floor to prevent zero)
        combined_score = (max(screen_kappa, 0.01) * max(stage_bacc, 0.01) * max(subtype_bacc, 0.01)) ** (1/3)
        print(f"  Fold {fold+1} Combined Score: {combined_score:.4f}" + (" [COLLAPSED]" if fold_collapsed else ""))

        # Only update best model if fold didn't collapse
        if combined_score > best_val_score and not fold_collapsed:
            best_val_score = combined_score
            best_model_state = model.state_dict()
            # Persist alongside the in-memory copy so downstream analyses
            # (e.g. experiments/29_feature_stream_ablation.py) can reload it.
            best_ckpt_dir = PROJECT_ROOT / "outputs" / f"models{RUN_SUFFIX}"
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_state, best_ckpt_dir / "ncg_v2_best.pt")
            print(f"  ✓ New best model saved (score: {combined_score:.4f}) → outputs/models{RUN_SUFFIX}/ncg_v2_best.pt")
        elif fold_collapsed:
            print(f"  ✗ Fold {fold} excluded from best model selection due to collapse")

    # Aggregated CV Results with proper statistical reporting
    df_res = pd.DataFrame(results)
    _results_dir = PROJECT_ROOT.joinpath("outputs", f"results{RUN_SUFFIX}")
    _results_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(_results_dir / "cv_results.csv")

    # Identify collapsed folds based on training history and final thresholds
    collapsed_folds = []
    for i, row in df_res.iterrows():
        fold_idx = int(row['fold'])
        # Check if fold was marked as collapsed during training
        training_collapsed = row.get('collapsed', False)
        # Also check final metric thresholds as backup detection
        metric_collapsed = row.get('cohens_kappa_screen', 1) < 0.3 or row.get('bacc_stage', 1) < 0.55

        if training_collapsed or metric_collapsed:
            collapsed_folds.append(fold_idx)
            reason = "training" if training_collapsed else "metrics"
            print(f"  ⚠️ Fold {fold_idx+1} marked as collapsed (detected via {reason})")

    # Calculate statistics for all folds and good folds
    print("\n===== Cross-Validation Results (Train/Val) =====")
    print(f"Total Folds: 5 | Collapsed Folds: {collapsed_folds if collapsed_folds else 'None'}")

    key_metrics = ['acc_screen', 'cohens_kappa_screen', 'bacc_stage', 'Balanced_Acc_Subtype']
    print("\n--- All Folds (Mean ± Std) ---")
    for metric in key_metrics:
        if metric in df_res.columns:
            mean = df_res[metric].mean()
            std = df_res[metric].std()
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")

    # Report excluding collapsed folds (more representative of best model)
    if collapsed_folds:
        good_folds = df_res[~df_res['fold'].isin(collapsed_folds)]
        print(f"\n--- Excluding Collapsed Folds {collapsed_folds} (Mean ± Std) ---")
        for metric in key_metrics:
            if metric in good_folds.columns:
                mean = good_folds[metric].mean()
                std = good_folds[metric].std()
                print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    # --- Prepare Development JSON Data ---
    # Aggregate dev predictions
    if len(dev_subject_results) > 0:
        dev_true = [d['true'] for d in dev_subject_results]
        dev_pred = [d['pred'] for d in dev_subject_results]
        classes = ['AD', 'FTD', 'CN', 'MCI']

        dev_report = classification_report(dev_true, dev_pred, target_names=classes, output_dict=True)
        cm_dev = confusion_matrix(dev_true, dev_pred)

        # Calculate PER-FOLD flat 4-class accuracy (the correct CV metric)
        # Separate collapsed and non-collapsed folds for proper reporting
        fold_accuracies_all = []
        fold_accuracies_valid = []  # Only non-collapsed folds

        for fold_idx in range(5):
            fold_results = [d for d in dev_subject_results if d['fold'] == fold_idx]
            if len(fold_results) > 0:
                fold_true = [d['true'] for d in fold_results]
                fold_pred = [d['pred'] for d in fold_results]
                fold_acc = np.mean(np.array(fold_true) == np.array(fold_pred))
                fold_accuracies_all.append((fold_idx, fold_acc))

                # Only include non-collapsed folds in the primary metric
                if fold_idx not in collapsed_folds:
                    fold_accuracies_valid.append(fold_acc)

        # Report both metrics - valid folds is the primary metric
        if fold_accuracies_valid:
            mean_cv_accuracy = float(np.mean(fold_accuracies_valid))
            std_cv_accuracy = float(np.std(fold_accuracies_valid))
            print(f"\n  📊 CV Accuracy (excluding {len(collapsed_folds)} collapsed folds): {mean_cv_accuracy:.3f} ± {std_cv_accuracy:.3f}")
        else:
            # Fallback if all folds collapsed
            mean_cv_accuracy = float(np.mean([acc for _, acc in fold_accuracies_all])) if fold_accuracies_all else 0.0
            std_cv_accuracy = float(np.std([acc for _, acc in fold_accuracies_all])) if fold_accuracies_all else 0.0
            print(f"\n  ⚠️ All folds collapsed! Using all-folds accuracy: {mean_cv_accuracy:.3f}")

        # Also report all-folds accuracy for transparency
        all_folds_acc = float(np.mean([acc for _, acc in fold_accuracies_all])) if fold_accuracies_all else 0.0
        print(f"  📊 CV Accuracy (all folds): {all_folds_acc:.3f}")

        # Calculate proper CV statistics
        cv_stats = {}
        for metric in ['acc_screen', 'cohens_kappa_screen', 'bacc_stage', 'Balanced_Acc_Subtype']:
            if metric in df_res.columns:
                cv_stats[metric] = {
                    'mean': float(df_res[metric].mean()),
                    'std': float(df_res[metric].std()),
                    'min': float(df_res[metric].min()),
                    'max': float(df_res[metric].max())
                }

        # Add flat 4-class accuracy to cv_stats (primary = non-collapsed folds)
        cv_stats['flat_accuracy'] = {
            'mean': mean_cv_accuracy,
            'std': std_cv_accuracy,
            'min': float(min(fold_accuracies_valid)) if fold_accuracies_valid else 0.0,
            'max': float(max(fold_accuracies_valid)) if fold_accuracies_valid else 0.0,
            'n_valid_folds': len(fold_accuracies_valid)
        }
        # Also report all-folds for transparency
        cv_stats['flat_accuracy_all_folds'] = {
            'mean': all_folds_acc,
            'std': float(np.std([acc for _, acc in fold_accuracies_all])) if len(fold_accuracies_all) > 1 else 0.0,
            'min': float(min(acc for _, acc in fold_accuracies_all)) if fold_accuracies_all else 0.0,
            'max': float(max(acc for _, acc in fold_accuracies_all)) if fold_accuracies_all else 0.0,
            'n_folds': len(fold_accuracies_all)
        }

        # Calculate Cohen's Kappa for dev set
        try:
            dev_kappa = cohen_kappa_score(dev_true, dev_pred)
        except:
            dev_kappa = 0.0

        dev_json_data = {
            'n_subjects': len(np.unique([d['subject'] for d in dev_subject_results])),
            'n_folds': 5,
            'n_valid_folds': 5 - len(collapsed_folds),
            'collapsed_folds': collapsed_folds,
            # USE MEAN CV ACCURACY from non-collapsed folds - proper metric
            'accuracy': mean_cv_accuracy,  # This is what should be reported!
            'accuracy_all_folds': all_folds_acc,  # For transparency
            'balanced_accuracy': dev_report['macro avg']['recall'],
            'f1_macro': dev_report['macro avg']['f1-score'],
            'f1_weighted': dev_report['weighted avg']['f1-score'],
            'cohens_kappa': dev_kappa,
            # Legacy keys for backwards compatibility
            'aggregated_accuracy': dev_report['accuracy'],
            'aggregated_balanced_accuracy': dev_report['macro avg']['recall'],
            'aggregated_f1_macro': dev_report['macro avg']['f1-score'],
            # Proper CV statistics (mean ± std)
            'cv_statistics': cv_stats,
            'confusion_matrix': cm_dev.tolist(),
            'per_class': {
                cls: {
                    'sensitivity': dev_report[cls]['recall'],
                    'precision': dev_report[cls]['precision'],
                    'f1': dev_report[cls]['f1-score'],
                    'support': int(dev_report[cls]['support'])
                } for cls in classes
            },
            'subject_results': dev_subject_results
        }

        # ============ DETAILED CLINICAL METRICS SUMMARY (DEV SET) ============
        print("\n" + "=" * 70)
        print("DEVELOPMENT SET - DETAILED CLINICAL METRICS SUMMARY")
        print("=" * 70)

        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Total Subjects: {len(np.unique([d['subject'] for d in dev_subject_results]))}")
        print(f"   Total Folds: 5 | Valid Folds: {5 - len(collapsed_folds)} | Collapsed: {collapsed_folds if collapsed_folds else 'None'}")

        print(f"\n📈 OVERALL PERFORMANCE (4-Class Flat):")
        print(f"   Accuracy (Valid Folds):    {mean_cv_accuracy*100:.2f}% ± {std_cv_accuracy*100:.2f}%")
        print(f"   Accuracy (All Folds):      {all_folds_acc*100:.2f}%")
        print(f"   Balanced Accuracy:         {dev_report['macro avg']['recall']*100:.2f}%")
        print(f"   F1-Score (Macro):          {dev_report['macro avg']['f1-score']:.4f}")
        print(f"   F1-Score (Weighted):       {dev_report['weighted avg']['f1-score']:.4f}")
        print(f"   Cohen's Kappa:             {dev_kappa:.4f}")

        print(f"\n🔬 HIERARCHICAL STAGE METRICS (Mean ± Std across folds):")
        for metric_name, display_name in [
            ('acc_screen', 'Screening Accuracy (CN vs Impaired)'),
            ('cohens_kappa_screen', "Screening Cohen's Kappa"),
            ('bacc_stage', 'Staging Balanced Accuracy (MCI vs Dem)'),
            ('Balanced_Acc_Subtype', 'Subtyping Balanced Accuracy (AD vs FTD)')
        ]:
            if metric_name in cv_stats:
                stats = cv_stats[metric_name]
                print(f"   {display_name}: {stats['mean']*100:.2f}% ± {stats['std']*100:.2f}%")

        print(f"\n🏥 CLASS-SPECIFIC CLINICAL METRICS:")
        for cls in classes:
            sens = dev_report[cls]['recall'] * 100
            prec = dev_report[cls]['precision'] * 100
            f1 = dev_report[cls]['f1-score']
            support = int(dev_report[cls]['support'])
            print(f"   [{cls}] Sensitivity: {sens:.1f}% | PPV: {prec:.1f}% | F1: {f1:.3f} | N={support}")

        # Compute additional clinical metrics
        # Screen: CN detection (Specificity) and Impaired detection (Sensitivity)
        cn_idx = 2  # CN is class 2
        impaired_mask = np.array(dev_true) != cn_idx
        pred_impaired_mask = np.array(dev_pred) != cn_idx
        screen_sens = np.sum((impaired_mask) & (pred_impaired_mask)) / max(np.sum(impaired_mask), 1)
        screen_spec = np.sum((~impaired_mask) & (~pred_impaired_mask)) / max(np.sum(~impaired_mask), 1)

        print(f"\n📋 SCREENING PERFORMANCE (CN vs Impaired):")
        print(f"   Sensitivity (Impaired Detection): {screen_sens*100:.1f}%")
        print(f"   Specificity (CN Detection):       {screen_spec*100:.1f}%")

        # Confusion matrix summary
        print(f"\n📊 CONFUSION MATRIX (Row=True, Col=Pred):")
        print(f"         AD    FTD    CN    MCI")
        for i, cls in enumerate(classes):
            row = "   " + cls.ljust(4) + " "
            for j in range(4):
                row += f"{cm_dev[i, j]:5d} "
            print(row)

        print("=" * 70)

    else:
        # Fallback if no dev results collected (shouldn't happen with new loop)
        dev_json_data = {}

    # 4. Final Evaluation on Hold-out Test Set
    holdout_json_data = {}
    
    if best_model_state is not None:
        print("\n===== Evaluating Best Model on Hold-out Test Set =====")
        model.load_state_dict(best_model_state)
        
        # Get Full Predictions for Detailed Report
        model.eval()
        all_screen, all_stage, all_subtype, all_targets = [], [], [], []
        
        holdout_subj_results = []
        test_subjs_list = groups[test_idx]
        curr_idx = 0
        
        with torch.no_grad():
            for batch in test_loader:
                 x = batch['x'].to(DEVICE)
                 y = batch['label'].to(DEVICE)
                 meta = batch['metadata'].to(DEVICE)
                 band_features = {k: v.to(DEVICE) for k, v in batch['band_features'].items()}
                 clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}
                 streams = _streams_to_device(batch, DEVICE)

                 out = model(x, band_features=band_features, clinical_data=clinical,
                             feature_streams=streams)
                 all_screen.append(out['probs_screen'].cpu().numpy())
                 all_stage.append(out['probs_stage'].cpu().numpy())
                 all_subtype.append(out['probs_subtype'].cpu().numpy())
                 all_targets.append(y.cpu().numpy())
                 
                 # Flat Probs with improved calculation to prevent chain rule dilution
                 p_h = out['probs_screen'][:, 0].cpu().numpy()
                 p_i = out['probs_screen'][:, 1].cpu().numpy()
                 p_m_g_i = out['probs_stage'][:, 0].cpu().numpy()
                 p_d_g_i = out['probs_stage'][:, 1].cpu().numpy()
                 p_a_g_d = out['probs_subtype'][:, 0].cpu().numpy()
                 p_f_g_d = out['probs_subtype'][:, 1].cpu().numpy()

                 # GEOMETRIC MEAN CHAIN RULE - compensates for depth-based probability dilution
                 # Standard chain rule creates bias: P(AD) = p1*p2*p3 is much smaller than P(CN) = p1
                 # Fix: Raise each chain to power of 1/depth to equalize scales
                 eps = 1e-8

                 # Depth 1: CN only needs P(Healthy)
                 P_CN = p_h

                 # Depth 2: MCI needs P(Impaired) * P(MCI|Impaired) -> take sqrt
                 P_MCI = np.power(p_i * p_m_g_i + eps, 1/2)

                 # Depth 3: AD needs P(Impaired) * P(Dementia) * P(AD|Dem) -> take cube root
                 P_AD = np.power(p_i * p_d_g_i * p_a_g_d + eps, 1/3)

                 # Depth 3: FTD needs P(Impaired) * P(Dementia) * P(FTD|Dem) -> take cube root
                 P_FTD = np.power(p_i * p_d_g_i * p_f_g_d + eps, 1/3)

                 # Normalize to sum to 1
                 t_probs_raw = np.stack([P_AD, P_FTD, P_CN, P_MCI], axis=1)
                 t_probs = t_probs_raw / (t_probs_raw.sum(axis=1, keepdims=True) + eps)

                 t_preds = np.argmax(t_probs, axis=1)
                 
                 batch_n = x.size(0)
                 # Handle last batch if size mismatch index out of bounds?
                 # test_subjs_list length matches dataset length.
                 t_subjs = test_subjs_list[curr_idx : curr_idx + batch_n]
                 curr_idx += batch_n
                 
                 for i in range(batch_n):
                     holdout_subj_results.append({
                         'subject': str(t_subjs[i]),
                         'true': int(y[i].item()),
                         'pred': int(t_preds[i]),
                         'probs': t_probs[i].tolist(),
                         'correct': bool(t_preds[i] == y[i].item())
                     })
                
        # Prepare Softmax Probs
        p_screen_probs = np.concatenate(all_screen)
        p_stage_probs = np.concatenate(all_stage)
        p_subtype_probs = np.concatenate(all_subtype)
        targets = np.concatenate(all_targets)
        
        # --- 1. Generate Hierarchical Report (Screen, Stage, Subtype) ---
        test_metrics = generate_detailed_report(
            targets, p_screen_probs, p_stage_probs, p_subtype_probs,
            output_dir=PROJECT_ROOT / "outputs" / f"results{RUN_SUFFIX}",
            prefix="test_"
        )
        
        print("Hold-out Test Results (Hierarchical):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        # --- 2. Generate Flat 4-Class Report ---
        classes = ['AD', 'FTD', 'CN', 'MCI']
        
        # Re-calc flat preds globally with improved chain rule
        p_healthy = p_screen_probs[:, 0]
        p_i = p_screen_probs[:, 1]
        p_m = p_stage_probs[:, 0]
        p_d = p_stage_probs[:, 1]
        p_a = p_subtype_probs[:, 0]
        p_f = p_subtype_probs[:, 1]

        # GEOMETRIC MEAN CHAIN RULE - compensates for depth-based probability dilution
        eps = 1e-8

        # Depth 1: CN only needs P(Healthy)
        P_CN = p_healthy

        # Depth 2: MCI needs P(Impaired) * P(MCI|Impaired) -> take sqrt
        P_MCI = np.power(p_i * p_m + eps, 1/2)

        # Depth 3: AD needs P(Impaired) * P(Dementia) * P(AD|Dem) -> take cube root
        P_AD = np.power(p_i * p_d * p_a + eps, 1/3)

        # Depth 3: FTD needs P(Impaired) * P(Dementia) * P(FTD|Dem) -> take cube root
        P_FTD = np.power(p_i * p_d * p_f + eps, 1/3)

        # Normalize to sum to 1
        flat_probs_raw = np.stack([P_AD, P_FTD, P_CN, P_MCI], axis=1)
        flat_probs = flat_probs_raw / (flat_probs_raw.sum(axis=1, keepdims=True) + eps)
        flat_preds = np.argmax(flat_probs, axis=1)
        
        print("\n===== Flat 4-Class Performance (Hold-out) =====")
        print(classification_report(targets, flat_preds, target_names=classes, digits=4))
        
        cm_flat = confusion_matrix(targets, flat_preds)
        print("Confusion Matrix (AD, FTD, CN, MCI):")
        print(cm_flat)
        
        # Save Flat Metrics
        flat_report = classification_report(targets, flat_preds, target_names=classes, output_dict=True)
        pd.DataFrame(flat_report).transpose().to_csv(PROJECT_ROOT / "outputs" / "results" / "test_flat_classification_report.csv")
            
        # Save Test Results (Hierarchical Summary)
        pd.Series(test_metrics).to_csv(PROJECT_ROOT / "outputs" / "results" / "test_holdout_results_detailed.csv")
        
        # Calculate Cohen's Kappa for holdout
        try:
            holdout_kappa = cohen_kappa_score(targets, flat_preds)
        except:
            holdout_kappa = 0.0

        # Holdout JSON with comprehensive metrics
        holdout_json_data = {
            'n_subjects': len(np.unique([s['subject'] for s in holdout_subj_results])),
            'accuracy': flat_report['accuracy'],
            'balanced_accuracy': flat_report['macro avg']['recall'],
            'f1_macro': flat_report['macro avg']['f1-score'],
            'f1_weighted': flat_report['weighted avg']['f1-score'],
            'cohens_kappa': holdout_kappa,
            # Hierarchical stage metrics
            'screen_accuracy': test_metrics.get('acc_screen', 0),
            'screen_kappa': test_metrics.get('cohens_kappa_screen', 0),
            'stage_balanced_accuracy': test_metrics.get('bacc_stage', 0),
            'subtype_balanced_accuracy': test_metrics.get('Balanced_Acc_Subtype', 0),
            'confusion_matrix': cm_flat.tolist(),
            'per_class': {
                 cls: {
                     'sensitivity': flat_report[cls]['recall'],
                     'precision': flat_report[cls]['precision'],
                     'f1': flat_report[cls]['f1-score'],
                     'support': int(flat_report[cls]['support'])
                 } for cls in classes
            },
            'subject_results': holdout_subj_results
        }

        # ============ DETAILED CLINICAL METRICS SUMMARY (HOLDOUT SET) ============
        print("\n" + "=" * 70)
        print("HOLDOUT SET - DETAILED CLINICAL METRICS SUMMARY")
        print("=" * 70)

        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Total Subjects: {len(np.unique([s['subject'] for s in holdout_subj_results]))}")

        print(f"\n📈 OVERALL PERFORMANCE (4-Class Flat):")
        print(f"   Accuracy:                  {flat_report['accuracy']*100:.2f}%")
        print(f"   Balanced Accuracy:         {flat_report['macro avg']['recall']*100:.2f}%")
        print(f"   F1-Score (Macro):          {flat_report['macro avg']['f1-score']:.4f}")
        print(f"   F1-Score (Weighted):       {flat_report['weighted avg']['f1-score']:.4f}")
        print(f"   Cohen's Kappa:             {holdout_kappa:.4f}")

        print(f"\n🔬 HIERARCHICAL STAGE METRICS:")
        print(f"   Screening Accuracy (CN vs Impaired): {test_metrics.get('acc_screen', 0)*100:.2f}%")
        print(f"   Screening Cohen's Kappa:             {test_metrics.get('cohens_kappa_screen', 0):.4f}")
        print(f"   Staging Balanced Accuracy:           {test_metrics.get('bacc_stage', 0)*100:.2f}%")
        print(f"   Subtyping Balanced Accuracy:         {test_metrics.get('Balanced_Acc_Subtype', 0)*100:.2f}%")

        print(f"\n🏥 CLASS-SPECIFIC CLINICAL METRICS:")
        for cls in classes:
            sens = flat_report[cls]['recall'] * 100
            prec = flat_report[cls]['precision'] * 100
            f1 = flat_report[cls]['f1-score']
            support = int(flat_report[cls]['support'])
            print(f"   [{cls}] Sensitivity: {sens:.1f}% | PPV: {prec:.1f}% | F1: {f1:.3f} | N={support}")

        # Screen performance
        cn_idx = 2
        impaired_mask = targets != cn_idx
        pred_impaired_mask = flat_preds != cn_idx
        hold_screen_sens = np.sum((impaired_mask) & (pred_impaired_mask)) / max(np.sum(impaired_mask), 1)
        hold_screen_spec = np.sum((~impaired_mask) & (~pred_impaired_mask)) / max(np.sum(~impaired_mask), 1)

        print(f"\n📋 SCREENING PERFORMANCE (CN vs Impaired):")
        print(f"   Sensitivity (Impaired Detection): {hold_screen_sens*100:.1f}%")
        print(f"   Specificity (CN Detection):       {hold_screen_spec*100:.1f}%")

        # Staging performance (among impaired)
        impaired_idx = np.where(impaired_mask)[0]
        if len(impaired_idx) > 0:
            stage_true = np.isin(targets[impaired_idx], [0, 1]).astype(int)  # Dementia=1
            stage_pred = np.isin(flat_preds[impaired_idx], [0, 1]).astype(int)
            stage_acc = np.mean(stage_true == stage_pred)
            print(f"\n📋 STAGING PERFORMANCE (MCI vs Dementia, among impaired):")
            print(f"   Accuracy: {stage_acc*100:.1f}%")

        # Subtyping performance (among dementia)
        dementia_idx = np.where(np.isin(targets, [0, 1]))[0]
        if len(dementia_idx) > 0:
            sub_true = targets[dementia_idx]
            sub_pred = flat_preds[dementia_idx]
            sub_acc = np.mean(sub_true == sub_pred)
            print(f"\n📋 SUBTYPING PERFORMANCE (AD vs FTD, among dementia):")
            print(f"   Accuracy: {sub_acc*100:.1f}%")

        print(f"\n📊 CONFUSION MATRIX (Row=True, Col=Pred):")
        print(f"         AD    FTD    CN    MCI")
        for i, cls in enumerate(classes):
            row = "   " + cls.ljust(4) + " "
            for j in range(4):
                row += f"{cm_flat[i, j]:5d} "
            print(row)

        print("=" * 70)

    # --- Aggregate per-fold interpretability stats ---
    valid_alphas = [a for a in fold_alpha_values if not (isinstance(a, float) and a != a)]  # drop NaN
    alpha_stats = {
        'per_fold': fold_alpha_values,
        'mean': float(np.mean(valid_alphas)) if valid_alphas else None,
        'std': float(np.std(valid_alphas)) if valid_alphas else None,
        'min': float(np.min(valid_alphas)) if valid_alphas else None,
        'max': float(np.max(valid_alphas)) if valid_alphas else None,
        'note': (
            'alpha controls wPLI-prior vs learned-adjacency blend '
            '(sigmoid-squashed; 1.0=full prior, 0.0=full learned).'
        )
    }
    print(f"\n  wPLI alpha (per fold): {fold_alpha_values}")
    print(f"  alpha mean±std: {alpha_stats['mean']:.3f}±{alpha_stats['std']:.3f}"
          if alpha_stats['mean'] is not None else "  alpha: unavailable")

    # --- SAVE RESULTS.JSON ---
    final_results = {
        'development': dev_json_data,
        'holdout': holdout_json_data,
        'training_history': training_histories,
        'interpretability': {
            'alpha_per_fold': alpha_stats,
            'band_coupling_per_fold': fold_band_coupling,
        }
    }
    
    out_dir = PROJECT_ROOT / "outputs" / f"results{RUN_SUFFIX}" / "v3_holdout_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=4)
        print(f"\nSaved results to {out_dir / 'results.json'}")

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, early_stopper=None, epochs=20):
    """Train model with optional LR scheduler and early stopping.

    Returns:
        dict: Training history with losses, accuracies, best score, and collapse indicator
    """
    best_model_state = None
    best_val_score = 0.0

    # Track training dynamics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc_screen': [],
        'val_acc_stage': [],
        'val_acc_subtype': [],
        'val_bacc_subtype': [],
        'val_kappa_screen': [],  # Track kappa for collapse detection
        'val_bacc_stage': [],    # Track stage bacc for collapse detection
        'learning_rate': [],
        'collapsed': False,      # Collapse indicator
    }

    # Collapse detection thresholds - RELAXED for curriculum learning
    # Note: With curriculum, staging is not active until epoch 7, so we only check screening early on
    COLLAPSE_KAPPA_THRESHOLD = 0.1   # Screen kappa below this indicates collapse
    COLLAPSE_BACC_THRESHOLD = 0.55   # Stage balanced accuracy below this indicates collapse
    COLLAPSE_CONSECUTIVE_EPOCHS = 4  # INCREASED from 3 - more patience for curriculum
    EARLY_WARNING_EPOCH = 5          # INCREASED from 2 - give model time to learn
    EARLY_WARNING_KAPPA = 0.02       # REDUCED from 0.05 - only flag severe collapse

    # Curriculum learning phase boundaries (match HierarchicalLoss)
    # EXTENDED for better per-phase learning
    CURRICULUM_PHASE1_END = 12  # Screening only until epoch 12 (extended for stream-gate stabilization)
    CURRICULUM_PHASE2_END = 20  # Screening + staging until epoch 20

    # Track Phase 3 best model separately to ensure subtype is trained
    phase3_best_score = 0.0
    phase3_best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc_sum = 0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            x = batch['x'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            meta = batch['metadata'].to(DEVICE)

            # Pass band features
            band_features = {k: v.to(DEVICE) for k, v in batch['band_features'].items()}
            streams = _streams_to_device(batch, DEVICE)

            optimizer.zero_grad()
            clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}

            outputs = model(x, band_features=band_features, clinical_data=clinical,
                            feature_streams=streams)
            # Pass current epoch for curriculum learning (1-indexed)
            loss, loss_comp = criterion(outputs, y, epoch=epoch + 1)

            if torch.isnan(loss):
                print(f"nan loss at epoch {epoch+1}, batch {pbar.n}")
                print(f"Loss comps: {loss_comp}")
                optimizer.zero_grad()
                continue

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check for NaNs in gradients
            has_nan = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    break

            if has_nan:
                print(f"WARNING: NaN gradient detected at epoch {epoch+1}, batch {pbar.n}. Skipping step.")
                optimizer.zero_grad()
                continue

            optimizer.step()

            # Step LR scheduler per batch (required for OneCycleLR)
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            pred_screen = torch.argmax(outputs['probs_screen'], dim=1)
            target_screen = (y != 2).long()
            acc_batch = (pred_screen == target_screen).float().mean()
            train_acc_sum += acc_batch.item()

            # Show RUNNING AVERAGE instead of last batch (more stable display)
            avg_loss = train_loss / n_batches
            avg_acc = train_acc_sum / n_batches
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{avg_acc:.2f}"})

        # Get current LR for logging
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # End of Epoch Validation
        torch.cuda.empty_cache()
        val_metrics = evaluate_hierarchical(model, val_loader)
        
        # Record history
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_acc = train_acc_sum / max(n_batches, 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc_screen'].append(val_metrics['acc_screen'])
        history['val_acc_stage'].append(val_metrics['acc_stage'])
        history['val_acc_subtype'].append(val_metrics.get('acc_subtype', 0))
        history['val_bacc_subtype'].append(val_metrics['Balanced_Acc_Subtype'])
        history['val_kappa_screen'].append(val_metrics.get('cohens_kappa_screen', 0))
        history['val_bacc_stage'].append(val_metrics.get('bacc_stage', 0))
        history['learning_rate'].append(current_lr)

        # ========== EARLY WARNING DETECTION ==========
        # At epoch 2, check if screening is starting to collapse (very low kappa)
        # This allows early exit before wasting 3 epochs on a doomed run
        if epoch + 1 == EARLY_WARNING_EPOCH:
            current_kappa = val_metrics.get('cohens_kappa_screen', 0)
            if current_kappa < EARLY_WARNING_KAPPA:
                print(f"\n  ⚠️ EARLY WARNING: Screen kappa={current_kappa:.4f} at epoch {epoch+1}")
                print(f"     Model likely to collapse. Triggering early exit for retry.")
                history['collapsed'] = True
                history['collapse_epoch'] = epoch + 1
                history['collapse_type'] = 'EARLY_SCREEN'
                return history  # Early exit for retry with different seed

        # ========== CURRICULUM-AWARE COLLAPSE DETECTION ==========
        # During curriculum learning, only check active heads:
        # - Phase 1 (epochs 1-6): Only check screening
        # - Phase 2 (epochs 7-12): Check screening + staging
        # - Phase 3 (epochs 13+): Check all three
        current_epoch = epoch + 1

        if len(history['val_kappa_screen']) >= COLLAPSE_CONSECUTIVE_EPOCHS:
            recent_kappas = history['val_kappa_screen'][-COLLAPSE_CONSECUTIVE_EPOCHS:]

            # Always check screen collapse (active in all phases)
            screen_collapsed = all(k < COLLAPSE_KAPPA_THRESHOLD for k in recent_kappas)

            # Only check stage collapse after phase 1 (when staging is active)
            stage_collapsed = False
            if current_epoch > CURRICULUM_PHASE1_END:
                recent_baccs = history['val_bacc_stage'][-COLLAPSE_CONSECUTIVE_EPOCHS:]
                stage_collapsed = all(b < COLLAPSE_BACC_THRESHOLD for b in recent_baccs)

            if screen_collapsed or stage_collapsed:
                collapse_type = "SCREEN" if screen_collapsed else "STAGE"
                print(f"\n  ⚠️ COLLAPSE DETECTED ({collapse_type}):")
                print(f"     Recent screen kappas: {[f'{k:.3f}' for k in recent_kappas]}")
                if current_epoch > CURRICULUM_PHASE1_END:
                    recent_baccs = history['val_bacc_stage'][-COLLAPSE_CONSECUTIVE_EPOCHS:]
                    print(f"     Recent stage baccs: {[f'{b:.3f}' for b in recent_baccs]}")
                history['collapsed'] = True
                history['collapse_epoch'] = epoch + 1
                history['collapse_type'] = collapse_type
                return history  # Early exit on collapse

        print(f"Epoch {epoch+1} Summary (LR: {current_lr:.2e}):")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Screen Acc: {val_metrics['acc_screen']:.4f} (Kappa: {val_metrics.get('cohens_kappa_screen', 0):.4f})")
        print(f"  Val Stage Acc:  {val_metrics['acc_stage']:.4f} (Bal: {val_metrics['bacc_stage']:.4f})")
        print(f"  Val Subtype B-Acc:{val_metrics['Balanced_Acc_Subtype']:.4f}")
        print(f"    [AD] F1: {val_metrics.get('F1_AD', 0):.4f}, Sens: {val_metrics.get('Sens_AD', 0):.4f}")
        print(f"    [FTD] F1: {val_metrics.get('F1_FTD', 0):.4f}, Sens: {val_metrics.get('Sens_FTD', 0):.4f}")

        # Track best model using CURRICULUM-AWARE scoring
        # Only include active heads in the score calculation
        screen_score = max(val_metrics.get('cohens_kappa_screen', 0), 0.01)
        stage_score = max(val_metrics.get('bacc_stage', 0), 0.01)
        subtype_score = max(val_metrics.get('Balanced_Acc_Subtype', 0), 0.01)

        # Curriculum-aware combined score
        if current_epoch <= CURRICULUM_PHASE1_END:
            # Phase 1: Only screening matters
            val_score = screen_score
            score_desc = "Screen only"
        elif current_epoch <= CURRICULUM_PHASE2_END:
            # Phase 2: Screening + Staging
            val_score = (screen_score * stage_score) ** 0.5
            score_desc = "Screen×Stage"
        else:
            # Phase 3: All three
            val_score = (screen_score * stage_score * subtype_score) ** (1/3)
            score_desc = "Screen×Stage×Subtype"

        print(f"  Combined Score: {val_score:.4f} ({score_desc})")

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # PHASE 3 MODEL TRACKING: Ensure subtype is properly trained
        # The issue: Early stopping in Phase 2 selects a model without trained subtype
        # Fix: Track Phase 3 best model separately and prefer it for final selection
        if current_epoch > CURRICULUM_PHASE2_END:
            phase3_score = (screen_score * stage_score * subtype_score) ** (1/3)
            if phase3_score > phase3_best_score:
                phase3_best_score = phase3_score
                phase3_best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New Phase 3 best model (score: {phase3_score:.4f})")

        # Early stopping check - but ONLY allow after Phase 3 has started
        # Reason: Early stopping in Phase 1/2 would select a model without trained subtype head
        MIN_PHASE3_EPOCHS = 4  # Require at least 4 epochs in Phase 3 before early stop allowed
        can_early_stop = current_epoch >= (CURRICULUM_PHASE2_END + MIN_PHASE3_EPOCHS)

        if early_stopper is not None and can_early_stop:
            if early_stopper(val_score, epoch):
                print(f"Restoring best model from epoch {early_stopper.best_epoch + 1}")
                # Prefer Phase 3 model if available
                if phase3_best_model_state is not None:
                    print(f"Using Phase 3 best model (score: {phase3_best_score:.4f})")
                    model.load_state_dict({k: v.to(DEVICE) for k, v in phase3_best_model_state.items()})
                elif best_model_state is not None:
                    model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
                break
        elif early_stopper is not None and not can_early_stop and epoch >= CURRICULUM_PHASE2_END:
            # Still track for early stopping, but don't trigger yet
            early_stopper(val_score, epoch)

    # CRITICAL: Prefer Phase 3 best model to ensure subtype head is trained
    # Phase 3 model has all three heads optimized, while Phase 1/2 models lack subtype training
    if phase3_best_model_state is not None:
        print(f"Restoring Phase 3 best model (score: {phase3_best_score:.4f}) - ensures trained subtype head")
        model.load_state_dict({k: v.to(DEVICE) for k, v in phase3_best_model_state.items()})
        history['best_score'] = phase3_best_score
    elif best_model_state is not None:
        print(f"Restoring best model (score: {best_val_score:.4f}) - WARNING: May lack subtype training")
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
        history['best_score'] = best_val_score
    else:
        history['best_score'] = 0.0

    return history

def evaluate_hierarchical(model, loader, export_plots=False, prefix="Val"):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score, cohen_kappa_score
    
    model.eval()
    all_probs_screen = []
    all_probs_subtype = []
    all_preds_screen = []
    all_preds_stage = []
    all_preds_subtype = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(DEVICE)
            y = batch['label'].to(DEVICE)
            meta = batch['metadata'].to(DEVICE)
            band_features = {k: v.to(DEVICE) for k, v in batch['band_features'].items()}
            clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}
            streams = _streams_to_device(batch, DEVICE)

            outputs = model(x, band_features=band_features, clinical_data=clinical,
                            feature_streams=streams)

            all_probs_screen.append(outputs['probs_screen'][:, 1].cpu()) # Prob of Impairment
            all_probs_subtype.append(outputs['probs_subtype'].cpu())
            
            all_preds_screen.append(torch.argmax(outputs['probs_screen'], dim=1).cpu())
            all_preds_stage.append(torch.argmax(outputs['probs_stage'], dim=1).cpu())
            all_preds_subtype.append(torch.argmax(outputs['probs_subtype'], dim=1).cpu())
            all_targets.append(y.cpu())
            
    # Concatenate
    probs_screen = torch.cat(all_probs_screen).numpy()
    probs_subtype = torch.cat(all_probs_subtype).numpy()
    
    p_screen = torch.cat(all_preds_screen).numpy()
    p_stage = torch.cat(all_preds_stage).numpy()
    p_subtype = torch.cat(all_preds_subtype).numpy()
    targets = torch.cat(all_targets).numpy()
    
    # Debug: Check for Collapse
    unique, counts = np.unique(p_subtype, return_counts=True)
    print(f"    Subtype Pred Distribution: {dict(zip(unique, counts))}")
    
    metrics = {}
    
    # --- 1. Screening (Binary: CN=0 vs Impaired=1) ---
    t_screen = (targets != 2).astype(int)
    
    metrics['acc_screen'] = accuracy_score(t_screen, p_screen)
    metrics['screen_sens'] = recall_score(t_screen, p_screen) # Recall (Impaired detection)
    metrics['screen_spec'] = recall_score(t_screen, p_screen, pos_label=0) # Specificity (CN detection)
    metrics['screen_f1'] = f1_score(t_screen, p_screen)
    try:
        metrics['screen_auc'] = roc_auc_score(t_screen, probs_screen)
    except:
        metrics['screen_auc'] = 0.0

    # Clinical Screening Metrics: PPV/NPV
    # Critical for population screening - tells clinicians how to interpret results
    metrics['screen_ppv'] = precision_score(t_screen, p_screen, pos_label=1, zero_division=0)  # P(Impaired|Predicted Impaired)
    metrics['screen_npv'] = precision_score(t_screen, p_screen, pos_label=0, zero_division=0)  # P(Healthy|Predicted Healthy)
        
    if export_plots:
        cm = confusion_matrix(t_screen, p_screen)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'Impaired'], yticklabels=['CN', 'Impaired'])
        plt.title(f"{prefix} Screening Confusion Matrix")
        plt.savefig(PROJECT_ROOT / "outputs" / "figures" / f"{prefix}_cm_screen.png")
        plt.close()

    # --- 2. Subtype (AD vs FTD) ---
    mask_dementia = np.isin(targets, [0, 1])
    if mask_dementia.sum() > 0:
        t_sub = targets[mask_dementia] # 0=AD, 1=FTD
        p_sub = p_subtype[mask_dementia]
        prob_sub = probs_subtype[mask_dementia] # [N, 2]
        
        metrics['acc_subtype'] = accuracy_score(t_sub, p_sub)
        metrics['subtype_sens'] = recall_score(t_sub, p_sub, pos_label=0) # AD sensitivity
        metrics['subtype_spec'] = recall_score(t_sub, p_sub, pos_label=1) # FTD sensitivity
        metrics['subtype_f1'] = f1_score(t_sub, p_sub, average='weighted')
        try:
            # Multi-class AUC or just binary since 2 classes?
            # t_sub is 0/1. probs is [N, 2]. Use column 1.
            metrics['subtype_auc'] = roc_auc_score(t_sub, prob_sub[:, 1])
        except:
            metrics['subtype_auc'] = 0.0
            
        if export_plots:
            cm = confusion_matrix(t_sub, p_sub)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['AD', 'FTD'], yticklabels=['AD', 'FTD'])
            plt.title(f"{prefix} Subtype Confusion Matrix")
            plt.savefig(PROJECT_ROOT / "outputs" / "figures" / f"{prefix}_cm_subtype.png")
            plt.close()
    else:
        metrics['acc_subtype'] = 0.0
        metrics['subtype_f1'] = 0.0
        
    # --- 3. Staging (MCI vs Dementia) ---
    mask_impaired = (targets != 2)
    if mask_impaired.sum() > 0:
        t_stage = (targets[mask_impaired] != 3).astype(int) # 0=MCI, 1=Dem
        p_stg = p_stage[mask_impaired]
        
        
        metrics['acc_stage'] = accuracy_score(t_stage, p_stg)
        metrics['bacc_stage'] = balanced_accuracy_score(t_stage, p_stg) # Added Balanced Accuracy
        metrics['stage_f1'] = f1_score(t_stage, p_stg)

        # Clinical Staging Metrics: PPV/NPV for MCI vs Dementia
        # Critical for determining patient care pathway
        metrics['stage_ppv'] = precision_score(t_stage, p_stg, pos_label=1, zero_division=0)  # P(Dementia|Predicted Dementia)
        metrics['stage_npv'] = precision_score(t_stage, p_stg, pos_label=0, zero_division=0)  # P(MCI|Predicted MCI)
        metrics['stage_sens_dem'] = recall_score(t_stage, p_stg, pos_label=1, zero_division=0)  # Dementia sensitivity
        metrics['stage_sens_mci'] = recall_score(t_stage, p_stg, pos_label=0, zero_division=0)  # MCI sensitivity
    else:
        metrics['acc_stage'] = 0.0
        metrics['bacc_stage'] = 0.0
        metrics['stage_f1'] = 0.0
        metrics['stage_ppv'] = 0.0
        metrics['stage_npv'] = 0.0
        metrics['stage_sens_dem'] = 0.0
        metrics['stage_sens_mci'] = 0.0

    # Detailed Subtype Metrics (Per Class)
    if mask_dementia.sum() > 0:
        # 0=AD, 1=FTD
        # Explicitly calculate for each class
        f1_per_class = f1_score(t_sub, p_sub, average=None, labels=[0, 1])
        if len(f1_per_class) == 2:
            metrics['F1_AD'] = f1_per_class[0]
            metrics['F1_FTD'] = f1_per_class[1]
        else:
            metrics['F1_AD'] = 0.0
            metrics['F1_FTD'] = 0.0
            
        # Clinical Metrics
        tn, fp, fn, tp = confusion_matrix(t_sub, p_sub, labels=[0, 1]).ravel()
        # AD is 0 (Negative?), FTD is 1 (Positive?) -> Let's be explicit
        # Let's define Sensitivity relative to FTD(1) detection? 
        # Usually AD is the 'baseline' pathology here relative to FTD?
        # Let's list SPECIFIC Sens/Spec:
        
        # Sens/Spec for AD (Class 0)
        # Treated Class 0 as Positive: TP=tn, FN=fp
        metrics['Sens_AD'] = tn / (tn + fp + 1e-6) # Recall of Class 0

        # Sens/Spec for FTD (Class 1)
        # Treated Class 1 as Positive: TP=tp, FN=fn
        metrics['Sens_FTD'] = tp / (tp + fn + 1e-6) # Recall of Class 1

        metrics['Balanced_Acc_Subtype'] = (metrics['Sens_AD'] + metrics['Sens_FTD']) / 2

        # Clinical Validity Metrics
        # PPV/NPV: Critical for clinical decision-making
        # PPV = TP / (TP + FP) - "If model predicts FTD, how likely is it correct?"
        # NPV = TN / (TN + FN) - "If model predicts AD, how likely is it correct?"
        metrics['PPV_FTD'] = tp / (tp + fp + 1e-6)  # Precision for FTD
        metrics['NPV_FTD'] = tn / (tn + fn + 1e-6)  # Precision for AD (negative = not FTD)
        metrics['PPV_AD'] = tn / (tn + fn + 1e-6)   # Precision for AD
        metrics['NPV_AD'] = tp / (tp + fp + 1e-6)   # Precision for FTD (negative = not AD)

        # Cohen's Kappa for inter-rater agreement interpretation
        try:
            metrics['cohens_kappa_subtype'] = cohen_kappa_score(t_sub, p_sub)
        except:
            metrics['cohens_kappa_subtype'] = 0.0

    else:
        metrics['F1_AD'] = 0.0
        metrics['F1_FTD'] = 0.0
        metrics['Sens_AD'] = 0.0
        metrics['Sens_FTD'] = 0.0
        metrics['Balanced_Acc_Subtype'] = 0.0
        metrics['PPV_FTD'] = 0.0
        metrics['NPV_FTD'] = 0.0
        metrics['PPV_AD'] = 0.0
        metrics['NPV_AD'] = 0.0
        metrics['cohens_kappa_subtype'] = 0.0

    # Overall Clinical Metrics: Cohen's Kappa for screening
    try:
        metrics['cohens_kappa_screen'] = cohen_kappa_score(t_screen, p_screen)
    except:
        metrics['cohens_kappa_screen'] = 0.0

    return metrics

if __name__ == '__main__':
    main()
