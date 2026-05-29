"""
Cross-Dataset Validation (Leave-One-Dataset-Out).

Tests true cross-site generalization by training on 4 datasets
and evaluating on the held-out dataset. 5 rounds total.

Improvements over naive LODO:
1. Pre-trained model initialization  - fine-tune from best checkpoint
2. DANN regularization               - adversarial dataset-identity suppression
3. HierarchicalLoss + curriculum     - proper loss matching main training
4. Correct hyperparameters           - lr, batch, accumulation matching main run
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, classification_report
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
from src.utils.reproducibility import DEFAULT_SEED, set_global_seed
SEED = int(os.environ.get("NCG_SEED", DEFAULT_SEED))
set_global_seed(SEED)

from src.config.config import DEVICE, DATASET_CONFIG, DATALOADER_CONFIG
from src.data.dataset_factory import DatasetFactory
from src.models.v2.neuro_chrono_graph_v2 import create_neuro_chrono_graph_v2
from src.models.losses import HierarchicalLoss
from src.training.domain_adaptation import DomainDiscriminator

DATASET_PATHS = {
    'ds004504': PROJECT_ROOT / "datasets" / "openneuro_ds004504",
    'ds006036': PROJECT_ROOT / "datasets" / "ds006036",
    'Alz_EEG': PROJECT_ROOT / "datasets" / "Alz_EEG_data",
    'Mendeley': PROJECT_ROOT / "datasets" / "Mendeley Dataset",
    'MCI_Dataset': PROJECT_ROOT / "datasets" / "mci dataset",
}

CLASS_NAMES = ['AD', 'FTD', 'CN', 'MCI']
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Training hyperparameters matching main training (18_train_hierarchical.py) ──
LODO_CONFIG = {
    'epochs': 30,
    'max_lr': 5e-5,
    'weight_decay': 0.05,
    'batch_size': 8,
    'grad_accum_steps': 4,       # Effective batch = 32
    'grad_clip': 1.0,
    'dropout': 0.5,              # Slightly relaxed from 0.6 for fine-tuning
    # DANN regularization
    'lambda_dann_max': 0.3,      # Peak GRL reversal strength
    'dann_warmup_epochs': 5,     # Ramp up domain adversarial pressure gradually
}


def get_dataset_source(subject_id):
    """Determine source dataset from subject ID naming convention."""
    sid = str(subject_id)
    if 'ds004504' in sid:
        return 'ds004504'
    elif 'ds006036' in sid:
        return 'ds006036'
    elif 'AlzEEG' in sid:
        return 'Alz_EEG'
    elif 'Mendeley' in sid:
        return 'Mendeley'
    elif 'MCIData' in sid:
        return 'MCI_Dataset'
    return 'unknown'


def load_best_checkpoint(model, checkpoints_dir: Path) -> bool:
    """
    Load the best available checkpoint to warm-start LODO fine-tuning.
    Returns True if a checkpoint was loaded successfully.
    """
    checkpoint_files = sorted(checkpoints_dir.glob("hierarchical_model_fold*.pt"))
    if not checkpoint_files:
        print("    [init] No checkpoints found — training from random init.")
        return False

    # Try each fold checkpoint; load the first that parses cleanly
    for ckpt_path in checkpoint_files:
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            # Checkpoints may be saved as full state dicts or as {'model_state_dict': ...}
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state, strict=False)
            print(f"    [init] Loaded pre-trained weights from {ckpt_path.name}")
            return True
        except Exception as e:
            print(f"    [init] Could not load {ckpt_path.name}: {e}")
            continue

    print("    [init] All checkpoint loads failed — training from random init.")
    return False


def train_one_round(model, train_loader, dataset_source_map, n_datasets,
                    epochs=30, device='cpu'):
    """
    Fine-tune model for one LODO round.

    Improvements:
    - HierarchicalLoss with full curriculum (phases 1→2→3)
    - DANN regularization: GRL + DomainDiscriminator suppresses dataset identity
    - OneCycleLR, gradient accumulation matching main training
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LODO_CONFIG['max_lr'] / 10,   # Start lower for fine-tuning
        weight_decay=LODO_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LODO_CONFIG['max_lr'],
        epochs=epochs,
        steps_per_epoch=max(len(train_loader) // LODO_CONFIG['grad_accum_steps'], 1),
        pct_start=0.2
    )

    # Hierarchical loss with curriculum (same as main training)
    criterion = HierarchicalLoss(
        weights={'screening': 1.0, 'staging': 1.5, 'subtype': 3.0},
        use_curriculum=True
    )

    # Domain discriminator: classifies which training dataset a sample came from
    # Embedding dim = 512 (NCG fused representation)
    domain_disc = DomainDiscriminator(
        input_dim=512,
        hidden_dim=256,
        n_subjects=n_datasets
    ).to(device)
    domain_optimizer = torch.optim.Adam(domain_disc.parameters(), lr=1e-4)
    domain_criterion = nn.CrossEntropyLoss()

    accum_steps = LODO_CONFIG['grad_accum_steps']

    model.train()
    for epoch in range(1, epochs + 1):
        criterion.set_epoch(epoch)
        total_loss = 0.0
        total_domain_loss = 0.0
        n_batches = 0

        # Schedule DANN lambda: ramp from 0 → lambda_dann_max over warmup epochs
        dann_warmup = LODO_CONFIG['dann_warmup_epochs']
        lambda_dann = LODO_CONFIG['lambda_dann_max'] * min(epoch / dann_warmup, 1.0)
        domain_disc.set_lambda(lambda_dann)

        optimizer.zero_grad()
        domain_optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            y = batch['label'].to(device)
            meta = batch['metadata'].to(device)
            band_features = {k: v.to(device) for k, v in batch['band_features'].items()}
            clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}

            # Forward pass
            out = model(x, band_features=band_features, clinical_data=clinical)

            # ── Hierarchical loss ──
            hier_loss, _ = criterion(out, y, epoch=epoch)
            hier_loss = hier_loss / accum_steps

            # ── DANN domain loss ──
            # Source dataset labels for this batch
            subj_ids = batch['subject_id']
            domain_labels = torch.tensor(
                [dataset_source_map.get(get_dataset_source(sid), 0) for sid in subj_ids],
                dtype=torch.long, device=device
            )

            # Get penultimate embedding for domain discrimination
            # NCG fuses to a 512-d representation before the classification heads
            embedding = out.get('embedding', None)
            if embedding is not None:
                # GRL is built into domain_disc.set_lambda; apply reversal
                domain_logits = domain_disc(embedding, apply_grl=True)
                domain_loss = domain_criterion(domain_logits, domain_labels) * lambda_dann
                total_domain_loss += domain_loss.item()
            else:
                domain_loss = torch.tensor(0.0, device=device)

            loss = hier_loss + domain_loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), LODO_CONFIG['grad_clip'])
                optimizer.step()
                scheduler.step()
                domain_optimizer.step()
                optimizer.zero_grad()
                domain_optimizer.zero_grad()

            total_loss += hier_loss.item() * accum_steps
            n_batches += 1

        # Flush any remaining accumulated gradients at end of epoch
        remaining = len(train_loader) % accum_steps
        if remaining != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), LODO_CONFIG['grad_clip'])
            optimizer.step()
            domain_optimizer.step()
            optimizer.zero_grad()
            domain_optimizer.zero_grad()

        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = total_loss / max(n_batches, 1)
            avg_dann = total_domain_loss / max(n_batches, 1)
            print(f"      Epoch {epoch:2d}/{epochs} | Loss={avg_loss:.4f} "
                  f"| DANN={avg_dann:.4f} | λ_dann={lambda_dann:.3f}")

    return model


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model and return predictions."""
    model.eval()
    all_preds, all_true, all_subjs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y = batch['label']
            meta = batch['metadata'].to(device)
            band_features = {k: v.to(device) for k, v in batch['band_features'].items()}
            clinical = {'mmse': meta[:, 2], 'age': meta[:, 0], 'sex': meta[:, 1]}

            out = model(x, band_features=band_features, clinical_data=clinical)

            p_h = out['probs_screen'][:, 0].cpu().numpy()
            p_i = out['probs_screen'][:, 1].cpu().numpy()
            p_m = out['probs_stage'][:, 0].cpu().numpy()
            p_d = out['probs_stage'][:, 1].cpu().numpy()
            p_a = out['probs_subtype'][:, 0].cpu().numpy()
            p_f = out['probs_subtype'][:, 1].cpu().numpy()

            eps = 1e-8
            P_CN  = p_h
            P_MCI = np.power(p_i * p_m + eps, 1/2)
            P_AD  = np.power(p_i * p_d * p_a + eps, 1/3)
            P_FTD = np.power(p_i * p_d * p_f + eps, 1/3)

            probs = np.stack([P_AD, P_FTD, P_CN, P_MCI], axis=1)
            probs = probs / (probs.sum(axis=1, keepdims=True) + eps)
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_true.extend(y.numpy())
            if 'subject_id' in batch:
                all_subjs.extend(batch['subject_id'])

    return np.array(all_true), np.array(all_preds), all_subjs


def main():
    print("=" * 70)
    print("CROSS-DATASET VALIDATION (LEAVE-ONE-DATASET-OUT)")
    print("Improvements: pre-trained init | DANN | HierarchicalLoss | proper LR")
    print("=" * 70)

    # 1. Load all datasets and track source per sample
    print("\n[1] Loading datasets...")
    factory = DatasetFactory()
    for name, path in DATASET_PATHS.items():
        if path.exists():
            factory.add_dataset(name, path)

    dataset, groups, labels = factory.create_torch_datasets(config=DATASET_CONFIG)
    n_total = len(dataset)
    print(f"  Total samples: {n_total}")

    sources = np.array([get_dataset_source(g) for g in groups])
    unique_sources = np.unique(sources)
    print(f"  Sources found: {list(unique_sources)}")

    for src in unique_sources:
        mask = sources == src
        src_labels = labels[mask]
        dist = {CLASS_NAMES[i]: int((src_labels == i).sum()) for i in range(4)}
        print(f"    {src}: {mask.sum()} windows, classes: {dist}")

    # 2. LODO Cross-Validation
    print("\n[2] Running Leave-One-Dataset-Out CV...")
    results_all = []

    for test_source in unique_sources:
        print(f"\n  === Held-out: {test_source} ===")

        test_mask  = sources == test_source
        train_mask = ~test_mask

        test_indices  = np.where(test_mask)[0]
        train_indices = np.where(train_mask)[0]

        test_labels_src  = labels[test_indices]
        train_labels_src = labels[train_indices]

        test_classes = np.unique(test_labels_src)
        train_classes = np.unique(train_labels_src)
        print(f"    Train: {len(train_indices)} windows | Test: {len(test_indices)} windows")
        print(f"    Test classes present:  {[CLASS_NAMES[c] for c in test_classes]}")
        print(f"    Train classes present: {[CLASS_NAMES[c] for c in train_classes]}")

        # Warn if test has classes absent from training (fundamental LODO problem)
        missing = set(test_classes) - set(train_classes)
        if missing:
            print(f"    ⚠️  Classes in test but NOT in training: "
                  f"{[CLASS_NAMES[c] for c in missing]} — accuracy floor is ~chance for these classes")

        if len(test_indices) < 10:
            print(f"    ⚠️ Skipping {test_source}: too few test samples")
            continue

        # Data loaders
        train_ds = Subset(dataset, train_indices)
        test_ds  = Subset(dataset, test_indices)

        train_loader = DataLoader(
            train_ds, batch_size=LODO_CONFIG['batch_size'],
            shuffle=True, **DATALOADER_CONFIG
        )
        test_loader = DataLoader(
            test_ds, batch_size=32,
            shuffle=False, **DATALOADER_CONFIG
        )

        # Build source → dataset index map for training samples only
        # (n_datasets - 1 training datasets for this round)
        train_sources = np.unique(sources[train_indices])
        train_source_to_idx = {src: i for i, src in enumerate(train_sources)}

        # Initialize fresh model and load pre-trained weights
        model = create_neuro_chrono_graph_v2({
            'n_classes': 3,
            'hidden_dim': 128,
            'dropout': LODO_CONFIG['dropout']
        }).to(DEVICE)
        load_best_checkpoint(model, CHECKPOINTS_DIR)

        # Train with DANN + HierarchicalLoss
        print(f"    Fine-tuning ({LODO_CONFIG['epochs']} epochs, "
              f"lr={LODO_CONFIG['max_lr']}, DANN λ_max={LODO_CONFIG['lambda_dann_max']})...")
        model = train_one_round(
            model, train_loader,
            dataset_source_map=train_source_to_idx,
            n_datasets=len(train_sources),
            epochs=LODO_CONFIG['epochs'],
            device=DEVICE
        )

        # Evaluate
        y_true, y_pred, _ = evaluate_model(model, test_loader, device=DEVICE)

        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
        bacc  = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0

        # Per-class recall
        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )

        result = {
            'held_out_dataset': test_source,
            'n_train': int(len(train_indices)),
            'n_test': int(len(test_indices)),
            'accuracy': float(acc),
            'f1_macro': float(f1),
            'balanced_accuracy': float(bacc),
            'cohens_kappa': float(kappa),
            'test_classes': [CLASS_NAMES[c] for c in test_classes],
            'missing_train_classes': [CLASS_NAMES[c] for c in missing],
            'per_class': {
                cls: {
                    'precision': report[cls]['precision'],
                    'recall': report[cls]['recall'],
                    'f1': report[cls]['f1-score'],
                }
                for cls in CLASS_NAMES if cls in report
            }
        }
        results_all.append(result)

        print(f"    Result: Acc={acc:.1%}, F1={f1:.3f}, BACC={bacc:.1%}, κ={kappa:.3f}")

        del model
        torch.cuda.empty_cache()

    # 3. Aggregate results
    print("\n" + "=" * 70)
    print("CROSS-DATASET VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<15} {'N_test':>7} {'Accuracy':>10} {'F1':>8} "
          f"{'BACC':>8} {'κ':>8} {'Missing'}")
    print("-" * 75)
    for r in results_all:
        missing_str = ','.join(r['missing_train_classes']) if r['missing_train_classes'] else '—'
        print(f"{r['held_out_dataset']:<15} {r['n_test']:>7} {r['accuracy']:>9.1%} "
              f"{r['f1_macro']:>8.3f} {r['balanced_accuracy']:>8.1%} "
              f"{r['cohens_kappa']:>8.3f}  {missing_str}")

    if len(results_all) > 1:
        mean_acc  = np.mean([r['accuracy'] for r in results_all])
        mean_f1   = np.mean([r['f1_macro'] for r in results_all])
        mean_bacc = np.mean([r['balanced_accuracy'] for r in results_all])
        print("-" * 75)
        print(f"{'Mean':<15} {'':>7} {mean_acc:>9.1%} {mean_f1:>8.3f} {mean_bacc:>8.1%}")

    # Note: LODO is fundamentally limited by class availability per dataset
    print("\n⚠️  Fundamental LODO limitation:")
    print("   Datasets with unique classes (e.g. MCI_Dataset=MCI only,")
    print("   ds004504=only source of FTD) will always yield near-chance accuracy")
    print("   when held out, regardless of training strategy, because the training")
    print("   set lacks those class examples. This is a data collection constraint,")
    print("   not a modelling failure.")

    # Save
    df = pd.DataFrame(results_all)
    df.to_csv(RESULTS_DIR / "cross_dataset_validation.csv", index=False)
    with open(RESULTS_DIR / "cross_dataset_validation.json", 'w') as f:
        json.dump({'config': LODO_CONFIG, 'results': results_all}, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'cross_dataset_validation.csv'}")
    print("Done!")


if __name__ == '__main__':
    main()
