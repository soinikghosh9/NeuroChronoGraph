"""
Hold-Out Test Set Splitter for NeuroChronoGraph.

Creates a stratified hold-out test set BEFORE any training begins.
This ensures truly unbiased evaluation with no data leakage.

Usage:
    from src.data.holdout_splitter import create_holdout_split, load_holdout_split
    
    # Create split (once)
    split = create_holdout_split(subjects, labels, holdout_ratio=0.15)
    
    # Load split (for subsequent runs)
    split = load_holdout_split()
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

# Default paths
DEFAULT_SPLIT_PATH = Path(__file__).parent.parent.parent / "outputs" / "holdout_split.json"


def create_holdout_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    holdout_ratio: float = 0.15,
    min_per_class: int = 3,
    random_seed: int = 42,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Create a stratified hold-out split ensuring all classes are represented.
    
    Args:
        subjects: Array of subject IDs (matching epochs)
        labels: Array of class labels (0=AD, 1=FTD, 2=CN)
        holdout_ratio: Fraction of subjects for hold-out (~15-20%)
        min_per_class: Minimum subjects per class in hold-out
        random_seed: For reproducibility
        save_path: Where to save the split JSON
        
    Returns:
        Dictionary with 'development' and 'holdout' subject lists
    """
    np.random.seed(random_seed)
    
    if save_path is None:
        save_path = DEFAULT_SPLIT_PATH
    
    # Get unique subjects and their labels
    unique_subjects = np.unique(subjects)
    subject_to_label = {}
    
    for subj in unique_subjects:
        mask = subjects == subj
        # All epochs from same subject should have same label
        subj_labels = labels[mask]
        if len(np.unique(subj_labels)) > 1:
            warnings.warn(f"Subject {subj} has multiple labels, using mode")
        subject_to_label[subj] = int(np.bincount(subj_labels).argmax())
    
    # Group subjects by class
    class_subjects = defaultdict(list)
    for subj, label in subject_to_label.items():
        class_subjects[label].append(subj)
    
    # Verify all classes present
    class_names = {0: 'AD', 1: 'FTD', 2: 'CN'}
    for cls in [0, 1, 2]:
        if cls not in class_subjects or len(class_subjects[cls]) == 0:
            raise ValueError(f"Class {class_names[cls]} has no subjects!")
    
    print(f"\n{'='*60}")
    print(" Creating Stratified Hold-Out Split")
    print(f"{'='*60}")
    print(f"Total subjects: {len(unique_subjects)}")
    for cls, name in class_names.items():
        print(f"  {name}: {len(class_subjects[cls])} subjects")
    
    # Calculate how many to hold out per class
    holdout_subjects = []
    development_subjects = []
    
    for cls in [0, 1, 2]:
        cls_subjs = class_subjects[cls].copy()
        np.random.shuffle(cls_subjs)
        
        n_holdout = max(min_per_class, int(len(cls_subjs) * holdout_ratio))
        # Don't take more than half of any class
        n_holdout = min(n_holdout, len(cls_subjs) // 2)
        
        holdout_subjects.extend(cls_subjs[:n_holdout])
        development_subjects.extend(cls_subjs[n_holdout:])
    
    # Create split info
    split = {
        'random_seed': random_seed,
        'holdout_ratio': holdout_ratio,
        'development': sorted(development_subjects),
        'holdout': sorted(holdout_subjects),
        'development_count': {
            class_names[cls]: len([s for s in development_subjects if subject_to_label[s] == cls])
            for cls in [0, 1, 2]
        },
        'holdout_count': {
            class_names[cls]: len([s for s in holdout_subjects if subject_to_label[s] == cls])
            for cls in [0, 1, 2]
        }
    }
    
    print(f"\nHold-out set ({len(holdout_subjects)} subjects):")
    for cls, name in class_names.items():
        count = split['holdout_count'][name]
        print(f"  {name}: {count} subjects")
    
    print(f"\nDevelopment set ({len(development_subjects)} subjects):")
    for cls, name in class_names.items():
        count = split['development_count'][name]
        print(f"  {name}: {count} subjects")
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"\n✓ Split saved to: {save_path}")
    
    return split


def load_holdout_split(split_path: Optional[Path] = None) -> Dict:
    """Load existing hold-out split."""
    if split_path is None:
        split_path = DEFAULT_SPLIT_PATH
    
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Hold-out split not found: {split_path}\n"
            "Run create_holdout_split() first."
        )
    
    with open(split_path) as f:
        return json.load(f)


def get_split_masks(
    subjects: np.ndarray,
    split: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get boolean masks for development and hold-out sets.
    
    Args:
        subjects: Array of subject IDs (per epoch)
        split: Split dictionary from create/load functions
        
    Returns:
        (development_mask, holdout_mask) - boolean arrays
    """
    development_set = set(split['development'])
    holdout_set = set(split['holdout'])
    
    dev_mask = np.array([s in development_set for s in subjects])
    holdout_mask = np.array([s in holdout_set for s in subjects])
    
    return dev_mask, holdout_mask


def verify_split(subjects: np.ndarray, labels: np.ndarray, split: Dict) -> bool:
    """Verify split integrity - all classes present, no overlap."""
    dev_mask, holdout_mask = get_split_masks(subjects, split)
    
    # Check no overlap
    if np.any(dev_mask & holdout_mask):
        print("ERROR: Overlap between development and holdout!")
        return False
    
    # Check all data covered
    if not np.all(dev_mask | holdout_mask):
        print("ERROR: Some subjects not in either set!")
        return False
    
    # Check all classes in holdout
    holdout_labels = labels[holdout_mask]
    for cls in [0, 1, 2]:
        if not np.any(holdout_labels == cls):
            print(f"ERROR: Class {cls} not in holdout set!")
            return False
    
    print("✓ Split verification passed")
    return True


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing holdout splitter...")
    
    # Simulate 88 subjects with ~100 epochs each
    subjects = np.repeat([f"sub-{i:03d}" for i in range(1, 89)], 100)
    # AD: 1-36, CN: 37-65, FTD: 66-88
    labels = np.zeros(len(subjects), dtype=int)
    for i, s in enumerate(subjects):
        num = int(s.split('-')[1])
        if num <= 36:
            labels[i] = 0  # AD
        elif num <= 65:
            labels[i] = 2  # CN
        else:
            labels[i] = 1  # FTD
    
    split = create_holdout_split(subjects, labels, holdout_ratio=0.15)
    verify_split(subjects, labels, split)
