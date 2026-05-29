"""
Experiment 01: Preprocess All Subjects

This script loads and preprocesses all EEG data from the dataset,
preparing it for feature extraction.

Optimized for HP Omen 16 (Intel Ultra 9, 32GB RAM)
"""

import sys
from pathlib import Path
import pickle
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import (
    DATA_ROOT, PREPROCESSED_PATH, 
    PREPROCESSING, set_seed, RANDOM_SEED, N_JOBS
)
from src.data.bids_loader import BIDSDataset
from src.data.preprocessor import (
    apply_preprocessing_pipeline,
    create_epochs,
    apply_preprocessing_pipeline,
    create_epochs,
    check_data_quality
)

print(f"DEBUG: Loaded preprocessor from: {sys.modules['src.data.preprocessor'].__file__}")


def preprocess_subject(dataset, subject_id, output_dir, dataset_name, skip_existing=True):
    """
    Preprocess a single subject's data.
    
    Args:
        dataset: BIDSDataset instance
        subject_id: Subject identifier
        output_dir: Directory to save preprocessed data
        dataset_name: Name of the dataset (for namespacing)
        skip_existing: Skip if output already exists
        
    Returns:
        Tuple of (subject_id, success, message)
    """
    output_file = output_dir / f"{dataset_name}_{subject_id}_preprocessed.pkl"
    
    # Skip if already exists
    if skip_existing and output_file.exists():
        return (subject_id, True, "Already exists, skipped")
    
    try:
        # Load raw data
        raw = dataset.load_raw(subject_id)
        
        # Check data quality
        quality = check_data_quality(raw)
        
        # Apply preprocessing pipeline
        # Note: derivatives data already has filtering applied
        if PREPROCESSING['use_derivatives']:
            # Minimal additional processing
            raw_preprocessed = apply_preprocessing_pipeline(
                raw,
                apply_filter=False,  # Already filtered
                apply_notch=False,   # Already done
                apply_reference=True,
                apply_csd=True      # Apply Surface Laplacian
            )
        else:
            # Full preprocessing for raw data
            raw_preprocessed = apply_preprocessing_pipeline(
                raw,
                apply_filter=True,
                apply_notch=True,
                apply_reference=True,
                apply_csd=True
            )
        
        # Create epochs
        epochs = create_epochs(
            raw_preprocessed,
            duration=PREPROCESSING['epoch_duration'],
            overlap=PREPROCESSING['epoch_overlap']
        )
        
        # Get subject info
        subject_info = dataset.get_subject_info(subject_id)
        
        # Save preprocessed data
        data_to_save = {
            'epochs_data': epochs.get_data(),
            'sfreq': epochs.info['sfreq'],
            'ch_names': epochs.ch_names,
            'n_epochs': len(epochs),
            'subject_info': subject_info,
            'quality': quality
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        msg = f"Created {len(epochs)} epochs, Duration: {quality['duration_sec']:.1f}s"
        return (subject_id, True, msg)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (subject_id, False, str(e))


def main():
    """Main preprocessing script."""
    parser = argparse.ArgumentParser(description='Preprocess EEG data')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Specific subjects to process (e.g., sub-001 sub-002)')
    parser.add_argument('--n-subjects', type=int, default=None,
                        help='Process first N subjects only')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip subjects that already have preprocessed data')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of all subjects')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='Number of parallel jobs')
    args = parser.parse_args()
    
    print("="*60)
    print("EEG Preprocessing Pipeline")
    print("="*60)
    
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Create output directory
    output_dir = PREPROCESSED_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading datasets from: {DATA_ROOT}")
    from src.data.dataset_factory import DatasetFactory
    
    factory = DatasetFactory()
    
    # Define datasets to allow - mirrors 18_train_hierarchical.py
    datasets_to_load = {
        'ds004504': DATA_ROOT / "openneuro_ds004504",
        'ds006036': DATA_ROOT / "ds006036",
        'Alz_EEG': DATA_ROOT / "Alz_EEG_data",
        'Mendeley': DATA_ROOT / "Mendeley Dataset",
        'MCI_Dataset': DATA_ROOT / "mci dataset"  # New MCI data source
    }
    
    for name, path in datasets_to_load.items():
        if path.exists():
            print(f"Adding dataset: {name}")
            factory.add_dataset(name, path)
        else:
             print(f"Dataset not found: {path} (Skipping)")

    if not factory.datasets:
        print("CRITICAL: No datasets loaded. Check path structure.")
        sys.exit(1)
    
    # Process subjects from ALL datasets
    total_success = 0
    total_fail = 0
    total_skipped = 0
    
    for dataset_obj in factory.datasets:
        ds_name = getattr(dataset_obj, 'dataset_name', 'Unknown')
        print(f"\n[{ds_name}] Scanning subjects...")
        
        try:
            subject_ids = dataset_obj.get_subject_ids()
        except Exception as e:
            print(f"Error getting subjects for {ds_name}: {e}")
            continue
            
        print(f"Found {len(subject_ids)} subjects in {ds_name}")
        
        # Determine specific subjects to process if args provided?
        # Argument filtering is tricky with multiple datasets.
        # Let's simplify: if args.subjects provided, check if in this dataset.
        
        subjects_to_process = subject_ids
        if args.subjects:
            subjects_to_process = [s for s in subject_ids if s in args.subjects]
            if not subjects_to_process:
                continue # None of the requested subjects are in this dataset
        elif args.n_subjects:
            subjects_to_process = subject_ids[:args.n_subjects]

        skip_existing = args.skip_existing and not args.force
        
        print(f"Processing {len(subjects_to_process)} subjects from {ds_name}...")
        
        for subject_id in subjects_to_process:
            print(f"  Subject: {subject_id}")
            
            # Use the specific dataset object!
            sid, success, msg = preprocess_subject(
                dataset_obj, subject_id, output_dir, ds_name, skip_existing
            )
            
            if success:
                if "skipped" in msg.lower():
                    total_skipped += 1
                    print(f"    - {msg}")
                else:
                    total_success += 1
                    print(f"    + {msg}")
            else:
                total_fail += 1
                if "Participants file not found" in msg:
                    print(f"    ! ERROR: {msg}")
                else:
                    print(f"    ! ERROR: {msg}")
                    # DEBUG: Print the simplified traceback captured in msg?
                    # preprocess_subject only returns str(e).
                    # We need to print traceback INSIDE preprocess_subject or here if we change return.
                    # Let's rely on preprocess_subject printing?
                    # Actually, let's just make preprocess_subject print traceback on error.

    # Summary
    print("\n" + "="*60)
    print("Preprocssing Pipeline Finished")
    print("="*60)
    print(f"Total Successful: {total_success}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Failed: {total_fail}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
