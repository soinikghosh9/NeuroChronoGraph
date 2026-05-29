"""
BIDS Data Loader for OpenNeuro Datasets (ds004504, ds006036).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import mne
import warnings
from sklearn.preprocessing import RobustScaler

from .base_loader import BaseDataset
from ..config.config import (
    DATA_ROOT, CHANNEL_NAMES, SAMPLING_RATE, OUTPUT_ROOT
)

# Standard channel mapping for unification
STANDARD_CHANNELS = CHANNEL_NAMES

class BIDSDataset(BaseDataset):
    """
    BIDS Dataset loader for OpenNeuro ds004504 (EC) and ds006036 (EO).
    """
    
    def __init__(self, 
                 root_path: Path,
                 dataset_name: str = "ds004504",
                 use_derivatives: bool = True,
                 verbose: bool = True):
        super().__init__(root_path)
        self.dataset_name = dataset_name
        self.use_derivatives = use_derivatives
        self.verbose = verbose
        
        # Load participants information
        self.participants = self._load_participants()
        self.subject_ids = self.participants['participant_id'].tolist()
        
        if self.verbose:
            print(f"Loaded {self.dataset_name} with {len(self.subject_ids)} subjects")
            print(f"  - AD: {sum(self.participants['Group'] == 'A')}")
            print(f"  - FTD: {sum(self.participants['Group'] == 'F')}")
            print(f"  - CN: {sum(self.participants['Group'] == 'C')}")
    
    def _load_participants(self) -> pd.DataFrame:
        """Load and parse the participants.tsv file."""
        participants_file = self.root_path / "participants.tsv"
        
        if not participants_file.exists():
            raise FileNotFoundError(f"Participants file not found: {participants_file}")
        
        df = pd.read_csv(participants_file, sep='\t')
        
        # Clean column names (handle trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Map Groups to Hierarchical Labels
        # 'label': 0=AD, 1=FTD, 2=CN, 3=MCI (no MCI in BIDS datasets usually)
        label_map = {'A': 0, 'F': 1, 'C': 2}
        if 'Group' not in df.columns:
             # Fallback if specific naming differs
             raise KeyError(f"Column 'Group' not found. Available: {df.columns}")
             
        df['label'] = df['Group'].map(label_map)
        
        # Normalize metadata
        # Handle cases where MMSE might be missing or non-numeric?
        # Assuming clean for now, but whitespace fix `strip()` above solves the main user error
        if 'MMSE' in df.columns:
            df['MMSE_normalized'] = df['MMSE'] / 30.0
        else:
            print("Warning: MMSE column not found even after strip. Using default.")
            df['MMSE_normalized'] = 0.5
            
        df['Sex_encoded'] = (df['Gender'] == 'M').astype(int)
        df['Age_normalized'] = (df['Age'] - 40) / 50.0
        df['Age_normalized'] = df['Age_normalized'].clip(0, 1)
        
        return df
    
    def get_subject_ids(self) -> List[str]:
        return self.subject_ids
    
    def get_subject_info(self, subject_id: str) -> Dict:
        row = self.participants[self.participants['participant_id'] == subject_id]
        if len(row) == 0:
            raise ValueError(f"Subject {subject_id} not found")
        row = row.iloc[0]
        
        return {
            'subject_id': subject_id,
            'dataset': self.dataset_name,
            'age': row['Age'],
            'sex': row['Gender'],
            'group': row['Group'],
            'mmse': row['MMSE'] if 'MMSE' in row else -1.0,
            'label': row['label'],
            'condition': 'eyes_open' if 'ds006036' in self.dataset_name else 'eyes_closed',
            'metadata': np.array([
                row['Age_normalized'],
                row['Sex_encoded'],
                row['MMSE'] / 30.0 if 'MMSE' in row else -1.0 # -1.0 signals missing to conditioner
            ], dtype=np.float32)
        }
    
    def load_raw(self, subject_id: str) -> mne.io.Raw:
        # Optimization: Check for preprocessed PKL (Generic for any BIDS dataset)
        pkl_path = OUTPUT_ROOT / "preprocessed" / f"{self.dataset_name}_{subject_id}_preprocessed.pkl"
        if pkl_path.exists():
            try:
                # print(f"Loading cached: {pkl_path.name}") # verbose off to keep clean?
                return self._load_from_pkl(pkl_path)
            except Exception as e:
                print(f"Warning: Failed to load cache {pkl_path}: {e}")

        # Resolve path handling 'sub-' prefix
        sub_folder = subject_id if subject_id.startswith('sub-') else f'sub-{subject_id}'
        set_files = []
        
        if self.use_derivatives:
            # 1. Try standard BIDS derivatives/sub-XXX structure
            search_path = self.root_path / "derivatives" / sub_folder
            set_files = list(search_path.rglob("*.set"))
            
            # 2. If not found, try recursive search in derivatives for this subject
            if not set_files and (self.root_path / "derivatives").exists():
                set_files = list((self.root_path / "derivatives").rglob(f"{sub_folder}/**/*.set"))
                
        # 3. Fallback: Search in root (standard BIDS raw)
        if not set_files:
            search_path = self.root_path / sub_folder
            set_files = list(search_path.rglob("*.set"))
        
        if not set_files:
             # 4. Final Fallback (anywhere)
             set_files = list(self.root_path.rglob(f"{sub_folder}*.set"))

        if not set_files:
            raise FileNotFoundError(f"No .set files found for {subject_id} in {self.root_path} (Derivatives: {self.use_derivatives})")
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*boundary.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*limited.*")
            raw = mne.io.read_raw_eeglab(set_files[0], preload=True, verbose=False)
        
        # 1. Standardize Channels
        raw = self._standardize_channels(raw)
        
        # 2. Resample
        if raw.info['sfreq'] != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)
            
        # 3. Robust Scaling with nan check
        data = raw.get_data()
        
        # FIX: Ensure channels are marked as EEG before further processing
        # This prevents picking errors in preprocessor
        ch_type_map = {}
        for ch in raw.ch_names:
            if ch in STANDARD_CHANNELS:
                ch_type_map[ch] = 'eeg'
        if ch_type_map:
            try:
                raw.set_channel_types(ch_type_map)
            except Exception as e:
                print(f"Warning: Failed to set channel types in bids_loader: {e}")

        scaler = RobustScaler(quantile_range=(25, 75))
        data_scaled = scaler.fit_transform(data.T).T
        
        if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
            # Replace NaNs and Infs with 0
            data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
        raw._data = data_scaled
            
        return raw

    def load_raw_info(self, subject_id: str) -> Dict:
        """Fast load of metadata without preloading data."""
        # Optimization: Check for preprocessed PKL (Only for ds004504 currently)
        if self.dataset_name == 'ds004504':
             pkl_path = Path(DATA_ROOT).parent.parent / "outputs" / "preprocessed" / f"{subject_id}_preprocessed.pkl"
             if pkl_path.exists():
                 import pickle
                 with open(pkl_path, 'rb') as f:
                     # Optimization: Don't read whole pickle if possible? 
                     # Pickle doesn't support random access easily.
                     # But it's faster than EEGLAB.
                     # Let's trust PKL speed or just use EEGLAB header if PKL is slow.
                     # Actually, for PKL, we might need to load it. 
                     # Let's fallback to EEGLAB header if derivative exists, else load PKL.
                     pass 

        # Logic: Use EEGLAB header (preload=False)
        sub_folder = subject_id if subject_id.startswith('sub-') else f'sub-{subject_id}'
        
        if self.use_derivatives:
            search_path = self.root_path / "derivatives" / sub_folder
            set_files = list(search_path.rglob("*.set"))
        else:
            search_path = self.root_path / sub_folder
            set_files = list(search_path.rglob("*.set"))
            
        if not set_files:
             set_files = list(self.root_path.rglob(f"{sub_folder}*.set"))
             
        if not set_files:
             # If no EEGLAB but PKL exists (handled inside load_raw), we might fallback.
             # But here we assume set files exist for now or we must load PKL.
             if self.dataset_name == 'ds004504':
                 pkl_path = Path(DATA_ROOT).parent.parent / "outputs" / "preprocessed" / f"{subject_id}_preprocessed.pkl"
                 if pkl_path.exists():
                     # Must load
                     import pickle
                     with open(pkl_path, 'rb') as f:
                         data = pickle.load(f)
                         # Use 'epochs_data' key
                         n_samples = data['epochs_data'].shape[0] * data['epochs_data'].shape[2] # windows * time
                         return {'n_times': n_samples, 'sfreq': data.get('sfreq', SAMPLING_RATE)}

             raise FileNotFoundError(f"No .set files found for {subject_id}")

        # Fast header read
        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             raw = mne.io.read_raw_eeglab(set_files[0], preload=False, verbose=False)
        
        # Calculate n_samples after resampling
        orig_sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        
        if orig_sfreq != SAMPLING_RATE:
            # Resampling changes sample count
            ratio = SAMPLING_RATE / orig_sfreq
            n_samples = int(n_samples * ratio)
            
        return {'n_times': n_samples, 'sfreq': SAMPLING_RATE}

    def _load_from_pkl(self, pkl_path: Path) -> mne.io.Raw:
        """Load data from preprocessed PKL and convert to MNE Raw."""
        import pickle
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # Structure: {'epochs_data': np.array, 'sfreq': ...}
        # If 'epochs_data' is numpy array: (N_epochs, Channels, Time)
        # We need to flatten back to continuous for consistency with 'load_raw' contract
        
        epochs_data = data_dict['epochs_data'] # Correct key from preprocessor
        
        # Flatten: (Ch, N*T)
        n_epochs, n_ch, n_times = epochs_data.shape
        data_flat = np.hstack(epochs_data) # Stack horizontally -> (Ch, N*T)
        
        # Create Info
        ch_names = data_dict.get('ch_names', STANDARD_CHANNELS)
        sfreq = data_dict.get('sfreq', SAMPLING_RATE)
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data_flat, info, verbose=False)
        
        # Montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # Robust Scaling?
        # "outputs/preprocessed" might ALREADY be scaled?
        # User says "load easily", "mostly those are preprocessed".
        # Safe to assume it might be cleaner, but RobustScaler is idempotent-ish if already scaled around median.
        # Let's Apply it to be safe and consistent with other sources.
        
        data = raw.get_data()
        scaler = RobustScaler(quantile_range=(25, 75))
        data_scaled = scaler.fit_transform(data.T).T
        raw._data = data_scaled
        
        return raw

    def _standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        # 1. Rename common variations
        rename_dict = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
        raw.rename_channels({k: v for k, v in rename_dict.items() if k in raw.ch_names})
        
        # 2. Set Montage (10-20)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # 3. Pick/Interpolate to Standard 19
        raw.pick(picks=[ch for ch in STANDARD_CHANNELS if ch in raw.ch_names])
        
        # If missing channels, add them as flat and warn (or interpolation if we had locations)
        # For these datasets, we usually have all 19.
        if len(raw.ch_names) < len(STANDARD_CHANNELS):
             # Logic to handle missing channels if strictly needed
             pass

        # Reorder to standard order
        raw.reorder_channels([ch for ch in STANDARD_CHANNELS if ch in raw.ch_names])
        
        return raw

def get_train_test_split(dataset: BIDSDataset, 
                         test_subject: str) -> Tuple[List[str], List[str]]:
    """
    Get train/test split for LOSO cross-validation (Legacy support).
    
    Args:
        dataset: BIDSDataset instance
        test_subject: Subject ID to use for testing
        
    Returns:
        (train_subjects, test_subjects) tuple of subject ID lists
    """
    all_subjects = dataset.get_subject_ids()
    train_subjects = [s for s in all_subjects if s != test_subject]
    test_subjects = [test_subject]
    
    return train_subjects, test_subjects
