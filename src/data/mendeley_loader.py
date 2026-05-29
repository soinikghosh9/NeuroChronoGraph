"""
Loader for Mendeley Dataset (AD, MCI, Normal).

Structure:
datasets/Mendeley Dataset/
  - AD.mat (struct with EEG data)
  - MCI.mat
  - normal.mat
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, List
import mne
from sklearn.preprocessing import RobustScaler

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

class MendeleyDataset(BaseDataset):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.groups = {'AD': 0, 'MCI': 3, 'normal': 2} # AD, MCI, CN
        self.subjects_cache = self._load_all_data_metadata()
        self.subject_ids = list(self.subjects_cache.keys())
        print(f"Loaded Mendeley Dataset with {len(self.subject_ids)} subjects")

    def _load_all_data_metadata(self) -> Dict:
        """
        Since these are single .mat files containing all subjects,
        we preload metadata but lazily access data if possible,
        or just load chunks.
        Actually, .mat files might be small enough to keep in memory or
        memmapped.
        """
        info_map = {}
        
        for grp_name, label_id in self.groups.items():
            mat_file = self.root_path / f"{grp_name}.mat"
            if not mat_file.exists():
                continue
                
            try:
                # Load MAT file
                mat = scipy.io.loadmat(str(mat_file))
                # Structure inspection: usually key matches filename or 'val'
                # Assumption: keys are subject IDs or a big matrix
                # Common Mendeley format: struct with 'AD', 'MCI', 'CMS' keys?
                # Or simply variable names.
                
                # Let's inspect keys excluding headers
                keys = [k for k in mat.keys() if not k.startswith('__')]
                
                # Assuming the main key holds the data struct
                # For this specific dataset, data is often (Channels x Samples x Subjects) OR (Subjects struct)
                # We'll assume a list of arrays for now.
                
                # Check for the main data key
                data_key = keys[0] # Aggressive assumption
                data_obj = mat[data_key]
                
                # Parse based on shape
                # Structure is typically (1, N_subjects) where each element is a struct
                
                n_subs = data_obj.shape[1] if len(data_obj.shape) == 2 else data_obj.shape[0]
                
                for i in range(n_subs):
                    sid = f"Mendeley_{grp_name}_{i}"
                    info_map[sid] = {
                        "file": mat_file,
                        "key": data_key,
                        "index": i,
                        "group": grp_name,
                        "label": label_id,
                        "data_ref": None
                    }
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
                
        return info_map

    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.subjects_cache[subject_id]
        return {
            'subject_id': subject_id,
            'dataset': 'Mendeley',
            'group': info['group'],
            'label': info['label'],
            'age': 70.0, # Dummy
            'sex': 'Unknown',
            'mmse': -1.0, # Standardized Missing
            'metadata': np.array([0.5, 0.5, -1.0], dtype=np.float32)
        }

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        # Reloading MAT file is inefficient but robust.
        info = self.subjects_cache[subject_id]
        mat = scipy.io.loadmat(str(info['file']))
        data_obj = mat[info['key']]
        
        # Extract specific subject
        # Shape: (1, N_subs) -> Element at [0, i]
        subj_struct = data_obj[0, info['index']]
        
        # Extract 'epoch' data: (4, 600, Trials)
        if 'epoch' in subj_struct.dtype.names:
            epochs_data = subj_struct['epoch'] # (4, 600, N)
        else:
             raise ValueError(f"Field 'epoch' not found for {subject_id}")
             
        # Flatten trials to continuous time
        # (4, 600, N) -> (N, 4, 600) -> (4, N*600)
        n_ch, n_time, n_trials = epochs_data.shape
        data = epochs_data.transpose(2, 0, 1) # (N, 4, 600)
        data = np.hstack([data[i] for i in range(n_trials)]) # (4, TotalTime)
        
        # Create MNE Raw
        # We only have 4 channels. 
        # Map to first 4 standard channels as placeholders? 
        # Or Fp1, Fp2, C3, C4? Without labels this is a guess.
        # Mapping to first 4 for pipeline compatibility.
        current_ch_names = CHANNEL_NAMES[:n_ch]
        
        sfreq_in = 250.0 # Assumption based on 600 samples ~ 2.4s?
        
        info_mne = mne.create_info(ch_names=current_ch_names, sfreq=sfreq_in, ch_types='eeg')
        raw = mne.io.RawArray(data, info_mne, verbose=False)
        
        # Standardize (Pad to 19)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # Custom standardize to add missing channels as zeros
        raw = self._pad_channels(raw)
        
        # Resample to Global SAMPLING_RATE
        if raw.info['sfreq'] != SAMPLING_RATE:
             print(f"  Resampling {subject_id} from {raw.info['sfreq']} to {SAMPLING_RATE} Hz")
             raw.resample(SAMPLING_RATE)

        # Robust Scaling
        data = raw.get_data()
        scaler = RobustScaler(quantile_range=(25, 75))
        data_scaled = scaler.fit_transform(data.T).T
        
        # Handle potential NaNs from flat channels (IQR=0)
        if np.isnan(data_scaled).any():
            data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
        raw._data = data_scaled
        
        return raw

    def _pad_channels(self, raw):
        """Pad missing channels with zeros to match STANDARD_CHANNELS."""
        existing = set(raw.ch_names)
        missing = [ch for ch in CHANNEL_NAMES if ch not in existing]
        
        if missing:
            # Add zero channels
            n_missing = len(missing)
            n_times = raw.n_times
            zeros = np.zeros((n_missing, n_times))
            
            info_add = mne.create_info(ch_names=missing, sfreq=raw.info['sfreq'], ch_types='eeg')
            raw_add = mne.io.RawArray(zeros, info_add, verbose=False)
            
            raw.add_channels([raw_add], force_update_info=True)
            
        # Reorder
        raw.pick(CHANNEL_NAMES)
        return raw

