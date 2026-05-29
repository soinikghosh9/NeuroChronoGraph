"""
Loader for MCI Dataset (Figshare EEG Dataset).

Structure:
datasets/mci dataset/
  - MCI/
      - MCI1/
          - 1.mat, 2.mat, ... (each MAT contains 'export' key with EEG data)
  - AD/
      - AD1/
          - 1.mat, 2.mat, ...
  - CONTROL/
      - 10057fir.mat, ... (single MAT files)
      - normal1/
          - 1.mat, 2.mat, ...

Dataset properties:
- Sampling rates: 128Hz (some) or 256Hz (some)
- 19 channels in 10-20 system
- MAT files contain EEG data in 'export' key [samples x 19] or 'segmenty' key [trials x samples]
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List
import mne
from sklearn.preprocessing import RobustScaler

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

# Sampling rates from readme.txt
# MCI1-4, AD1-34: 256Hz
# MCI5-7, AD35-59, normal1-2: 128Hz
SFREQ_256HZ_SUBJECTS = set([
    'MCI1', 'MCI2', 'MCI3', 'MCI4',
    *[f'AD{i}' for i in range(1, 35)]
])
SFREQ_128HZ_SUBJECTS = set([
    'MCI5', 'MCI6', 'MCI7',
    *[f'AD{i}' for i in range(35, 60)],
    'normal1', 'normal2'
])

class MCIDataset(BaseDataset):
    """
    Loader for the Figshare MCI-AD-Control EEG Dataset.
    """
    
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        # Labels: 0=AD, 1=FTD (none here), 2=CN, 3=MCI
        self.group_labels = {'AD': 0, 'CONTROL': 2, 'MCI': 3}
        self.info_map = self._scan_dataset()
        self.subject_ids = list(self.info_map.keys())
        
        # Print summary
        counts = {'AD': 0, 'MCI': 0, 'CONTROL': 0}
        for sid, info in self.info_map.items():
            counts[info['group']] += 1
        print(f"Loaded MCI Dataset with {len(self.subject_ids)} subjects")
        print(f"  - AD: {counts['AD']}, MCI: {counts['MCI']}, CONTROL: {counts['CONTROL']}")
        
    def _scan_dataset(self) -> Dict:
        """
        Scan directory structure for subjects.
        Returns dict: subject_id -> {path, group, label, sfreq, files}
        """
        info_map = {}
        
        # Process each group
        for group_name, label in self.group_labels.items():
            group_path = self.root_path / group_name
            if not group_path.exists():
                continue
            
            if group_name == 'CONTROL':
                # CONTROL has mixed structure: direct MAT files + subdirectories
                # Process direct MAT files
                for mat_file in group_path.glob("*.mat"):
                    sid = f"MCIData_CONTROL_{mat_file.stem}"
                    sfreq = 128  # Most CONTROL files are 128Hz
                    info_map[sid] = {
                        'path': group_path,
                        'group': group_name,
                        'label': label,
                        'sfreq': sfreq,
                        'files': [mat_file],
                        'original_id': mat_file.stem,
                        'is_single_file': True,
                        'data_key': 'segmenty'  # CONTROL files use different key
                    }
                
                # Process subdirectories (normal1, normal2)
                for subdir in group_path.iterdir():
                    if subdir.is_dir():
                        mat_files = list(subdir.glob("*.mat"))
                        if mat_files:
                            orig_id = subdir.name
                            sfreq = 128 if orig_id in SFREQ_128HZ_SUBJECTS else 256
                            sid = f"MCIData_CONTROL_{orig_id}"
                            info_map[sid] = {
                                'path': subdir,
                                'group': group_name,
                                'label': label,
                                'sfreq': sfreq,
                                'files': mat_files,
                                'original_id': orig_id,
                                'is_single_file': False,
                                'data_key': 'export'
                            }
            else:
                # AD and MCI have subdirectory structure
                for subdir in group_path.iterdir():
                    if subdir.is_dir():
                        mat_files = list(subdir.glob("*.mat"))
                        if mat_files:
                            orig_id = subdir.name  # e.g., 'MCI1', 'AD1'
                            sfreq = 256 if orig_id in SFREQ_256HZ_SUBJECTS else 128
                            sid = f"MCIData_{group_name}_{orig_id}"
                            info_map[sid] = {
                                'path': subdir,
                                'group': group_name,
                                'label': label,
                                'sfreq': sfreq,
                                'files': mat_files,
                                'original_id': orig_id,
                                'is_single_file': False,
                                'data_key': 'export'
                            }
        
        return info_map
    
    def get_subject_ids(self) -> List[str]:
        return self.subject_ids
    
    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.info_map[subject_id]
        # Use dummy metadata (not available in this dataset)
        # Could be enhanced by parsing readme.txt for cognitive scores
        
        # Standardized Missing Value: -1.0
        # Prevents label leakage via dummy MMSE values
        mmse_val = -1.0
        
        return {
            'subject_id': subject_id,
            'dataset': 'MCI_Dataset',
            'group': info['group'],
            'label': info['label'],
            'age': 70.0, # Dummy mean
            'sex': 'Unknown',
            'mmse': mmse_val,
            'metadata': np.array([0.5, 0.5, -1.0], dtype=np.float32) # age_norm, sex, mmse_norm
        }
            
        return {
            'subject_id': subject_id,
            'dataset': 'MCI_Dataset',
            'group': info['group'],
            'label': info['label'],
            'age': 70.0,  # Dummy
            'sex': 'Unknown',
            'mmse': mmse_approx,
            'metadata': np.array([0.5, 0.5, mmse_approx / 30.0], dtype=np.float32)
        }
    
    def load_raw(self, subject_id: str) -> mne.io.Raw:
        """
        Load and concatenate all MAT files for a subject, return MNE Raw.
        
        Data formats in this dataset:
        - AD/MCI/normal1/normal2 subdirs: 'export' key, shape [samples x 19 channels]
        - CONTROL single files (xxxfir.mat): 'segmenty' key, shape [22-25 channels x samples]
          where first 19 channels are standard 10-20 EEG
        """
        info = self.info_map[subject_id]
        sfreq = info['sfreq']
        is_single_file = info.get('is_single_file', False)
        
        all_data = []
        
        for mat_file in sorted(info['files']):
            try:
                mat_data = sio.loadmat(mat_file)
                
                # Find the data array
                segment = None
                for key in ['export', 'segmenty']:
                    if key in mat_data:
                        segment = mat_data[key]
                        break
                
                if segment is None:
                    # Try to find any suitable 2D array
                    for key, val in mat_data.items():
                        if isinstance(val, np.ndarray) and val.ndim == 2 and not key.startswith('__'):
                            segment = val
                            break
                
                if segment is None or segment.ndim != 2:
                    continue
                
                # Handle different formats based on source type
                if is_single_file:
                    # CONTROL single files: shape is [channels x samples]
                    # First 19 channels are standard 10-20 EEG
                    n_ch = segment.shape[0]
                    n_samples = segment.shape[1]
                    
                    if n_ch >= 19 and n_samples > n_ch:
                        # Take first 19 channels, transpose to [samples x 19]
                        segment = segment[:19, :].T
                    else:
                        print(f"  Skipping {mat_file.name}: unexpected shape ({n_ch}, {n_samples})")
                        continue
                else:
                    # Subdirectory files (AD/MCI/normal): shape is [samples x channels]
                    n_samples = segment.shape[0]
                    n_ch = segment.shape[1]
                    
                    if n_ch >= 17 and n_ch <= 21:
                        # Standard format, pad or truncate to 19 channels
                        if n_ch < 19:
                            # Pad with zeros
                            padding = np.zeros((n_samples, 19 - n_ch))
                            segment = np.hstack([segment, padding])
                        elif n_ch > 19:
                            # Truncate
                            segment = segment[:, :19]
                    elif n_samples >= 17 and n_samples <= 25 and n_ch > n_samples:
                        # Might be transposed: [channels x samples]
                        segment = segment[:19, :].T if segment.shape[0] >= 19 else segment.T[:, :19]
                    else:
                        print(f"  Skipping {mat_file.name}: unexpected shape ({n_samples}, {n_ch})")
                        continue
                
                # Final validation: must be [samples x 19]
                if segment.shape[1] != 19:
                    print(f"  Skipping {mat_file.name}: final channel count {segment.shape[1]} != 19")
                    continue
                    
                # Check for valid data
                if np.isnan(segment).any() or np.all(segment == 0):
                    continue
                    
                all_data.append(segment)
                        
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
                continue
        
        if not all_data:
            raise FileNotFoundError(f"No valid data files for {subject_id}")
        
        # Concatenate all segments: [total_samples x 19]
        data = np.concatenate(all_data, axis=0)
        
        # Transpose to [channels x samples] for MNE
        data = data.T
        
        # Final validation
        if data.shape[0] != 19:
            raise ValueError(f"Expected 19 channels, got {data.shape[0]} for {subject_id}")
        
        # Create MNE Raw
        info_mne = mne.create_info(
            ch_names=CHANNEL_NAMES,
            sfreq=float(sfreq),
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info_mne, verbose=False)
        
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # Resample to Global SAMPLING_RATE
        if raw.info['sfreq'] != SAMPLING_RATE:
             print(f"  Resampling {subject_id} from {raw.info['sfreq']} to {SAMPLING_RATE} Hz")
             raw.resample(SAMPLING_RATE)

        # Robust scaling
        data = raw.get_data()
        scaler = RobustScaler(quantile_range=(25, 75))
        data_scaled = scaler.fit_transform(data.T).T
        
        # Handle NaNs/Infs
        if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
            data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        raw._data = data_scaled
        
        return raw


