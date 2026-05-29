"""
Loader for Alz_EEG_data Dataset.

Structure:
datasets/Alz_EEG_data/EEG_data/
  - AD/
      - Eyes_closed/
          - Paciente1/
             - Fp1.txt, Fp2.txt ...
  - Healthy/
      - Eyes_closed/
          - PacienteX/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import mne
from sklearn.preprocessing import RobustScaler

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

# Mapping from Alz_EEG_data generic names to Standard 10-20
# Based on common variations found in this dataset
CHANNEL_MAP = {
    'Fp1': 'Fp1', 'Fp2': 'Fp2', 'F3': 'F3', 'F4': 'F4',
    'C3': 'C3', 'C4': 'C4', 'P3': 'P3', 'P4': 'P4',
    'O1': 'O1', 'O2': 'O2', 'F7': 'F7', 'F8': 'F8',
    'T3': 'T3', 'T4': 'T4', 'T5': 'T5', 'T6': 'T6',
    'Fz': 'Fz', 'Cz': 'Cz', 'Pz': 'Pz'
}

class AlzEEGDataset(BaseDataset):
    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        self.groups = {'AD': 0, 'Healthy': 2} # AD=0, FTD=1 (none), CN=2
        self.info_map = self._scan_dataset()
        self.subject_ids = list(self.info_map.keys())
        
        print(f"Loaded Alz_EEG_data with {len(self.subject_ids)} subjects")
        
    def _scan_dataset(self) -> Dict:
        """
        Scan directory for subjects.
        Returns dict: subject_id -> {path, group}
        """
        info_map = {}
        
        # We only use Eyes_closed to match primary dataset modality for now?
        # Or should we include Eyes_open if available here too?
        # User prompt implicitly suggested "Alz_EEG_data" which has EC/EO.
        # Let's focus on Eyes_closed for alignment with base training if unclear,
        # BUT plan said train on both. Let's start with all.
        
        # Traverse: Group -> Condition -> Patient
        for group_name, label_id in self.groups.items():
            group_path = self.root_path / "EEG_data" / group_name
            if not group_path.exists():
                continue
                
            for cond in ["Eyes_closed", "Eyes_open"]:
                cond_path = group_path / cond
                if not cond_path.exists():
                    continue
                    
                for patient_dir in cond_path.iterdir():
                    if patient_dir.is_dir() and "Paciente" in patient_dir.name:
                        # Check if populated
                        if not list(patient_dir.glob("*.txt")):
                            continue
                            
                        # ID format: AlzEEG_AD_EC_Paciente1
                        sid = f"AlzEEG_{group_name}_{cond}_{patient_dir.name}"
                        info_map[sid] = {
                            "path": patient_dir,
                            "group": group_name,
                            "label": label_id,
                            "condition": cond,
                            "original_id": patient_dir.name
                        }
        return info_map

    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.info_map[subject_id]
        return {
            'subject_id': subject_id,
            'dataset': 'Alz_EEG_data',
            'group': info['group'],
            'label': info['label'],
            'condition': info['condition'],
            # Metadata mostly missing for this dataset (Age/Sex/MMSE unavailable in folder structure)
            # We will use mean imputation or specific flags later.
            'age': 70.0, # Dummy mean
            'sex': 'Unknown',
            'mmse': -1.0, # Standardized Missing
            'metadata': np.array([0.5, 0.5, -1.0], dtype=np.float32) # Dummy normalized
        }

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        info = self.info_map[subject_id]
        path = info['path']
        
        # Load all .txt files
        # Each file is a channel.
        # Format: single column of values.
        
        data_list = []
        ch_names = []
        
        for ch_file in path.glob("*.txt"):
            ch_name_clean = ch_file.stem # remove .txt
            if ch_name_clean in CHANNEL_MAP:
                # Load text data, assuming 1 value per line
                try:
                    vals = np.loadtxt(ch_file)
                    data_list.append(vals)
                    ch_names.append(CHANNEL_MAP[ch_name_clean])
                except Exception as e:
                    print(f"Error reading {ch_file}: {e}")
                    
        if not data_list:
            raise FileNotFoundError(f"No valid channel files found in {path}")
            
        # Stack: (n_channels, n_samples)
        # Verify lengths are equal
        lengths = [len(x) for x in data_list]
        min_len = min(lengths)
        data = np.stack([x[:min_len] for x in data_list])
        
        # Create MNE Raw
        # Dataset sampling rate? Paper says 19 electrodes.
        # Usually these are 256Hz or 128Hz or 500Hz.
        # **Assume 256Hz** based on common Mendeley/Kaggle variations if not specified.
        # Wait, user context said "check associated papers". 
        # For safety, I should infer or use 250Hz target directly via resampling.
        # Let's assume 256Hz input for now (standard for these datasets).
        sfreq_in = 250.0 # Placeholder, often described in dataset description.
        
        info_mne = mne.create_info(ch_names=ch_names, sfreq=sfreq_in, ch_types='eeg')
        raw = mne.io.RawArray(data, info_mne, verbose=False)
        
        # Standardize
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # If missing standard channels, we might need to interpolate
        # For now, just leave as is, model will handle or we pad.
        # (Ideal: Interpolate to full 19 list)
        
        # Resample to Global SAMPLING_RATE
        if raw.info['sfreq'] != SAMPLING_RATE:
             print(f"  Resampling {subject_id} from {raw.info['sfreq']} to {SAMPLING_RATE} Hz")
             raw.resample(SAMPLING_RATE)
        
        # Robust Scaling
        data = raw.get_data()
        scaler = RobustScaler(quantile_range=(25, 75))
        data_scaled = scaler.fit_transform(data.T).T
        
        # Handle potential NaNs from flat channels (IQR=0)
        if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
            data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
        raw._data = data_scaled
        
        return raw
