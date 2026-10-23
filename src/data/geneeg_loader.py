"""Loader for the GENEEG dataset (Mishra et al. 2025, "detecting MCI in the wild").

Source: https://huggingface.co/datasets/aamixsh/GENEEG  (gated, MIT)
Paper:  https://arxiv.org/abs/2501.17871

Real-world MCI vs Control EEG collected in general-practice clinics -- the external, in-the-wild
test set for our blind cross-domain MCI benchmark (train on CAUEEG, test blind on GENEEG). The
paper reports that CAUEEG-trained models detect 0.00 of GENEEG MCI patients, so this is a hard,
honest deployment-shift test that showcases NCG's calibration / abstention / OOD.

Structure (after huggingface download into datasets/GENEEG/):
    datasets/GENEEG/
        Control_raw_files/  ctrl_0001.eeg .. ctrl_0035.eeg  (+ .art, .evt)   35 Control
        MCI_raw_files/      Y_0001.eeg    .. Y_0028.eeg      (+ .art, .evt)   28 MCI

Raw .eeg: 17 space-separated columns (uV), 250 Hz native, ~245 s (4-min eyes-closed protocol).
Channels (17): FP1 FP2 F3 F4 F7 F8 C3 C4 P3 P4 O1 O2 T3 T4 FZ CZ PZ  -- i.e. our 19-channel
10-20 montage MINUS T5, T6, which we zero-fill so the tensor shape matches the pipeline.

Labels (project convention 0=AD,1=FTD,2=CN,3=MCI): GENEEG contributes CN + MCI only.
Demographics (age/sex/MMSE) are not distributed per subject -> standardized missing (mmse=-1).
The loader returns a 19-channel / SAMPLING_RATE Raw; EEGDatasetWrapper windows it identically to
every other cohort, so the only train/test difference is acquisition site, not preprocessing.
"""
import numpy as np
from pathlib import Path
from typing import Dict, List
import mne
from src.data.scaling import robust_scale   # centralised: see src/data/scaling.py

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

# GENEEG column order (uppercase), per the dataset README.
_GENEEG_CH = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4',
              'O1', 'O2', 'T3', 'T4', 'FZ', 'CZ', 'PZ']
_GENEEG_IDX = {name: i for i, name in enumerate(_GENEEG_CH)}
_NATIVE_SFREQ = 250.0  # raw acquisition rate (61,376 samples / 245 s)


class GENEEGDataset(BaseDataset):
    """GENEEG real-world MCI-vs-Control loader (17->19 ch, zero-fill T5/T6)."""

    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        self.info_map = self._scan_dataset()
        self.subject_ids = list(self.info_map.keys())
        counts = {"CONTROL": 0, "MCI": 0}
        for info in self.info_map.values():
            counts[info["group"]] += 1
        print(f"Loaded GENEEG Dataset with {len(self.subject_ids)} subjects "
              f"(CONTROL/CN: {counts['CONTROL']}, MCI: {counts['MCI']})")

    def _scan_dataset(self) -> Dict:
        info_map = {}
        for sub, group, label in [("Control_raw_files", "CONTROL", 2), ("MCI_raw_files", "MCI", 3)]:
            d = self.root_path / sub
            if not d.exists():
                continue
            for eeg in sorted(d.glob("*.eeg")):
                sid = f"GENEEG_{'CN' if label == 2 else 'MCI'}_{eeg.stem}"
                info_map[sid] = {"path": eeg, "group": group, "label": label}
        return info_map

    # ---- BaseDataset interface --------------------------------------------
    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.info_map[subject_id]
        # No per-subject demographics distributed -> standardized missing (mmse=-1 == masked in FiLM).
        return {
            "subject_id": subject_id,
            "dataset": "GENEEG",
            "group": info["group"],
            "label": info["label"],
            "age": 73.0,           # cohort mean (Mishra et al.); not per-subject
            "sex": "Unknown",
            "mmse": -1.0,
            "metadata": np.array([0.5, 0.5, -1.0], dtype=np.float32),  # age_norm, sex, mmse_norm
        }

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        info = self.info_map[subject_id]
        arr = np.loadtxt(info["path"], dtype=np.float64)          # [T, 17]
        if arr.ndim != 2 or arr.shape[1] != len(_GENEEG_CH):
            raise ValueError(f"{subject_id}: expected [T,17] .eeg, got {arr.shape}")
        T = arr.shape[0]

        # Map the 17 GENEEG channels into our 19-channel order; T5/T6 stay zero.
        data = np.zeros((len(CHANNEL_NAMES), T), dtype=np.float64)
        for i, ch in enumerate(CHANNEL_NAMES):
            gi = _GENEEG_IDX.get(ch.upper())
            if gi is not None:
                data[i] = arr[:, gi]

        raw = mne.io.RawArray(
            data * 1e-6,                                          # uV -> V for MNE
            mne.create_info(list(CHANNEL_NAMES), sfreq=_NATIVE_SFREQ, ch_types="eeg"),
            verbose="ERROR",
        )
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

        print(f"loaded {subject_id}: native {_NATIVE_SFREQ:g} Hz -> resample {SAMPLING_RATE} Hz")
        if _NATIVE_SFREQ != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)

        # Robust per-channel scaling (matches misp/mci/mendeley loaders). Zero channels (T5/T6)
        # have IQR 0 -> RobustScaler yields NaN -> nan_to_num restores 0, preserving the zero-fill.
        raw._data = robust_scale(raw.get_data())
        return raw
