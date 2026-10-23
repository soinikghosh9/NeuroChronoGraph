"""CAUEEG dataset loader (Chung-Ang University Hospital EEG; ~1.4k subjects).

Maps the multi-label `symptom` annotations to our 4-class taxonomy (0=AD,1=FTD,2=CN,3=MCI),
EXCLUDING non-taxonomy dementias (vascular, Parkinsonian, NPH, mixed) and ambiguous labels
(SMI, TGA). EDF signals are 19-channel 10-20 (average-referenced, "<ch>-AVG") at 200 Hz plus
EKG/Photic; channels are renamed/picked to the canonical 19 and resampled to SAMPLING_RATE.

Why this matters: CAUEEG adds a large INDEPENDENT cohort that (a) makes FTD multi-cohort
(OpenNeuro + CAUEEG) so AD-vs-FTD can be cross-site validated, (b) adds new-site MCI to break the
MCI/site confound, and (c) enables a strong leave-CAUEEG-out external generalization test.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import mne
from src.data.scaling import robust_scale   # centralised: see src/data/scaling.py

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

# 4-class assignment from the CAUEEG `symptom` token set (clinical priority).
_FTD = {'ftd', 'bvftd', 'non_fluent_aphasia', 'semantic_aphasia'}
_AD = {'ad', 'load', 'eoad'}
_NORMAL = {'normal', 'cb_normal', 'hc_normal'}
_EXCL_DEM = {'vd', 'sivd', 'ad_vd_mixed'}          # mixed/vascular -> drop even if AD/FTD co-occurs
_LABEL = {'AD': 0, 'FTD': 1, 'CN': 2, 'MCI': 3}


def assign_class(symptoms) -> str:
    """Return 'AD'|'FTD'|'CN'|'MCI'|'EXCL' for a CAUEEG symptom list."""
    s = set(symptoms or [])
    has_mci = any(str(t).startswith('mci') for t in s)
    if s & _FTD:
        return 'EXCL' if (s & _EXCL_DEM) else 'FTD'
    if (s & _AD) and not has_mci:                  # AD *dementia* (prodromal mci_ad stays MCI)
        return 'EXCL' if (s & _EXCL_DEM) else 'AD'
    if has_mci:
        return 'MCI'
    if s & _NORMAL:
        return 'CN'
    return 'EXCL'                                   # smi/tga/pure-vascular/PD/nph -> not in taxonomy


class CAUEEGDataset(BaseDataset):
    """Loader for the CAUEEG EDF corpus restricted to the AD/FTD/CN/MCI taxonomy."""

    def __init__(self, root_path: Path, verbose: bool = True):
        super().__init__(root_path)
        self.edf_dir = self.root_path / "signal" / "edf"
        ann = json.load(open(self.root_path / "annotation.json", encoding="utf-8"))
        self.info_map = {}
        counts = {'AD': 0, 'FTD': 0, 'CN': 0, 'MCI': 0, 'EXCL': 0}
        for rec in ann["data"]:
            cls = assign_class(rec.get("symptom"))
            counts[cls] += 1
            if cls == 'EXCL':
                continue
            serial = str(rec["serial"])
            if not (self.edf_dir / f"{serial}.edf").exists():
                continue
            sid = f"CAUEEG_{cls}_{serial}"          # person-level id; cohort_of -> CAUEEG
            self.info_map[sid] = {'serial': serial, 'class': cls,
                                  'label': _LABEL[cls], 'age': float(rec.get("age", -1))}
        self.subject_ids = list(self.info_map.keys())
        if verbose:
            print(f"Loaded CAUEEG with {len(self.subject_ids)} subjects (taxonomy AD/FTD/CN/MCI)")
            print(f"  - AD: {counts['AD']}, FTD: {counts['FTD']}, CN: {counts['CN']}, "
                  f"MCI: {counts['MCI']} | excluded(non-taxonomy): {counts['EXCL']}")

    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.info_map[subject_id]
        age = info['age']
        age_norm = float(np.clip((age - 40) / 50.0, 0, 1)) if age > 0 else 0.5
        # CAUEEG annotation has no sex/MMSE -> sex unknown (0), MMSE missing (-1 signals the conditioner)
        return {
            'subject_id': subject_id,
            'dataset': 'CAUEEG',
            'group': info['class'],
            'label': info['label'],
            'age': age,
            'sex': 'U',
            'mmse': -1.0,
            'metadata': np.array([age_norm, 0.0, -1.0], dtype=np.float32),
        }

    def _standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        # strip the "-AVG" reference suffix and fix uppercase-Z midline names
        fix = {'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'}
        rename = {}
        for ch in raw.ch_names:
            base = ch.replace('-AVG', '').replace('-REF', '').strip()
            rename[ch] = fix.get(base.upper(), base)
        raw.rename_channels(rename)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        keep = [ch for ch in CHANNEL_NAMES if ch in raw.ch_names]   # drops EKG/Photic
        raw.pick(keep)
        raw.reorder_channels([ch for ch in CHANNEL_NAMES if ch in raw.ch_names])
        return raw

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        serial = self.info_map[subject_id]['serial']
        edf = self.edf_dir / f"{serial}.edf"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
        raw = self._standardize_channels(raw)
        native = float(raw.info['sfreq'])
        print(f"loaded {subject_id}: native {native:g} Hz -> resample {SAMPLING_RATE} Hz")
        if native != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)
        try:
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch in CHANNEL_NAMES})
        except Exception:
            pass
        raw._data = robust_scale(raw.get_data())
        return raw

    def load_raw_info(self, subject_id: str) -> Dict:
        serial = self.info_map[subject_id]['serial']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(str(self.edf_dir / f"{serial}.edf"), preload=False, verbose="ERROR")
        n = raw.n_times
        if float(raw.info['sfreq']) != SAMPLING_RATE:
            n = int(n * SAMPLING_RATE / float(raw.info['sfreq']))
        return {'n_times': n, 'sfreq': SAMPLING_RATE}
