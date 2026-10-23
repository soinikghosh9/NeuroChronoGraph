"""
Loader for the MISP / Isfahan EEG dataset (Kashefpoor et al. 2016 & 2019).

Source: Medical Image and Signal Processing Research Center, Isfahan
        https://misp.mui.ac.ir/en/eeg-data-0

Structure (after extracting the two Google-Drive zips into datasets/MISP/):
    datasets/MISP/
        set1/  1.edf .. 27.edf  + states.xlsx   (Kashefpoor 2016: 16 NORMAL + 11 MCI)
        set2/  1.edf .. 34.edf  + States.xlsx   (Kashefpoor 2019: 16 NORMAL + 18 MCI)

Each .edf is a ~30 min eyes-closed resting recording, 19 channels (exact 10-20 set),
256 Hz native (matches the pipeline SAMPLING_RATE, so no resampling). The .xlsx maps
'file number' -> 'status' in {NORMAL, MCI}.

Labels (project convention): 0=AD, 1=FTD, 2=CN, 3=MCI. MISP contributes CN + MCI only.

CAVEAT (subject overlap): it is not documented whether set1 (2016) and set2 (2019) share
people. They are given distinct subject ids here. If the two sets are later confirmed to
overlap, restrict to one set via the `sets` argument to avoid cross-fold subject leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import mne
from src.data.scaling import robust_scale   # centralised: see src/data/scaling.py

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

_STATUS_TO_GROUP = {"NORMAL": ("CONTROL", 2), "MCI": ("MCI", 3)}


class MISPDataset(BaseDataset):
    """Loader for the MISP/Isfahan resting-state EEG dataset (NORMAL vs MCI)."""

    def __init__(self, root_path: Path, sets=("set1", "set2"), mci_only: bool = True):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        self.sets = tuple(sets)
        # mci_only (default True): include ONLY MISP MCI subjects, not the NORMAL group.
        # Rationale (per-cohort hold-out 2026-06-12): MISP NORMAL are memory-clinic "normals"
        # (borderline) that the model reads as MCI, tanking the hold-out (subj acc 0.44, kappa 0.12);
        # MISP MCI classifies correctly and fixed the staging collapse. We have ample CN elsewhere
        # (158), so keep MISP's valuable MCI and drop its non-comparable controls.
        self.mci_only = mci_only
        self.info_map = self._scan_dataset()
        self.subject_ids = list(self.info_map.keys())

        counts = {"CONTROL": 0, "MCI": 0}
        for info in self.info_map.values():
            counts[info["group"]] += 1
        print(f"Loaded MISP Dataset with {len(self.subject_ids)} subjects "
              f"(sets={list(self.sets)})")
        print(f"  - CONTROL(CN): {counts['CONTROL']}, MCI: {counts['MCI']}")

    # ---- label table -------------------------------------------------------
    @staticmethod
    def _read_labels(set_dir: Path) -> Dict[int, str]:
        """Return {file_number -> STATUS} from the set's xlsx (case-insensitive filename)."""
        xlsx = None
        for cand in set_dir.glob("*.xlsx"):
            xlsx = cand
            break
        if xlsx is None:
            return {}
        df = pd.read_excel(xlsx)
        # normalise column names
        cols = {c.lower().strip(): c for c in df.columns}
        fcol = cols.get("file number") or cols.get("file") or list(df.columns)[0]
        scol = cols.get("status") or list(df.columns)[1]
        out = {}
        for _, row in df.iterrows():
            try:
                out[int(row[fcol])] = str(row[scol]).strip().upper()
            except (ValueError, TypeError):
                continue
        return out

    def _scan_dataset(self) -> Dict:
        info_map = {}
        for set_name in self.sets:
            set_dir = self.root_path / set_name
            if not set_dir.exists():
                continue
            labels = self._read_labels(set_dir)
            for edf in sorted(set_dir.glob("*.edf"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
                try:
                    fnum = int(edf.stem)
                except ValueError:
                    continue
                status = labels.get(fnum)
                if status not in _STATUS_TO_GROUP:
                    continue  # skip files with no/unknown label
                if self.mci_only and status != "MCI":
                    continue  # drop MISP NORMAL (non-comparable controls); keep MCI only
                group, label = _STATUS_TO_GROUP[status]
                # Globally-unique, person-level id (one recording == one person here).
                sid = f"MISP_{set_name}_{status}_{fnum:02d}"
                info_map[sid] = {
                    "path": edf,
                    "group": group,
                    "label": label,
                    "set": set_name,
                }
        return info_map

    # ---- BaseDataset interface --------------------------------------------
    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.info_map[subject_id]
        # No age/sex/MMSE provided -> standardized missing (mirrors mci_loader; -1.0 == missing).
        return {
            "subject_id": subject_id,
            "dataset": "MISP",
            "group": info["group"],
            "label": info["label"],
            "age": 70.0,
            "sex": "Unknown",
            "mmse": -1.0,
            "metadata": np.array([0.5, 0.5, -1.0], dtype=np.float32),  # age_norm, sex, mmse_norm
        }

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        info = self.info_map[subject_id]
        raw = mne.io.read_raw_edf(info["path"], preload=True, verbose="ERROR")

        # Keep exactly the pipeline's 19 channels, in CHANNEL_NAMES order.
        present = [c for c in CHANNEL_NAMES if c in raw.ch_names]
        if len(present) != len(CHANNEL_NAMES):
            missing = [c for c in CHANNEL_NAMES if c not in raw.ch_names]
            raise ValueError(f"{subject_id}: missing channels {missing}; have {raw.ch_names}")
        raw.pick(present)
        raw.reorder_channels(CHANNEL_NAMES)

        # Montage
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

        # Resample to the global rate (MISP is 256 Hz native -> usually a no-op). (AUDIT T4.7)
        native_sfreq = float(raw.info["sfreq"])
        print(f"loaded {subject_id}: native {native_sfreq:g} Hz -> resample {SAMPLING_RATE} Hz")
        if native_sfreq != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)

        # Robust per-channel scaling (matches mci_loader/mendeley_loader conditioning).
        raw._data = robust_scale(raw.get_data())
        return raw
