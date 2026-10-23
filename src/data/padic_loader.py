"""
Loader for the P-ADIC MCI cohort (Shor & Benninger, PLOS ONE 2021; Dryad 10.5061/dryad.8gtht76pw).

Source file: datasets/P-ADIC/p-adic mci/mci_c1_new.mat  (MATLAB v7.3 / HDF5, ~2.5 GB).
Structure: struct `mci_r` of shape (2 rows x 28 cols) = 28 MCI subjects, up to 2 recordings each.
  - field `G`  -> EEG signal, dataset [n_samples x 19]   (19 channels)
  - field `g`  -> sampling rate for that recording (500 Hz for most, 200 Hz for a couple)
  - field `eeg_files` -> original filename (e.g. 'DEM66_FC0900YQ.EEG'); 'DEM##' identifies the subject
  Empty cells (subjects with a single recording) carry placeholder (2,)/'' values and are skipped.

Acquisition (paper): Nihon Kohden, 19-electrode international 10-20, 500 Hz, resting (eyes open+closed),
50 Hz notch + 1 Hz high-pass already applied. ~25 clinically-defined MCI subjects.

Channel order: NOT labelled in the file or the paper, but CONFIRMED data-driven (resting posterior alpha
peaks at O1/O2/T5/T6/P3; frontal Fp1/Fp2 carry max delta/EOG) to be the standard 10-20 order, i.e. the
pipeline's CHANNEL_NAMES order -- so the 19 columns map 1:1 with no reordering.

Labels (project convention): 0=AD, 1=FTD, 2=CN, 3=MCI. P-ADIC contributes MCI only.

Caveats: recordings mix eyes-open + eyes-closed (not separable here) -> the full resting segment is used;
P-ADIC is a distinct site -> a domain-shift cohort (monitor per-cohort performance).
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List
import h5py
import mne
from src.data.scaling import robust_scale   # centralised: see src/data/scaling.py

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

_MIN_SAMPLES = 5000  # a valid recording must be longer than this (placeholders are tiny)


class PADICDataset(BaseDataset):
    """Loader for the P-ADIC MCI cohort (MATLAB v7.3 HDF5)."""

    def __init__(self, root_path: Path):
        super().__init__(root_path)
        self.root_path = Path(root_path)
        self.mat_path = self._find_mat()
        self.info_map = self._scan_dataset()
        self.subject_ids = list(self.info_map.keys())
        print(f"Loaded P-ADIC Dataset with {len(self.subject_ids)} subjects (MCI)")

    def _find_mat(self) -> Path:
        cands = list(self.root_path.rglob("mci_c1_new.mat")) or list(self.root_path.rglob("*.mat"))
        if not cands:
            raise FileNotFoundError(f"No P-ADIC .mat under {self.root_path}")
        return cands[0]

    @staticmethod
    def _demid(fname: str, col: int) -> str:
        m = re.search(r"DEM\d+", str(fname))
        return m.group(0) if m else f"col{col}"

    def _scan_dataset(self) -> Dict:
        """One subject per column; pick that subject's LONGEST valid recording (avoids mixed-Fs concat)."""
        info_map = {}
        with h5py.File(self.mat_path, "r") as h:
            g = h["mci_r"]
            n_rows, n_cols = g["G"].shape
            for c in range(n_cols):
                best = None  # (n_samples, row, fs, fname)
                demid = None
                for r in range(n_rows):
                    try:
                        G = h[g["G"][r, c]]
                        if not isinstance(G, h5py.Dataset) or G.ndim != 2 or 19 not in G.shape:
                            continue
                        n_samp = max(G.shape)
                        if n_samp < _MIN_SAMPLES:
                            continue
                        fs = float(np.array(h[g["g"][r, c]]).ravel()[0])
                        fname = self._decode_str(h, g["eeg_files"][r, c])
                        if demid is None and fname:
                            demid = self._demid(fname, c)
                        if best is None or n_samp > best[0]:
                            best = (n_samp, r, fs, fname)
                    except Exception:
                        continue
                if best is None:
                    continue
                _, row, fs, fname = best
                if demid is None:
                    demid = f"col{c}"
                sid = f"PADIC_MCI_{demid}_{c}"
                info_map[sid] = {"col": c, "row": row, "sfreq": fs, "fname": fname,
                                 "group": "MCI", "label": 3}
        return info_map

    @staticmethod
    def _decode_str(h, ref) -> str:
        try:
            d = np.array(h[ref][()]).ravel()
            return "".join(chr(int(v)) for v in d if 0 < int(v) < 128)
        except Exception:
            return ""

    # ---- BaseDataset interface --------------------------------------------
    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        info = self.info_map[subject_id]
        return {
            "subject_id": subject_id,
            "dataset": "P-ADIC",
            "group": info["group"],
            "label": info["label"],
            "age": 70.0,
            "sex": "Unknown",
            "mmse": -1.0,
            "metadata": np.array([0.5, 0.5, -1.0], dtype=np.float32),  # age_norm, sex, mmse_norm
        }

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        info = self.info_map[subject_id]
        with h5py.File(self.mat_path, "r") as h:
            g = h["mci_r"]
            G = np.array(h[g["G"][info["row"], info["col"]]][()])  # [n_samples x 19]
        if G.ndim != 2:
            raise ValueError(f"{subject_id}: unexpected EEG shape {G.shape}")
        if G.shape[0] == 19 and G.shape[1] != 19:
            G = G.T
        if G.shape[1] != 19:
            raise ValueError(f"{subject_id}: expected 19 channels, got {G.shape}")
        data = G.T.astype(np.float64)  # -> [19 x n_samples], already in CHANNEL_NAMES order

        info_mne = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=float(info["sfreq"]), ch_types="eeg")
        raw = mne.io.RawArray(data, info_mne, verbose=False)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

        native = float(raw.info["sfreq"])
        print(f"loaded {subject_id}: native {native:g} Hz -> resample {SAMPLING_RATE} Hz")
        if native != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)

        raw._data = robust_scale(raw.get_data())
        return raw
