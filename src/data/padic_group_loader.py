"""P-ADIC loader for any diagnostic-group file (AD, controls, MCI) of the Dryad release.

Shor & Benninger et al., PLOS ONE 2021; Dryad doi:10.5061/dryad.8gtht76pw (CC0). Each group is one
MATLAB v7.3 (HDF5) file holding a struct array whose fields are cell arrays: ``G`` the EEG
(n_samples x 19), ``g`` the sampling rate, ``age``, ``sex`` (char code, 70='F', 77='M'), and for the
patient files ``eeg_files``, ``birth``, ``eegdate``. The original `padic_loader.py` handled only
``mci_c1_new.mat``, which is why the project recorded P-ADIC as single-class MCI.

Two layout quirks the loader handles:
  * the grid has several rows per subject (up to three); only one holds a real recording, the others are
    2-element placeholders. The longest valid recording per column is used, so a subject is never
    represented by a stub and the two sampling rates are never concatenated.
  * in the controls file each ``G`` cell is an object reference to a further object reference, so the
    dereference has to be applied until a 2-D dataset appears.

Acquisition (paper): Nihon Kohden, 19-electrode 10-20 montage, 500 Hz (some recordings 200 Hz),
routine clinical EEG between 08:00 and 13:00, resting with eyes open and closed (not separable),
50 Hz notch and 1 Hz high-pass already applied, two medical centres in Israel. Channel order is the
pipeline's CHANNEL_NAMES order (10-20); exp 224 re-verifies it per file with a posterior-alpha check.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import mne
import numpy as np

from ..config.config import CHANNEL_NAMES

_MIN_SAMPLES = 5000
GROUP_LABEL = {"AD": 0, "CN": 2, "MCI": 3}


class PADICGroupDataset:
    def __init__(self, mat_path: Path, group: str, prefix: str):
        self.mat_path = Path(mat_path)
        self.group, self.prefix = group, prefix
        with h5py.File(self.mat_path, "r") as h:
            keys = [k for k in h.keys() if not k.startswith("#")]
            self.struct = next(k for k in keys if isinstance(h[k], h5py.Group) and "G" in h[k])
            self.info_map = self._scan(h)
        self.subject_ids = list(self.info_map)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _deref(h, ref, depth: int = 4) -> Optional[h5py.Dataset]:
        """Follow object references until a 2-D 19-channel dataset appears."""
        obj = ref
        for _ in range(depth):
            try:
                d = h[obj] if not isinstance(obj, h5py.Dataset) else obj
            except Exception:                                            # noqa: BLE001
                return None
            if isinstance(d, h5py.Dataset) and d.ndim == 2 and 19 in d.shape and max(d.shape) >= _MIN_SAMPLES:
                return d
            if isinstance(d, h5py.Dataset) and d.dtype == object and d.size >= 1:
                obj = np.array(d[()]).ravel()[0]
                continue
            return None
        return None

    @staticmethod
    def _scalar(h, ref):
        try:
            v = np.array(h[ref][()]).ravel()
            return float(v[0]) if v.size else None
        except Exception:                                                # noqa: BLE001
            return None

    @staticmethod
    def _string(h, ref) -> str:
        try:
            d = np.array(h[ref][()]).ravel()
            return "".join(chr(int(v)) for v in d if 0 < int(v) < 128)
        except Exception:                                                # noqa: BLE001
            return ""

    def _scan(self, h) -> Dict:
        grp = h[self.struct]
        n_rows, n_cols = grp["G"].shape
        out = {}
        for c in range(n_cols):
            best = None
            for r in range(n_rows):
                d = self._deref(h, grp["G"][r, c])
                if d is None:
                    continue
                n = max(d.shape)
                fs = self._scalar(h, grp["g"][r, c]) or 500.0
                if best is None or n > best[0]:
                    best = (n, r, fs)
            if best is None:
                continue
            n, row, fs = best
            age = next((self._scalar(h, grp["age"][r, c]) for r in range(n_rows)
                        if (self._scalar(h, grp["age"][r, c]) or 0) > 0), None)
            sex_code = next((self._scalar(h, grp["sex"][r, c]) for r in range(n_rows)
                             if (self._scalar(h, grp["sex"][r, c]) or 0) > 0), None)
            sex = {70: "F", 77: "M"}.get(int(sex_code)) if sex_code else None
            fname = self._string(h, grp["eeg_files"][row, c]) if "eeg_files" in grp else ""
            m = re.search(r"[A-Z]{2,4}\d+", fname or "")
            sid = f"{self.prefix}_{m.group(0) if m else 'col'}_{c}"
            out[sid] = {"col": c, "row": row, "sfreq": fs, "n_samples": int(n), "fname": fname,
                        "group": self.group, "label": GROUP_LABEL[self.group], "age": age, "sex": sex}
        return out

    # ---------------------------------------------------------------- interface
    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, sid: str) -> Dict:
        i = self.info_map[sid]
        return {"subject_id": sid, "dataset": "P-ADIC", "group": self.group, "label": i["label"],
                "age": i["age"], "sex": i["sex"], "sfreq": i["sfreq"]}

    def load_raw(self, sid: str) -> mne.io.Raw:
        i = self.info_map[sid]
        with h5py.File(self.mat_path, "r") as h:
            d = self._deref(h, h[self.struct]["G"][i["row"], i["col"]])
            if d is None:
                raise ValueError(f"{sid}: no usable recording")
            G = np.array(d[()])
        if G.shape[0] == 19 and G.shape[1] != 19:
            G = G.T
        raw = mne.io.RawArray(G.T.astype(np.float64),
                              mne.create_info(list(CHANNEL_NAMES), float(i["sfreq"]), "eeg"), verbose=False)
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        return raw
