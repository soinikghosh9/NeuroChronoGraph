"""BrainLat resting-state EEG loader (candidate third external site).

BrainLat (Prado et al., Sci Data 2023; Synapse syn51549340) records AD, bvFTD, HC (and PD, MS) on a
128-electrode Biosemi montage across five Latin-American sites. This loader exposes the resting-state
recordings in the same interface as the P-ADIC loader, and subsamples the 128 Biosemi electrodes to the
pipeline's 19-channel 10-20 set by nearest electrode position, so the harmonisation pipeline can treat it
like every other cohort.

The download is spread over many arbitrarily-named folders, files split into ``.set`` + ``.fdt`` that may
live in different folders, and both single-file (embedded) and split recordings; the loader scans the whole
tree recursively, keeps one resting recording per subject preferring a copy whose data is present, and
stages ``.set`` + ``.fdt`` together before reading. ``organize`` writes a clean class-wise copy.

Only resting recordings (``rs`` / ``rs-HEP``) of the target classes (CN, AD, FTD) are used; task
(exteroception/interoception) files and the PD and MS cohorts are ignored. Diagnosis, age and sex come from
the Demographics CSVs.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from src.config.config import CHANNEL_NAMES  # 19-channel 10-20 order the pipeline expects

mne.set_log_level("ERROR")

_ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}   # CAUEEG naming -> 10-20 position lookup
_TARGET_CLASSES = ("CN", "AD", "FTD")
_EMBED_BYTES = 20_000_000                                    # a .set larger than this embeds its data


def _paradigm(name: str) -> str:
    n = name.lower()
    if "pd_reject" in n or "_pd_" in n:
        return "PD"
    if "rs-hep" in n:
        return "rs-HEP"
    if "_rs_" in n or "_rs-" in n or "rs_eeg" in n:
        return "rs"
    if "extero" in n:
        return "extero"
    if "intero" in n:
        return "intero"
    return "other"


def _dx_from_id(sid: str) -> str:
    m = re.search(r"sub-(\d+)", sid or "")
    if not m:
        return "?"
    return {"10": "CN", "20": "FTD", "30": "AD", "40": "PD"}.get(m.group(1)[:2], "?")


def biosemi128_to_1020() -> Dict[str, str]:
    """Map each of the 19 target 10-20 labels to its nearest Biosemi-128 electrode by 3D position."""
    bio = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    ten = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    bio_names = list(bio)
    bio_xyz = np.array([bio[n] for n in bio_names])
    out = {}
    for target in CHANNEL_NAMES:
        pos = ten.get(_ALIAS.get(target, target))
        if pos is None:
            continue
        j = int(np.argmin(np.linalg.norm(bio_xyz - np.asarray(pos), axis=1)))
        out[target] = bio_names[j]
    return out


class BrainLatLoader:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.demo = self._load_demographics()
        self.fdt: Dict[str, Path] = {}
        for p in self.root.rglob("*.fdt"):
            self.fdt.setdefault(p.stem, p)
        # one resting .set per subject, preferring a copy whose data is present
        self.sets: Dict[str, Tuple[Path, bool]] = {}
        for p in self.root.rglob("*.set"):
            if _paradigm(p.name) not in ("rs", "rs-HEP"):
                continue
            m = re.search(r"(sub-\d+)", p.name)
            if not m:
                continue
            sid = m.group(1)
            has = (p.stem in self.fdt) or p.stat().st_size > _EMBED_BYTES
            cur = self.sets.get(sid)
            if cur is None or (has and not cur[1]):
                self.sets[sid] = (p, has)
        self.map1020 = biosemi128_to_1020()

    def _load_demographics(self) -> Dict[str, Dict]:
        demo = {}
        for c in self.root.glob("**/Demographics_*_EEG_data.csv"):
            for row in csv.DictReader(open(c, encoding="latin-1")):
                k = {(kk.strip().lower() if kk else "_"): vv for kk, vv in row.items()}
                sid = (k.get("id eeg") or k.get("id_eeg") or "").strip()
                if sid:
                    demo[sid] = {"diagnosis": k.get("diagnosis", "").strip(),
                                 "age": k.get("age", ""), "sex": k.get("sex", "")}
        return demo

    def _dx(self, sid: str) -> str:
        return self.demo.get(sid, {}).get("diagnosis") or _dx_from_id(sid)

    def _has_data(self, sid: str) -> bool:
        return self.sets[sid][1]

    def get_subject_ids(self, group: Optional[str] = None, with_data: bool = True) -> List[str]:
        ids = [s for s in self.sets if self._dx(s) in _TARGET_CLASSES and (not with_data or self._has_data(s))]
        if group:
            ids = [s for s in ids if self._dx(s) == group]
        return sorted(ids)

    def get_subject_info(self, sid: str) -> Dict:
        d = self.demo.get(sid, {})
        return {"subject_id": sid, "dataset": "BrainLat", "label": self._dx(sid),
                "age": d.get("age", ""), "sex": d.get("sex", ""), "has_data": self._has_data(sid)}

    def load_raw(self, sid: str) -> mne.io.Raw:
        """Return a Raw restricted to the 19 pipeline channels (10-20 names), montage set."""
        setp = self.sets[sid][0]
        if setp.stem in self.fdt and not (setp.parent / f"{setp.stem}.fdt").exists():
            tmp = Path(tempfile.mkdtemp(prefix="brainlat_"))
            shutil.copy2(setp, tmp / setp.name)
            shutil.copy2(self.fdt[setp.stem], tmp / f"{setp.stem}.fdt")
            setp = tmp / setp.name
        raw = mne.io.read_raw_eeglab(str(setp), preload=True)
        pick = {bio: tgt for tgt, bio in self.map1020.items() if bio in raw.ch_names}
        raw.pick(list(pick))
        raw.rename_channels(pick)
        raw.reorder_channels([c for c in CHANNEL_NAMES if c in raw.ch_names])
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        return raw

    def scan_availability(self) -> Dict:
        rep = {"present": {}, "missing": {}}
        for cls in _TARGET_CLASSES:
            rep["present"][cls] = sorted(s for s in self.sets if self._dx(s) == cls and self._has_data(s))
            rep["missing"][cls] = sorted(s for s in self.sets if self._dx(s) == cls and not self._has_data(s))
        return rep

    def organize(self, dest: str | Path, link: bool = True) -> Dict:
        """Write a clean class-wise copy: dest/<class>/<subject>.set (+ .fdt if split) + manifest.csv.

        Uses hardlinks by default (instant, no extra disk on the same volume); falls back to copy.
        """
        dest = Path(dest)
        man_rows = []
        counts = {c: 0 for c in _TARGET_CLASSES}
        for sid in self.get_subject_ids():
            cls = self._dx(sid)
            outdir = dest / cls
            outdir.mkdir(parents=True, exist_ok=True)
            setp = self.sets[sid][0]
            targets = [setp]
            if setp.stem in self.fdt:                       # split recording: bring its data along
                targets.append(self.fdt[setp.stem])
            placed = []
            for src in targets:
                dst = outdir / src.name
                if dst.exists():
                    dst.unlink()
                try:
                    os.link(src, dst) if link else shutil.copy2(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                placed.append(dst.name)
            info = self.get_subject_info(sid)
            man_rows.append([sid, cls, info["age"], info["sex"], placed[0],
                             placed[1] if len(placed) > 1 else "", str(setp.parent)])
            counts[cls] += 1
        with open(dest / "manifest.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "class", "age", "sex", "set_file", "fdt_file", "source_folder"])
            w.writerows(sorted(man_rows, key=lambda r: (r[1], r[0])))
        return {"dest": str(dest), "counts": counts, "manifest": str(dest / "manifest.csv")}


if __name__ == "__main__":
    import sys
    L = BrainLatLoader(r"C:\Users\user\Downloads\Alzfiles\datasets\BrainLat Data")
    av = L.scan_availability()
    for cls in ("AD", "FTD", "CN"):
        print(f"{cls}: {len(av['present'][cls])} with data, {len(av['missing'][cls])} missing")
    if av["missing"]["CN"]:
        print("  missing CN:", av["missing"]["CN"])
    if "--organize" in sys.argv:
        res = L.organize(L.root / "organized")
        print("\norganized ->", res["dest"], res["counts"])
        sid = L.get_subject_ids(group="CN")[0]
        raw = L.load_raw(sid)
        print(f"loaded CN {sid}: {len(raw.ch_names)} ch, sfreq={raw.info['sfreq']}, dur={raw.n_times/raw.info['sfreq']:.0f}s")
