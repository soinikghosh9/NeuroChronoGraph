"""Lever 1: transfer-robust extended features, appended to the cached 12 node features.

Adds four per-electrode descriptors designed to survive site shift, computed from the same harmonised
recordings and the exact windows the node12 cache used (matched by subject + start sample):

    12  IAF-referenced theta relative power   power in [IAF-4, IAF-2] / total (band follows the subject's
                                              own alpha peak, removing age/site variation in peak location)
    13  IAF-referenced alpha relative power   power in [IAF-2, IAF+2] / total
    14  aperiodic offset                      intercept of the log-log 1/f fit (2-40 Hz, 7-14 excluded);
                                              the periodic/aperiodic split that band ratios conflate
    15  alpha inter-hemispheric asymmetry     log(alpha_c / alpha_homolog_c), 0 on the midline

Writes node_ext4.npy (N,19,4) beside node12.npy in each feature tag; 234 concatenates to 16 when
--extended. Gain-invariant features (12,13,15) are unaffected by scaling; the offset (14) is computed on the
same per-channel robust-scaled window as node12, so it is comparable across cohorts.

    python experiments/235_extract_extended.py
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from src.config.config import CHANNEL_NAMES
from src.data.scaling import robust_scale

spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
FEAT, HARM, ALZ = C.FEAT, C.ALZ / "harmonised", C.ALZ
SFREQ, WINDOW, TRUNC = 256.0, 1024, int(235.0 * 256)

# homologous electrode index for alpha asymmetry (midline maps to itself)
_PAIRS = {"Fp1": "Fp2", "F3": "F4", "C3": "C4", "P3": "P4", "O1": "O2", "F7": "F8",
          "T3": "T4", "T5": "T6", "Fz": "Fz", "Cz": "Cz", "Pz": "Pz"}
HOMOLOG = []
for ch in CHANNEL_NAMES:
    h = _PAIRS.get(ch) or {v: k for k, v in _PAIRS.items()}.get(ch, ch)
    HOMOLOG.append(CHANNEL_NAMES.index(h))
HOMOLOG = np.array(HOMOLOG)

TAG_HARM_PREFIX = {"channel_clean": None, "padic_channel_clean": "PADIC_", "brainlat_channel_clean": "BRAINLAT_"}


def extras_for_window(w):
    """w: (19, 1024) robust-scaled. Returns (19, 4)."""
    f, psd = welch(w, fs=SFREQ, nperseg=512, axis=-1)                 # (19, F)
    total = psd.sum(-1) + 1e-12
    out = np.zeros((19, 4), np.float32)
    # alpha peak per channel (6-14 Hz), parabolic
    band = (f >= 6) & (f <= 14); fb = f[band]; pb = np.log(psd[:, band] + 1e-12)
    k = pb.argmax(1).clip(1, pb.shape[1] - 2)
    y0 = pb[np.arange(19), k - 1]; y1 = pb[np.arange(19), k]; y2 = pb[np.arange(19), k + 1]
    den = (y0 - 2 * y1 + y2)
    off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0).clip(-1, 1)
    pf = fb[k] + off * (f[1] - f[0])                                  # (19,)
    for c in range(19):
        tlo, thi = pf[c] - 4, pf[c] - 2; alo, ahi = pf[c] - 2, pf[c] + 2
        out[c, 0] = psd[c, (f >= max(tlo, 0.5)) & (f < thi)].sum() / total[c]
        out[c, 1] = psd[c, (f >= max(alo, 0.5)) & (f < ahi)].sum() / total[c]
    # aperiodic offset: intercept of log10 psd vs log10 f (2-40, excl 7-14)
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14)) & (f > 0)
    lf = np.log10(f[fit]); Y = np.log10(psd[:, fit] + 1e-12)
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, Y.T, rcond=None)                    # (2, 19): slope, intercept
    out[:, 2] = (coef[1] + 12.0)                                      # intercept, shifted to ~O(1)
    # alpha asymmetry
    apow = psd[:, (f >= 8) & (f < 13)].sum(-1) + 1e-12
    out[:, 3] = np.log(apow / apow[HOMOLOG])
    return out


def window_starts(n, sid):
    if n < WINDOW:
        return []
    starts = list(range(0, n - WINDOW + 1, WINDOW))
    if len(starts) > 50:
        seed = int(hashlib.md5(str(sid).encode()).hexdigest()[:8], 16)
        starts = sorted(np.random.default_rng(seed).choice(starts, 50, replace=False))
    return starts


def harm_path(tag, sid):
    if tag == "channel_clean":
        return HARM / f"{sid}.npy"       # CAUEEG_*, sub-*, GENEEG_*, etc. stored by raw id
    return HARM / f"{sid}.npy"           # PADIC_*/BRAINLAT_* already prefixed in id


def process(tag):
    meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    groups = np.array([str(x) for x in meta["groups"]]); starts = np.array(meta["start"])
    node12 = np.load(FEAT / tag / "node12.npy", mmap_mode="r")
    ext = np.zeros((len(groups), 19, 4), np.float32)
    cache = {}
    done = 0
    for sid in np.unique(groups):
        p = harm_path(tag, sid)
        if not p.exists():
            print(f"  [miss] {sid}"); continue
        data = np.load(p)[:, :TRUNC]
        rows = np.flatnonzero(groups == sid)
        for r in rows:
            s = int(starts[r]); w = data[:, s:s + WINDOW]
            if w.shape[1] < WINDOW:
                continue
            ext[r] = extras_for_window(robust_scale(w, "channel"))
        done += 1
        if done % 200 == 0:
            print(f"  {tag}: {done} subjects", flush=True)
    np.save(FEAT / tag / "node_ext4.npy", ext)
    print(f"wrote {FEAT / tag / 'node_ext4.npy'}  shape {ext.shape}  (node12 {node12.shape})", flush=True)


if __name__ == "__main__":
    import sys
    tags = sys.argv[1:] or ["channel_clean", "padic_channel_clean", "brainlat_channel_clean"]
    for t in tags:
        print(f"=== {t} ===", flush=True); process(t)
