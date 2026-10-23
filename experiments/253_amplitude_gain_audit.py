"""Amplitude (gain) audit for every cohort, and the BrainLat export-variant check.

BrainLat releases controls as `*_rs_eeg.set` and every AD/FTD recording as `*_rs-HEP_eeg.set`; the two
export variants differ by a near-constant gain (patients/controls amplitude ratio 3.5-4.3 at every frequency
from 1 to 40 Hz), so diagnosis and export gain coincide. This script
  1. measures subject broadband RMS (uV, matched windows, CAR) and its case-control AUC at every cohort,
  2. measures the patients/controls amplitude ratio per frequency at BrainLat (flat = gain, not physiology),
  3. checks that the per-window per-channel robust scaling applied before every canonical feature is exactly
     invariant to a constant gain (features recomputed on BrainLat windows multiplied by 4),
  4. lists which scored arms consume physical-unit input (gain-sensitive).
-> outputs/results/canonical/amplitude_gain_audit.json

    python experiments/253_amplitude_gain_audit.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("f216", ROOT / "experiments" / "216_fm_matched.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
C = F.C


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y, s, b=2000, seed=20261003):
    y = np.asarray(y); s = np.asarray(s, float); rng = np.random.default_rng(seed); out = []
    for _ in range(b):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(auc(y[i], s[i]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def subject_rms(tag, cohorts):
    from src.config.config import CHANNEL_NAMES
    m = F.meta_for(tag)
    idx, X = F.load_windows(m, list(CHANNEL_NAMES), cohorts)
    g = m["groups"][idx]; lab = np.array([C.CLASS[str(x).upper()] for x in m["labels"][idx]])
    coh = np.array([str(c) for c in m["cohort"][idx]])
    out = {}
    for s in np.unique(g):
        k = g == s
        out[s] = (coh[k][0], lab[k][0], float(np.sqrt(np.median((X[k] ** 2).mean(-1)))))
    return out, X, g, lab


def main():
    rep = {"rms_auc": {}, "rms_median_uV": {}}
    rows = {}
    r_main, *_ = subject_rms("channel_clean", ("CAUEEG", "ds004504_raw", "GENEEG"))
    rows.update(r_main)
    r_pa, *_ = subject_rms("padic_channel_clean", ("P-ADIC",))
    pmeta = json.loads((C.OUT / "padic_subject_meta.json").read_text())
    for s, (c, l, v) in r_pa.items():
        grp = pmeta.get(str(s), {}).get("group"); age = pmeta.get(str(s), {}).get("age") or 0
        if grp in ("AD", "MCI") or (grp == "CN" and age >= 55):
            rows[s] = (c, grp, v)
    r_bl, Xb, gb, lb = subject_rms("brainlat_channel_clean", ("BrainLat",))
    rows.update(r_bl)
    for coh in ("CAUEEG", "ds004504_raw", "P-ADIC", "BrainLat", "GENEEG"):
        sub = {s: v for s, v in rows.items() if v[0] == coh}
        rep["rms_median_uV"][coh] = {k: round(float(np.median([v[2] for v in sub.values() if v[1] == k])), 2)
                                     for k in sorted({v[1] for v in sub.values()})}
        for name, pos in (("CN_vs_dementia", ("AD", "FTD")), ("CN_vs_AD", ("AD",)), ("CN_vs_MCI", ("MCI",))):
            ss = [s for s, v in sub.items() if v[1] in pos + ("CN",)]
            y = np.array([sub[s][1] in pos for s in ss]).astype(int)
            if 0 < y.sum() < len(y):
                v = [sub[s][2] for s in ss]
                rep["rms_auc"].setdefault(coh, {})[name] = {"auc": round(auc(y, v), 3), "ci95_boot": boot_ci(y, v),
                                                            "n": len(ss), "n_pos": int(y.sum())}

    # BrainLat amplitude ratio by frequency (patients / controls)
    from scipy.signal import welch
    f, P = welch(Xb, fs=256, nperseg=512, axis=-1)
    subs = np.unique(gb); grp = {s: lb[gb == s][0] for s in subs}
    Ps = {s: np.median(P[gb == s], 0).mean(0) for s in subs}
    med = {k: np.median(np.stack([Ps[s] for s in subs if grp[s] == k]), 0) for k in ("CN", "AD", "FTD")}
    sel = [1, 2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40]
    rep["brainlat_amplitude_ratio_vs_controls"] = {
        k: {str(fr): round(float(np.sqrt(med[k][np.argmin(abs(f - fr))] / med["CN"][np.argmin(abs(f - fr))])), 2) for fr in sel}
        for k in ("AD", "FTD")}

    # gain invariance of the canonical preprocessing (robust scaling precedes every feature)
    from src.data.scaling import robust_scale
    w = Xb[:200] * 1e-6
    d = max(float(np.max(np.abs(robust_scale(4.0 * x, "channel") - robust_scale(x, "channel")))) for x in w)
    rep["robust_scale_gain4_max_abs_diff"] = d
    rep["gain_sensitive_arms"] = ("frozen-encoder probes and fine-tuned encoders whose selected input is physical "
                                  "units (uV/100): eegpt_probe_physical (selected for EEGPT); every other encoder arm "
                                  "uses per-window z-scored input and every canonical model uses robust-scaled windows")
    rep["brainlat_files"] = {"CN": "s6_sub-1xxxx_rs_eeg.set (32/32)", "AD": "s6_sub-3xxxx_rs-HEP_eeg.set (30/30)",
                             "FTD": "s6_sub-2xxxx_rs-HEP_eeg.set (18/18)",
                             "pipeline_both": "0.5-40 Hz, 512 Hz, average reference, ASR burst 5, ICA pruning",
                             "rs-HEP_note": "patients' files were already filtered 0.5-40 Hz and eye-cleaned before the pipeline"}
    dest = C.OUT / "amplitude_gain_audit.json"
    dest.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("rms_auc", "rms_median_uV", "robust_scale_gain4_max_abs_diff")}, indent=1))


if __name__ == "__main__":
    main()
