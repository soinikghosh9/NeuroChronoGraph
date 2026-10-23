"""Cohort table and acquisition-confound audit for the revised Methods (review W-cohorts).

The extraction summary (127) stored only median rejection rates, so the paper could not state per-class
retention under the 24-window quota, nor the residual artefact-burden confound that survives it. This
recomputes, for every cohort in the matched corpus, directly from the harmonised cache:

  * recordings offered, recordings retained by the K=24 clean-window quota, per class;
  * recording duration as acquired, and the AUC of duration alone for each contrast BEFORE matching;
  * the surviving window-rejection rate within the truncated 235 s span, and its AUC per contrast
    AFTER the quota (the quota equalises window COUNT, not artefact burden, so this is the residual
    confound the paper must disclose);
  * ICA components removed and variance removed, per class (from the stored QC).

    python experiments/226_corpus_confounds.py
Out: outputs/results/canonical/corpus_confounds.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALZ = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles"))
HARM = ALZ / "harmonised"
FEAT = ROOT / "outputs" / "harmonised_features" / "channel_clean"
OUT = ROOT / "outputs" / "results" / "canonical"
WINDOW, STRIDE, MAX_WINDOWS, REJECT_V, SFREQ, K = 1024, 1024, 50, 200e-6, 256.0, 24
TRUNC = {"CAUEEG": 235.0, "ds004504_raw": 235.0, "GENEEG": 235.0, "MISP": 235.0, "P-ADIC": 235.0,
         "MCI_Dataset": 50.0}
CLASS = {"AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD", "CN": "CN", "C": "CN", "CONTROL": "CN", "MCI": "MCI"}
CONTRASTS = {"CN_vs_AD": (("AD",), ("CN",)), "CN_vs_dementia": (("AD", "FTD"), ("CN",)),
             "CN_vs_impaired": (("AD", "FTD", "MCI"), ("CN",)), "CN_vs_MCI": (("MCI",), ("CN",))}


def starts(n, sid):
    if n < WINDOW:
        return []
    s = list(range(0, n - WINDOW + 1, STRIDE))
    if len(s) > MAX_WINDOWS:
        seed = int(hashlib.md5(str(sid).encode()).hexdigest()[:8], 16)
        s = sorted(np.random.default_rng(seed).choice(s, MAX_WINDOWS, replace=False))
    return s


def auc(y, x):
    from scipy.stats import rankdata
    y = np.asarray(y); x = np.asarray(x, float)
    if len(np.unique(y)) < 2:
        return None
    r = rankdata(x, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, x, n=2000, seed=11):
    """Subject bootstrap interval for an AUC, so a bar can carry its uncertainty."""
    y = np.asarray(y); x = np.asarray(x, float)
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(auc(y[i], x[i]))
    if not out:
        return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def wilson(k, n, z=1.959963985):
    """Wilson score interval for a retention proportion (percent)."""
    if not n:
        return None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [float(100 * max(0.0, c - h)), float(100 * min(1.0, c + h))]


def main():
    retained = set()
    for tag in ("channel_clean", "padic_channel_clean"):
        p = ROOT / "outputs" / "harmonised_features" / tag / "meta.npz"
        if p.exists():
            retained |= set(map(str, np.unique(np.load(p, allow_pickle=False)["groups"])))
    rows = []
    for side in sorted(HARM.glob("*.json")):
        m = json.loads(side.read_text())
        coh, grp, sid = m.get("cohort"), CLASS.get(str(m.get("group", "?")).upper()), m["subject"]
        if coh not in TRUNC or grp is None:
            continue
        qc = m.get("qc", {})
        npy = side.with_suffix(".npy")
        if not npy.exists():
            continue
        a = np.load(npy, mmap_mode="r")
        dur = a.shape[1] / SFREQ
        span = a[:, :int(TRUNC[coh] * SFREQ)]
        kept = rej = 0
        for s in starts(span.shape[1], sid):
            w = np.asarray(span[:, s:s + WINDOW])
            if float(np.ptp(w, axis=1).max()) > REJECT_V:
                rej += 1
            else:
                kept += 1
        rows.append({"sid": sid, "cohort": coh, "cls": grp, "duration_s": dur, "n_kept": kept,
                     "n_rejected": rej, "rejection_rate": rej / max(1, kept + rej),
                     "retained": sid in retained, "ica_removed": len(qc.get("ica_excluded", []) or []),
                     "variance_removed": qc.get("variance_removed")})
    rep = {"n_recordings": len(rows), "cohorts": {}}
    by_coh = defaultdict(list)
    for r in rows:
        by_coh[r["cohort"]].append(r)
    for coh, rs in by_coh.items():
        cls_counts = defaultdict(lambda: [0, 0])
        for r in rs:
            cls_counts[r["cls"]][0] += 1
            cls_counts[r["cls"]][1] += int(r["retained"])
        entry = {"offered": len(rs), "retained": sum(r["retained"] for r in rs),
                 "by_class": {c: {"offered": v[0], "retained": v[1], "retention": round(v[1] / v[0], 3),
                                  "retention_pct_ci95": wilson(v[1], v[0])}
                              for c, v in sorted(cls_counts.items())},
                 "duration_s_median_by_class": {c: float(np.median([r["duration_s"] for r in rs if r["cls"] == c]))
                                                for c in sorted(cls_counts)},
                 "rejection_rate_median_by_class_retained": {
                     c: float(np.median([r["rejection_rate"] for r in rs if r["cls"] == c and r["retained"]] or [np.nan]))
                     for c in sorted(cls_counts)},
                 "ica_removed_median_by_class": {c: float(np.median([r["ica_removed"] for r in rs if r["cls"] == c]))
                                                 for c in sorted(cls_counts)},
                 "confound_auc": {}}
        for name, (pos, neg) in CONTRASTS.items():
            sel = [r for r in rs if r["cls"] in pos + neg]
            y = np.array([int(r["cls"] in pos) for r in sel])
            selr = [r for r in sel if r["retained"]]
            yr = np.array([int(r["cls"] in pos) for r in selr])
            if len(np.unique(y)) < 2:
                continue
            entry["confound_auc"][name] = {
                "duration_all_offered": auc(y, [r["duration_s"] for r in sel]),
                "duration_all_offered_ci": auc_ci(y, [r["duration_s"] for r in sel]),
                "duration_retained": auc(yr, [r["duration_s"] for r in selr]) if len(np.unique(yr)) > 1 else None,
                "rejection_rate_retained": auc(yr, [r["rejection_rate"] for r in selr]) if len(np.unique(yr)) > 1 else None,
                "rejection_rate_retained_ci": auc_ci(yr, [r["rejection_rate"] for r in selr]) if len(np.unique(yr)) > 1 else None,
                "n_offered": int(len(sel)), "n_retained": int(len(selr))}
        rep["cohorts"][coh] = entry
        print(f"\n== {coh}: offered {entry['offered']} retained {entry['retained']}")
        print("   by class:", {c: f"{v['retained']}/{v['offered']}" for c, v in entry["by_class"].items()})
        print("   duration median by class:", {c: round(v) for c, v in entry["duration_s_median_by_class"].items()})
        for n, v in entry["confound_auc"].items():
            print(f"   {n:15s} duration AUC offered {v['duration_all_offered']:.3f} retained "
                  f"{(v['duration_retained'] if v['duration_retained'] is not None else float('nan')):.3f} | "
                  f"rejection-rate AUC retained {(v['rejection_rate_retained'] if v['rejection_rate_retained'] is not None else float('nan')):.3f}")
    (OUT / "corpus_confounds.json").write_text(json.dumps(rep, indent=1, default=float))
    print("\nwrote", OUT / "corpus_confounds.json")


if __name__ == "__main__":
    main()
