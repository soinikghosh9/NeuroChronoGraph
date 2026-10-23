"""BrainLat third-site analysis: site distance, the three-site transfer summary, and the
leave-one-site-out transferability index. Reads the shards scored by 232 and the cached features.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
FEAT, CAN = C.FEAT, C.OUT


def subj_qeeg(tag, want_cohort=None, want_label=None):
    meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    g = np.array([str(x) for x in meta["groups"]]); coh = np.array([str(x) for x in meta["cohort"]])
    lab = np.array([str(x) for x in meta["labels"]])
    S = np.load(FEAT / tag / "spectral.npy", mmap_mode="r")
    rows, ids = [], []
    for s in np.unique(g):
        i = np.flatnonzero(g == s)
        if want_cohort and coh[i[0]] != want_cohort:
            continue
        if want_label and lab[i[0]] not in want_label:
            continue
        rows.append(np.nan_to_num(np.nanmedian(np.asarray(S[i]), 0))); ids.append(s)
    return np.array(rows), ids


def cv_auc(X, y, seed=42):
    X = np.nan_to_num(np.asarray(X, float)); y = np.asarray(y, int)
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[a])
        f = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(X[a]), y[a])
        oof[b] = f.predict_proba(sc.transform(X[b]))[:, 1]
    return float(roc_auc_score(y, oof))


def site_distance():
    """Controls-only decodability of BrainLat vs CAUEEG controls (the site-distance measure)."""
    cau, _ = subj_qeeg("channel_clean", "CAUEEG", {"CN"})
    bl, _ = subj_qeeg("brainlat_channel_clean", "BrainLat", {"CN"})
    X = np.vstack([cau, bl]); y = np.r_[np.zeros(len(cau)), np.ones(len(bl))]
    return {"n_caueeg_ctrl": len(cau), "n_brainlat_ctrl": len(bl), "decodability_auc": round(cv_auc(X, y), 3)}


def transfer_table(task):
    rows = {}
    for s in sorted((CAN / task).glob("*__seed42.json")):
        j = json.loads(s.read_text()); n = s.stem.replace("__seed42", "")
        if j.get("auc_brainlat") is None:
            continue
        rows[n] = {"internal": j.get("auc_internal_oof"), "ds004504": j.get("auc_external"),
                   "P-ADIC": j.get("auc_padic"), "BrainLat": j.get("auc_brainlat")}
    return rows


def summarise(task):
    rows = transfer_table(task)
    by_site = {site: np.array([r[site] for r in rows.values() if r[site] is not None])
               for site in ("internal", "ds004504", "P-ADIC", "BrainLat")}
    out = {"n_models": len(rows)}
    for site, v in by_site.items():
        out[site] = {"mean": round(float(v.mean()), 3), "min": round(float(v.min()), 3),
                     "max": round(float(v.max()), 3), "spread": round(float(v.max() - v.min()), 3)}
    # per-model drop ds004504 -> BrainLat
    drops = [r["ds004504"] - r["BrainLat"] for r in rows.values() if r["ds004504"] and r["BrainLat"]]
    out["ds_to_brainlat_drop"] = {"mean": round(float(np.mean(drops)), 3),
                                  "min": round(float(np.min(drops)), 3), "max": round(float(np.max(drops)), 3)}
    return out, rows


def transferability_index(rows):
    """LOSO: predict a site's mean external AUC from its controls-only site distance."""
    dec = {"ds004504": 0.96, "P-ADIC": 0.99, "BrainLat": site_distance()["decodability_auc"]}
    y = {site: float(np.mean([r[site] for r in rows.values() if r[site] is not None])) for site in dec}
    sites = list(dec)
    X = np.array([[dec[s]] for s in sites]); Y = np.array([y[s] for s in sites])
    preds = {}
    for i, s in enumerate(sites):
        tr = [j for j in range(len(sites)) if j != i]
        lr = LinearRegression().fit(X[tr], Y[tr])
        preds[s] = {"decodability": dec[s], "actual_mean_auc": round(y[s], 3),
                    "predicted_mean_auc": round(float(lr.predict(X[i:i + 1])[0]), 3)}
    return preds


def main():
    sd = site_distance()
    print("== BrainLat site distance (controls only) ==")
    print(f"  decodability AUC vs CAUEEG controls: {sd['decodability_auc']} "
          f"(n={sd['n_caueeg_ctrl']} CAUEEG, {sd['n_brainlat_ctrl']} BrainLat)")
    rep = {"site_distance": sd, "tasks": {}}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        out, rows = summarise(task)
        rep["tasks"][task] = out
        print(f"\n== {task} across sites (n_models={out['n_models']}) ==")
        for site in ("internal", "ds004504", "P-ADIC", "BrainLat"):
            print(f"  {site:9s} mean {out[site]['mean']}  range {out[site]['min']}-{out[site]['max']} (spread {out[site]['spread']})")
        print(f"  ds004504 -> BrainLat drop: mean {out['ds_to_brainlat_drop']['mean']} "
              f"({out['ds_to_brainlat_drop']['min']}-{out['ds_to_brainlat_drop']['max']})")
        if task == "CN_vs_dementia":
            idx = transferability_index(rows)
            rep["transferability_index"] = idx
            print("  transferability index (LOSO, site distance -> mean external AUC):")
            for s, d in idx.items():
                print(f"    {s:9s} distance {d['decodability']}  actual {d['actual_mean_auc']}  predicted {d['predicted_mean_auc']}")
    (CAN / "brainlat_analysis.json").write_text(json.dumps(rep, indent=1))
    print("\nwrote", CAN / "brainlat_analysis.json")


if __name__ == "__main__":
    main()
