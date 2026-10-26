"""Classical arm of the canonical benchmark (CPU): same folds, internal OOF + external scores.

Companion to 211. Writes shards with the same schema to outputs/results/canonical/<task>/ so the
merge (225) treats classical and deep models identically.

Models
    rf123        400-tree class-weighted random forest on the 123-d QEEG block (the published reference)
    rf_node      the same forest on Spectral-75k's OWN input: the 12 spectral node features per
                 electrode (228-d), subject median over windows. This is the fair "same features,
                 different estimator" comparator the montage grid lacked (review C4.2).
    gb, svm, logistic   as in 161 (scikit-learn defaults; SVM RBF; L2 logistic), standardised inputs
    concept      the sign-constrained concept logistic, C=0.5 (opt-in via --models; it belongs to the companion
                 construct paper and is not part of this paper's benchmark)

Unlike 161 the scaler is fitted inside each training fold, so internal OOF scores carry no
scaling leakage. External scores are the mean of the five fold models, as for the networks.

    python experiments/212_classical_benchmark.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

ALZ = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles"))
FEAT = ROOT / "outputs" / "harmonised_features"
OUT = ROOT / "outputs" / "results" / "canonical"
SPLIT_SEED = 42
CLASS = {"AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD", "CN": "CN", "C": "CN",
         "CONTROL": "CN", "MCI": "MCI"}
TASKS = {"CN_vs_dementia": (("AD", "FTD"), ("CN",), "ds004504_raw"),
         "CN_vs_AD": (("AD",), ("CN",), "ds004504_raw"),
         "CN_vs_impaired": (("AD", "FTD", "MCI"), ("CN",), "ds004504_raw"),
         "CN_vs_MCI": (("MCI",), ("CN",), "GENEEG")}
DIR = {"pdr_frequency": -1, "pdr_prominence": -1, "pdr_variability": +1,
       "posterior_maximality": +1, "ap_gradient_alpha": -1, "ap_gradient_freq": -1,
       "asymmetry_alpha": 0, "asymmetry_freq": 0, "theta_alpha_ratio": +1,
       "slowing_ratio": +1, "rel_alpha": -1, "rel_theta": +1,
       "aperiodic_exponent": 0, "spectral_edge": -1}


def windows_path(tag):
    for base in (ROOT / "outputs" / "harmonised_windows", ALZ / "harmonised_windows"):
        if (base / tag / "windows.npy").exists():
            return base / tag / "windows.npy"
    raise SystemExit(f"windows.npy for {tag} not found")


def node_features(tag, force=False):
    """Window-level Spectral-75k node features [N, 19, 12], cached next to the other features."""
    path = FEAT / tag / "node12.npy"
    if path.exists() and not force:
        return np.load(path, mmap_mode="r")
    import torch
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xw = np.load(windows_path(tag), mmap_mode="r")
    ex = SpectralNodeFeatures(sfreq=256.0).to(dev).eval()
    F = np.zeros((len(Xw), 19, 12), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(Xw), 512):
            xb = torch.from_numpy(np.ascontiguousarray(Xw[i:i + 512])).float().to(dev)
            F[i:i + 512] = ex(xb).cpu().numpy()
    np.save(path, F)
    print(f"node features {F.shape} in {time.time()-t0:.0f}s ({dev})", flush=True)
    return np.load(path, mmap_mode="r")


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return None
    r = rankdata(s, method="average"); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--models", default="rf123,rf_node,gb,svm,logistic")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--rf-seeds", default="42,43,44")
    args = ap.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from src.models.v6 import SignConstrainedLogistic

    spec = importlib.util.spec_from_file_location("tb", ROOT / "experiments" / "157_task_battery.py")
    tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
    T = tb.subject_table(args.tag)
    subs, coh, cls_ = np.asarray(T["subjects"]), T["cohort"], T["cls"]
    Xq, Xc = T["X_spectral"], T["X_concept"]
    signs = [DIR[n] for n in T["names"]]

    meta = np.load(FEAT / args.tag / "meta.npz", allow_pickle=False)
    g = meta["groups"]
    F = node_features(args.tag)
    order = {s: np.flatnonzero(g == s) for s in subs}
    Xn = np.stack([np.nanmedian(np.asarray(F[order[s]]).reshape(len(order[s]), -1), axis=0)
                   for s in subs])
    tr_mask_all = coh == "CAUEEG"
    Xn = np.where(np.isfinite(Xn), Xn, np.nanmedian(Xn[tr_mask_all], axis=0))

    rf_seeds = [int(s) for s in args.rf_seeds.split(",") if s]
    ctor = {
        "rf123": (lambda sd: RandomForestClassifier(400, random_state=sd, n_jobs=6,
                                                    class_weight="balanced"), Xq, rf_seeds, False),
        "rf_node": (lambda sd: RandomForestClassifier(400, random_state=sd, n_jobs=6,
                                                      class_weight="balanced"), Xn, rf_seeds, False),
        "gb": (lambda sd: GradientBoostingClassifier(random_state=sd), Xq, [42], True),
        "svm": (lambda sd: SVC(probability=True, random_state=sd), Xq, [42], True),
        "logistic": (lambda sd: LogisticRegression(max_iter=2000), Xq, [42], True),
        "concept": (lambda sd: SignConstrainedLogistic(signs, C=0.5), Xc, [42], False),
    }

    for task in [t for t in args.tasks.split(",") if t]:
        pos, neg, site = TASKS[task]
        km = np.isin(cls_, pos + neg)
        y = np.isin(cls_, pos).astype(int)
        tr = km & (coh == "CAUEEG"); te = km & (coh == site)
        tr_sub, te_sub = subs[tr], subs[te]
        tr_y, te_y = y[tr], y[te]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(tr_sub, tr_y))
        fold_of_subject = np.full(len(tr_sub), -1)
        for k, (_, b) in enumerate(folds):
            fold_of_subject[b] = k
        print(f"\n=== {task}: train {len(tr_sub)} | {site} {len(te_sub)} ({int(te_y.sum())} pos)", flush=True)
        for name in [m for m in args.models.split(",") if m]:
            make, X, seeds, scale = ctor[name]
            for sd in seeds:
                dest = OUT / task / f"{name}__seed{sd}.npz"
                if dest.exists():
                    continue
                t0 = time.time()
                Xtr, Xte = X[tr], X[te]
                oof = np.full(len(tr_sub), np.nan, np.float32); ext = []
                for a, b in folds:
                    if scale:
                        sc = StandardScaler().fit(Xtr[a]); f = make(sd).fit(sc.transform(Xtr[a]), tr_y[a])
                        oof[b] = f.predict_proba(sc.transform(Xtr[b]))[:, 1]
                        ext.append(f.predict_proba(sc.transform(Xte))[:, 1])
                    else:
                        f = make(sd).fit(Xtr[a], tr_y[a])
                        oof[b] = f.predict_proba(Xtr[b])[:, 1]
                        ext.append(f.predict_proba(Xte)[:, 1])
                ext = np.mean(ext, 0).astype(np.float32)
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, train_subjects=tr_sub, train_y=tr_y, oof_score=oof,
                                    fold_of_subject=fold_of_subject, ext_site=site,
                                    ext_subjects=te_sub, ext_y=te_y, ext_score=ext)
                info = {"task": task, "model": name, "seed": sd, "auc_internal_oof": auc(tr_y, oof),
                        "auc_external": auc(te_y, ext), "ext_site": site, "n_train": int(len(tr_sub)),
                        "n_ext": int(len(te_sub)), "features": {"rf123": "QEEG-123", "rf_node": "node12x19",
                        "concept": "concepts-14"}.get(name, "QEEG-123"),
                        "seconds": round(time.time() - t0, 1)}
                dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
                print(f"   {name:10s} seed {sd}: internal {info['auc_internal_oof']:.4f} | "
                      f"external {info['auc_external']:.4f} | {info['seconds']}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
