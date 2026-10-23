"""Deployment checks for SpectraLite-12 (the 12-feature model fixed before external scoring), mirroring the
16-feature checks so either model can be reported as primary. CPU only (the GPU runs the seed job).

  attribution  Integrated Gradients exactly as 239_attribution16.py (32 steps, mean development-control baseline,
               15 seed-fold checkpoints), on the 12 base features.
  montage      fixed-weight SpectraLite-12 on electrode subsets cut from the cleaned 19-channel ds004504
               recordings and re-referenced to their own average (montages of 242), against a 400-tree forest on
               the same 12 features retrained for each subset; noise robustness (pink / 20-45 Hz, 5-20 uV RMS) on
               b19, muse and ganglion, as 242.

    python experiments/259_spectralite12_deployment.py --parts attribution,montage
Out: outputs/results/canonical/attribution12.json, montage_spectralite12.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("c211_259", "211_canonical_benchmark.py")
M242 = _load("m242_259", "242_montage_spectralite16.py")
CAN, CKPT, FEAT = C.OUT, C.CKPT, C.FEAT
DEV = "cpu"; torch.set_num_threads(6)
NAMES = ["rel delta", "rel theta", "rel alpha", "rel beta", "rel gamma", "IAF", "alpha prominence",
         "spectral edge", "aperiodic exponent", "theta/alpha", "slowing", "spectral entropy"]
SEEDS = (42, 43, 44)


def load(task, seed, k):
    from src.models.spectral75k import Spectral75k
    m = Spectral75k(n_classes=2).to(DEV)
    m.load_state_dict(torch.load(CKPT / task / "spectral75k" / f"seed{seed}" / f"fold{k}.pt", map_location=DEV, weights_only=True))
    return m.eval()


def part_attribution():
    task = "CN_vs_dementia"
    z = np.load(CAN / task / "spectral75k__seed42.npz", allow_pickle=True)
    subs, y = z["train_subjects"].astype(str), z["train_y"].astype(int)
    meta = np.load(FEAT / "channel_clean" / "meta.npz", allow_pickle=False); g = meta["groups"].astype(str)
    N = np.load(FEAT / "channel_clean" / "node12.npy")
    idx = {s: np.flatnonzero(g == s) for s in subs}
    X = np.array([np.nan_to_num(np.nanmean(N[idx[s]], 0)) for s in subs], np.float32)
    Xt = torch.tensor(X); base = torch.tensor(X[y == 0].mean(0, keepdims=True))
    vecs = []
    for seed in SEEDS:
        for k in range(5):
            m = load(task, seed, k); total = torch.zeros_like(Xt)
            for a in np.linspace(0, 1, 32):
                xi = (base + float(a) * (Xt - base)).clone().requires_grad_(True)
                gr, = torch.autograd.grad(m(node_features=xi)[:, 1].sum(), xi); total += gr
            vecs.append(((Xt - base) * total / 32).abs().mean(dim=(0, 1)).numpy())
    A = np.array(vecs); imp = A.mean(0); imp = imp / imp.sum(); order = np.argsort(-imp)
    rhos = [spearmanr(A[i], A[j]).correlation for i in range(len(A)) for j in range(i + 1, len(A))]
    rep = {"task": task, "model": "SpectraLite-12", "n_models": len(A),
           "feature_importance_share": {NAMES[i]: round(float(imp[i]), 4) for i in order},
           "stability_spearman_median": round(float(np.median(rhos)), 3)}
    (CAN / "attribution12.json").write_text(json.dumps(rep, indent=1))
    print("[attribution12]", rep["stability_spearman_median"], list(rep["feature_importance_share"].items())[:5], flush=True)


def part_montage():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from src.config.config import CHANNEL_NAMES
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    CH = list(CHANNEL_NAMES); task = "CN_vs_dementia"
    meta = np.load(FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    g_all = meta["groups"].astype(str); st = meta["start"].astype(int)
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]]); coh = meta["cohort"].astype(str)
    pos, neg, site = C.TASKS[task]
    rows_idx = np.flatnonzero(np.isin(coh, ["CAUEEG", site]) & np.isin(cls, pos + neg))
    g = g_all[rows_idx]; c = coh[rows_idx]; yw = np.isin(cls[rows_idx], pos).astype(int)
    tr_sub = np.unique(g[c == "CAUEEG"]); te_sub = np.unique(g[c == site])
    tr_y = np.array([yw[np.flatnonzero(g == s)[0]] for s in tr_sub]); te_y = np.array([yw[np.flatnonzero(g == s)[0]] for s in te_sub])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr_sub, tr_y))
    t0 = time.time(); X = np.zeros((len(rows_idx), 19, 1024), np.float32)
    for sid in np.unique(g):
        a = np.load(C.ALZ / "harmonised" / f"{sid}.npy")
        for j in np.flatnonzero(g == sid):
            X[j] = a[:, st[rows_idx[j]]:st[rows_idx[j]] + 1024]
    X *= 1e6; print(f"windows {X.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
    ex = SpectralNodeFeatures(sfreq=256.0).eval()
    N12 = np.load(FEAT / "channel_clean" / "node12.npy", mmap_mode="r")

    def feats(x, montage, clean_rows=None):
        if montage == "b19" and clean_rows is not None:
            return np.asarray(N12[clean_rows], np.float32), CH
        xd, elec = M242.derive_static(x, montage, CH) if hasattr(M242, "derive_static") else derive(x, montage)
        out = []
        with torch.no_grad():
            for i in range(0, len(xd), 512):
                out.append(ex(torch.from_numpy(np.ascontiguousarray(xd[i:i + 512]))).numpy())
        return np.concatenate(out).astype(np.float32), elec

    def derive(x, montage):
        elec, ref = M242.MONTAGES[montage]
        if elec is None:
            return x, CH
        ci = [CH.index(e) for e in elec]; sub = x[:, ci, :]
        if ref is None:
            return (sub - sub.mean(1, keepdims=True)).astype(np.float32), elec
        ri = [CH.index(e) for e in ref]
        return (sub - x[:, ri, :].mean(1, keepdims=True)).astype(np.float32), elec

    te_mask = np.isin(g, te_sub)

    def subj(F, gg, subs):
        Fw = F.reshape(len(F), -1); return np.stack([np.nanmedian(Fw[gg == s], 0) for s in subs])

    def forest(Ftr_w, Fte_w, seed):
        Fs_tr = subj(Ftr_w, g[~te_mask], tr_sub); Fs_te = subj(Fte_w, g[te_mask], te_sub)
        med = np.nanmedian(Fs_tr, 0); Fs_tr = np.where(np.isfinite(Fs_tr), Fs_tr, med); Fs_te = np.where(np.isfinite(Fs_te), Fs_te, med)
        fits = [RandomForestClassifier(400, random_state=seed, n_jobs=6, class_weight="balanced").fit(Fs_tr[a], tr_y[a]) for a, _ in folds]
        return np.mean([f.predict_proba(Fs_te)[:, 1] for f in fits], 0)

    def sl_scores(F, chans):
        per = []
        for sd in SEEDS:
            W = []
            for k in range(5):
                m = load(task, sd, k)
                with torch.no_grad():
                    W.append(torch.softmax(m(node_features=torch.from_numpy(F), channel_names=chans), 1)[:, 1].numpy())
            per.append(C.subject_mean(np.mean(W, 0), g[te_mask], te_sub))
        return np.asarray(per)

    rep = {"model": "SpectraLite-12", "task": task, "n_eval": int(len(te_sub)), "n_pos": int(te_y.sum()), "montages": {}, "noise": {}}
    pv, keys = [], []
    for mname in M242.MONTAGES:
        F, chans = feats(X, mname, clean_rows=rows_idx)
        s = sl_scores(F[te_mask], chans); rf = np.asarray([forest(F[~te_mask], F[te_mask], sd) for sd in SEEDS])
        row = {"spectralite12_mean": float(np.mean([C.auc(te_y, v) for v in s])), "spectralite12_sd": float(np.std([C.auc(te_y, v) for v in s], ddof=1)),
               "forest12_mean": float(np.mean([C.auc(te_y, v) for v in rf])), "forest12_sd": float(np.std([C.auc(te_y, v) for v in rf], ddof=1)),
               "paired_sl12_minus_forest12": M242.paired(te_y, s.mean(0), rf.mean(0))}
        rep["montages"][mname] = row; pv.append(row["paired_sl12_minus_forest12"]["p"]); keys.append(mname)
        print(f"  {mname:13s} SL-12 {row['spectralite12_mean']:.3f} | forest-12 {row['forest12_mean']:.3f}", flush=True)
        if mname in M242.NOISE_MONTAGES:
            for nz, (kind, rms) in M242.NOISE.items():
                ss, rr = [], []
                for sd in SEEDS:
                    Xn = X[te_mask] + M242.make_noise(kind, rms, X[te_mask].shape, np.random.default_rng(sd))
                    Fn, cn = feats(Xn, mname)
                    W = []
                    for k in range(5):
                        m = load(task, sd, k)
                        with torch.no_grad():
                            W.append(torch.softmax(m(node_features=torch.from_numpy(Fn), channel_names=cn), 1)[:, 1].numpy())
                    ss.append(C.subject_mean(np.mean(W, 0), g[te_mask], te_sub)); rr.append(forest(F[~te_mask], Fn, sd))
                ss, rr = np.asarray(ss), np.asarray(rr)
                rep["noise"][f"{mname}+{nz}"] = {"spectralite12_mean": float(np.mean([C.auc(te_y, v) for v in ss])),
                                                 "forest12_mean": float(np.mean([C.auc(te_y, v) for v in rr])),
                                                 "paired_sl12_minus_forest12": M242.paired(te_y, ss.mean(0), rr.mean(0))}
                print(f"    noise {mname}+{nz}: SL-12 {rep['noise'][f'{mname}+{nz}']['spectralite12_mean']:.3f} | forest-12 "
                      f"{rep['noise'][f'{mname}+{nz}']['forest12_mean']:.3f}", flush=True)
    for k_, adj in zip(keys, M242.holm(pv)):
        rep["montages"][k_]["paired_sl12_minus_forest12"]["p_holm_within_task"] = float(adj)
    (CAN / "montage_spectralite12.json").write_text(json.dumps(rep, indent=1))
    print("wrote montage_spectralite12.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="attribution,montage"); a = ap.parse_args()
    for p in a.parts.split(","):
        {"attribution": part_attribution, "montage": part_montage}[p]()
