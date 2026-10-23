"""Montage analysis, corrected: fair forest, test-time electrode removal, device-reference proxies, noise.

Fixes review C4 on exp 210:
  (1) the 210 "random forest" saw 5 relative band powers per channel from a raw periodogram while
      Spectral-75k saw 12 spectral node features per channel. `rf_node` here uses Spectral-75k's own
      12 features per channel (subject median), so model and feature set are no longer confounded.
      The legacy forest is kept for continuity (`rf_band`).
  (2) 210 RETRAINED Spectral-75k per montage although the text claimed re-evaluation. Both are now
      reported: `s75_retrain` (a model trained on each montage; reused from 210, identical code/folds/
      seeds) and `s75_noretrain` (one model trained on the 19-electrode array, then evaluated on each
      subset through its coordinate montage embedding, no retraining -- the P2 claim tested directly).
  (3) Subset common-average referencing is not what a headset does. Two proxies are added:
      muse_fpz  = T5, Fp1, Fp2, T6 referenced to mean(Fp1, Fp2) (Fpz proxy; Muse reference is Fpz)
      ganglion_ear = Fp1, Fp2, T3, T4 referenced to mean(T3, T4) (linked-ear proxy; nearest scalp sites)
  (4) Additive noise at test time (no retraining): 1/f (pink) noise and 20-45 Hz band-limited
      "EMG-like" noise at fixed per-channel RMS, the dominant degradations of dry commodity electrodes.
  (5) Paired Spectral-75k-vs-rf_node bootstrap at each montage x task, Holm-corrected across the grid.

    python experiments/215_montage_v2.py
Out: outputs/results/canonical/montage_v2.json (+ subject scores npz)
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

OUT = C.OUT
CONT = C.ALZ / "harmonised"
WINDOW = 1024
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
SEEDS = (42, 43, 44)
# name -> (electrodes, reference electrodes or None for subset CAR)
MONTAGES = {"b19": (None, None),
            "b8": (["F3", "F4", "C3", "C4", "T5", "T6", "O1", "O2"], None),
            "b4": (["T5", "T6", "O1", "O2"], None),
            "b2": (["T5", "T6"], None),
            "muse": (["T5", "Fp1", "Fp2", "T6"], None),
            "ganglion": (["Fp1", "Fp2", "T3", "T4"], None),
            "muse_fpz": (["T5", "Fp1", "Fp2", "T6"], ["Fp1", "Fp2"]),
            "ganglion_ear": (["Fp1", "Fp2", "T3", "T4"], ["T3", "T4"])}
NOISE = {"pink5": ("pink", 5.0), "pink10": ("pink", 10.0), "pink20": ("pink", 20.0),
         "emg5": ("emg", 5.0), "emg10": ("emg", 10.0)}
NOISE_MONTAGES = ("b19", "muse", "ganglion")


def paired(y, a, b, n=4000, seed=42):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(C.auc(y[i], a[i]) - C.auc(y[i], b[i]))
    d = np.asarray(d)
    return {"delta": C.auc(y, a) - C.auc(y, b), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def make_noise(kind, rms, shape, rng):
    white = rng.standard_normal(shape).astype(np.float32)
    f = np.fft.rfftfreq(shape[-1], 1 / 256.0)
    spec = np.fft.rfft(white, axis=-1)
    if kind == "pink":
        g = np.where(f > 0, 1 / np.sqrt(np.maximum(f, 1e-3)), 0.0)
    else:
        g = ((f >= 20) & (f <= 45)).astype(float)
    x = np.fft.irfft(spec * g, n=shape[-1], axis=-1).astype(np.float32)
    x /= (x.std(axis=-1, keepdims=True) + 1e-12)
    return x * rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()

    import torch
    C.set_determinism(torch)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from src.config.config import CHANNEL_NAMES
    from src.models.spectral75k import Spectral75k
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    e161 = C.load_161()
    device = "cuda"; assert torch.cuda.is_available()
    CH = list(CHANNEL_NAMES)

    meta = np.load(C.FEAT / args.tag / "meta.npz", allow_pickle=False)
    groups, start = meta["groups"], meta["start"].astype(int)
    win_cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh = np.array([str(x) for x in meta["cohort"]])
    keep_any = np.isin(win_coh, ["CAUEEG", "ds004504_raw"])
    idx_all = np.flatnonzero(keep_any)
    print(f"building microvolt CAR windows for {len(idx_all)} windows ...", flush=True)
    t0 = time.time()
    Xuv = np.zeros((len(idx_all), 19, WINDOW), np.float32)
    pos_of = {}
    for j, i in enumerate(idx_all):
        pos_of[i] = j
    for s in np.unique(groups[idx_all]):
        a = np.load(CONT / f"{s}.npy")
        for i in idx_all[groups[idx_all] == s]:
            Xuv[pos_of[i]] = a[:, start[i]:start[i] + WINDOW]
    Xuv *= 1e6
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    ex = SpectralNodeFeatures(sfreq=256.0).to(device).eval()

    def derive(X, montage):
        elec, ref = MONTAGES[montage]
        if elec is None:
            return X, CH
        ci = [CH.index(e) for e in elec]; sub = X[:, ci, :]
        if ref is None:
            return (sub - sub.mean(1, keepdims=True)).astype(np.float32), elec
        ri = [CH.index(e) for e in ref]
        return (sub - X[:, ri, :].mean(1, keepdims=True)).astype(np.float32), elec

    def node_feats(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                out.append(ex(torch.from_numpy(np.ascontiguousarray(X[i:i + 512])).to(device)).cpu().numpy())
        return np.concatenate(out)

    dest = OUT / "montage_v2.json"
    report = json.loads(dest.read_text()) if dest.exists() else {"tasks": {}}
    legacy = json.loads((ROOT / "outputs/results/audit/montage_grid_multiseed.json").read_text())["tasks"]
    score_dir = OUT / "montage_v2_scores"; score_dir.mkdir(parents=True, exist_ok=True)

    for task in [t for t in args.tasks.split(",") if t]:
        pos, neg, site = C.TASKS[task]
        kmask = np.isin(win_cls[idx_all], pos + neg)
        loc = np.flatnonzero(kmask)                       # positions into Xuv
        g = groups[idx_all][loc]; coh = win_coh[idx_all][loc]
        yw = np.isin(win_cls[idx_all][loc], pos).astype(int)
        tr_sub = np.unique(g[coh == "CAUEEG"]); te_sub = np.unique(g[coh == site])
        tr_y = np.array([yw[np.flatnonzero(g == s)[0]] for s in tr_sub])
        te_y = np.array([yw[np.flatnonzero(g == s)[0]] for s in te_sub])
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr_sub, tr_y))
        Xt = Xuv[loc]
        trep = report["tasks"].get(task, {"n_eval": int(len(te_sub)), "n_pos": int(te_y.sum()), "montages": {}})
        print(f"\n=== {task}: train {len(tr_sub)} | ext {len(te_sub)}", flush=True)

        # --- forests on node features (fair) per montage, per seed ---
        rfs = {}
        for mname in MONTAGES:
            Xd, elec = derive(Xt, mname)
            Fw = node_feats(Xd).reshape(len(Xd), -1)
            Fs_tr = np.stack([np.nanmedian(Fw[g == s], 0) for s in tr_sub])
            Fs_te = np.stack([np.nanmedian(Fw[g == s], 0) for s in te_sub])
            med = np.nanmedian(Fs_tr, 0); Fs_tr = np.where(np.isfinite(Fs_tr), Fs_tr, med)
            Fs_te = np.where(np.isfinite(Fs_te), Fs_te, med)
            per_seed = []
            for sd in SEEDS:
                fits = [RandomForestClassifier(400, random_state=sd, n_jobs=6, class_weight="balanced")
                        .fit(Fs_tr[a], tr_y[a]) for a, _ in folds]
                per_seed.append(np.mean([f.predict_proba(Fs_te)[:, 1] for f in fits], 0))
            rfs[mname] = np.array(per_seed)
            np.save(score_dir / f"{task}__rf_node__{mname}.npy", rfs[mname])
            print(f"   rf_node {mname:13s} AUC {np.mean([C.auc(te_y, s) for s in per_seed]):.3f}", flush=True)

        # --- Spectral-75k trained on b19 (no retrain) and on the two reference proxies (retrain) ---
        def train_eval_s75(train_montage, eval_specs, noise_specs=()):
            """Returns {eval_key: [n_seeds x n_te]} ensembles over folds."""
            Xd_tr, elec_tr = derive(Xt, train_montage)
            is_tr = np.isin(g, tr_sub)
            out = {}
            for sd in SEEDS:
                fold_models = []
                for a, _ in folds:
                    m = np.isin(g, tr_sub[a])
                    torch.manual_seed(sd); np.random.seed(sd)
                    model = Spectral75k(n_classes=2, channel_names=elec_tr).to(device)
                    from torch.utils.data import DataLoader, TensorDataset
                    ytr = yw[m]
                    cls, cnt = np.unique(ytr, return_counts=True); w = np.zeros(2, np.float32)
                    w[cls] = len(ytr) / (len(cls) * cnt)
                    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
                    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)
                    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR,
                                                                total_steps=C.EPOCHS * max(1, len(ytr) // C.BATCH + 1))
                    gen = torch.Generator(); gen.manual_seed(sd)
                    dl = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(Xd_tr[m])),
                                                  torch.from_numpy(ytr.astype(np.int64))),
                                    batch_size=C.BATCH, shuffle=True, generator=gen)
                    model.train()
                    for _ in range(C.EPOCHS):
                        for xb, yb in dl:
                            opt.zero_grad(); loss = crit(model(xb.to(device)), yb.to(device))
                            loss.backward(); opt.step(); sched.step()
                    model.eval(); fold_models.append(model)
                te_mask = np.isin(g, te_sub)
                for key, (mname, noise) in {**{k: (k, None) for k in eval_specs},
                                            **{f"{mm}+{nz}": (mm, nz) for mm in noise_specs for nz in NOISE}}.items():
                    X_eval = Xt[te_mask]
                    if noise is not None:
                        kind, rms = NOISE[noise]
                        X_eval = X_eval + make_noise(kind, rms, X_eval.shape, np.random.default_rng(sd))
                    Xe, elec_e = derive(X_eval, mname)
                    probs = []
                    with torch.no_grad():
                        for model in fold_models:
                            p = []
                            for i in range(0, len(Xe), 256):
                                xb = torch.from_numpy(np.ascontiguousarray(Xe[i:i + 256])).to(device)
                                p.append(torch.softmax(model(xb, channel_names=elec_e), 1)[:, 1].cpu().numpy())
                            probs.append(np.concatenate(p))
                    s = C.subject_mean(np.mean(probs, 0), g[te_mask], te_sub)
                    out.setdefault(key, []).append(s)
                del fold_models; torch.cuda.empty_cache()
            return {k: np.array(v) for k, v in out.items()}

        t1 = time.time()
        nore = train_eval_s75("b19", list(MONTAGES), noise_specs=NOISE_MONTAGES)
        print(f"   s75 trained on b19, evaluated everywhere ({(time.time()-t1)/60:.1f} min)", flush=True)
        retr = {}
        for mname in ("muse_fpz", "ganglion_ear"):
            retr.update(train_eval_s75(mname, [mname]))
        for k, v in {**{f"noretrain__{k}": v for k, v in nore.items()},
                     **{f"retrain__{k}": v for k, v in retr.items()}}.items():
            np.save(score_dir / f"{task}__s75__{k}.npy", v)

        # rf_node under noise (trained clean, tested noisy), for the noise montages
        rf_noise = {}
        te_mask = np.isin(g, te_sub)
        for mname in NOISE_MONTAGES:
            Xd_tr, _ = derive(Xt[~te_mask], mname)
            Fw_tr = node_feats(Xd_tr).reshape(len(Xd_tr), -1); g_tr = g[~te_mask]
            Fs_tr = np.stack([np.nanmedian(Fw_tr[g_tr == s], 0) for s in tr_sub])
            med = np.nanmedian(Fs_tr, 0); Fs_tr = np.where(np.isfinite(Fs_tr), Fs_tr, med)
            for nz, (kind, rms) in NOISE.items():
                per = []
                for sd in SEEDS:
                    Xn = Xt[te_mask] + make_noise(kind, rms, Xt[te_mask].shape, np.random.default_rng(sd))
                    Xd, _ = derive(Xn, mname)
                    Fw = node_feats(Xd).reshape(len(Xd), -1); ge = g[te_mask]
                    Fs_te = np.stack([np.nanmedian(Fw[ge == s], 0) for s in te_sub])
                    Fs_te = np.where(np.isfinite(Fs_te), Fs_te, med)
                    fits = [RandomForestClassifier(400, random_state=sd, n_jobs=6, class_weight="balanced")
                            .fit(Fs_tr[a], tr_y[a]) for a, _ in folds]
                    per.append(np.mean([f.predict_proba(Fs_te)[:, 1] for f in fits], 0))
                rf_noise[f"{mname}+{nz}"] = np.array(per)

        # --- assemble ---
        rows, pvals, keys = {}, [], []
        for mname in MONTAGES:
            row = {"n_electrodes": 19 if MONTAGES[mname][0] is None else len(MONTAGES[mname][0]),
                   "reference": "CAR (19)" if MONTAGES[mname][0] is None else
                   ("subset CAR" if MONTAGES[mname][1] is None else f"mean({','.join(MONTAGES[mname][1])})")}
            rf = rfs[mname]
            row["rf_node_mean"] = float(np.mean([C.auc(te_y, s) for s in rf]))
            row["rf_node_sd"] = float(np.std([C.auc(te_y, s) for s in rf], ddof=1))
            so = nore[mname]
            row["s75_noretrain_mean"] = float(np.mean([C.auc(te_y, s) for s in so]))
            row["s75_noretrain_sd"] = float(np.std([C.auc(te_y, s) for s in so], ddof=1))
            leg = legacy.get(task, {}).get("montages", {}).get(mname)
            if leg:
                row["s75_retrain_mean"] = leg["spectral_only_mean"]; row["s75_retrain_sd"] = leg["spectral_only_sd"]
                row["rf_band_legacy_mean"] = leg["spectral_rf_mean"]; row["rf_band_legacy_sd"] = leg["spectral_rf_sd"]
            if f"retrain__{mname}" in {f"retrain__{k}" for k in retr}:
                sr = retr[mname]
                row["s75_retrain_mean"] = float(np.mean([C.auc(te_y, s) for s in sr]))
                row["s75_retrain_sd"] = float(np.std([C.auc(te_y, s) for s in sr], ddof=1))
            row["paired_s75noretrain_minus_rfnode"] = paired(te_y, so.mean(0), rf.mean(0))
            pvals.append(row["paired_s75noretrain_minus_rfnode"]["p"]); keys.append(mname)
            rows[mname] = row
        for k, adj in zip(keys, holm(pvals)):
            rows[k]["paired_s75noretrain_minus_rfnode"]["p_holm_within_task"] = float(adj)
        noise_rows = {}
        for key, so in nore.items():
            if "+" not in key:
                continue
            rf = rf_noise[key]
            noise_rows[key] = {"s75_noretrain_mean": float(np.mean([C.auc(te_y, s) for s in so])),
                               "rf_node_mean": float(np.mean([C.auc(te_y, s) for s in rf]))}
        trep["montages"] = rows; trep["noise"] = noise_rows
        report["tasks"][task] = trep
        report["note"] = ("s75_retrain for b19..ganglion reused from 210 (identical model/folds/seeds); "
                          "muse_fpz/ganglion_ear retrained here. Holm across the montage family per task; "
                          "the across-task Holm is applied in 225.")
        dest.write_text(json.dumps(report, indent=1))
        for mname, r in rows.items():
            print(f"   {mname:13s} s75-noretrain {r['s75_noretrain_mean']:.3f} | s75-retrain "
                  f"{r.get('s75_retrain_mean', float('nan')):.3f} | rf_node {r['rf_node_mean']:.3f} | "
                  f"p_holm {r['paired_s75noretrain_minus_rfnode']['p_holm_within_task']:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
