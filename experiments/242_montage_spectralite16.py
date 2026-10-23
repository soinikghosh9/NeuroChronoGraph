"""Montage and test-time-noise analysis for the released 16-feature SpectraLite only.

This is the paper-facing replacement for the historical 12-feature montage run.  It uses the
canonical 75,971-parameter checkpoints from experiment 237 for fixed-weight re-evaluation after
electrode removal or device-style re-referencing.  The forest is refit on the same sixteen
per-electrode features for each clean montage.  No historical SpectraLite-S result is read.

    python experiments/242_montage_spectralite16.py --tasks CN_vs_dementia,CN_vs_AD,CN_vs_impaired

Output: outputs/results/canonical/montage_v2.json and montage_v2_scores/*spectralite16*.
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

OUT = C.OUT
CONT = C.ALZ / "harmonised"
WINDOW = 1024
SEEDS = (42, 43, 44)
PARAMS = 75_971
N_FEATURES = 16
MONTAGES = {
    "b19": (None, None),
    "b8": (["F3", "F4", "C3", "C4", "T5", "T6", "O1", "O2"], None),
    "b4": (["T5", "T6", "O1", "O2"], None),
    "b2": (["T5", "T6"], None),
    "muse": (["T5", "Fp1", "Fp2", "T6"], None),
    "ganglion": (["Fp1", "Fp2", "T3", "T4"], None),
    "muse_fpz": (["T5", "Fp1", "Fp2", "T6"], ["Fp1", "Fp2"]),
    "ganglion_ear": (["Fp1", "Fp2", "T3", "T4"], ["T3", "T4"]),
}
NOISE = {
    "pink5": ("pink", 5.0), "pink10": ("pink", 10.0), "pink20": ("pink", 20.0),
    "emg5": ("emg", 5.0), "emg10": ("emg", 10.0),
}
NOISE_MONTAGES = ("b19", "muse", "ganglion")

PAIR = {"Fp1": "Fp2", "F3": "F4", "C3": "C4", "P3": "P4", "O1": "O2",
        "F7": "F8", "T3": "T4", "T5": "T6", "Fz": "Fz", "Cz": "Cz", "Pz": "Pz"}
PAIR.update({v: k for k, v in list(PAIR.items())})


def paired(y, a, b, n=4000, seed=42):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(C.auc(y[i], a[i]) - C.auc(y[i], b[i]))
    d = np.asarray(d)
    return {"delta": C.auc(y, a) - C.auc(y, b),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def make_noise(kind, rms, shape, rng):
    white = rng.standard_normal(shape).astype(np.float32)
    f = np.fft.rfftfreq(shape[-1], 1 / 256.0)
    spec = np.fft.rfft(white, axis=-1)
    gain = 1 / np.sqrt(np.maximum(f, 1e-3)) if kind == "pink" else ((f >= 20) & (f <= 45)).astype(float)
    gain[0] = 0.0
    x = np.fft.irfft(spec * gain, n=shape[-1], axis=-1).astype(np.float32)
    x /= (x.std(axis=-1, keepdims=True) + 1e-12)
    return x * rms


def extended_features_torch(x, ex, channels):
    """CUDA implementation of the same four descriptors used by node_ext4.npy."""
    med = x.median(dim=-1, keepdim=True).values
    q = torch.quantile(x, torch.tensor((0.25, 0.75), device=x.device, dtype=x.dtype), dim=-1)
    iqr = (q[1] - q[0]).unsqueeze(-1)
    w = torch.nan_to_num((x - med) / iqr)
    f, psd = ex._welch(w)
    # scipy.signal.welch, which produced the released node_ext4 cache, applies the
    # one-sided factor of two to non-DC/non-Nyquist bins.
    if psd.shape[-1] > 2:
        psd = psd.clone(); psd[..., 1:-1] *= 2.0
    total = psd.sum(-1) + 1e-12
    logpsd = torch.log(psd + 1e-12)
    pf, _ = ex._alpha_peak(f, logpsd)
    ff = f.view(1, 1, -1)
    out = torch.zeros((*x.shape[:2], 4), device=x.device, dtype=psd.dtype)
    out[..., 0] = (psd * ((ff >= torch.maximum(pf[..., None] - 4, torch.tensor(0.5, device=x.device))) &
                           (ff < pf[..., None] - 2))).sum(-1) / total
    out[..., 1] = (psd * ((ff >= torch.maximum(pf[..., None] - 2, torch.tensor(0.5, device=x.device))) &
                           (ff < pf[..., None] + 2))).sum(-1) / total
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14)) & (f > 0)
    lf = torch.log10(f[fit]); lc = lf - lf.mean()
    y = torch.log10(psd[..., fit] + 1e-12)
    slope = ((y - y.mean(-1, keepdim=True)) * lc).sum(-1) / (lc.square().sum() + 1e-12)
    out[..., 2] = y.mean(-1) - slope * lf.mean() + 12.0
    alpha = psd[..., (f >= 8) & (f < 13)].sum(-1) + 1e-12
    pos = {ch: i for i, ch in enumerate(channels)}
    for i, ch in enumerate(channels):
        j = pos.get(PAIR.get(ch, ch), i)
        if j != i:
            out[:, i, 3] = torch.log(alpha[:, i] / alpha[:, j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--tasks", default="CN_vs_dementia,CN_vs_AD,CN_vs_impaired")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from src.config.config import CHANNEL_NAMES
    from src.models.spectral75k import SpectraLite
    from src.models.v5.spectral_nodes import SpectralNodeFeatures

    C.set_determinism(torch)
    if not torch.cuda.is_available():
        raise SystemExit("SpectraLite-16 montage run requires the configured CUDA GPU")
    # Feature extraction is inference-only.  The canonical training runs remain deterministic;
    # disabling the warning-only global guard here avoids repeated CUDA median warnings and lets
    # the batched descriptor path use the available GPU efficiently.
    torch.use_deterministic_algorithms(False)
    device = "cuda"; CH = list(CHANNEL_NAMES)
    meta = np.load(C.FEAT / args.tag / "meta.npz", allow_pickle=False)
    groups_all = np.array([str(x) for x in meta["groups"]])
    starts = meta["start"].astype(int)
    win_cls_all = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh_all = np.array([str(x) for x in meta["cohort"]])
    idx_all = np.flatnonzero(np.isin(win_coh_all, ["CAUEEG", "ds004504_raw"]))

    print(f"building raw windows for {len(idx_all)} windows ...", flush=True)
    t0 = time.time(); Xuv = np.zeros((len(idx_all), 19, WINDOW), np.float32)
    pos_of = {i: j for j, i in enumerate(idx_all)}
    for sid in np.unique(groups_all[idx_all]):
        a = np.load(CONT / f"{sid}.npy")
        for i in idx_all[groups_all[idx_all] == sid]:
            Xuv[pos_of[i]] = a[:, starts[i]:starts[i] + WINDOW]
    Xuv *= 1e6
    print(f"raw windows ready in {(time.time() - t0) / 60:.1f} min", flush=True)

    ex = SpectralNodeFeatures(sfreq=256.0).to(device).eval()
    node16_cache = np.concatenate([
        np.load(C.FEAT / args.tag / "node12.npy", mmap_mode="r"),
        np.load(C.FEAT / args.tag / "node_ext4.npy", mmap_mode="r")], -1)

    def derive(x, montage):
        elec, ref = MONTAGES[montage]
        if elec is None:
            return x, CH
        ci = [CH.index(e) for e in elec]; sub = x[:, ci, :]
        if ref is None:
            return (sub - sub.mean(1, keepdims=True)).astype(np.float32), elec
        ri = [CH.index(e) for e in ref]
        return (sub - x[:, ri, :].mean(1, keepdims=True)).astype(np.float32), elec

    def node16(x, channels):
        base, extra = [], []
        with torch.no_grad():
            for i in range(0, len(x), 1024):
                xb = torch.from_numpy(np.ascontiguousarray(x[i:i + 1024])).to(device)
                base.append(ex(xb).cpu().numpy())
                extra.append(extended_features_torch(xb, ex, channels).cpu().numpy())
        return np.concatenate([np.concatenate(base, 0), np.concatenate(extra, 0)], -1).astype(np.float32, copy=False)

    def feature16(x, montage, global_rows=None, clean=False):
        if clean and montage == "b19" and global_rows is not None:
            return np.asarray(node16_cache[global_rows], dtype=np.float32), CH
        xd, elec = derive(x, montage)
        return node16(xd, elec), elec

    def subject_scores(models, feats, channels, g, subjects):
        vals = []
        with torch.no_grad():
            for model in models:
                model.eval(); p = []
                for i in range(0, len(feats), 2048):
                    xb = torch.from_numpy(np.ascontiguousarray(feats[i:i + 2048])).to(device)
                    p.append(torch.softmax(model(node_features=xb, channel_names=channels), 1)[:, 1].cpu().numpy())
                vals.append(C.subject_mean(np.concatenate(p), g, subjects))
        return np.asarray(vals)

    def load_models(task, seed):
        models = []
        for k in range(5):
            path = C.CKPT / task / "spectralite16" / f"seed{seed}" / f"fold{k}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            m = SpectraLite(n_classes=2, n_features=16).to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True)); m.eval()
            models.append(m)
        return models

    dest = OUT / "montage_v2.json"; score_dir = OUT / "montage_v2_scores"; score_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": "SpectraLite", "parameters": PARAMS, "node_features": N_FEATURES,
              "tasks": {}, "protocol": "fixed canonical 16-feature checkpoints; no retraining after electrode removal"}

    # The three montage tasks share the same windows.  Extract each clean montage once, then slice
    # the cached feature tensor per task instead of repeating seven Welch passes for every contrast.
    print("precomputing shared clean 16-feature montage tensors ...", flush=True)
    global_clean = {}; global_channels = {}
    for mname in MONTAGES:
        print(f"  shared features {mname}", flush=True)
        global_clean[mname], global_channels[mname] = feature16(Xuv, mname, idx_all, clean=True)

    for task in [t for t in args.tasks.split(",") if t]:
        pos, neg, site = C.TASKS[task]
        kmask = np.isin(win_cls_all[idx_all], pos + neg); loc = np.flatnonzero(kmask)
        global_rows = idx_all[loc]
        g = groups_all[global_rows]; coh = win_coh_all[global_rows]
        yw = np.isin(win_cls_all[global_rows], pos).astype(int)
        tr_sub = np.unique(g[coh == "CAUEEG"]); te_sub = np.unique(g[coh == site])
        tr_y = np.array([yw[np.flatnonzero(g == s)[0]] for s in tr_sub])
        te_y = np.array([yw[np.flatnonzero(g == s)[0]] for s in te_sub])
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr_sub, tr_y))
        Xt = Xuv[loc]; te_mask = np.isin(g, te_sub)
        print(f"\n=== {task}: train {len(tr_sub)} | external {len(te_sub)} ===", flush=True)

        clean_feats = {mname: global_clean[mname][loc] for mname in MONTAGES}
        channels = global_channels

        rf_clean = {}
        for mname, F in clean_feats.items():
            Fw_tr = F[~te_mask].reshape(len(F[~te_mask]), -1)
            Fw_te = F[te_mask].reshape(len(F[te_mask]), -1)
            g_tr = g[~te_mask]
            Fs_tr = np.stack([np.nanmedian(Fw_tr[g_tr == s], 0) for s in tr_sub])
            # The external window groups are already selected by te_mask.
            g_te = g[te_mask]
            Fs_te = np.stack([np.nanmedian(Fw_te[g_te == s], 0) for s in te_sub])
            med = np.nanmedian(Fs_tr, 0); Fs_tr = np.where(np.isfinite(Fs_tr), Fs_tr, med)
            Fs_te = np.where(np.isfinite(Fs_te), Fs_te, med)
            per = []
            for sd in SEEDS:
                fits = [RandomForestClassifier(400, random_state=sd, n_jobs=6, class_weight="balanced").fit(Fs_tr[a], tr_y[a])
                        for a, _ in folds]
                per.append(np.mean([f.predict_proba(Fs_te)[:, 1] for f in fits], 0))
            rf_clean[mname] = np.asarray(per)
            np.save(score_dir / f"{task}__rf_node__{mname}.npy", rf_clean[mname])

        spectral_clean = {m: [] for m in MONTAGES}
        for sd in SEEDS:
            models = load_models(task, sd)
            for mname in MONTAGES:
                spectral_clean[mname].append(subject_scores(models, clean_feats[mname][te_mask], channels[mname],
                                                            g[te_mask], te_sub))
            del models; torch.cuda.empty_cache()
        spectral_clean = {m: np.asarray(v)[:, :, :].mean(1) for m, v in spectral_clean.items()}
        # `subject_scores` returns one row per seed and five fold-model scores averaged at subject level.
        for mname in MONTAGES:
            np.save(score_dir / f"{task}__spectralite16__noretrain__{mname}.npy", spectral_clean[mname])

        noise_rows = {}; noise_spec = {}; noise_rf = {}
        for mname in NOISE_MONTAGES:
            for nz, (kind, rms) in NOISE.items():
                key = f"{mname}+{nz}"; noise_spec[key] = []; noise_rf[key] = []
                for sd in SEEDS:
                    Xn = Xt[te_mask] + make_noise(kind, rms, Xt[te_mask].shape, np.random.default_rng(sd))
                    Fn, cn = feature16(Xn, mname)
                    models = load_models(task, sd)
                    noise_spec[key].append(subject_scores(models, Fn, cn, g[te_mask], te_sub).mean(0))
                    del models; torch.cuda.empty_cache()
                    Fw_tr = clean_feats[mname][~te_mask].reshape(len(clean_feats[mname][~te_mask]), -1)
                    Fw_te = Fn.reshape(len(Fn), -1); g_tr = g[~te_mask]; g_te = g[te_mask]
                    Fs_tr = np.stack([np.nanmedian(Fw_tr[g_tr == s], 0) for s in tr_sub])
                    Fs_te = np.stack([np.nanmedian(Fw_te[g_te == s], 0) for s in te_sub])
                    med = np.nanmedian(Fs_tr, 0); Fs_tr = np.where(np.isfinite(Fs_tr), Fs_tr, med)
                    Fs_te = np.where(np.isfinite(Fs_te), Fs_te, med)
                    fits = [RandomForestClassifier(400, random_state=sd, n_jobs=6, class_weight="balanced").fit(Fs_tr[a], tr_y[a])
                            for a, _ in folds]
                    noise_rf[key].append(np.mean([f.predict_proba(Fs_te)[:, 1] for f in fits], 0))
                noise_spec[key] = np.asarray(noise_spec[key]); noise_rf[key] = np.asarray(noise_rf[key])
                np.save(score_dir / f"{task}__spectralite16__noretrain__{key}.npy", noise_spec[key])
                np.save(score_dir / f"{task}__rf_node__{key}.npy", noise_rf[key])
                noise_rows[key] = {
                    "spectralite_mean": float(np.mean([C.auc(te_y, s) for s in noise_spec[key]])),
                    "rf_node_mean": float(np.mean([C.auc(te_y, s) for s in noise_rf[key]])),
                }

        rows = {}; pvals = []; keys = []
        for mname, scores in spectral_clean.items():
            rf = rf_clean[mname]
            row = {"n_electrodes": 19 if MONTAGES[mname][0] is None else len(MONTAGES[mname][0]),
                   "reference": "CAR (19)" if MONTAGES[mname][0] is None else
                   ("subset CAR" if MONTAGES[mname][1] is None else f"mean({','.join(MONTAGES[mname][1])})"),
                   "spectralite_mean": float(np.mean([C.auc(te_y, s) for s in scores])),
                   "spectralite_sd": float(np.std([C.auc(te_y, s) for s in scores], ddof=1)),
                   "rf_node_mean": float(np.mean([C.auc(te_y, s) for s in rf])),
                   "rf_node_sd": float(np.std([C.auc(te_y, s) for s in rf], ddof=1))}
            row["paired_spectralite_minus_rfnode"] = paired(te_y, scores.mean(0), rf.mean(0))
            pvals.append(row["paired_spectralite_minus_rfnode"]["p"]); keys.append(mname); rows[mname] = row
        for k, adj in zip(keys, holm(pvals)):
            rows[k]["paired_spectralite_minus_rfnode"]["p_holm_within_task"] = float(adj)
        report["tasks"][task] = {"n_eval": int(len(te_sub)), "n_pos": int(te_y.sum()), "montages": rows,
                                 "noise": noise_rows}
        for mname, row in rows.items():
            print(f"  {mname:13s} SpectraLite {row['spectralite_mean']:.3f} | forest {row['rf_node_mean']:.3f} | "
                  f"p_Holm {row['paired_spectralite_minus_rfnode']['p_holm_within_task']:.3f}", flush=True)

    (OUT / "montage_v2.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "montage_v2.json", flush=True)


if __name__ == "__main__":
    main()
