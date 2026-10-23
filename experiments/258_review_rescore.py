"""Inference-only rescoring for the review revision (CPU; the GPU is reserved for the seed run).

  perfold   SpectraLite-12 (spectral75k) per-fold-model window scores at P-ADIC and BrainLat, which the canonical
            shards store only as the fold ensemble. Written to outputs/results/canonical/review_rescore/ for
            257 --sections procedure. Checked: the mean over folds reproduces the shard's ensemble score.
  geneeg17  GENEEG recorded 17 electrodes; T5 and T6 were zero-filled and then average-referenced, so every model
            received two identical synthetic channels (-mean of the others). SpectraLite masks absent electrodes,
            so it is rescored here on the 17 real electrodes only (re-referenced to their own average, the same
            derivation as the montage experiments), with the fixed CAUEEG checkpoints.

    python experiments/258_review_rescore.py --parts perfold,geneeg17
Out: outputs/results/canonical/review_rescore/*.npz and review_rescore.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("c211_258", "211_canonical_benchmark.py")
M242 = _load("m242_258", "242_montage_spectralite16.py")
CAN = C.OUT
RES = CAN / "review_rescore"; RES.mkdir(exist_ok=True)
DEV = "cpu"
torch.set_num_threads(4)


def load_model(task, name, seed, k):
    from src.models.spectral75k import Spectral75k, SpectraLite
    m = (SpectraLite(n_classes=2, n_features=16) if name == "spectralite16" else Spectral75k(n_classes=2)).to(DEV)
    m.load_state_dict(torch.load(C.CKPT / task / name / f"seed{seed}" / f"fold{k}.pt", map_location=DEV, weights_only=True))
    return m.eval()


def probs(model, feats, channels=None, bs=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            xb = torch.from_numpy(np.ascontiguousarray(feats[i:i + bs])).float()
            out.append(torch.softmax(model(node_features=xb, channel_names=channels), 1)[:, 1].numpy())
    return np.concatenate(out)


def part_perfold():
    rep = {}
    stores = {"padic": "padic_channel_clean", "brainlat": "brainlat_channel_clean"}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for f in sorted((CAN / task).glob("spectral75k__seed*.npz")):
            z = np.load(f, allow_pickle=True); seed = int(f.stem.split("__seed")[1])
            for p, tag in stores.items():
                dest = RES / f"{task}__{f.stem}__{p}.npz"
                meta = np.load(C.FEAT / tag / "meta.npz", allow_pickle=False); g = meta["groups"].astype(str)
                N = np.load(C.FEAT / tag / "node12.npy", mmap_mode="r")
                subs = z[f"{p}_subjects"].astype(str); idx = np.flatnonzero(np.isin(g, subs))
                F = np.asarray(N[idx], np.float32).reshape(len(idx), 19, -1); G = g[idx]
                W = np.stack([probs(load_model(task, "spectral75k", seed, k), F) for k in range(5)])
                np.savez_compressed(dest, win_folds=W.astype(np.float32), win_groups=G)
                ens = np.array([W.mean(0)[G == s].mean() for s in subs])
                err = float(np.max(np.abs(ens - z[f"{p}_score"].astype(float))))
                rep[f"{task}/{f.stem}/{p}"] = {"max_abs_diff_vs_shard_ensemble": err, "n_windows": int(len(idx))}
                print(f"[perfold] {task} {f.stem} {p}: {len(idx)} windows, max |ens - shard| = {err:.2e}", flush=True)
    return rep


def part_geneeg17():
    from src.config.config import CHANNEL_NAMES
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    from scipy.stats import rankdata
    CH = list(CHANNEL_NAMES); keep = [c for c in CH if c not in ("T5", "T6")]; ci = [CH.index(c) for c in keep]
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    g = meta["groups"].astype(str); coh = meta["cohort"].astype(str); st = meta["start"].astype(int)
    rep = {}
    z0 = np.load(CAN / "CN_vs_MCI" / "spectralite16__seed42.npz", allow_pickle=True)
    subs = z0["ext_subjects"].astype(str); y = z0["ext_y"].astype(int)
    idx = np.flatnonzero((coh == "GENEEG") & np.isin(g, subs))
    X = np.zeros((len(idx), 19, 1024), np.float32)
    for sid in np.unique(g[idx]):
        a = np.load(C.ALZ / "harmonised" / f"{sid}.npy")
        for j in np.flatnonzero(g[idx] == sid):
            X[j] = a[:, st[idx[j]]:st[idx[j]] + 1024]
    X *= 1e6
    t5 = CH.index("T5"); t6 = CH.index("T6")
    rep["t5_t6_identical"] = bool(np.allclose(X[:, t5], X[:, t6]))
    rep["t5_equals_minus_mean_of_real_channels_corr"] = float(np.corrcoef(X[:200, t5].ravel(), -X[:200][:, ci].mean(1).ravel())[0, 1])
    ex = SpectralNodeFeatures(sfreq=256.0).eval()
    Xs = X[:, ci, :]; Xs = (Xs - Xs.mean(1, keepdims=True)).astype(np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(Xs), 256):
            xb = torch.from_numpy(Xs[i:i + 256])
            feats.append(torch.cat([ex(xb), M242.extended_features_torch(xb, ex, keep)], -1).numpy())
    F17 = np.concatenate(feats).astype(np.float32); G = g[idx]

    def auc(yy, ss):
        r = rankdata(ss, method="average"); n1 = yy.sum(); n0 = yy.size - n1
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    for name, nf in (("spectralite16", 16), ("spectral75k", 12)):
        sc = []
        for f in sorted((CAN / "CN_vs_MCI").glob(f"{name}__seed*.npz")):
            seed = int(f.stem.split("__seed")[1])
            W = np.stack([probs(load_model("CN_vs_MCI", name, seed, k), F17[..., :nf], keep) for k in range(5)]).mean(0)
            sc.append(np.array([W[G == s].mean() for s in subs]))
        s17 = np.mean(sc, 0)
        s19 = np.mean([np.load(f, allow_pickle=True)["ext_score"] for f in sorted((CAN / "CN_vs_MCI").glob(f"{name}__seed*.npz"))], 0)
        rng = np.random.default_rng(258); d = []
        for _ in range(4000):
            ii = np.r_[rng.choice(np.flatnonzero(y == 1), y.sum()), rng.choice(np.flatnonzero(y == 0), (y == 0).sum())]
            d.append(auc(y[ii], s17[ii]) - auc(y[ii], s19[ii]))
        rep[name] = {"auc_17_masked": auc(y, s17), "auc_19_zero_filled": auc(y, s19),
                     "delta": auc(y, s17) - auc(y, s19), "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "n": int(y.size), "n_pos": int(y.sum())}
        np.savez_compressed(RES / f"CN_vs_MCI__{name}__geneeg17.npz", subjects=subs, y=y, score17=s17, score19=s19)
        print(f"[geneeg17] {name}: AUC 17-ch masked {rep[name]['auc_17_masked']:.3f} vs 19-ch zero-filled "
              f"{rep[name]['auc_19_zero_filled']:.3f} (delta {rep[name]['delta']:+.3f} {np.round(rep[name]['delta_ci95'], 3)})", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="perfold,geneeg17"); a = ap.parse_args()
    out_f = CAN / "review_rescore.json"
    rep = json.loads(out_f.read_text()) if out_f.exists() else {}
    for part in a.parts.split(","):
        rep[part] = {"perfold": part_perfold, "geneeg17": part_geneeg17}[part]()
        out_f.write_text(json.dumps(rep, indent=1))
