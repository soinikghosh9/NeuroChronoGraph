"""Promote the 16-feature SpectraLite to a first-class canonical model.

The released SpectraLite is the transfer-robust 16-feature model (the 12 spectral node features plus
individual-alpha-referenced theta/alpha, an aperiodic offset and an inter-hemispheric alpha asymmetry).
This trains it under the exact canonical recipe (matched to 211: five-fold subject-grouped, AdamW one-cycle
3e-3, 25 epochs, batch 128, class-weighted CE, final-epoch weights, deterministic; no augmentation), and
writes shards and per-fold checkpoints in the same format as every other model so 225 (benchmark,
equivalence), 218 (edge), 221/223 (calibration), 219 (attribution) can use it. The 12-feature model keeps
its shards as the "spectral-only" ablation.

Subjects, labels and folds are taken from the existing spectral75k shard, so the 16-feature model is scored
on identical subjects at internal out-of-fold, ds004504, P-ADIC (age-restricted, the benchmark set) and
BrainLat.

    python experiments/237_canonical_spectralite16.py --task CN_vs_dementia
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.spectral75k import SpectraLite

spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
CAN, CKPT, FEAT = C.OUT, C.CKPT, C.FEAT
TAGS = ("channel_clean", "padic_channel_clean", "brainlat_channel_clean")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NFEAT = 16
EPOCHS, BS = 25, 128


def load_feats():
    feats = {}
    for tag in TAGS:
        meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
        g = np.array([str(x) for x in meta["groups"]])
        N = np.concatenate([np.load(FEAT / tag / "node12.npy"), np.load(FEAT / tag / "node_ext4.npy")], -1)
        for s in np.unique(g):
            feats[s] = np.nan_to_num(N[g == s].astype(np.float32))
    return feats


def stack(subs, ys, feats):
    Xs, gs = [], []
    for sid, y in zip(subs, ys):
        w = feats.get(str(sid))
        if w is None:
            continue
        Xs.append(w); gs.append([str(sid)] * len(w))
    return np.concatenate(Xs), np.concatenate(gs)


def subj_mean(win, groups, subs):
    return np.array([win[groups == str(s)].mean() if (groups == str(s)).any() else np.nan for s in subs])


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, int); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def train(Xtr, ytr, seed, ckpt_path):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SpectraLite(n_classes=2, n_features=NFEAT).to(DEV)
    w = torch.tensor([1.0 / max((ytr == c).sum(), 1) for c in (0, 1)], dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w / w.sum() * 2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, device=DEV)
    steps = (len(Xt) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, epochs=EPOCHS, steps_per_epoch=steps)
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            opt.zero_grad(); crit(model(node_features=Xt[idx]), yt[idx]).backward(); opt.step(); sched.step()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 512):
        out.append(torch.softmax(model(node_features=torch.tensor(X[i:i + 512], device=DEV)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def run(task, seeds=(42, 43, 44)):
    z = dict(np.load(CAN / task / "spectral75k__seed42.npz", allow_pickle=True))
    tr, ytr_all, fold = z["train_subjects"], z["train_y"], z["fold_of_subject"]
    feats = load_feats()
    # evaluation sites present in the reference shard (padic = age-restricted benchmark set)
    site_keys = [("ext", "ext_subjects", "ext_y"), ("padic", "padic_subjects", "padic_y"),
                 ("padic_all", "padic_all_subjects", "padic_all_y"), ("brainlat", "brainlat_subjects", "brainlat_y")]
    sites = {n: (z[ks], z[ky]) for n, ks, ky in site_keys if ks in z}
    site_X = {n: stack(subj, y, feats) for n, (subj, y) in sites.items()}
    for seed in seeds:
        oof = np.full(len(tr), np.nan)
        # Keep the window-level predictions needed by the reliability and
        # recording-length analysis.  The subject-level scores remain the
        # primary benchmark artifact.
        train_groups = np.concatenate([np.full(len(feats[str(s)]), str(s)) for s in tr])
        oof_win = np.full(len(train_groups), np.nan, dtype=np.float32)
        acc = {n: [] for n in sites}
        win_acc = {n: [] for n in sites}
        for k in range(5):
            trm, tem = fold != k, fold == k
            Xtr, _ = stack(tr[trm], ytr_all[trm], feats)
            ywin = _win_labels(tr[trm], ytr_all[trm], feats)
            model = train(Xtr, ywin, seed, CKPT / task / "spectralite16" / f"seed{seed}" / f"fold{k}.pt")
            Xoof, goof = stack(tr[tem], ytr_all[tem], feats)
            poof = predict(model, Xoof)
            oof[tem] = subj_mean(poof, goof, tr[tem])
            for sid in tr[tem]:
                oof_win[train_groups == str(sid)] = poof[goof == str(sid)]
            for n, (subj, _) in sites.items():
                X, gw = site_X[n]
                pext = predict(model, X)
                win_acc[n].append(pext)
                acc[n].append(subj_mean(pext, gw, subj))
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()
        out = dict(train_subjects=tr, train_y=ytr_all, fold_of_subject=fold,
                   oof_score=oof.astype(np.float32), win_train_groups=train_groups,
                   oof_win=oof_win)
        for n, (subj, y) in sites.items():
            score = np.mean(acc[n], 0).astype(np.float32)
            pref = {"ext": "ext", "padic": "padic", "padic_all": "padic_all", "brainlat": "brainlat"}[n]
            out[f"{pref}_subjects"] = subj; out[f"{pref}_y"] = y; out[f"{pref}_score"] = score
            out[f"win_{pref}_groups"] = site_X[n][1]
            out[f"{pref}_win_folds"] = np.asarray(win_acc[n], dtype=np.float32)
        np.savez_compressed(CAN / task / f"spectralite16__seed{seed}.npz", **out)
        info = {"task": task, "model": "spectralite16", "seed": seed, "params": 75971,
                "auc_internal_oof": auc(ytr_all, oof),
                "auc_external": auc(sites["ext"][1], out["ext_score"]) if "ext" in sites else None,
                "auc_padic": auc(sites["padic"][1], out["padic_score"]) if "padic" in sites else None,
                "auc_brainlat": auc(sites["brainlat"][1], out["brainlat_score"]) if "brainlat" in sites else None,
                "n_brainlat": int(len(sites["brainlat"][0])) if "brainlat" in sites else None}
        (CAN / task / f"spectralite16__seed{seed}.json").write_text(json.dumps(info, indent=1))
        def show(v):
            return f"{v:.3f}" if v is not None else "NA"
        print(f"  [{task} seed{seed}] internal {show(info['auc_internal_oof'])} ds {show(info['auc_external'])} "
              f"P-ADIC {show(info['auc_padic'])} BrainLat {show(info['auc_brainlat'])}", flush=True)


def _win_labels(subs, ys, feats):
    """Per-window labels aligned to stack()'s window order."""
    lab = []
    for sid, y in zip(subs, ys):
        w = feats.get(str(sid))
        if w is not None:
            lab.append(np.full(len(w), int(y)))
    return np.concatenate(lab)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="CN_vs_dementia")
    a = ap.parse_args()
    print(f"== canonical SpectraLite-16 [{a.task}] on {DEV} ==", flush=True)
    run(a.task)


if __name__ == "__main__":
    main()
