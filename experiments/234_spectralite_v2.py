"""SpectraLite-v2: transfer-improvement levers, trained on CAUEEG, evaluated on three external sites.

The paper's ceiling is site shift, so every lever here targets *transfer* and keeps the 75k-parameter,
gain-invariant, montage-agnostic design. Because the features are already gain-invariant, raw gain
randomisation is a no-op; the cross-site failure is a shift in the *feature distribution* (relative theta /
delta / entropy move by |g|>1.1 at P-ADIC), so domain randomisation and test-time calibration act in
feature space.

Levers (composable via flags):
  --aug      domain randomisation: per-recording feature offsets ~ N(0, s.std) and scale jitter, plus
             electrode dropout (replace an electrode's features with the training mean). Simulates the
             site-to-site distribution shift the model must survive.
  --calib    test-time feature calibration: recentre each feature by (target-control mean - CAUEEG-control
             mean), estimated from the receiving site's healthy controls only (no patient labels), the
             representation-level analogue of the split-conformal threshold.
  --extended transfer-robust features (lever 1) -- added in stage two once the extended cache is built.

Trains SpectraLite (src/models/spectral75k.py) on the cached node features (no re-extraction), five-fold
subject-grouped on CAUEEG, matched recipe; the five fold models are applied unchanged at each external
site. Reports internal OOF AUC and external AUC at ds004504 / P-ADIC / BrainLat, per config, vs the
released SpectraLite as the baseline.

    python experiments/234_spectralite_v2.py --task CN_vs_dementia --configs base,aug,aug_calib
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from src.models.spectral75k import SpectraLite

spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
FEAT, CAN = C.FEAT, C.OUT
TAGS = ("channel_clean", "padic_channel_clean", "brainlat_channel_clean")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def load_all_features(extended=False):
    """subject_id -> (n_windows, 19, F) float32, across every cohort tag (F=12, or 16 if extended)."""
    feats = {}
    for tag in TAGS:
        meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
        g = np.array([str(x) for x in meta["groups"]])
        N = np.load(FEAT / tag / "node12.npy")
        if extended:
            N = np.concatenate([N, np.load(FEAT / tag / "node_ext4.npy")], axis=-1)
        for s in np.unique(g):
            feats[s] = np.nan_to_num(N[g == s].astype(np.float32))
    return feats


def stack(subjects, labels, feats):
    Xs, ys, gs = [], [], []
    for sid, y in zip(subjects, labels):
        w = feats.get(str(sid))
        if w is None:
            continue
        Xs.append(w); ys.append(np.full(len(w), int(y))); gs.append([str(sid)] * len(w))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def subject_mean(win_prob, groups, subjects):
    return np.array([win_prob[groups == str(s)].mean() if (groups == str(s)).any() else np.nan for s in subjects])


class Aug:
    """Feature-space domain randomisation, fitted on the training windows."""
    def __init__(self, Xtr, offset=0.8, scale=0.10, drop_p=0.10):
        self.std = torch.tensor(Xtr.std(0), device=DEV)          # (19,12) per-channel per-feature std
        self.mean = torch.tensor(Xtr.mean(0), device=DEV)
        self.offset, self.scale, self.drop_p = offset, scale, drop_p

    def __call__(self, xb):                                       # xb: (B,19,F) on DEV
        B, Ch, Fd = xb.shape
        s = torch.rand(B, 1, 1, device=DEV) * self.offset         # per-sample strength in [0, offset]
        off = torch.randn(B, 1, Fd, device=DEV) * self.std.mean(0) * s     # per-(sample,feature), broadcast over electrodes
        xb = xb + off
        xb = xb * (1 + torch.randn(B, Ch, Fd, device=DEV) * self.scale)
        if self.drop_p > 0:                                       # electrode dropout -> training mean
            m = (torch.rand(B, Ch, 1, device=DEV) < self.drop_p)
            xb = torch.where(m, self.mean.unsqueeze(0), xb)
        return xb


EPOCHS = 25


def train_fold(Xtr, ytr, Xte_list, seed, aug=None, epochs=None, bs=128, n_features=12):
    epochs = EPOCHS if epochs is None else epochs
    torch.manual_seed(seed); np.random.seed(seed)
    model = SpectraLite(n_classes=2, n_features=n_features).to(DEV)
    w = torch.tensor([1.0 / max((ytr == c).sum(), 1) for c in (0, 1)], dtype=torch.float32, device=DEV)
    w = w / w.sum() * 2
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, device=DEV)
    steps = (len(Xt) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, epochs=epochs, steps_per_epoch=steps)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; xb = Xt[idx]
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad()
            loss = crit(model(node_features=xb), yt[idx]); loss.backward(); opt.step(); sched.step()
    model.eval()
    outs = []
    with torch.no_grad():
        for Xte in Xte_list:
            p = []
            for i in range(0, len(Xte), 512):
                p.append(torch.softmax(model(node_features=torch.tensor(Xte[i:i + 512], device=DEV)), 1)[:, 1].cpu().numpy())
            outs.append(np.concatenate(p) if p else np.array([]))
    return outs


def run(task, configs, seeds, extended=False):
    feats = load_all_features(extended)
    nf = 16 if extended else 12
    z = np.load(CAN / task / "spectral75k__seed42.npz", allow_pickle=True)
    tr_subj = z["train_subjects"]; tr_y = z["train_y"]; fold = z["fold_of_subject"]
    sites = {"ds004504": (z["ext_subjects"], z["ext_y"]),
             "P-ADIC": (z["padic_all_subjects"], z["padic_all_y"]),
             "BrainLat": (z["brainlat_subjects"], z["brainlat_y"])}
    # external window tensors + per-subject controls (for calibration)
    site_data = {}
    for name, (subj, y) in sites.items():
        Xe, ye_w, ge = stack(subj, y, feats)
        site_data[name] = dict(subj=subj, y=y, X=Xe, gw=ge)
    rep = {"task": task, "device": DEV, "configs": {}}
    for cfg in configs:
        use_aug = cfg in ("aug", "aug_calib")
        use_calib = cfg == "aug_calib"
        oof = np.full(len(tr_subj), np.nan)
        ext_fold = {name: [] for name in sites}
        for seed in seeds:
            for k in range(5):
                tr_mask = fold != k; te_mask = fold == k
                Xtr, ytr, _ = stack(tr_subj[tr_mask], tr_y[tr_mask], feats)
                Xoof, _, goof = stack(tr_subj[te_mask], tr_y[te_mask], feats)
                aug = Aug(Xtr) if use_aug else None
                # control-mean of the CAUEEG training fold (for calibration reference)
                ctrl_ref = Xtr[stack(tr_subj[tr_mask], tr_y[tr_mask], feats)[1] == 0].mean(0) if use_calib else None
                Xte_list = [Xoof] + [site_data[n]["X"] for n in sites]
                if use_calib:
                    shifted = []
                    for n in sites:
                        d = site_data[n]; ctrl = d["X"][np.isin(d["gw"], [str(s) for s, yy in zip(d["subj"], d["y"]) if yy == 0])]
                        shift = ctrl.mean(0) - ctrl_ref if len(ctrl) else 0.0
                        shifted.append(d["X"] - shift)
                    Xte_list = [Xoof] + shifted
                outs = train_fold(Xtr, ytr, Xte_list, seed, aug=aug, n_features=nf)
                if seed == seeds[0]:
                    oof[te_mask] = subject_mean(outs[0], goof, tr_subj[te_mask])
                for j, n in enumerate(sites):
                    ext_fold[n].append(subject_mean(outs[1 + j], site_data[n]["gw"], site_data[n]["subj"]))
        res = {"internal": round(auc(tr_y, oof), 3), "scores": {}}
        for n in sites:
            score = np.nanmean(ext_fold[n], 0)
            res[n] = round(auc(sites[n][1], score), 3)
            res["scores"][n] = {"y": [int(v) for v in sites[n][1]], "p": [float(v) for v in score]}
        rep["configs"][cfg] = res
        print(f"  {cfg:10s} internal {res['internal']}  ds {res['ds004504']}  P-ADIC {res['P-ADIC']}  BrainLat {res['BrainLat']}", flush=True)
    suffix = "_ext" if extended else ""
    (CAN / f"spectralite_v2_{task}{suffix}.json").write_text(json.dumps(rep, indent=1))
    print("wrote", CAN / f"spectralite_v2_{task}{suffix}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CN_vs_dementia")
    ap.add_argument("--configs", default="base,aug,aug_calib")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--extended", action="store_true", help="use the 16-feature transfer-robust set (lever 1)")
    args = ap.parse_args()
    global EPOCHS; EPOCHS = args.epochs
    print(f"== SpectraLite-v2 [{args.task}] on {DEV} ({EPOCHS} epochs, extended={args.extended}) ==", flush=True)
    run(args.task, args.configs.split(","), [int(s) for s in args.seeds.split(",")], extended=args.extended)


if __name__ == "__main__":
    main()
