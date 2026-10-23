"""Integrated-gradient attribution for the released 16-feature SpectraLite.

Shows how much each of the sixteen per-electrode features drives the case-minus-control decision, and in
particular whether the four transfer-robust additions (indices 12-15) carry weight. Uses the canonical
16-feature checkpoints from 237; attribution is the mean over folds and seeds of the integrated gradient of
the positive-class logit with respect to each feature, baseline the mean development-control features,
averaged over electrodes and subjects. Stability is the rank correlation of the 16-vector across folds/seeds.

    python experiments/239_attribution16.py
Out: outputs/results/canonical/attribution16.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from src.models.spectral75k import SpectraLite

spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
FEAT, CKPT, CAN = C.FEAT, C.CKPT, C.OUT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = ["rel delta", "rel theta", "rel alpha", "rel beta", "rel gamma", "IAF", "alpha prominence",
         "spectral edge", "aperiodic exponent", "theta/alpha", "slowing", "spectral entropy",
         "IAF-ref theta", "IAF-ref alpha", "aperiodic offset", "alpha asymmetry"]
NEW = [12, 13, 14, 15]


def subject_features16(task):
    """CAUEEG development subjects: subject-mean 16-feature vectors, labels, and control mean (baseline)."""
    z = np.load(CAN / task / "spectralite16__seed42.npz", allow_pickle=True)
    subs, y = z["train_subjects"], z["train_y"]
    meta = np.load(FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    g = np.array([str(x) for x in meta["groups"]])
    N = np.concatenate([np.load(FEAT / "channel_clean" / "node12.npy"),
                        np.load(FEAT / "channel_clean" / "node_ext4.npy")], -1)   # (win,19,16)
    feat = {}
    for s in np.unique(g):
        i = np.flatnonzero(g == s)
        feat[str(s)] = np.nan_to_num(np.nanmean(N[i], 0))   # (19,16) subject mean
    X = np.array([feat[str(s)] for s in subs if str(s) in feat], dtype=np.float32)
    yy = np.array([int(v) for s, v in zip(subs, y) if str(s) in feat])
    return X, yy


def integrated_gradients(model, X, baseline, steps=32):
    """X: (B,19,16) tensor. Returns mean |IG| over subjects and electrodes -> (16,)."""
    B = X.shape[0]
    total = torch.zeros_like(X)
    for a in np.linspace(0, 1, steps):
        xi = (baseline + float(a) * (X - baseline)).clone().requires_grad_(True)
        logit = model(node_features=xi)[:, 1].sum()
        g, = torch.autograd.grad(logit, xi)
        total += g
    ig = (X - baseline) * total / steps          # (B,19,16)
    return ig.abs().mean(dim=(0, 1)).detach().cpu().numpy()   # (16,)


def main():
    task = "CN_vs_dementia"
    X, y = subject_features16(task)
    Xt = torch.tensor(X, device=DEV)
    baseline = torch.tensor(X[y == 0].mean(0, keepdims=True), device=DEV)   # mean control (1,19,16)
    vecs = []
    for seed in (42, 43, 44):
        for k in range(5):
            ck = CKPT / task / "spectralite16" / f"seed{seed}" / f"fold{k}.pt"
            if not ck.exists():
                continue
            m = SpectraLite(n_classes=2, n_features=16).to(DEV)
            m.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True)); m.eval()
            vecs.append(integrated_gradients(m, Xt, baseline))
    A = np.array(vecs)                      # (n_models, 16)
    imp = A.mean(0); imp = imp / imp.sum()  # normalised importance share
    # stability: rank correlation across the model instances
    rhos = [spearmanr(A[i], A[j]).correlation for i in range(len(A)) for j in range(i + 1, len(A))]
    order = np.argsort(-imp)
    rep = {"task": task, "n_models": len(A), "feature_importance_share": {NAMES[i]: round(float(imp[i]), 4) for i in order},
           "new_feature_indices": NEW, "new_feature_share": round(float(imp[NEW].sum()), 4),
           "new_feature_ranks": {NAMES[i]: int(np.where(order == i)[0][0] + 1) for i in NEW},
           "stability_spearman_median": round(float(np.median(rhos)), 3)}
    (CAN / "attribution16.json").write_text(json.dumps(rep, indent=1))
    print(f"attribution over {len(A)} checkpoints; stability Spearman median {rep['stability_spearman_median']}")
    print("ranked feature importance share:")
    for i in order:
        tag = "  <-- transfer-robust" if i in NEW else ""
        print(f"  {NAMES[i]:20s} {imp[i]*100:5.1f}%{tag}")
    print(f"\nfour transfer-robust features carry {rep['new_feature_share']*100:.1f}% of attribution; "
          f"ranks {rep['new_feature_ranks']}")


if __name__ == "__main__":
    main()
