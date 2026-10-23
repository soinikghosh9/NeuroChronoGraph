"""What Spectral-75k uses: electrode attention and feature attribution, with fold/seed stability (E10).

For each dementia-side task and every available canonical Spectral-75k checkpoint (seeds x 5 folds):
  * electrode attention: the gated-attention pool weights over the 19 electrodes, averaged over a
    subject's windows, then over controls and cases (internal held-out folds and the external site);
  * integrated gradients (32 steps) of the case-minus-control logit with respect to the 12 x 19 node
    features, baseline = mean node features of CAUEEG training controls; summarised per feature
    (summed over electrodes) and per electrode (summed over features) as mean |attribution|, plus the
    signed case-minus-control difference;
  * stability: Spearman correlation of the 12-feature importance vector between every pair of
    (seed, fold) models;
  * agreement with the random forest on the same features: grouped permutation importance (all 19
    columns of one feature permuted together) of the canonical rf_node folds at the external site.

    python experiments/219_attribution.py
Out: outputs/results/canonical/attribution.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
CAN = ROOT / "outputs" / "results" / "canonical"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
FEATURES = ("rel. delta", "rel. theta", "rel. alpha", "rel. beta", "rel. gamma", "alpha peak freq.",
            "alpha peak prominence", "spectral edge (95%)", "aperiodic exponent", "theta/alpha", "slowing ratio",
            "spectral entropy")


def main():
    import torch
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestClassifier
    from src.config.config import CHANNEL_NAMES
    from src.models.spectral75k import Spectral75k
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = np.load(C.FEAT / "channel_clean" / "node12.npy", mmap_mode="r")
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False); g = meta["groups"]
    rep = {"features": FEATURES, "electrodes": list(CHANNEL_NAMES), "tasks": {}}
    for task in TASKS:
        shards = sorted((CAN / task).glob("spectral75k__seed*.npz"))
        if not shards:
            continue
        z0 = np.load(shards[0], allow_pickle=True)
        tr, ytr, fold, te, yte = z0["train_subjects"], z0["train_y"], z0["fold_of_subject"], z0["ext_subjects"], z0["ext_y"]
        win_idx = {s: np.flatnonzero(g == s) for s in np.concatenate([tr, te])}
        base = torch.from_numpy(np.nanmean(np.concatenate([np.asarray(F[win_idx[s]]) for s in tr[ytr == 0]]), 0)).float().to(dev)
        att = {"internal": {0: [], 1: []}, "external": {0: [], 1: []}}
        ig_feat, ig_elec, ig_signed = [], [], {0: [], 1: []}
        for sh in shards:
            seed = int(sh.stem.split("seed")[1])
            for k in range(5):
                ck = C.CKPT / task / "spectral75k" / f"seed{seed}" / f"fold{k}.pt"
                if not ck.exists():
                    continue
                m = Spectral75k(n_classes=2).to(dev); m.load_state_dict(torch.load(ck, map_location=dev, weights_only=True)); m.eval()
                sets = (("internal", tr[fold == k], ytr[fold == k]), ("external", te, yte))
                feat_imp = np.zeros(12); elec_imp = np.zeros(19)
                for side, subs, ys in sets:
                    for s, yy in zip(subs, ys):
                        x = torch.from_numpy(np.nan_to_num(np.asarray(F[win_idx[s]]))).float().to(dev)
                        with torch.no_grad():
                            _, w = m(node_features=x, return_attention=True)
                        att[side][int(yy)].append(w.mean(0).cpu().numpy())
                        if side == "external":
                            alphas = torch.linspace(0, 1, 32, device=dev).view(-1, 1, 1, 1)
                            path = (base + alphas * (x.unsqueeze(0) - base)).reshape(-1, 19, 12).requires_grad_(True)
                            out = m(node_features=path)
                            (out[:, 1] - out[:, 0]).sum().backward()
                            grad = path.grad.reshape(32, len(x), 19, 12).mean(0)
                            ig = ((x - base) * grad).detach().cpu().numpy().mean(0)       # [19, 12]
                            feat_imp += np.abs(ig).sum(0); elec_imp += np.abs(ig).sum(1)
                            ig_signed[int(yy)].append(ig)
                ig_feat.append(feat_imp / len(te)); ig_elec.append(elec_imp / len(te))
        stab = [spearmanr(a, b).statistic for a, b in combinations(ig_feat, 2)]
        # random forest grouped permutation importance at the external site
        rf_imp = None
        f212 = CAN / task / "rf_node__seed42.npz"
        if f212.exists():
            med = lambda s: np.nan_to_num(np.nanmedian(np.asarray(F[win_idx[s]]).reshape(-1, 228), 0))
            Xtr = np.stack([med(s) for s in tr]); Xte = np.stack([med(s) for s in te])
            fits = [RandomForestClassifier(400, random_state=42, n_jobs=6, class_weight="balanced").fit(Xtr[fold != k], ytr[fold != k]) for k in range(5)]
            pred = lambda X: np.mean([f.predict_proba(X)[:, 1] for f in fits], 0)
            base_auc = C.auc(yte, pred(Xte)); rng = np.random.default_rng(0); rf_imp = []
            for j in range(12):
                cols = [c * 12 + j for c in range(19)]; drops = []
                for _ in range(20):
                    Xp = Xte.copy(); perm = rng.permutation(len(Xp)); Xp[:, cols] = Xp[perm][:, cols]
                    drops.append(base_auc - C.auc(yte, pred(Xp)))
                rf_imp.append(float(np.mean(drops)))
        mean_feat = np.mean(ig_feat, 0)
        rep["tasks"][task] = {
            "n_models": len(ig_feat),
            "attention": {side: {"control": np.mean(att[side][0], 0).tolist(), "case": np.mean(att[side][1], 0).tolist()}
                          for side in att},
            "ig_feature_importance_mean": mean_feat.tolist(), "ig_feature_importance_sd": np.std(ig_feat, 0).tolist(),
            "ig_electrode_importance_mean": np.mean(ig_elec, 0).tolist(),
            "ig_signed_case": np.mean(ig_signed[1], 0).tolist(), "ig_signed_control": np.mean(ig_signed[0], 0).tolist(),
            "stability_spearman_feature_importance": {"median": float(np.median(stab)), "min": float(np.min(stab))},
            "rf_node_permutation_importance": rf_imp,
            "spearman_ig_vs_rf": float(spearmanr(mean_feat, rf_imp).statistic) if rf_imp else None}
        order = np.argsort(-mean_feat)
        print(f"{task}: top features {[FEATURES[i] for i in order[:4]]} | stability rho median "
              f"{np.median(stab):.2f} | IG vs RF rho {rep['tasks'][task]['spearman_ig_vs_rf']}", flush=True)
        (CAN / "attribution.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
