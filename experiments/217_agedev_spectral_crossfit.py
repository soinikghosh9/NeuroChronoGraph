"""Age-normative referencing for Spectral-75k, with the anchor fitted OUT OF SAMPLE.

Review C1: exp 207 fitted the target anchor on all 29 ds004504 controls and scored specificity on the
same controls. The anchor only transforms TEST inputs, so the trained fold models can be reused and
the anchor refitted per split. This script:
  1. replicates 207's age-deviation arm (node features replaced by z = (x - mu(age)) / sigma(age), the
     normative model fitted on CAUEEG training controls at the subject-median level, applied per
     window) and trains the five-fold Spectral-75k on it (seed 42), recording source OOF scores;
  2. for k in {5,10,15,20,25} and 200 shared splits (_calib_splits), refits the robust anchor
     (median/MAD of the k anchor controls' deviations) and scores all external subjects.
The operating-point metrics for every method are computed in 221 on the same splits.

    python experiments/217_agedev_spectral_crossfit.py
Out: outputs/results/canonical/calibration/agedev_s75__<task>.npz
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("c211", "211_canonical_benchmark.py")
A = _load("a207", "207_agecond_spectral.py")          # norm_fit / norm_dev / fit_anchor / load_ages / keyf
from _calib_splits import K_VALUES, N_SPLITS, anchor_splits  # noqa: E402

OUT = C.OUT / "calibration"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")


def main():
    import torch
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.spectral75k import Spectral75k
    C.set_determinism(torch)
    dev = "cuda"; assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    win_cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh = np.array([str(x) for x in meta["cohort"]])
    F = np.load(C.FEAT / "channel_clean" / "node12.npy", mmap_mode="r")
    amap = A.load_ages()
    win_age = np.array([amap.get(A.keyf(g, c), np.nan) for g, c in zip(groups, win_coh)], float)

    for task in TASKS:
        dest = OUT / f"agedev_s75__{task}.npz"
        if dest.exists():
            continue
        t0 = time.time()
        pos, neg, site = C.TASKS[task]
        keep = np.isin(win_cls, pos + neg) & np.isfinite(win_age)
        y = np.isin(win_cls, pos).astype(int)
        tr_sub = np.unique(groups[keep & (win_coh == "CAUEEG")]); te_sub = np.unique(groups[keep & (win_coh == site)])
        tr_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in tr_sub])
        te_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in te_sub])
        idx = np.flatnonzero(keep); gk = groups[idx]; Fk = np.asarray(F[idx]).reshape(len(idx), -1); ak = win_age[idx]
        s_med = {s: np.nanmedian(Fk[gk == s], 0) for s in np.unique(gk)}
        s_age = {s: ak[gk == s][0] for s in np.unique(gk)}
        ctrl_tr = [s for s, yy in zip(tr_sub, tr_y) if yy == 0]
        nm = A.norm_fit(np.stack([s_med[s] for s in ctrl_tr]), np.array([s_age[s] for s in ctrl_tr]))
        Z = A.norm_dev(nm, Fk, ak).reshape(-1, 19, 12).astype(np.float32)
        is_tr = np.isin(gk, tr_sub); is_te = np.isin(gk, te_sub)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
        models, oof_w = [], np.full(len(idx), np.nan, np.float32)
        for a, b in folds:
            fit = np.isin(gk, tr_sub[a]); held = np.isin(gk, tr_sub[b]); ytr = y[idx][fit]
            torch.manual_seed(42); m = Spectral75k(n_classes=2).to(dev)
            cls, cnt = np.unique(ytr, return_counts=True); w = np.zeros(2, np.float32); w[cls] = len(ytr) / (len(cls) * cnt)
            crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))
            opt = torch.optim.AdamW(m.parameters(), lr=C.LR, weight_decay=C.WD)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR, total_steps=C.EPOCHS * (len(ytr) // C.BATCH + 1))
            gen = torch.Generator(); gen.manual_seed(42)
            dl = DataLoader(TensorDataset(torch.from_numpy(Z[fit]), torch.from_numpy(ytr.astype(np.int64))),
                            batch_size=C.BATCH, shuffle=True, generator=gen)
            m.train()
            for _ in range(C.EPOCHS):
                for xb, yb in dl:
                    opt.zero_grad(); loss = crit(m(node_features=xb.to(dev)), yb.to(dev)); loss.backward(); opt.step(); sched.step()
            m.eval()
            with torch.no_grad():
                oof_w[held] = torch.softmax(m(node_features=torch.from_numpy(Z[held]).to(dev)), 1)[:, 1].cpu().numpy()
            models.append(m)
        oof = C.subject_mean(oof_w, gk, tr_sub)
        Fte, gte, ate = Fk[is_te], gk[is_te], ak[is_te]

        def score(anchor):
            Zt = torch.from_numpy(A.norm_dev(nm, Fte, ate, anchor=anchor).reshape(-1, 19, 12).astype(np.float32)).to(dev)
            with torch.no_grad():
                p = np.mean([torch.softmax(m(node_features=Zt), 1)[:, 1].cpu().numpy() for m in models], 0)
            return C.subject_mean(p, gte, te_sub)

        unanchored = score(None)
        ctrl_te = [s for s, yy in zip(te_sub, te_y) if yy == 0]
        anchored = np.zeros((len(K_VALUES), N_SPLITS, len(te_sub)), np.float32)
        for ki, k in enumerate(K_VALUES):
            for r, (anc, _) in enumerate(anchor_splits(ctrl_te, k)):
                anchor = A.fit_anchor(nm, np.stack([s_med[s] for s in anc]), np.array([s_age[s] for s in anc]))
                anchored[ki, r] = score(anchor)
        np.savez_compressed(dest, train_subjects=tr_sub, train_y=tr_y, oof_score=oof, ext_subjects=te_sub,
                            ext_y=te_y, ext_unanchored=unanchored, ext_anchored=anchored, k_values=np.array(K_VALUES))
        print(f"{task}: internal OOF {C.auc(tr_y, oof):.3f} | external unanchored {C.auc(te_y, unanchored):.3f} "
              f"| anchored k=20 mean AUC {np.mean([C.auc(te_y, s) for s in anchored[K_VALUES.index(20)]]):.3f} "
              f"| {(time.time()-t0)/60:.1f} min", flush=True)
        del models; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
