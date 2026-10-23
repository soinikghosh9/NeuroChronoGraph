"""SpectraLite transfer levers under a locked protocol (model_paper/notes/IMPROVEMENT_PROTOCOL_2026-10-02.md).

Regime R1 (CAUEEG-only training; same subjects, folds, seeds and recipe as 237 unless stated):
  sl16_dr     feature-space domain randomisation (Aug of script 234, its fixed defaults)
  sl16_tuned  peak LR 1e-3, <=50 epochs, early stopping on a 15% inner-validation split (patience 8)
  f16         class-balanced random forest on subject-median 19x16 features
Regime R2 (leave-one-cohort-out multi-cohort training; the held-out cohort is never touched):
  sl16_mc     SpectraLite-16 on the CAUEEG training fold + all task-eligible subjects of the other two
              external cohorts, cohort- and class-balanced sample weights
  f16_mc      the forest on the same pooled data and weights

Scores are written per seed in the canonical shard format (train_subjects, train_y, fold_of_subject,
oof_score, <site>_subjects/_y/_score) under outputs/results/canonical/improvement/<task>/, so the
clinical-metrics script reads them exactly like the benchmark shards. Every unit is skipped if its shard
exists, so the run can be restarted after an interruption.

    python experiments/249_spectralite_transfer_levers.py --task CN_vs_dementia --part r1
    python experiments/249_spectralite_transfer_levers.py --task CN_vs_dementia --part r2
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
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.spectral75k import SpectraLite

spec = importlib.util.spec_from_file_location("c211", ROOT / "experiments" / "211_canonical_benchmark.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
spec = importlib.util.spec_from_file_location("v2_234", ROOT / "experiments" / "234_spectralite_v2.py")
V2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V2)
CAN, FEAT = C.OUT, C.FEAT
OUT = CAN / "improvement"
STATUS = ROOT / "outputs" / "logs" / "improvement_status.txt"
TAGS = ("channel_clean", "padic_channel_clean", "brainlat_channel_clean")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (42, 43, 44)
NFEAT, BS = 16, 128
# external evaluation sets in the canonical shards: (shard prefix, display name)
SITES = (("ext", "ds004504"), ("padic", "P-ADIC"), ("brainlat", "BrainLat"))


def status(msg):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def auc(y, s):
    return V2.auc(y, s)


def load_feats():
    feats = {}
    for tag in TAGS:
        meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
        g = np.array([str(x) for x in meta["groups"]])
        N = np.concatenate([np.load(FEAT / tag / "node12.npy"), np.load(FEAT / tag / "node_ext4.npy")], -1)
        for s in np.unique(g):
            feats[s] = np.nan_to_num(N[g == s].astype(np.float32))
    return feats


def stack(subs, ys, feats, cohort_id=None):
    """Window tensor, window labels, window groups (and window cohort ids when given)."""
    Xs, yw, gs, cw = [], [], [], []
    for sid, y in zip(subs, ys):
        w = feats.get(str(sid))
        if w is None:
            continue
        Xs.append(w); yw.append(np.full(len(w), int(y))); gs.append(np.full(len(w), str(sid), dtype=object))
        if cohort_id is not None:
            cw.append(np.full(len(w), cohort_id))
    out = (np.concatenate(Xs), np.concatenate(yw), np.concatenate(gs).astype(str))
    return out + ((np.concatenate(cw),) if cohort_id is not None else ())


def subj_mean(win, groups, subs):
    return np.array([win[groups == str(s)].mean() if (groups == str(s)).any() else np.nan for s in subs])


def balanced_weights(y, cohort):
    """Every (cohort, class) cell gets equal total weight; mean weight is one."""
    w = np.zeros(len(y), np.float32)
    cells = [(c, k) for c in np.unique(cohort) for k in (0, 1) if ((cohort == c) & (y == k)).any()]
    for c, k in cells:
        m = (cohort == c) & (y == k); w[m] = 1.0 / m.sum()
    return w * len(y) / w.sum()


@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 1024):
        out.append(torch.softmax(model(node_features=torch.tensor(X[i:i + 1024], device=DEV)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_net(Xtr, ytr, seed, *, aug=False, sample_w=None, lr=3e-3, epochs=25, inner=None):
    """Canonical recipe (237) by default. sample_w replaces class weights by per-window weights.
    inner = (Xval, gval, val_sub, val_y): early stopping on inner-validation subject AUC, patience 8."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SpectraLite(n_classes=2, n_features=NFEAT).to(DEV)
    if sample_w is None:
        w = torch.tensor([1.0 / max((ytr == c).sum(), 1) for c in (0, 1)], dtype=torch.float32, device=DEV)
        crit = nn.CrossEntropyLoss(weight=w / w.sum() * 2)
    else:
        crit_raw = nn.CrossEntropyLoss(reduction="none")
        sw = torch.tensor(sample_w, device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, device=DEV)
    steps = (len(Xt) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=steps)
    A = V2.Aug(Xtr) if aug else None
    best, best_state, since, used = -1.0, None, 0, epochs
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]; xb = Xt[idx]
            if A is not None:
                xb = A(xb)
            opt.zero_grad()
            logits = model(node_features=xb)
            loss = crit(logits, yt[idx]) if sample_w is None else (crit_raw(logits, yt[idx]) * sw[idx]).mean()
            loss.backward(); opt.step(); sched.step()
        if inner is not None:
            Xv, gv, sv, yv = inner
            a = auc(yv, subj_mean(predict(model, Xv), gv, sv))
            if a > best + 1e-4:
                best, since, used = a, 0, ep + 1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                since += 1
                if since >= 8:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, used


def forest_scores(Xtr_subj, ytr_subj, Xte_list, seed, sample_w=None):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(400, random_state=seed, n_jobs=3,
                                class_weight=None if sample_w is not None else "balanced")
    rf.fit(Xtr_subj, ytr_subj, sample_weight=sample_w)
    return [rf.predict_proba(X)[:, 1] for X in Xte_list]


def subj_medians(subs, feats):
    return np.stack([np.median(feats[str(s)], 0).reshape(-1) for s in subs])


def canonical(task):
    z = dict(np.load(CAN / task / "spectralite16__seed42.npz", allow_pickle=True))
    tr, ytr, fold = z["train_subjects"], z["train_y"], z["fold_of_subject"]
    sites = {pref: (z[f"{pref}_subjects"], z[f"{pref}_y"]) for pref, _ in SITES}
    return tr, ytr, fold, sites


def save(path, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out); os.replace(tmp, path)


def run_r1(task, feats, cands):
    tr, ytr, fold, sites = canonical(task)
    siteX = {p: stack(s, y, feats) for p, (s, y) in sites.items()}
    for cand in cands:
        for seed in SEEDS:
            path = OUT / task / f"{cand}__seed{seed}.npz"
            if path.exists():
                status(f"skip {task} {cand} seed{seed} (exists)"); continue
            t0 = time.time(); oof = np.full(len(tr), np.nan); acc = {p: [] for p in sites}; epochs_used = []
            for k in range(5):
                trm, tem = fold != k, fold == k
                if cand == "f16":
                    Xs = subj_medians(tr[trm], feats)
                    te = [subj_medians(tr[tem], feats)] + [subj_medians(sites[p][0], feats) for p in sites]
                    outs = forest_scores(Xs, ytr[trm], te, seed)
                    oof[tem] = outs[0]
                    for j, p in enumerate(sites):
                        acc[p].append(outs[1 + j])
                    continue
                inner = None; fit_sub, fit_y = tr[trm], ytr[trm]
                if cand == "sl16_tuned":
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=seed * 10 + k)
                    fi, vi = next(sss.split(fit_sub, fit_y))
                    Xv, _, gv = stack(fit_sub[vi], fit_y[vi], feats)
                    inner = (Xv, gv, fit_sub[vi], fit_y[vi]); fit_sub, fit_y = fit_sub[fi], fit_y[fi]
                Xtr, yw, _ = stack(fit_sub, fit_y, feats)
                model, used = train_net(Xtr, yw, seed, aug=(cand == "sl16_dr"),
                                        lr=1e-3 if cand == "sl16_tuned" else 3e-3,
                                        epochs=50 if cand == "sl16_tuned" else 25, inner=inner)
                epochs_used.append(used)
                Xo, _, go = stack(tr[tem], ytr[tem], feats)
                oof[tem] = subj_mean(predict(model, Xo), go, tr[tem])
                for p, (s, _) in sites.items():
                    acc[p].append(subj_mean(predict(model, siteX[p][0]), siteX[p][2], s))
                del model; torch.cuda.empty_cache()
            out = dict(train_subjects=tr, train_y=ytr, fold_of_subject=fold, oof_score=oof.astype(np.float32),
                       epochs_used=np.asarray(epochs_used))
            for p, (s, y) in sites.items():
                out[f"{p}_subjects"] = s; out[f"{p}_y"] = y; out[f"{p}_score"] = np.mean(acc[p], 0).astype(np.float32)
            save(path, out)
            status(f"done {task} {cand} seed{seed} in {time.time() - t0:.0f}s | int {auc(ytr, oof):.3f} "
                   + " ".join(f"{n} {auc(sites[p][1], out[p + '_score']):.3f}" for p, n in SITES))


def run_r2(task, feats, cands):
    tr, ytr, fold, sites = canonical(task)
    for held, held_name in SITES:
        donors = [p for p, _ in SITES if p != held]
        for cand in cands:
            for seed in SEEDS:
                path = OUT / task / f"{cand}__holdout-{held_name}__seed{seed}.npz"
                if path.exists():
                    status(f"skip {task} {cand} holdout {held_name} seed{seed} (exists)"); continue
                t0 = time.time(); oof = np.full(len(tr), np.nan); acc = []
                for k in range(5):
                    trm, tem = fold != k, fold == k
                    parts_sub = [tr[trm]] + [sites[p][0] for p in donors]
                    parts_y = [ytr[trm]] + [sites[p][1] for p in donors]
                    if cand == "f16_mc":
                        subs = np.concatenate(parts_sub); ys = np.concatenate(parts_y)
                        coh = np.concatenate([np.full(len(s), i) for i, s in enumerate(parts_sub)])
                        w = balanced_weights(ys, coh)
                        outs = forest_scores(subj_medians(subs, feats), ys,
                                             [subj_medians(tr[tem], feats), subj_medians(sites[held][0], feats)],
                                             seed, sample_w=w)
                        oof[tem] = outs[0]; acc.append(outs[1]); continue
                    Xs, ys, cs = [], [], []
                    for i, (s, y) in enumerate(zip(parts_sub, parts_y)):
                        X_, y_, _, c_ = stack(s, y, feats, cohort_id=i)
                        Xs.append(X_); ys.append(y_); cs.append(c_)
                    Xtr, yw, cw = np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs)
                    model, _ = train_net(Xtr, yw, seed, sample_w=balanced_weights(yw, cw))
                    Xo, _, go = stack(tr[tem], ytr[tem], feats)
                    oof[tem] = subj_mean(predict(model, Xo), go, tr[tem])
                    Xh, _, gh = stack(sites[held][0], sites[held][1], feats)
                    acc.append(subj_mean(predict(model, Xh), gh, sites[held][0]))
                    del model; torch.cuda.empty_cache()
                s, y = sites[held]
                out = dict(train_subjects=tr, train_y=ytr, fold_of_subject=fold, oof_score=oof.astype(np.float32),
                           **{f"{held}_subjects": s, f"{held}_y": y, f"{held}_score": np.mean(acc, 0).astype(np.float32)},
                           donors=np.asarray([n for p, n in SITES if p != held]))
                save(path, out)
                status(f"done {task} {cand} holdout {held_name} seed{seed} in {time.time() - t0:.0f}s | "
                       f"int {auc(ytr, oof):.3f} {held_name} {auc(y, out[held + '_score']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CN_vs_dementia")
    ap.add_argument("--part", choices=("r1", "r2"), default="r1")
    ap.add_argument("--cands", default=None)
    args = ap.parse_args()
    C.set_determinism(torch)
    if DEV != "cuda":
        status("WARNING: CUDA not available; refusing to train on CPU"); sys.exit(2)
    feats = load_feats()
    status(f"start {args.task} {args.part} on {DEV}")
    if args.part == "r1":
        run_r1(args.task, feats, (args.cands or "sl16_dr,sl16_tuned,f16").split(","))
    else:
        run_r2(args.task, feats, (args.cands or "sl16_mc,f16_mc").split(","))
    status(f"finished {args.task} {args.part}")


if __name__ == "__main__":
    main()
