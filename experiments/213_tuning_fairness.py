"""Tuning fairness: does a modest, internally-selected hyperparameter search change the capacity result?

Review C5: every network was trained with one recipe (max LR 3e-3, 25 epochs, final weights) and no
tuning, which invites "the high-capacity models were undertrained". Here each network gets a small
grid, selected ONLY on inner-validation subjects inside the training cohort; the external site is
touched once, after selection.

Grid: max LR in {3e-4, 1e-3, 3e-3} x regime in {fixed25: final weights after 25 epochs,
es50: up to 50 epochs keeping the epoch with the best inner-validation subject AUC, patience 8}.
Inside every outer training fold (same five folds as 211), 15% of subjects (stratified, seeded) are
held out as the inner validation set, and ALL configurations train on the same remaining 85%, so the
configurations are compared on equal data. The selected configuration is the one with the highest
mean inner-validation AUC across folds. We report internal OOF and external AUC for every
configuration (full transparency) and for the selected one.

    python experiments/213_tuning_fairness.py --task CN_vs_dementia --models spectral75k,ncg_v5,atcnet,conformer
Out: outputs/results/canonical/tuning/<task>/<model>__<config>.npz and tuning_<task>.json
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

OUT = C.OUT / "tuning"
LRS = (3e-4, 1e-3, 3e-3)
REGIMES = {"fixed25": (25, None), "es50": (50, 8)}
if os.environ.get("REV_EPOCHS"):                                         # smoke test only
    REGIMES = {"fixed25": (1, None), "es50": (2, 1)}; LRS = (3e-3,)


def run_config(torch, name, Xfit, yfit, Xval, gval, val_sub, val_y, Xeval_list, device, seed, e161,
               lr, epochs, patience):
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    model = C.build(name, device, e161)
    cls, cnt = np.unique(yfit, return_counts=True)
    w = np.zeros(2, np.float32); w[cls] = len(yfit) / (len(cls) * cnt)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * max(1, len(yfit) // C.BATCH + 1))
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(Xfit)),
                                  torch.from_numpy(yfit.astype(np.int64))),
                    batch_size=C.BATCH, shuffle=True, generator=g)
    best_auc, best_state, best_ep, since = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(e161.logits(model, xb.float().to(device)), yb.to(device))
            loss.backward(); opt.step()
            try:
                sched.step()
            except Exception:                                            # noqa: BLE001
                pass
        if patience is not None or ep == epochs - 1:
            v = C.subject_mean(C.predict(torch, model, Xval, device, e161), gval, val_sub)
            a = C.auc(val_y, v) or 0.0
            if patience is None:
                best_auc, best_ep = a, ep
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            elif a > best_auc + 1e-4:
                best_auc, best_ep, since = a, ep, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
    model.load_state_dict(best_state)
    preds = [C.predict(torch, model, X, device, e161) for X in Xeval_list]
    del model; torch.cuda.empty_cache()
    return best_auc, best_ep + 1, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--task", default="CN_vs_dementia")
    ap.add_argument("--models", default="spectral75k,ncg_v5,atcnet,conformer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    C.set_determinism(torch)
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    e161 = C.load_161()
    device = "cuda"
    assert torch.cuda.is_available()
    Xw = np.load(C.windows_path(args.tag), mmap_mode="r")
    meta = np.load(C.FEAT / args.tag / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    win_cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh = np.array([str(x) for x in meta["cohort"]])
    pos, neg, site = C.TASKS[args.task]
    keep = np.isin(win_cls, pos + neg); y = np.isin(win_cls, pos).astype(int)
    tr_sub = np.unique(groups[keep & (win_coh == "CAUEEG")])
    te_sub = np.unique(groups[keep & (win_coh == site)])
    tr_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in tr_sub])
    te_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in te_sub])
    idx_tr = np.flatnonzero(keep & (win_coh == "CAUEEG")); idx_te = np.flatnonzero(keep & (win_coh == site))
    Xtr_all = np.asarray(Xw[idx_tr]); gtr = groups[idx_tr]; ytr_all = y[idx_tr]
    Xte = np.asarray(Xw[idx_te]); gte = groups[idx_te]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
    out_dir = OUT / args.task; out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / f"tuning_{args.task}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for name in [m for m in args.models.split(",") if m]:
        for lr in LRS:
            for regime, (epochs, patience) in REGIMES.items():
                cfg = f"lr{lr:g}_{regime}"
                dest = out_dir / f"{name}__{cfg}.npz"
                if dest.exists():
                    continue
                t0 = time.time()
                oof = np.full(len(tr_sub), np.nan, np.float32); ext = []; vals = []; eps = []
                for k, (a, b) in enumerate(folds):
                    sub_a = tr_sub[a]; y_a = tr_y[a]
                    fit_i, val_i = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=100 + k)
                                        .split(sub_a, y_a))
                    fit_sub, val_sub = sub_a[fit_i], sub_a[val_i]
                    mfit = np.isin(gtr, fit_sub); mval = np.isin(gtr, val_sub); mheld = np.isin(gtr, tr_sub[b])
                    va, ep_used, (p_held, p_ext) = run_config(
                        torch, name, Xtr_all[mfit], ytr_all[mfit], Xtr_all[mval], gtr[mval], val_sub,
                        tr_y[np.searchsorted(tr_sub, val_sub)], [Xtr_all[mheld], Xte], device, args.seed,
                        e161, lr, epochs, patience)
                    oof[b] = C.subject_mean(p_held, gtr[mheld], tr_sub[b]); ext.append(p_ext)
                    vals.append(va); eps.append(ep_used)
                ext_sub = C.subject_mean(np.mean(ext, 0), gte, te_sub)
                np.savez_compressed(dest, train_subjects=tr_sub, train_y=tr_y, oof_score=oof,
                                    ext_subjects=te_sub, ext_y=te_y, ext_score=ext_sub)
                row = {"model": name, "config": cfg, "lr": lr, "regime": regime,
                       "inner_val_auc_mean": float(np.mean(vals)), "inner_val_auc_folds": vals,
                       "epochs_used": eps, "auc_internal_oof": C.auc(tr_y, oof),
                       "auc_external": C.auc(te_y, ext_sub), "minutes": round((time.time() - t0) / 60, 1)}
                summary.setdefault(name, {})[cfg] = row
                summary_path.write_text(json.dumps(summary, indent=1))
                print(f"   {name:12s} {cfg:16s} inner-val {row['inner_val_auc_mean']:.3f} | internal "
                      f"{row['auc_internal_oof']:.3f} | external {row['auc_external']:.3f} | epochs {eps} "
                      f"| {row['minutes']} min", flush=True)
        if name in summary and summary[name]:
            sel = max(summary[name].values(), key=lambda r: r["inner_val_auc_mean"])
            summary[name]["_selected"] = sel["config"]
            summary_path.write_text(json.dumps(summary, indent=1))
            print(f"   {name}: selected {sel['config']} -> external {sel['auc_external']:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
