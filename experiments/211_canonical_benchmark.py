"""Canonical benchmark: one training protocol, every network, internal AND external scores, checkpoints.

Why this exists (model-paper review 2026-09-17, C5/C9/W1). The published numbers for the same model
came from four separately-run experiments (161 per-task, 206 subtractive, 210 montage, 203 FMs) with
different absolute AUCs; Spectral-75k never entered the benchmark table; no deep model had internal
out-of-fold scores, so equivalence could only be tested at the n=87 external site; and no checkpoint
was saved, so a new external site, attribution, int8 export or test-time montage evaluation all
required retraining. This script fixes all four in one pass.

Protocol (identical to 161, which produced the published per-task arm): per task, binary training on
CAUEEG only; five-fold subject-grouped CV with the fold split fixed at seed 42 (StratifiedKFold over
the sorted subject list); AdamW, one-cycle to 3e-3, weight decay 1e-4, 25 epochs, batch 128,
class-weighted cross-entropy; the FINAL epoch's weights are used (no validation-based epoch
selection -- the manuscript previously said otherwise); subject score = mean window probability;
external score = mean over the five fold models. Unlike 161, cuDNN determinism is enforced.

Per (task, model, seed) it writes one shard with subject-level internal OOF scores, subject-level
external scores, and the window-level probabilities behind both (for reliability/recording-length
analyses), plus the five fold checkpoints (for scoring new sites without retraining).

    python experiments/211_canonical_benchmark.py --models spectral75k,ncg_v5 --seeds 42,43,44
Out: outputs/results/canonical/<task>/<model>__seed<s>.npz   (+ .json summary)
     <Alzfiles>/checkpoints/211/<task>/<model>/seed<s>/fold<k>.pt
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

ALZ = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles"))
FEAT = ROOT / "outputs" / "harmonised_features"
# REV_* environment overrides exist only for smoke tests; published runs use the defaults.
OUT = Path(os.environ.get("REV_OUT_ROOT", str(ROOT / "outputs" / "results" / "canonical")))
CKPT = Path(os.environ.get("REV_CKPT_ROOT", str(ALZ / "checkpoints" / "211")))
EPOCHS = int(os.environ.get("REV_EPOCHS", "25"))
BATCH, LR, WD, SPLIT_SEED = 128, 3e-3, 1e-4, 42

CLASS = {"AD": "AD", "A": "AD", "FTD": "FTD", "F": "FTD", "CN": "CN", "C": "CN",
         "CONTROL": "CN", "MCI": "MCI"}
TASKS = {"CN_vs_dementia": (("AD", "FTD"), ("CN",), "ds004504_raw"),
         "CN_vs_AD": (("AD",), ("CN",), "ds004504_raw"),
         "CN_vs_impaired": (("AD", "FTD", "MCI"), ("CN",), "ds004504_raw"),
         "CN_vs_MCI": (("MCI",), ("CN",), "GENEEG")}
ALL_MODELS = ("spectral75k", "ncg_v5", "eegnet", "shallow", "deep4", "eegitnet", "atcnet",
              "conformer", "eegnex", "ncg_v3", "ncg_v2")


def windows_path(tag):
    for base in (ROOT / "outputs" / "harmonised_windows", ALZ / "harmonised_windows"):
        if (base / tag / "windows.npy").exists():
            return base / tag / "windows.npy"
    raise SystemExit(f"windows.npy for {tag} not found")


def load_161():
    spec = importlib.util.spec_from_file_location("e161", ROOT / "experiments" / "161_deep_per_task.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build(name, device, e161):
    if name == "spectral75k":
        from src.models.spectral75k import Spectral75k
        return Spectral75k(n_classes=2).to(device)
    if name.startswith("ladder:"):
        from src.models.spectral75k import LADDER, LadderNet
        return LadderNet(**LADDER[name.split(":", 1)[1]]).to(device)
    return e161.make_binary(name, device)


def set_determinism(torch):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:                                                    # noqa: BLE001
        pass


def predict(torch, model, X, device, e161, bs=256):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).float().to(device)
            out.append(torch.softmax(e161.logits(model, xb), 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_one(torch, name, Xtr, ytr, device, seed, e161, lr=LR, epochs=EPOCHS, wd=WD):
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(name, device, e161)
    cls, cnt = np.unique(ytr, return_counts=True)
    w = np.zeros(2, np.float32); w[cls] = len(ytr) / (len(cls) * cnt)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * max(1, len(ytr) // BATCH + 1))
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(Xtr)),
                                  torch.from_numpy(ytr.astype(np.int64))),
                    batch_size=BATCH, shuffle=True, generator=g)
    model.train()
    last = float("nan")
    for _ in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(e161.logits(model, xb.float().to(device)), yb.to(device))
            loss.backward(); opt.step()
            try:
                sched.step()
            except Exception:                                            # noqa: BLE001
                pass
            tot += float(loss); nb += 1
        last = tot / max(nb, 1)
    return model, last


def subject_mean(scores, groups, subjects):
    return np.array([scores[groups == s].mean() for s in subjects], np.float32)


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return None
    r = rankdata(s, method="average"); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="channel_clean")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="smoke tests only")
    ap.add_argument("--ckpt-root", default=str(CKPT))
    args = ap.parse_args()
    ckpt_root = Path(args.ckpt_root)

    import torch
    set_determinism(torch)
    from sklearn.model_selection import StratifiedKFold
    e161 = load_161()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("CUDA not available -- refusing to silently CPU-train (see pipeline notes)")
    out_root = Path(args.out)
    print(f"device {device} | torch {torch.__version__}", flush=True)

    Xw = np.load(windows_path(args.tag), mmap_mode="r")
    meta = np.load(FEAT / args.tag / "meta.npz", allow_pickle=False)
    groups = meta["groups"]
    win_cls = np.array([CLASS[str(x).upper()] for x in meta["labels"]])
    win_coh = np.array([str(x) for x in meta["cohort"]])
    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    for task in [t for t in args.tasks.split(",") if t]:
        pos, neg, site = TASKS[task]
        todo = [(m, s) for m in models for s in seeds
                if not (out_root / task / f"{m.replace(':', '_')}__seed{s}.npz").exists()]
        if not todo:
            print(f"[{task}] all shards present", flush=True); continue
        keep = np.isin(win_cls, pos + neg)
        y = np.isin(win_cls, pos).astype(int)
        tr_sub = np.unique(groups[keep & (win_coh == "CAUEEG")])
        te_sub = np.unique(groups[keep & (win_coh == site)])
        tr_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in tr_sub])
        te_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in te_sub])
        t0 = time.time()
        idx_tr = np.flatnonzero(keep & (win_coh == "CAUEEG"))
        idx_te = np.flatnonzero(keep & (win_coh == site))
        X_tr_all = np.asarray(Xw[idx_tr]); g_tr_all = groups[idx_tr]; y_tr_all = y[idx_tr]
        X_te = np.asarray(Xw[idx_te]); g_te = groups[idx_te]
        print(f"\n=== {task}: train {len(tr_sub)} subj ({len(idx_tr)} win) | {site} {len(te_sub)} subj "
              f"({int(te_y.sum())} pos) | load {time.time()-t0:.0f}s", flush=True)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(tr_sub, tr_y))

        for name, seed in todo:
            t1 = time.time()
            oof_win = np.full(len(idx_tr), np.nan, np.float32)
            ext_win = np.zeros((5, len(idx_te)), np.float32)
            fold_of_subject = np.full(len(tr_sub), -1)
            losses = []
            for k, (a, b) in enumerate(folds):
                fit = np.isin(g_tr_all, tr_sub[a]); held = np.isin(g_tr_all, tr_sub[b])
                fold_of_subject[b] = k
                model, last = train_one(torch, name, X_tr_all[fit], y_tr_all[fit], device, seed, e161,
                                        epochs=args.epochs)
                losses.append(last)
                oof_win[held] = predict(torch, model, X_tr_all[held], device, e161)
                ext_win[k] = predict(torch, model, X_te, device, e161)
                if not args.no_ckpt:
                    cdir = ckpt_root / task / name.replace(":", "_") / f"seed{seed}"
                    cdir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), cdir / f"fold{k}.pt")
                del model; torch.cuda.empty_cache()
            oof_sub = subject_mean(oof_win, g_tr_all, tr_sub)
            ext_sub = subject_mean(ext_win.mean(0), g_te, te_sub)
            a_int, a_ext = auc(tr_y, oof_sub), auc(te_y, ext_sub)
            dest = out_root / task / f"{name.replace(':', '_')}__seed{seed}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dest, train_subjects=tr_sub, train_y=tr_y, oof_score=oof_sub,
                fold_of_subject=fold_of_subject, ext_site=site, ext_subjects=te_sub, ext_y=te_y,
                ext_score=ext_sub, oof_win=oof_win.astype(np.float16), win_train_groups=g_tr_all,
                ext_win_folds=ext_win.astype(np.float16), win_ext_groups=g_te)
            info = {"task": task, "model": name, "seed": seed, "auc_internal_oof": a_int,
                    "auc_external": a_ext, "ext_site": site, "n_train": int(len(tr_sub)),
                    "n_ext": int(len(te_sub)), "final_train_loss_per_fold": losses,
                    "minutes": round((time.time() - t1) / 60, 1), "epochs": args.epochs, "lr": LR,
                    "batch": BATCH, "weight_decay": WD, "split_seed": SPLIT_SEED,
                    "epoch_selection": "final epoch (none)", "deterministic": True,
                    "torch": torch.__version__}
            dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
            print(f"   {name:24s} seed {seed}: internal OOF AUC {a_int:.4f} | external {a_ext:.4f} "
                  f"| {info['minutes']} min", flush=True)
        del X_tr_all, X_te
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
