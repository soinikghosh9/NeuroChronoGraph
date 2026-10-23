"""Complete the foundation-model grid at the two later external cohorts (P-ADIC, BrainLat).

Exp 216 embedded P-ADIC for the frozen probes but never BrainLat, and its fine-tuned arms were scored only
at ds004504 (the fold models were not kept). This script fills both gaps under the 216 protocol unchanged:
same 24 matched 4-s windows per subject, same input normalisation (selected on internal OOF AUC,
fm_norm_selection.json), same CAUEEG folds (SPLIT_SEED), same probe grid, same fine-tuning recipe
(10 epochs, AdamW 5e-5 / 1e-3, OneCycle, batch 64, fp32).

LEAD is excluded at BrainLat and P-ADIC: its pretraining corpus includes both (third_party/LEAD/README.md).

Stages (restart-safe; every output is skipped if it exists)
  embed     BrainLat window embeddings for labram/eegpt/cbramod (physical and zscore)
  probe     refit every 216 probe (deterministic), check the refit reproduces the stored ds004504 and
            P-ADIC scores, then append brainlat_subjects/_y/_score to the canonical shard
  finetune  labram and cbramod fine-tuned per task and seed (42, 43, 44); each run scores ds004504,
            P-ADIC and BrainLat from the same fold models -> canonical/fm_ext/<task>/<fm>_ft__seed<k>.npz
            (canonical shards are not touched; 253 compares seed 42 with the published ds004504 shard)

    python experiments/252_fm_external_complete.py --stage embed,probe,finetune
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


F = _load("f216", "216_fm_matched.py")
C = F.C
OUT = C.OUT
FT_OUT = OUT / "fm_ext"
BL_TAG, PA_TAG = "brainlat_channel_clean", "padic_channel_clean"
BL_TASK = {"CN_vs_AD": (("AD",), ("CN",)), "CN_vs_dementia": (("AD", "FTD"), ("CN",)),
           "CN_vs_impaired": (("AD", "FTD"), ("CN",))}          # identical to 232 TASK_SITE
P_TASK = {"CN_vs_AD": ("AD",), "CN_vs_dementia": ("AD",), "CN_vs_impaired": ("AD", "MCI"), "CN_vs_MCI": ("MCI",)}
STATUS = ROOT / "outputs" / "logs" / "fm_ext_status.txt"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def brainlat_labels():
    m = F.meta_for(BL_TAG)
    g = m["groups"]; lab = np.array([str(x) for x in m["labels"]])
    return {s: lab[np.flatnonzero(g == s)[0]] for s in np.unique(g)}


def brainlat_subjects(task):
    grp = brainlat_labels(); pos, neg = BL_TASK[task]
    sb = np.array(sorted(s for s in grp if grp[s] in pos + neg))
    return sb, np.array([int(grp[s] in pos) for s in sb])


def padic_subjects(task, available):
    pmeta = json.loads((OUT / "padic_subject_meta.json").read_text())
    ppos = P_TASK[task]
    cand = [s for s in available if (pmeta.get(str(s), {}).get("group") in ppos)
            or (pmeta.get(str(s), {}).get("group") == "CN" and (pmeta.get(str(s), {}).get("age") or 0) >= 55.0)]
    p_sub = np.array(sorted(cand))
    return p_sub, np.array([int(pmeta[str(s)]["group"] in ppos) for s in p_sub])


def stage_embed(torch, CH, fms):
    m = F.meta_for(BL_TAG)
    suffix = f"__{BL_TAG}"
    enc = F.Encoders(torch, "cuda", CH); X = None
    for fm in fms:
        for norm in ("physical", "zscore"):
            path = F.EMB / f"{fm}_{norm}{suffix}.npy"
            if path.exists():
                continue
            if X is None:
                idx, X = F.load_windows(m, CH, ("BrainLat",))
                np.save(F.EMB / f"window_index{suffix}.npy", idx)
                log(f"embed: BrainLat windows {X.shape}")
            model, feat, fs, nt = getattr(enc, fm)(); model.eval(); outs = []; t0 = time.time()
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    xb = F.resample(X[i:i + 256], fs)[..., :nt]
                    xb = F.zscore(xb) if norm == "zscore" else xb / 100.0
                    outs.append(feat(torch.from_numpy(xb).to("cuda")).float().cpu().numpy())
            E = np.concatenate(outs).astype(np.float32); np.save(path, E)
            log(f"embed: {fm}/{norm} BrainLat {E.shape} in {(time.time()-t0)/60:.1f} min")
            del model; torch.cuda.empty_cache()


def stage_probe(meta, fms, tasks):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = np.load(F.EMB / "window_index.npy")
    groups = meta["groups"][idx]
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"][idx]])
    coh = np.array([str(c) for c in meta["cohort"][idx]])
    for fm in fms:
        for norm in ("physical", "zscore"):
            SB = F.subject_vectors(fm, norm, BL_TAG)
            if SB is None:
                log(f"probe: {fm}/{norm} BrainLat embeddings missing; skipped"); continue
            E = np.load(F.EMB / f"{fm}_{norm}.npy")
            subs = np.unique(groups)
            pos_of = {s: np.flatnonzero(groups == s) for s in subs}
            S = {s: E[pos_of[s]].mean(0) for s in subs}
            for task in tasks:
                if task not in BL_TASK:
                    continue
                name = f"{fm}_probe_{norm}"
                dest = OUT / task / f"{name}__seed42.npz"
                if not dest.exists():
                    log(f"probe: {dest.name} not in canonical; skipped"); continue
                z = dict(np.load(dest, allow_pickle=True))
                if "brainlat_score" in z:
                    continue
                pos, neg, site = C.TASKS[task]
                tr_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == "CAUEEG" and cls[pos_of[s][0]] in pos + neg}))
                te_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == site and cls[pos_of[s][0]] in pos + neg}))
                tr_y = np.array([int(cls[pos_of[s][0]] in pos) for s in tr_sub])
                Xtr = np.stack([S[s] for s in tr_sub]); Xte = np.stack([S[s] for s in te_sub])
                b_sub, b_y = brainlat_subjects(task)
                b_sub = np.array([s for s in b_sub if s in SB]); b_y = np.array([brainlat_labels()[s] in BL_TASK[task][0] for s in b_sub]).astype(int)
                Xb = np.stack([SB[s] for s in b_sub])
                folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
                ext, bl = [], []
                for a, _ in folds:
                    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
                        max_iter=5000, class_weight="balanced")), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(Xtr[a], tr_y[a])
                    ext.append(gs.predict_proba(Xte)[:, 1]); bl.append(gs.predict_proba(Xb)[:, 1])
                ext = np.mean(ext, 0); bl = np.mean(bl, 0)
                dev = float(np.max(np.abs(ext - z["ext_score"]))) if len(ext) == len(z["ext_score"]) else np.inf
                if not dev < 1e-4:
                    log(f"probe: REFIT MISMATCH {task} {name} (max |d| {dev}); BrainLat not appended"); continue
                z.update(brainlat_subjects=b_sub, brainlat_y=b_y, brainlat_score=bl.astype(np.float32))
                tmp = dest.with_name(dest.stem + ".tmp.npz"); np.savez_compressed(tmp, **z); os.replace(tmp, dest)
                info = json.loads(dest.with_suffix(".json").read_text())
                info.update(auc_brainlat=C.auc(b_y, bl), n_brainlat=int(len(b_sub)))
                dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
                log(f"probe: {task} {name} BrainLat AUC {info['auc_brainlat']:.3f} (n={len(b_sub)}; refit max |d| {dev:.1e})")


def stage_finetune(torch, meta, CH, fms, tasks, seeds):
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset
    sel = json.loads((OUT / "fm_norm_selection.json").read_text())["selected"]
    enc = F.Encoders(torch, "cuda", CH)
    idx, X = F.load_windows(meta, CH)
    groups = meta["groups"][idx]
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"][idx]])
    coh = np.array([str(c) for c in meta["cohort"][idx]])
    pm = F.meta_for(PA_TAG); p_idx, XP = F.load_windows(pm, CH, ("P-ADIC",)); gP = pm["groups"][p_idx]
    bm = F.meta_for(BL_TAG); b_idx, XB = F.load_windows(bm, CH, ("BrainLat",)); gB = bm["groups"][b_idx]
    log(f"finetune: windows main {X.shape}, P-ADIC {XP.shape}, BrainLat {XB.shape}")
    for fm in fms:
        norm = sel.get(fm, "zscore")
        _, _, fs, nt = getattr(enc, fm)()
        prep = lambda Z: (F.zscore(F.resample(Z, fs)[..., :nt]) if norm == "zscore" else F.resample(Z, fs)[..., :nt] / 100.0)
        batched = lambda Z: np.concatenate([prep(Z[i:i + 512]) for i in range(0, len(Z), 512)])
        for task in tasks:
            todo = [s for s in seeds if not (FT_OUT / task / f"{fm}_ft__seed{s}.npz").exists()]
            if not todo:
                continue
            pos, neg, site = C.TASKS[task]
            keep = np.isin(cls, pos + neg); y = np.isin(cls, pos).astype(int)
            tr_sub = np.unique(groups[keep & (coh == "CAUEEG")]); te_sub = np.unique(groups[keep & (coh == site)])
            tr_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in tr_sub])
            te_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in te_sub])
            p_sub, p_y = padic_subjects(task, set(np.unique(gP)))
            b_sub, b_y = brainlat_subjects(task)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
            m_tr = keep & (coh == "CAUEEG"); m_te = keep & (coh == site)
            mP = np.isin(gP, p_sub); mB = np.isin(gB, b_sub)
            t0 = time.time()
            Xtr, Xte, Xp, Xb = batched(X[m_tr]), batched(X[m_te]), batched(XP[mP]), batched(XB[mB])
            gtr, gte, gp, gb, ytr_w = groups[m_tr], groups[m_te], gP[mP], gB[mB], y[m_tr]
            log(f"finetune: {fm} ({norm}) {task}: prep {(time.time()-t0)/60:.1f} min; "
                f"train {len(tr_sub)} subj, ds {len(te_sub)}, P-ADIC {len(p_sub)}, BrainLat {len(b_sub)}")
            for seed in todo:
                dest = FT_OUT / task / f"{fm}_ft__seed{seed}.npz"; dest.parent.mkdir(parents=True, exist_ok=True)
                t1 = time.time()
                oof_w = np.full(len(Xtr), np.nan, np.float32); ext_w, pad_w, bl_w = [], [], []
                for k, (a, b) in enumerate(folds):
                    torch.manual_seed(seed)
                    backbone, feat, _, _ = getattr(enc, fm)()
                    with torch.no_grad():
                        d = feat(torch.from_numpy(Xtr[:2]).cuda()).shape[-1]
                    head = torch.nn.Linear(d, 2).cuda()
                    fit = np.isin(gtr, tr_sub[a]); held = np.isin(gtr, tr_sub[b])
                    yf = ytr_w[fit]; cw = np.zeros(2, np.float32); cc, cn = np.unique(yf, return_counts=True)
                    cw[cc] = len(yf) / (len(cc) * cn)
                    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw).cuda())
                    opt = torch.optim.AdamW([{"params": backbone.parameters(), "lr": 5e-5},
                                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.05)
                    epochs, bs = 10, 64
                    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 1e-3],
                                                                total_steps=epochs * (int(fit.sum()) // bs + 1))
                    gen = torch.Generator(); gen.manual_seed(seed)
                    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr[fit]), torch.from_numpy(yf.astype(np.int64))),
                                    batch_size=bs, shuffle=True, generator=gen, pin_memory=True)
                    for ep in range(epochs):
                        backbone.train(); head.train()
                        for xb, yb in dl:
                            opt.zero_grad()
                            loss = crit(head(feat(xb.cuda(non_blocking=True))), yb.cuda(non_blocking=True)); loss.backward()
                            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
                            opt.step(); sched.step()
                    backbone.eval(); head.eval()

                    def pred(Z):
                        out = []
                        with torch.no_grad():
                            for i in range(0, len(Z), 256):
                                out.append(torch.softmax(head(feat(torch.from_numpy(Z[i:i + 256]).cuda())), 1)[:, 1].cpu().numpy())
                        return np.concatenate(out)
                    oof_w[held] = pred(Xtr[held]); ext_w.append(pred(Xte)); pad_w.append(pred(Xp)); bl_w.append(pred(Xb))
                    del backbone, head, opt; torch.cuda.empty_cache()
                    log(f"finetune: {fm} {task} seed {seed} fold {k} done ({(time.time()-t1)/60:.1f} min)")
                oof = C.subject_mean(oof_w, gtr, tr_sub)
                ext = C.subject_mean(np.mean(ext_w, 0), gte, te_sub)
                pad = C.subject_mean(np.mean(pad_w, 0), gp, p_sub)
                bl = C.subject_mean(np.mean(bl_w, 0), gb, b_sub)
                tmp = dest.with_name(dest.stem + ".tmp.npz")
                np.savez_compressed(tmp, train_subjects=tr_sub, train_y=tr_y, oof_score=oof, ext_site=site,
                                    ext_subjects=te_sub, ext_y=te_y, ext_score=ext,
                                    padic_subjects=p_sub, padic_y=p_y, padic_score=pad,
                                    brainlat_subjects=b_sub, brainlat_y=b_y, brainlat_score=bl)
                os.replace(tmp, dest)
                info = {"task": task, "model": f"{fm}_ft", "seed": seed, "norm": norm,
                        "auc_internal_oof": C.auc(tr_y, oof), "auc_external": C.auc(te_y, ext),
                        "auc_padic": C.auc(p_y, pad), "auc_brainlat": C.auc(b_y, bl),
                        "n_ext": int(len(te_sub)), "n_padic": int(len(p_sub)), "n_brainlat": int(len(b_sub)),
                        "epochs": 10, "lr_backbone": 5e-5, "lr_head": 1e-3, "minutes": round((time.time() - t1) / 60, 1)}
                dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
                log(f"finetune: DONE {fm} {task} seed {seed}: internal {info['auc_internal_oof']:.3f} "
                    f"ds {info['auc_external']:.3f} P-ADIC {info['auc_padic']:.3f} BrainLat {info['auc_brainlat']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="embed,probe,finetune")
    ap.add_argument("--fms", default="labram,eegpt,cbramod")
    ap.add_argument("--ft-fms", default="labram,cbramod")
    ap.add_argument("--tasks", default="CN_vs_dementia,CN_vs_AD,CN_vs_impaired")
    ap.add_argument("--ft-tasks", default="CN_vs_dementia,CN_vs_AD")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    import torch
    C.set_determinism(torch)
    from src.config.config import CHANNEL_NAMES
    CH = list(CHANNEL_NAMES)
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    stages = args.stage.split(",")
    if "embed" in stages:
        stage_embed(torch, CH, args.fms.split(","))
    if "probe" in stages:
        stage_probe(meta, args.fms.split(","), args.tasks.split(","))
    if "finetune" in stages:
        stage_finetune(torch, meta, CH, args.ft_fms.split(","), args.ft_tasks.split(","),
                       [int(s) for s in args.seeds.split(",")])
    log("all stages done")


if __name__ == "__main__":
    main()
