"""EEG foundation models under the matched protocol: frozen linear probe AND per-task fine-tuning.

Fixes review C6. The published FM arm (55/56/203) read RAW recordings (up to 8x15 s or 16x4 s windows,
bypassing the duration/epoch-count matching every other model saw), used an older n=88 inclusion,
zero-filled 43 of EEGPT's 62 channel slots, never fine-tuned on the external binary tasks, labelled an
unpaired difference "paired", and omitted CBraMod and the Alzheimer's-specific LEAD. Here every FM sees
exactly the 24 matched 4-s windows per subject used by every other model (microvolt, common-average
referenced, from the harmonised cache at the stored window offsets), is scored on the same folds, and
writes shards in the canonical schema so 225 pairs it against the forest on identical subjects.

Encoders (all pretrained weights; see provenance in the output JSON):
  labram   braindecode/labram-pretrained  200 Hz, 800 samples, named 10-20 channel embeddings
  eegpt    braindecode/eegpt-pretrained   250 Hz, 1000 samples, named channel ids (no zero-fill)
  cbramod  braindecode/cbramod-pretrained 200 Hz, 800 samples
  lead     DL4mHealth/LEAD P-Base (LEADv2, d128 L12) 100 Hz, 400 samples, per-window z-score.
           NOTE: LEAD's pretraining corpus includes CAUEEG and P-ADIC; it is therefore scored only at
           ds004504 and GENEEG, which it did not see in pretraining.

Input normalisation for labram/eegpt/cbramod is chosen on INTERNAL out-of-fold AUC of the linear probe
(dementia task) between "physical" (uV/100, the published pretraining convention) and per-window
per-channel z-score. LEAD uses its own z-score convention.

Stages
  embed     window embeddings -> <Alzfiles>/fm_embeddings/<fm>_<norm>.npy
  probe     subject-mean embedding -> standardised L2 logistic (C by inner CV) -> canonical shards <fm>_probe
  finetune  encoder + linear head, 10 epochs, AdamW (backbone 5e-5, head 1e-3), final epoch -> <fm>_ft

    python experiments/216_fm_matched.py --stage embed,probe,finetune
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

EMB = Path(os.environ.get("REV_EMB_ROOT", str(C.ALZ / "fm_embeddings")))
OUT = C.OUT
LEAD_DIR = C.ALZ / "third_party" / "LEAD"
LEAD_CKPT = (C.ALZ / "third_party" / "LEAD_P-Base" / "LEADv2" / "pretrain_lead" / "LEADv2" / "P-Base" /
             "nh8_el12_dm128_df256_seed41" / "checkpoint.pth")
COHORTS = ("CAUEEG", "ds004504_raw", "GENEEG")
MAP1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
LEAD_ORDER = "Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2".split(",")


def resample(x, fs_out):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(256, fs_out)
    return resample_poly(x, fs_out // g, 256 // g, axis=-1).astype(np.float32)


def zscore(x):
    return ((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-7)).astype(np.float32)


class Encoders:
    def __init__(self, torch, device, CH):
        self.torch, self.device, self.CH = torch, device, CH

    def labram(self):
        from braindecode.models import Labram
        pre = Labram.from_pretrained("braindecode/labram-pretrained")
        # The checkpoint was saved at n_times=3000 (15 patches); our 4-s windows are 4 patches, so
        # the model is rebuilt at 800 samples and the temporal embedding truncated to the first 5
        # positions (cls + 4). Every other weight is copied unchanged.
        m = Labram(n_times=800, n_outputs=0, chs_info=pre.chs_info)
        sd = pre.state_dict()
        sd["temporal_embedding"] = sd["temporal_embedding"][:, :m.temporal_embedding.shape[1]]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"   LaBraM load: missing {missing} unexpected {unexpected}", flush=True)
        m = m.to(self.device); del pre
        names = [MAP1020.get(c, c).upper() for c in self.CH]

        def feat(x):
            out = m(x, ch_names=names, return_features=True)
            f = out["features"] if isinstance(out, dict) else out
            return f.mean(1) if f.dim() == 3 else f
        return m, feat, 200, 800

    def eegpt(self):
        from braindecode.models import EEGPT
        pre = EEGPT.from_pretrained("braindecode/eegpt-pretrained", n_chans=62, chan_proj_type="none")
        # Select the pretrained channel ids of our 19 electrodes instead of scattering 19 signals into
        # 62 zero-filled slots (review C6): the encoder indexes its channel embedding by chans_id, so a
        # 19-channel model with the 19 matching ids uses exactly the pretrained embeddings.
        names62 = [c["ch_name"].upper() for c in pre.chs_info]
        pick = [names62.index(MAP1020.get(c, c).upper()) for c in self.CH]
        m = EEGPT(n_chans=19, n_times=1000, sfreq=250, n_outputs=1, chan_proj_type="none",
                  chs_info=[pre.chs_info[i] for i in pick])
        sd = {k: v for k, v in pre.state_dict().items() if k != "chans_id"}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"   EEGPT load: missing {missing} unexpected {unexpected}", flush=True)
        m.chans_id = pre.chans_id[:, pick].clone()
        m = m.to(self.device); del pre

        def feat(x):
            f = m(x, return_features=True)["features"]          # [B, n_patches, embed_num*dim]
            return f.mean(1)
        return m, feat, 250, 1000

    def cbramod(self):
        from braindecode.models import CBraMod
        m = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_chans=19, n_times=800,
                                    sfreq=200, n_outputs=2).to(self.device)

        def feat(x):
            f = m(x, return_features=True)["features"]          # [B, C, P, D]
            return f.mean(dim=tuple(range(1, f.dim() - 1)))
        return m, feat, 200, 800

    def lead(self):
        torch = self.torch
        sys.path.insert(0, str(LEAD_DIR))
        from types import ModuleType, SimpleNamespace
        if "reformer_pytorch" not in sys.modules:
            # LEAD's attention module imports reformer_pytorch for a layer LEADv2 never uses; a stub
            # avoids installing a third-party package just to satisfy an unused import.
            stub = ModuleType("reformer_pytorch")
            stub.LSHSelfAttention = type("LSHSelfAttention", (), {})
            sys.modules["reformer_pytorch"] = stub
        from models.LEADv2 import Model
        cfg = SimpleNamespace(task_name="pretrain_lead", output_attention=False, patch_len=50, stride=50,
                              enc_in=19, seq_len=400, d_model=128, n_heads=8, e_layers=12, d_ff=256,
                              dropout=0.1, activation="gelu", channel_names=",".join(LEAD_ORDER),
                              montage_name="standard_1005", augmentations="none", num_class=2)
        m = Model(cfg).float().to(self.device)
        sd = torch.load(LEAD_CKPT, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        sd = {k.replace("module.", ""): v for k, v in sd.items() if k != "n_averaged"}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"   LEAD load: missing {len(missing)} unexpected {len(unexpected)}", flush=True)
        order = [self.CH.index({v: k for k, v in MAP1020.items()}.get(c, c)) for c in LEAD_ORDER]

        def feat(x):
            x = x[:, order, :].permute(0, 2, 1)                  # (B, T, C) in LEAD channel order
            fs = torch.full((x.shape[0],), 100, device=x.device)
            reprs_h, _ = m(x, None, None, None, fs, None)
            return reprs_h
        return m, feat, 100, 400


def meta_for(tag):
    p = C.FEAT / tag / "meta.npz"
    return np.load(p, allow_pickle=False) if p.exists() else None


def load_windows(meta, CH, cohorts=COHORTS):
    """uV CAR windows for the requested cohorts, at the stored window offsets."""
    groups, start, coh = meta["groups"], meta["start"].astype(int), np.array([str(c) for c in meta["cohort"]])
    idx = np.flatnonzero(np.isin(coh, cohorts))
    X = np.zeros((len(idx), 19, 1024), np.float32)
    pos = {i: j for j, i in enumerate(idx)}
    for s in np.unique(groups[idx]):
        a = np.load(C.ALZ / "harmonised" / f"{s}.npy")
        for i in idx[groups[idx] == s]:
            X[pos[i]] = a[:, start[i]:start[i] + 1024]
    X *= 1e6
    X -= X.mean(1, keepdims=True)
    return idx, X


def subject_vectors(fm, norm, tag):
    """subject -> mean embedding, for one cohort tag (None if that tag was not embedded)."""
    suffix = "" if tag == "channel_clean" else f"__{tag}"
    p = EMB / f"{fm}_{norm}{suffix}.npy"
    ip = EMB / f"window_index{suffix}.npy"
    m = meta_for(tag)
    if not (p.exists() and ip.exists() and m is not None):
        return None
    E = np.load(p); idx = np.load(ip); g = m["groups"][idx]
    return {s: E[g == s].mean(0) for s in np.unique(g)}


def stage_embed(args, torch, meta, CH):
    EMB.mkdir(parents=True, exist_ok=True)
    enc = Encoders(torch, "cuda", CH)
    for tag, cohorts in (("channel_clean", COHORTS), ("padic_channel_clean", ("P-ADIC",))):
        m = meta if tag == "channel_clean" else meta_for(tag)
        if m is None:
            print(f"[{tag}] not extracted yet; skipped", flush=True); continue
        suffix = "" if tag == "channel_clean" else f"__{tag}"
        X = None
        for fm in args.fms.split(","):
            # LEAD's pretraining corpus includes CAUEEG and P-ADIC, so it is never scored there
            if fm == "lead" and tag != "channel_clean":
                continue
            for norm in (("zscore",) if fm == "lead" else ("physical", "zscore")):
                path = EMB / f"{fm}_{norm}{suffix}.npy"
                if path.exists():
                    continue
                if X is None:
                    t0 = time.time(); idx, X = load_windows(m, CH, cohorts)
                    np.save(EMB / f"window_index{suffix}.npy", idx)
                    print(f"[{tag}] windows {X.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
                model, feat, fs, nt = getattr(enc, fm)()
                model.eval(); outs = []; t0 = time.time()
                with torch.no_grad():
                    for i in range(0, len(X), 256):
                        xb = resample(X[i:i + 256], fs)[..., :nt]
                        xb = zscore(xb) if norm == "zscore" else xb / 100.0
                        outs.append(feat(torch.from_numpy(xb).to("cuda")).float().cpu().numpy())
                E = np.concatenate(outs).astype(np.float32)
                np.save(path, E)
                print(f"   [{tag}] {fm}/{norm}: embeddings {E.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
                del model; torch.cuda.empty_cache()
        del X


def stage_probe(args, meta):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = np.load(EMB / "window_index.npy")
    groups = meta["groups"][idx]
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"][idx]])
    coh = np.array([str(c) for c in meta["cohort"][idx]])
    pmeta_p = OUT / "padic_subject_meta.json"
    pmeta = json.loads(pmeta_p.read_text()) if pmeta_p.exists() else {}
    P_TASK = {"CN_vs_AD": ("AD",), "CN_vs_dementia": ("AD",), "CN_vs_impaired": ("AD", "MCI"), "CN_vs_MCI": ("MCI",)}
    choice = {}
    for fm in args.fms.split(","):
        norms = ("zscore",) if fm == "lead" else ("physical", "zscore")
        for norm in norms:
            E = np.load(EMB / f"{fm}_{norm}.npy")
            subs = np.unique(groups)
            pos_of = {s: np.flatnonzero(groups == s) for s in subs}
            S = {s: E[pos_of[s]].mean(0) for s in subs}
            SP = None if fm == "lead" else subject_vectors(fm, norm, "padic_channel_clean")
            for task in args.tasks.split(","):
                pos, neg, site = C.TASKS[task]
                tr_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == "CAUEEG" and cls[pos_of[s][0]] in pos + neg}))
                te_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == site and cls[pos_of[s][0]] in pos + neg}))
                tr_y = np.array([int(cls[pos_of[s][0]] in pos) for s in tr_sub])
                te_y = np.array([int(cls[pos_of[s][0]] in pos) for s in te_sub])
                Xtr = np.stack([S[s] for s in tr_sub]); Xte = np.stack([S[s] for s in te_sub])
                # P-ADIC, controls age-restricted exactly as in exp 224
                p_sub = p_y = Xp = None
                if SP:
                    ppos = P_TASK[task]
                    cand = [s for s in SP if (pmeta.get(str(s), {}).get("group") in ppos)
                            or (pmeta.get(str(s), {}).get("group") == "CN"
                                and (pmeta.get(str(s), {}).get("age") or 0) >= 55.0)]
                    if len({pmeta[str(s)]["group"] in ppos for s in cand}) > 1:
                        p_sub = np.array(sorted(cand))
                        p_y = np.array([int(pmeta[str(s)]["group"] in ppos) for s in p_sub])
                        Xp = np.stack([SP[s] for s in p_sub])
                folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
                oof = np.full(len(tr_sub), np.nan, np.float32); ext = []; pad = []
                for a, b in folds:
                    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
                        max_iter=5000, class_weight="balanced")), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(Xtr[a], tr_y[a])
                    oof[b] = gs.predict_proba(Xtr[b])[:, 1]; ext.append(gs.predict_proba(Xte)[:, 1])
                    if Xp is not None:
                        pad.append(gs.predict_proba(Xp)[:, 1])
                ext = np.mean(ext, 0)
                a_int, a_ext = C.auc(tr_y, oof), C.auc(te_y, ext)
                name = f"{fm}_probe_{norm}"
                dest = OUT / task / f"{name}__seed42.npz"; dest.parent.mkdir(parents=True, exist_ok=True)
                arrays = dict(train_subjects=tr_sub, train_y=tr_y, oof_score=oof, ext_site=site,
                              ext_subjects=te_sub, ext_y=te_y, ext_score=ext)
                info = {"task": task, "model": name, "auc_internal_oof": a_int, "auc_external": a_ext,
                        "n_ext": int(len(te_sub))}
                if pad:
                    ps = np.mean(pad, 0)
                    arrays.update(padic_subjects=p_sub, padic_y=p_y, padic_score=ps.astype(np.float32))
                    info["auc_padic"] = C.auc(p_y, ps); info["n_padic"] = int(len(p_sub))
                np.savez_compressed(dest, **arrays)
                dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
                print(f"   probe {fm:8s} {norm:9s} {task:15s} internal {a_int:.3f} external {a_ext:.3f}"
                      + (f" P-ADIC {info['auc_padic']:.3f}" if "auc_padic" in info else ""), flush=True)
                if task == "CN_vs_dementia":
                    choice.setdefault(fm, {})[norm] = a_int
    sel = {fm: max(v, key=v.get) for fm, v in choice.items()}
    (OUT / "fm_norm_selection.json").write_text(json.dumps({"internal_oof_auc_dementia": choice, "selected": sel}, indent=1))
    print("normalisation selected on internal OOF (dementia):", sel, flush=True)


def stage_finetune(args, torch, meta, CH):
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset
    sel = json.loads((OUT / "fm_norm_selection.json").read_text())["selected"]
    idx, X = load_windows(meta, CH)
    groups = meta["groups"][idx]
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"][idx]])
    coh = np.array([str(c) for c in meta["cohort"][idx]])
    enc = Encoders(torch, "cuda", CH)
    for fm in args.ft_fms.split(","):
        norm = sel.get(fm, "zscore")
        for task in args.tasks.split(","):
            pos, neg, site = C.TASKS[task]
            name = f"{fm}_ft"
            dest = OUT / task / f"{name}__seed42.npz"
            if dest.exists() or (fm == "lead" and site not in ("ds004504_raw", "GENEEG")):
                continue
            keep = np.isin(cls, pos + neg); y = np.isin(cls, pos).astype(int)
            tr_sub = np.unique(groups[keep & (coh == "CAUEEG")]); te_sub = np.unique(groups[keep & (coh == site)])
            tr_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in tr_sub])
            te_y = np.array([y[np.flatnonzero(groups == s)[0]] for s in te_sub])
            folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
            m_tr = keep & (coh == "CAUEEG"); m_te = keep & (coh == site)
            _, _, fs, nt = getattr(enc, fm)()
            prep = lambda Z: (zscore(resample(Z, fs)[..., :nt]) if norm == "zscore" else resample(Z, fs)[..., :nt] / 100.0)
            t0 = time.time()
            Xtr = np.concatenate([prep(X[m_tr][i:i + 512]) for i in range(0, int(m_tr.sum()), 512)])
            Xte = np.concatenate([prep(X[m_te][i:i + 512]) for i in range(0, int(m_te.sum()), 512)])
            gtr, gte, ytr_w = groups[m_tr], groups[m_te], y[m_tr]
            print(f"\n=== FT {fm} ({norm}) {task}: prep {(time.time()-t0)/60:.1f} min", flush=True)
            oof_w = np.full(len(Xtr), np.nan, np.float32); ext_w = []
            for k, (a, b) in enumerate(folds):
                torch.manual_seed(42)
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
                epochs, bs = (1 if os.environ.get("REV_EPOCHS") else 10), 64
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 1e-3],
                                                            total_steps=epochs * (int(fit.sum()) // bs + 1))
                gen = torch.Generator(); gen.manual_seed(42)
                dl = DataLoader(TensorDataset(torch.from_numpy(Xtr[fit]), torch.from_numpy(yf.astype(np.int64))),
                                batch_size=bs, shuffle=True, generator=gen)
                for ep in range(epochs):
                    backbone.train(); head.train()
                    for xb, yb in dl:
                        opt.zero_grad()
                        loss = crit(head(feat(xb.cuda())), yb.cuda()); loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
                        opt.step(); sched.step()
                backbone.eval(); head.eval()

                def pred(Z):
                    out = []
                    with torch.no_grad():
                        for i in range(0, len(Z), 256):
                            out.append(torch.softmax(head(feat(torch.from_numpy(Z[i:i + 256]).cuda())), 1)[:, 1].cpu().numpy())
                    return np.concatenate(out)
                oof_w[held] = pred(Xtr[held]); ext_w.append(pred(Xte))
                del backbone, head; torch.cuda.empty_cache()
                print(f"   fold {k} done ({(time.time()-t0)/60:.1f} min)", flush=True)
            oof = C.subject_mean(oof_w, gtr, tr_sub); ext = C.subject_mean(np.mean(ext_w, 0), gte, te_sub)
            np.savez_compressed(dest, train_subjects=tr_sub, train_y=tr_y, oof_score=oof, ext_site=site,
                                ext_subjects=te_sub, ext_y=te_y, ext_score=ext)
            info = {"task": task, "model": name, "norm": norm, "auc_internal_oof": C.auc(tr_y, oof),
                    "auc_external": C.auc(te_y, ext), "epochs": 10, "lr_backbone": 5e-5, "lr_head": 1e-3,
                    "minutes": round((time.time() - t0) / 60, 1)}
            dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
            print(f"   FT {fm} {task}: internal {info['auc_internal_oof']:.3f} external {info['auc_external']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="embed,probe,finetune")
    ap.add_argument("--fms", default="labram,eegpt,cbramod,lead")
    ap.add_argument("--ft-fms", default="labram,cbramod,lead")
    ap.add_argument("--tasks", default="CN_vs_dementia,CN_vs_AD,CN_vs_impaired,CN_vs_MCI")
    ap.add_argument("--tag", default="channel_clean")
    args = ap.parse_args()
    import torch
    C.set_determinism(torch)
    from src.config.config import CHANNEL_NAMES
    meta = np.load(C.FEAT / args.tag / "meta.npz", allow_pickle=False)
    stages = args.stage.split(",")
    if "embed" in stages:
        stage_embed(args, torch, meta, list(CHANNEL_NAMES))
    if "probe" in stages:
        stage_probe(args, meta)
    if "finetune" in stages:
        stage_finetune(args, torch, meta, list(CHANNEL_NAMES))
    print("done", flush=True)


if __name__ == "__main__":
    main()
