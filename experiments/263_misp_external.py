"""MISP (Isfahan) as an external control-versus-MCI cohort. Evaluation only.

MISP releases 61 eyes-closed recordings (set 1: 16 normal + 11 MCI, Kashefpoor 2016; set 2: 16 normal + 18 MCI,
Kashefpoor 2019): 19 electrodes, 256 Hz, 30 min, cardiac-catheterisation patients aged 60-77, MCI by Petersen
criteria with MMSE 21-26, controls MMSE > 26. Earlier runs loaded MCI only (misp_loader mci_only=True), so the
control-versus-MCI task at MISP was never used for any model, feature, threshold or variant choice.

All 61 recordings are harmonised again here with the current harmonise() so both classes share one code version
(prefix MISPR_; the canonical MISP_ MCI files and the channel_clean cache are not touched). Window rules are those
of 224/260: first 235 s, 200 uV peak-to-peak rejection, 24 windows, md5 seeds, robust scaling.

The predictions and the confound gate were written before scoring (model_paper/PLAN_MISP_2026-10-13.md):
  P1  SpectraLite CN-vs-MCI AUC: 95% CI includes 0.5 and 0.40 <= point <= 0.65; refuted if >= 0.70 with CI > 0.5
  P2  median CN-vs-MCI AUC over classifiers within 0.45-0.65; the any-impairment model alike
  P3  CAUEEG threshold specificity on MISP controls < 0.80; split-conformal threshold from 20 local controls
      0.73-0.95 on held-out controls (only if >= 26 controls retained)
  Gate: retention, rejection rate, ICA variance removed, raw and residual line noise, amplitude, by class
        (AUC >= 0.75 or <= 0.25 is flagged), read before any model AUC

Stages (resumable; every output is skipped if it exists):
  harmonise  src.data.harmonise.harmonise() on all 61 recordings (12 processes)
  extract    windows, concepts, spectral stream, node12, node_ext4 -> harmonised_features/misp_all
  score      networks from fold checkpoints (GPU), the 16-feature variant, classical fold refits
  embed      frozen foundation-model embeddings (labram, eegpt, cbramod: physical + zscore; lead: zscore)
  probe      216 probes refitted on CAUEEG (refit must reproduce the stored external score), applied to MISP
  analyse    confound gate, P1-P3, set-stratified AUCs, paired differences from the forest, MCI-endpoint
             cohort/model decomposition over GENEEG, P-ADIC and MISP

    python experiments/263_misp_external.py --stage harmonise,extract,score,embed,probe,analyse
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
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


DS = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles")) / "datasets" / "MISP"
PREFIX, TAG, COHORT = "MISPR", "misp_all", "MISP"
RELAXED = False          # --relaxed: keep every recording, dropped ones scored from their 24 least-artefactual windows
OUTDIR = ROOT / "outputs" / "results" / "canonical" / "misp"
STATUS = ROOT / "outputs" / "logs" / "misp_status.txt"
PLAN = ROOT / "model_paper" / "PLAN_MISP_2026-10-13.md"
WINDOW, STRIDE, MAX_WINDOWS, REJECT_V, SFREQ, TRUNC_S, K = 1024, 1024, 50, 200e-6, 256.0, 235.0, 24
# positives at MISP are MCI for every task; the dementia model is scored for description only
TASKS = {"CN_vs_MCI": ("MCI",), "CN_vs_impaired": ("MCI",), "CN_vs_dementia": ("MCI",)}
SEEDS = (42, 43, 44)
REF = "rf123"
SCORES = "scores"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def recordings():
    """[(subject id, set, group, edf path)] from the two status sheets; group CN or MCI."""
    import pandas as pd
    out = []
    for st in ("set1", "set2"):
        d = DS / st
        df = pd.read_excel(next(d.glob("*.xlsx")))
        cols = {c.lower().strip(): c for c in df.columns}
        lab = {int(r[cols["file number"]]): str(r[cols["status"]]).strip().upper() for _, r in df.iterrows()}
        for e in sorted(d.glob("*.edf"), key=lambda p: int(p.stem)):
            s = lab.get(int(e.stem))
            if s in ("NORMAL", "MCI"):
                out.append((f"{PREFIX}_{st}_{s}_{int(e.stem):02d}", st, "CN" if s == "NORMAL" else "MCI", str(e)))
    return out


def _line_rms(x, sf):
    """Line-noise share (49-51 Hz power over 1-45 Hz power, channel median) and median channel RMS in uV."""
    from scipy.signal import welch
    f, p = welch(x, fs=sf, nperseg=int(2 * sf), axis=-1)
    line = p[:, (f >= 49) & (f <= 51)].sum(-1) / (p[:, (f >= 1) & (f <= 45)].sum(-1) + 1e-30)
    return float(np.median(line)), float(np.median(np.sqrt(np.mean((x - x.mean(1, keepdims=True)) ** 2, 1))) * 1e6)


# ---------------------------------------------------------------------------------------------- harmonise
def _harmonise_one(job):
    sid, st, grp, path = job
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    import mne
    mne.set_log_level("ERROR")
    C = _load("c211", "211_canonical_benchmark.py")
    from src.config.config import CHANNEL_NAMES
    from src.data.harmonise import harmonise
    H = C.ALZ / "harmonised"
    npy, side = H / f"{sid}.npy", H / f"{sid}.json"
    if npy.exists() and side.exists():
        return sid, "skip"
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.pick([c for c in CHANNEL_NAMES if c in raw.ch_names]); raw.reorder_channels(list(CHANNEL_NAMES))
        raw.set_channel_types({c: "eeg" for c in raw.ch_names})
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        sf = float(raw.info["sfreq"])
        seg = raw.get_data(start=0, stop=int(TRUNC_S * sf)) * 1e-6       # file units are microvolts
        raw_line, raw_rms = _line_rms(seg, sf)
        out, qc = harmonise(raw, run_ica=True)
        data = out.get_data().astype(np.float32)
        np.save(npy, data)
        side.write_text(json.dumps({"subject": sid, "cohort": COHORT, "set": st, "group": grp, "source": path,
                                    "channels": out.ch_names, "qc": qc, "raw_line_share_235s": raw_line,
                                    "raw_rms_uV_235s": raw_rms}, indent=1, default=float))
        return sid, "done"
    except Exception as e:                                                   # noqa: BLE001
        return sid, f"fail {type(e).__name__}: {str(e)[:100]}"


def stage_harmonise(limit=0):
    from multiprocessing import Pool
    jobs = recordings()
    if limit:
        jobs = jobs[:limit]
    log(f"harmonise: {len(jobs)} recordings")
    t0 = time.time(); n = 0
    with Pool(min(12, len(jobs))) as pool:
        for sid, st in pool.imap_unordered(_harmonise_one, jobs, chunksize=1):
            n += 1
            if st.startswith("fail"):
                log(f"  [{st}] {sid}")
            if n % 5 == 0 or n == len(jobs):
                el = (time.time() - t0) / 60
                log(f"harmonise {n}/{len(jobs)} | {el:.1f} min | ETA {el / n * (len(jobs) - n):.1f} min")


# ------------------------------------------------------------------------------------------------ extract
def window_starts(n, sid):
    if n < WINDOW:
        return []
    starts = list(range(0, n - WINDOW + 1, STRIDE))
    if len(starts) > MAX_WINDOWS:
        seed = int(hashlib.md5(str(sid).encode("utf-8")).hexdigest()[:8], 16)
        starts = sorted(np.random.default_rng(seed).choice(starts, MAX_WINDOWS, replace=False))
    return list(starts)


def stage_extract():
    import torch
    C = _load("c211", "211_canonical_benchmark.py")
    X235 = _load("x235", "235_extract_extended.py")
    from src.data.scaling import robust_scale
    from src.data.window_features import spectral_stream
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    from src.models.v6 import ConceptExtractor
    FEAT = C.FEAT / TAG; WIN = C.ALZ / "harmonised_windows" / TAG
    if (FEAT / "meta.npz").exists():
        log("extract: already done"); return
    FEAT.mkdir(parents=True, exist_ok=True); WIN.mkdir(parents=True, exist_ok=True)
    ex = ConceptExtractor(emg_free=True); nodes = SpectralNodeFeatures(sfreq=SFREQ).eval()
    W_all, C_rows, S_rows, N_rows, E_rows, groups, labels, starts, qc = [], [], [], [], [], [], [], [], {}
    for sid, st, grp, _ in recordings():
        side = C.ALZ / "harmonised" / f"{sid}.json"
        if not side.exists():
            log(f"extract: [missing harmonised] {sid}"); continue
        meta = json.loads(side.read_text())
        data = np.load(side.with_suffix(".npy"))
        dur_full = data.shape[1] / SFREQ
        data = data[:, :int(TRUNC_S * SFREQ)]
        post_line, post_rms = _line_rms(data, SFREQ)
        keep, n_rej, cand = [], 0, []
        for s in window_starts(data.shape[1], sid):
            w = data[:, s:s + WINDOW]
            cand.append((float(np.ptp(w, axis=1).max()), s, w))
            if cand[-1][0] > REJECT_V:
                n_rej += 1; continue
            keep.append((s, w))
        relaxed_fill = False
        if RELAXED and len(keep) < K:
            keep = [(s, w) for _, s, w in sorted(cand, key=lambda c: c[0])[:K]]; relaxed_fill = True
        q = meta["qc"]
        qc[sid] = {"set": st, "group": grp, "duration_s": dur_full, "n_kept": len(keep), "n_rejected": n_rej,
                   "rejection_rate": n_rej / max(len(keep) + n_rej, 1), "retained": len(keep) >= K,
                   "ica_variance_removed": q.get("variance_removed"), "ica_n_excluded": len(q.get("ica_excluded", [])),
                   "raw_line_share": meta.get("raw_line_share_235s"), "raw_rms_uV": meta.get("raw_rms_uV_235s"),
                   "post_line_share": post_line, "post_rms_uV": post_rms, "unit_correction": q.get("unit_correction"),
                   "relaxed_fill": relaxed_fill}
        if len(keep) < K:
            continue
        r = np.random.default_rng(int(hashlib.md5(("q" + str(sid)).encode("utf-8")).hexdigest()[:8], 16))
        keep = [keep[i] for i in sorted(r.choice(len(keep), K, replace=False))]
        W = np.stack([robust_scale(w, "channel") for _, w in keep]).astype(np.float32)
        with torch.no_grad():
            Wt = torch.from_numpy(W)
            C_rows.append(ex.window_concepts(Wt).cpu().numpy()); N_rows.append(nodes(Wt).cpu().numpy())
        S_rows.append(np.stack([spectral_stream(w, SFREQ) for w in W]))
        E_rows.append(np.stack([X235.extras_for_window(w) for w in W]))
        W_all.append(W)
        groups += [sid] * K; labels += [grp] * K; starts += [s for s, _ in keep]
    np.save(WIN / "windows.npy", np.concatenate(W_all))
    np.save(FEAT / "window_concepts.npy", np.concatenate(C_rows).astype(np.float32))
    np.save(FEAT / "spectral.npy", np.concatenate(S_rows).astype(np.float32))
    np.save(FEAT / "node12.npy", np.concatenate(N_rows).astype(np.float32))
    np.save(FEAT / "node_ext4.npy", np.concatenate(E_rows).astype(np.float32))
    np.savez_compressed(FEAT / "meta.npz", groups=np.array(groups), labels=np.array(labels),
                        cohort=np.array([COHORT] * len(groups)), start=np.array(starts, np.int64))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ret = {f"{s}/{g}": [sum(1 for v in qc.values() if v["set"] == s and v["group"] == g and v["retained"]),
                        sum(1 for v in qc.values() if v["set"] == s and v["group"] == g)]
           for s in ("set1", "set2") for g in ("CN", "MCI")}
    (OUTDIR / f"extract{'_relaxed' if RELAXED else ''}.json").write_text(json.dumps({"n_subjects": len(set(groups)), "retained_of_offered": ret,
                                                     "qc": qc}, indent=1, default=float))
    log(f"extract: {len(set(groups))} subjects retained; retained/offered {ret}")


# -------------------------------------------------------------------------------------------------- score
def _misp_feats(C, T, cols, tr_mask):
    from src.models.v6 import aggregate_subject
    FEAT = C.FEAT / TAG
    meta = np.load(FEAT / "meta.npz", allow_pickle=False)
    pg = meta["groups"]; plab = np.array([str(x) for x in meta["labels"]])
    subs = np.unique(pg); grp_of = {s: plab[np.flatnonzero(pg == s)[0]] for s in subs}
    Cw = np.load(FEAT / "window_concepts.npy"); Sw = np.load(FEAT / "spectral.npy")
    Nw = np.load(FEAT / "node12.npy"); Ew = np.load(FEAT / "node_ext4.npy")
    Cp = np.stack([aggregate_subject(Cw[pg == s]) for s in subs])[:, cols]
    Cp = np.where(np.isfinite(Cp), Cp, np.nanmedian(T["X_concept"][tr_mask], 0))
    Sp = np.stack([np.nanmedian(Sw[pg == s], 0) for s in subs]); Sp = np.where(np.isfinite(Sp), Sp, np.nanmedian(T["X_spectral"][tr_mask], 0))
    Np_ = np.stack([np.nanmedian(Nw[pg == s].reshape(-1, 228), 0) for s in subs])
    feats = {"concept": Cp, "rf123": Sp, "gb": Sp, "svm": Sp, "logistic": Sp, "rf_node": np.nan_to_num(Np_)}
    node16 = np.nan_to_num(np.concatenate([Nw, Ew], -1).astype(np.float32))
    return pg, subs, grp_of, feats, node16


def stage_score():
    import torch
    C = _load("c211", "211_canonical_benchmark.py")
    tb = _load("tb", "157_task_battery.py")
    c212 = _load("c212", "212_classical_benchmark.py")
    e161 = C.load_161()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C.set_determinism(torch)
    from src.models.spectral75k import SpectraLite
    from src.models.v6 import CONCEPT_NAMES
    T = tb.subject_table("channel_clean")
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    tr_mask = T["cohort"] == "CAUEEG"
    pg, subs, grp_of, feats, node16 = _misp_feats(C, T, cols, tr_mask)
    Xp = np.load(C.ALZ / "harmonised_windows" / TAG / "windows.npy", mmap_mode="r")
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from src.models.v6 import SignConstrainedLogistic
    subsT = np.asarray(T["subjects"]); pos_of = {s: i for i, s in enumerate(subsT)}
    for task, pos in TASKS.items():
        keep = np.array([grp_of[s] in pos + ("CN",) for s in subs]); sp = subs[keep]
        yp = np.array([int(grp_of[s] in pos) for s in sp])
        od = OUTDIR / SCORES / task; od.mkdir(parents=True, exist_ok=True)
        m_idx = np.flatnonzero(np.isin(pg, sp)); g = pg[m_idx]
        for shard in sorted((C.OUT / task).glob("*__seed*.npz")):
            name, seed = shard.stem.rsplit("__seed", 1); seed = int(seed)
            dst = od / f"{name}__seed{seed}.npz"
            if dst.exists():
                continue
            ck = C.CKPT / task / name / f"seed{seed}"
            score = None
            if name == "spectralite16" and (ck / "fold4.pt").exists():
                X = node16[m_idx]; probs = []
                for k in range(5):
                    model = SpectraLite(n_classes=2, n_features=16).to(dev)
                    model.load_state_dict(torch.load(ck / f"fold{k}.pt", map_location=dev, weights_only=True)); model.eval()
                    with torch.no_grad():
                        pw = np.concatenate([torch.softmax(model(node_features=torch.tensor(X[i:i + 512], device=dev)), 1)[:, 1].cpu().numpy()
                                             for i in range(0, len(X), 512)])
                    probs.append(np.array([pw[g == s].mean() for s in sp])); del model
                score = np.mean(probs, 0)
            elif (ck / "fold4.pt").exists():
                X = np.asarray(Xp[m_idx]); probs = []
                for k in range(5):
                    model = C.build(name.replace("ladder_", "ladder:"), dev, e161)
                    model.load_state_dict(torch.load(ck / f"fold{k}.pt", map_location=dev, weights_only=True))
                    probs.append(C.predict(torch, model, X, dev, e161)); del model
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                score = C.subject_mean(np.mean(probs, 0), g, sp)
            elif name in feats:
                z = dict(np.load(shard, allow_pickle=True))
                ti = np.array([pos_of[s] for s in z["train_subjects"]]); ytr = z["train_y"]
                if name == "rf_node":
                    Fw = np.load(C.FEAT / "channel_clean" / "node12.npy", mmap_mode="r")
                    gm = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)["groups"]
                    Xtr = np.nan_to_num(np.stack([np.nanmedian(np.asarray(Fw[gm == s]).reshape(-1, 228), 0) for s in z["train_subjects"]]))
                else:
                    Xtr = (T["X_concept"] if name == "concept" else T["X_spectral"])[ti]
                Xte = feats[name][keep]
                folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(z["train_subjects"], ytr))
                signs = [c212.DIR[n] for n in T["names"]]
                make = {"rf123": lambda: RandomForestClassifier(400, random_state=seed, n_jobs=12, class_weight="balanced"),
                        "rf_node": lambda: RandomForestClassifier(400, random_state=seed, n_jobs=12, class_weight="balanced"),
                        "gb": lambda: GradientBoostingClassifier(random_state=seed),
                        "svm": lambda: SVC(probability=True, random_state=seed),
                        "logistic": lambda: LogisticRegression(max_iter=2000),
                        "concept": lambda: SignConstrainedLogistic(signs, C=0.5)}[name]
                scale = name in ("gb", "svm", "logistic"); out = []
                for a, _ in folds:
                    if scale:
                        sc = StandardScaler().fit(Xtr[a]); out.append(make().fit(sc.transform(Xtr[a]), ytr[a]).predict_proba(sc.transform(Xte))[:, 1])
                    else:
                        out.append(make().fit(Xtr[a], ytr[a]).predict_proba(Xte)[:, 1])
                score = np.mean(out, 0)
            if score is None:
                continue
            np.savez_compressed(dst, subjects=sp, y=yp, score=np.asarray(score, np.float32))
            log(f"score {task:15s} {name:28s} seed {seed}: n={len(sp)}")


# ------------------------------------------------------------------------------------- embed and probe
FMS = ("labram", "eegpt", "cbramod", "lead")      # MISP is not in LEAD's pretraining corpus (third_party/LEAD/README.md)


def stage_embed():
    import torch
    F = _load("f216", "216_fm_matched.py")
    C = F.C
    C.set_determinism(torch)
    from src.config.config import CHANNEL_NAMES
    CH = list(CHANNEL_NAMES)
    m = F.meta_for(TAG); suffix = f"__{TAG}"
    enc = F.Encoders(torch, "cuda", CH); X = None
    for fm in FMS:
        for norm in (("zscore",) if fm == "lead" else ("physical", "zscore")):
            path = F.EMB / f"{fm}_{norm}{suffix}.npy"
            if path.exists():
                continue
            if X is None:
                idx, X = F.load_windows(m, CH, (COHORT,))
                np.save(F.EMB / f"window_index{suffix}.npy", idx)
                log(f"embed: windows {X.shape}")
            model, feat, fs, nt = getattr(enc, fm)(); model.eval(); outs = []; t0 = time.time()
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    xb = F.resample(X[i:i + 256], fs)[..., :nt]
                    xb = F.zscore(xb) if norm == "zscore" else xb / 100.0
                    outs.append(feat(torch.from_numpy(xb).to("cuda")).float().cpu().numpy())
            E = np.concatenate(outs).astype(np.float32); np.save(path, E)
            log(f"embed: {fm}/{norm} {E.shape} in {(time.time() - t0) / 60:.1f} min")
            del model; torch.cuda.empty_cache()


def stage_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    F = _load("f216", "216_fm_matched.py")
    C = F.C
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    idx = np.load(F.EMB / "window_index.npy")
    groups = meta["groups"][idx]
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"][idx]])
    coh = np.array([str(c) for c in meta["cohort"][idx]])
    mm = F.meta_for(TAG)
    mg = mm["groups"]; mlab = np.array([str(x) for x in mm["labels"]])
    grp_of = {s: mlab[np.flatnonzero(mg == s)[0]] for s in np.unique(mg)}
    checks = {}
    for fm in FMS:
        for norm in (("zscore",) if fm == "lead" else ("physical", "zscore")):
            SM = F.subject_vectors(fm, norm, TAG)
            if SM is None:
                log(f"probe: {fm}/{norm} MISP embeddings missing; skipped"); continue
            E = np.load(F.EMB / f"{fm}_{norm}.npy")
            subs = np.unique(groups)
            pos_of = {s: np.flatnonzero(groups == s) for s in subs}
            S = {s: E[pos_of[s]].mean(0) for s in subs}
            for task, mpos in TASKS.items():
                name = f"{fm}_probe_{norm}"
                src = C.OUT / task / f"{name}__seed42.npz"
                od = OUTDIR / "scores" / task; od.mkdir(parents=True, exist_ok=True)
                dst = od / f"{name}__seed42.npz"
                if dst.exists() or not src.exists():
                    continue
                z = dict(np.load(src, allow_pickle=True))
                pos, neg, site = C.TASKS[task]
                tr_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == "CAUEEG" and cls[pos_of[s][0]] in pos + neg}))
                te_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == site and cls[pos_of[s][0]] in pos + neg}))
                tr_y = np.array([int(cls[pos_of[s][0]] in pos) for s in tr_sub])
                Xtr = np.stack([S[s] for s in tr_sub]); Xte = np.stack([S[s] for s in te_sub])
                sp = np.array(sorted(s for s in SM if grp_of[s] in mpos + ("CN",)))
                yp = np.array([int(grp_of[s] in mpos) for s in sp]); Xm = np.stack([SM[s] for s in sp])
                folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
                ext, mis = [], []
                for a, _ in folds:
                    gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
                        max_iter=5000, class_weight="balanced")), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(Xtr[a], tr_y[a])
                    ext.append(gs.predict_proba(Xte)[:, 1]); mis.append(gs.predict_proba(Xm)[:, 1])
                ext = np.mean(ext, 0)
                dev = float(np.max(np.abs(ext - z["ext_score"]))) if len(ext) == len(z["ext_score"]) else float("inf")
                yte = np.asarray(z["ext_y"]).astype(int)
                dauc = abs(_auc(yte, ext) - _auc(yte, z["ext_score"])) if np.isfinite(dev) else float("inf")
                checks[f"{task}/{name}"] = {"max_abs_prob_diff": dev, "abs_auc_diff_canonical_site": dauc}
                # EEGPT's 2048-d probes reproduce to ~5e-4 in probability (solver tolerance), not 1e-4
                if not (dev < 1e-3 and dauc < 0.005):
                    log(f"probe: REFIT MISMATCH {task} {name} (max |d| {dev}, |dAUC| {dauc}); not scored"); continue
                np.savez_compressed(dst, subjects=sp, y=yp, score=np.mean(mis, 0).astype(np.float32))
                log(f"probe: {task} {name} scored (refit max |d| {dev:.1e}, n={len(sp)})")
    (OUTDIR / "probe_refit_checks.json").write_text(json.dumps(checks, indent=1))


# ------------------------------------------------------------------------------------------------ analyse
def _auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _strat_auc(y, s, sets):
    """Pair-weighted within-set AUC: concordance over CN-MCI pairs recorded in the same set."""
    num = den = 0.0
    for st in np.unique(sets):
        m = sets == st; yy = y[m]
        n1 = int(yy.sum()); n0 = int(m.sum()) - n1
        if n1 and n0:
            a = _auc(yy, s[m]); num += a * n1 * n0; den += n1 * n0
    return num / den if den else float("nan")


def _pct(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def _ensemble(task, name, sub="scores"):
    sc, subj, yv, n = [], None, None, 0
    for f in sorted((OUTDIR / sub / task).glob(f"{name}__seed*.npz")):
        z = np.load(f, allow_pickle=True)
        s_, y_, x_ = np.asarray(z["subjects"]).astype(str), np.asarray(z["y"]).astype(int), np.asarray(z["score"], float)
        o = np.argsort(s_); s_, y_, x_ = s_[o], y_[o], x_[o]
        if subj is None:
            subj, yv = s_, y_
        if not np.array_equal(s_, subj):
            continue
        sc.append(x_); n += 1
    return (subj, yv, np.mean(sc, 0), n) if sc else None


def _fisher(a, b, c, d):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, b], [c, d]])[1])


def stage_analyse(B=4000):
    C = _load("c211", "211_canonical_benchmark.py")
    R = _load("r257", "257_review_revision.py")
    ex = json.loads((OUTDIR / "extract.json").read_text())
    qc = ex["qc"]
    res = {"cohort": "MISP (Isfahan; set1 Kashefpoor 2016, set2 Kashefpoor 2019)",
           "plan_file_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(PLAN.stat().st_mtime)) if PLAN.exists() else None,
           "first_score_mtime": None, "retained_of_offered": ex["retained_of_offered"]}
    sc_files = sorted((OUTDIR / "scores").rglob("*.npz"), key=lambda p: p.stat().st_mtime)
    if sc_files:
        res["first_score_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(sc_files[0].stat().st_mtime))

    # ---- confound gate (recording properties; read before model AUCs)
    gate = {}
    sids = sorted(qc)
    yy = np.array([int(qc[s]["group"] == "MCI") for s in sids]); ss = np.array([qc[s]["set"] for s in sids])
    ret = np.array([bool(qc[s]["retained"]) for s in sids])
    for st in ("pooled", "set1", "set2"):
        m = np.ones(len(sids), bool) if st == "pooled" else ss == st
        a = int((ret & m & (yy == 1)).sum()); b = int((~ret & m & (yy == 1)).sum())
        c = int((ret & m & (yy == 0)).sum()); d = int((~ret & m & (yy == 0)).sum())
        gate.setdefault("retention", {})[st] = {"MCI_retained": a, "MCI_dropped": b, "CN_retained": c, "CN_dropped": d,
                                                "fisher_p": _fisher(a, b, c, d)}
    props = ("rejection_rate", "ica_variance_removed", "raw_line_share", "post_line_share", "raw_rms_uV", "post_rms_uV", "duration_s")
    flagged = []
    for p in props:
        v = np.array([np.nan if qc[s].get(p) is None else float(qc[s][p]) for s in sids])
        for st in ("pooled", "set1", "set2"):
            m = ret & np.isfinite(v) & ((ss == st) if st != "pooled" else True)
            a = _auc(yy[m], v[m])
            gate.setdefault("auc_mci_vs_cn_retained", {}).setdefault(p, {})[st] = a
            if p != "duration_s" and np.isfinite(a) and (a >= 0.75 or a <= 0.25):
                flagged.append(f"{p}/{st} {a:.2f}")
    gate["flagged"] = flagged
    gate["note"] = "duration is truncated to 235 s before windowing and cannot reach the features"
    res["confound_gate"] = gate
    log(f"gate: retention {gate['retention']['pooled']} | flagged {flagged}")

    # ---- reproducibility of the re-harmonised MCI files against the August cache
    H = C.ALZ / "harmonised"; rep = []
    for s in sids:
        if qc[s]["group"] != "MCI":
            continue
        old = H / s.replace(PREFIX + "_", "MISP_").replace("_NORMAL_", "_NORMAL_")
        old = old.with_suffix(".npy")
        if old.exists():
            a = np.load(old, mmap_mode="r")[:, :int(TRUNC_S * SFREQ)]; b = np.load(H / f"{s}.npy", mmap_mode="r")[:, :int(TRUNC_S * SFREQ)]
            if a.shape == b.shape:
                rep.append(float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1]))
    res["reharmonise_vs_august_corr"] = {"n": len(rep), "min": min(rep) if rep else None, "median": float(np.median(rep)) if rep else None}

    # ---- per-model AUCs
    rng = np.random.default_rng(263)
    out_t = {}
    for task in TASKS:
        names = sorted({f.stem.rsplit("__seed", 1)[0] for f in (OUTDIR / "scores" / task).glob("*.npz")})
        ens = {n: _ensemble(task, n) for n in names}
        ens = {n: e for n, e in ens.items() if e is not None}
        if not ens:
            continue
        subj = ens[REF][0] if REF in ens else next(iter(ens.values()))[0]
        ens = {n: e for n, e in ens.items() if np.array_equal(e[0], subj)}
        y = ens[next(iter(ens))][1]; sets = np.array([s.split("_")[1] for s in subj])
        strata = [np.flatnonzero((sets == st) & (y == c)) for st in ("set1", "set2") for c in (0, 1)]
        boot_idx = [np.concatenate([rng.choice(ix, len(ix)) for ix in strata if len(ix)]) for _ in range(B)]
        per = {}
        for n, (_, _, s, nseed) in ens.items():
            a = _auc(y, s); bs = np.array([_auc(y[i], s[i]) for i in boot_idx])
            sa = _strat_auc(y, s, sets); bss = np.array([_strat_auc(y[i], s[i], sets[i]) for i in boot_idx])
            per[n] = {"n_seeds": nseed, "auc": a, "auc_ci95": _pct(bs), "auc_set_stratified": sa, "auc_set_stratified_ci95": _pct(bss),
                      "auc_set1": _auc(y[sets == "set1"], s[sets == "set1"]), "auc_set2": _auc(y[sets == "set2"], s[sets == "set2"])}
            if n != REF and REF in ens:
                r = ens[REF][2]; d = a - _auc(y, r)
                bd = bs - np.array([_auc(y[i], r[i]) for i in boot_idx])
                per[n]["delta_vs_forest"] = d; per[n]["delta_vs_forest_ci95"] = _pct(bd)
                per[n]["p_two_sided"] = float(min(1.0, 2 * min(np.mean(bd <= 0), np.mean(bd >= 0))))
        ps = {n: v["p_two_sided"] for n, v in per.items() if "p_two_sided" in v}
        order = sorted(ps, key=ps.get); m_ = len(order); run = 0.0
        for i, n in enumerate(order):
            run = max(run, min(1.0, (m_ - i) * ps[n])); per[n]["p_holm"] = run
        A = np.array([v["auc"] for v in per.values()])
        out_t[task] = {"n": int(len(y)), "n_mci": int(y.sum()), "n_cn": int((y == 0).sum()),
                       "n_by_set": {st: [int(((sets == st) & (y == 0)).sum()), int(((sets == st) & (y == 1)).sum())] for st in ("set1", "set2")},
                       "per_model": per,
                       "across_models": {"n_models": len(A), "median": float(np.median(A)), "min": float(A.min()), "max": float(A.max()),
                                         "sd": float(A.std(ddof=1)) if len(A) > 1 else None}}
        log(f"[{task}] n={len(y)} ({int(y.sum())} MCI) | {len(A)} models median AUC {np.median(A):.3f} "
            f"({A.min():.3f}-{A.max():.3f}) | SpectraLite {per.get('spectral75k', {}).get('auc', float('nan')):.3f}")
    res["tasks"] = out_t

    # ---- predictions
    pr = {}
    t = out_t.get("CN_vs_MCI", {}).get("per_model", {}).get("spectral75k")
    if t:
        lo, hi = t["auc_ci95"]; a = t["auc"]
        holds = (lo <= 0.5 <= hi) and 0.40 <= a <= 0.65
        refuted = a >= 0.70 and lo > 0.5
        pr["P1"] = {"auc": a, "ci95": t["auc_ci95"], "set_stratified": t["auc_set_stratified"],
                    "verdict": "holds" if holds else ("refuted" if refuted else "neither (outside the predicted range, not refuted)")}
    for task in ("CN_vs_MCI", "CN_vs_impaired"):
        if task in out_t:
            med = out_t[task]["across_models"]["median"]
            pr.setdefault("P2", {})[task] = {"median": med, "verdict": "holds" if 0.45 <= med <= 0.65 else "fails"}
    res["predictions"] = pr

    # ---- P3 threshold transport
    thr = {}
    for task in ("CN_vs_MCI", "CN_vs_impaired"):
        e = _ensemble(task, "spectral75k")
        E = R.ens(task, "spectral75k")
        if e is None or E is None:
            continue
        subj, y, s, _ = e
        tau = R.src_threshold(E, 0.80)
        neg = np.flatnonzero(y == 0); posi = np.flatnonzero(y == 1)
        d = {"n_cn": int(len(neg)), "n_mci": int(len(posi)), "tau_caueeg": tau,
             "transferred": {"spec": float(np.mean(s[neg] < tau)), "sens": float(np.mean(s[posi] >= tau))}}
        kk = 20
        if len(neg) >= 26:
            rr = np.random.default_rng(7); spv, sev = [], []
            for _ in range(200):
                cal = rr.choice(neg, kk, replace=False); ev = np.setdiff1d(neg, cal)
                t_ = R.conformal_t(s[cal], 0.80)
                spv.append(float(np.mean(s[ev] < t_))); sev.append(float(np.mean(s[posi] >= t_)))
            d["conformal_k20"] = {"spec_mean": float(np.mean(spv)), "spec_p2_5_97_5": _pct(spv),
                                  "sens_mean": float(np.mean(sev)), "sens_p2_5_97_5": _pct(sev)}
        else:
            d["conformal_k20"] = f"not run: {len(neg)} retained controls (< 26)"
        thr[task] = d
    res["thresholds"] = thr
    if "CN_vs_MCI" in thr:
        d = thr["CN_vs_MCI"]; cf = d["conformal_k20"]
        res["predictions"]["P3"] = {"transferred_spec": d["transferred"]["spec"],
                                    "conformal_spec_mean": cf["spec_mean"] if isinstance(cf, dict) else None,
                                    "verdict": ("holds" if isinstance(cf, dict) and d["transferred"]["spec"] < 0.80 and 0.73 <= cf["spec_mean"] <= 0.95
                                                else "not testable" if not isinstance(cf, dict) else "fails")}

    # ---- MCI endpoint: cohort/model decomposition over GENEEG, P-ADIC, MISP (supplement only)
    task = "CN_vs_MCI"
    if task in out_t:
        data = {}
        for n in out_t[task]["per_model"]:
            E = R.ens(task, n)
            e = _ensemble(task, n)
            if E is None or e is None or "ext" not in E or "padic" not in E:
                continue
            data[n] = {"GENEEG": (E["ext"][1], E["ext"][2]), "P-ADIC": (E["padic"][1], E["padic"][2]), "MISP": (e[1], e[2])}
        models = list(data); sites = ["GENEEG", "P-ADIC", "MISP"]
        if len(models) >= 3:
            def decompose(M):
                gm = M.mean(); cm = M.mean(0); mm = M.mean(1)
                ss_c = M.shape[0] * np.sum((cm - gm) ** 2); ss_m = M.shape[1] * np.sum((mm - gm) ** 2)
                ss_t = np.sum((M - gm) ** 2); ss_r = ss_t - ss_c - ss_m
                return {"share_cohort": ss_c / ss_t, "share_model": ss_m / ss_t, "share_residual": ss_r / ss_t,
                        "cohort_means": dict(zip(sites, cm.tolist()))}
            M = np.array([[_auc(*data[m][st]) for st in sites] for m in models])
            dec = {"n_models": len(models), "models": models, "auc_matrix": M.tolist(), **decompose(M)}
            rb = np.random.default_rng(2572); bs = []
            for _ in range(1000):
                Mb = np.zeros_like(M)
                for j, st in enumerate(sites):
                    yv = data[models[0]][st][0]; i1 = np.flatnonzero(yv == 1); i0 = np.flatnonzero(yv == 0)
                    ii = np.r_[rb.choice(i1, i1.size), rb.choice(i0, i0.size)]
                    Mb[:, j] = [_auc(yv[ii], data[m][st][1][ii]) for m in models]
                bs.append(decompose(Mb))
            for k in ("share_cohort", "share_model", "share_residual"):
                dec[k + "_ci95"] = _pct([b[k] for b in bs])
            res["mci_variance"] = dec
            log(f"[MCI variance] {len(models)} models: cohort {dec['share_cohort']:.2f} model {dec['share_model']:.2f} "
                f"residual {dec['share_residual']:.2f}; cohort means {dec['cohort_means']}")
    # ---- sensitivity: every recording kept (dropped ones from their 24 least-artefactual windows)
    rel = {}
    if (OUTDIR / "scores_relaxed").exists():
        for task in TASKS:
            names = sorted({f.stem.rsplit("__seed", 1)[0] for f in (OUTDIR / "scores_relaxed" / task).glob("*.npz")})
            per = {}
            for n in names:
                e = _ensemble(task, n, "scores_relaxed")
                if e is None:
                    continue
                subj, y, s, _ = e; sets = np.array([x.split("_")[1] for x in subj])
                strata = [np.flatnonzero((sets == st) & (y == c)) for st in ("set1", "set2") for c in (0, 1)]
                bs = [_auc(y[i], s[i]) for i in (np.concatenate([rng.choice(ix, len(ix)) for ix in strata]) for _ in range(2000))]
                per[n] = {"n": int(len(y)), "n_mci": int(y.sum()), "auc": _auc(y, s), "auc_ci95": _pct(bs),
                          "auc_set1": _auc(y[sets == "set1"], s[sets == "set1"]), "auc_set2": _auc(y[sets == "set2"], s[sets == "set2"])}
            if per:
                A = np.array([v["auc"] for v in per.values()])
                rel[task] = {"per_model": per, "median": float(np.median(A)), "min": float(A.min()), "max": float(A.max())}
                log(f"[relaxed {task}] n={next(iter(per.values()))['n']} | {len(A)} models median {np.median(A):.3f} | "
                    f"SpectraLite {per.get('spectral75k', {}).get('auc', float('nan')):.3f}")
    res["sensitivity_all_recordings"] = rel
    (OUTDIR / "misp.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"analyse: wrote {OUTDIR / 'misp.json'} | predictions {json.dumps(res['predictions'], default=float)[:400]}")


def _keep_awake():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="harmonise,extract,score,embed,probe,analyse")
    ap.add_argument("--limit", type=int, default=0, help="harmonise only the first N recordings (smoke test)")
    ap.add_argument("--relaxed", action="store_true", help="sensitivity: keep every recording (extract and score only)")
    args = ap.parse_args()
    global TAG, SCORES, RELAXED
    if args.relaxed:
        TAG, SCORES, RELAXED = TAG + "_relaxed", "scores_relaxed", True
    _keep_awake()
    for st in args.stage.split(","):
        log(f"=== stage {st}")
        t0 = time.time()
        try:
            if st == "harmonise":
                stage_harmonise(args.limit)
            else:
                {"extract": stage_extract, "score": stage_score, "embed": stage_embed, "probe": stage_probe,
                 "analyse": stage_analyse}[st]()
        except Exception as e:                                           # noqa: BLE001
            import traceback
            log(f"ERROR in stage {st}: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
            raise
        log(f"=== stage {st} done in {(time.time() - t0) / 60:.1f} min")
    log("ALL DONE")


if __name__ == "__main__":
    main()
