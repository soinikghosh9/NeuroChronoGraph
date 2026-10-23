"""Recording-condition and preprocessing shift in the same subjects (ds004504 + ds006036).

ds006036 records the 88 ds004504 subjects at the same clinic, on the same amplifier, straight after the
eyes-closed session, during photic stimulation (mostly eyes open). Both releases provide the authors' own
preprocessed files (Butterworth 0.5-45 Hz, A1-A2 reference, ASR, ICA with ICLabel). Running both through our
pipeline gives three versions of the same subjects:

  raw   ds004504 raw files, eyes closed (the published external result; read from the canonical shards)
  ecd   ds004504 authors' derivatives, eyes closed           -> raw vs ecd isolates preprocessing
  pho   ds006036 authors' derivatives, photic session        -> ecd vs pho isolates recording condition

Nothing here is used to choose or tune a model. Scores go to outputs/results/canonical/protocol_shift/, never
into the canonical shards.

Stages (resumable):
  harmonise  src.data.harmonise.harmonise() on every derivative recording (parallel, CPU)
  extract    the matched-corpus rules of 224 (235 s, 200 uV rejection, 24 windows, md5 seeds, robust scaling)
  score      every network with fold checkpoints (GPU) and every classical model (fold refits, as in 224)
  analyse    per-model seed-ensemble AUCs, paired subject-bootstrap differences, threshold transport

    python experiments/260_protocol_shift.py --stage harmonise,extract,score,analyse
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


DS = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles")) / "datasets"
VERSIONS = {
    "ecd": {"glob": lambda sid: sorted((DS / "openneuro_ds004504" / "derivatives" / sid / "eeg").glob("*.set")),
            "prefix": "D4504D", "label": "ds004504 derivatives, eyes closed"},
    "pho": {"glob": lambda sid: sorted((DS / "ds006036" / "derivatives" / "eeglab" / sid / "eeg").glob("*.set")),
            "prefix": "D6036", "label": "ds006036 derivatives, photic session"},
}
GROUP = {"A": "AD", "F": "FTD", "C": "CN"}
OUTDIR = ROOT / "outputs" / "results" / "canonical" / "protocol_shift"
WINDOW, STRIDE, MAX_WINDOWS, REJECT_V, SFREQ, TRUNC_S, K = 1024, 1024, 50, 200e-6, 256.0, 235.0, 24
TASKS = {"CN_vs_dementia": ("AD", "FTD"), "CN_vs_AD": ("AD",)}


def participants():
    rows = {}
    lines = (DS / "ds006036" / "participants.tsv").read_text(encoding="utf-8-sig").splitlines()
    head = [h.strip() for h in lines[0].split("\t")]
    for ln in lines[1:]:
        v = dict(zip(head, [x.strip() for x in ln.split("\t")]))
        rows[v["participant_id"]] = {"group": GROUP[v["Group"]], "age": float(v["Age"]), "sex": v["Gender"],
                                     "mmse": float(v["MMSE"]) if v.get("MMSE") not in (None, "", "n/a") else None}
    return rows


def _harm_paths(C, version, sid):
    pre = VERSIONS[version]["prefix"]
    H = C.ALZ / "harmonised"
    return H / f"{pre}_{sid}.npy", H / f"{pre}_{sid}.json"


def _harmonise_one(args):
    version, sid, grp = args
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    import mne
    mne.set_log_level("ERROR")
    C = _load("c211", "211_canonical_benchmark.py")
    from src.config.config import CHANNEL_NAMES
    from src.data import scaling
    from src.data.harmonise import harmonise
    scaling.set_mode("none")
    npy, side = _harm_paths(C, version, sid)
    if npy.exists() and side.exists():
        return sid, "skip"
    files = VERSIONS[version]["glob"](sid)
    if not files:
        return sid, "missing"
    try:
        raw = mne.io.read_raw_eeglab(files[0], preload=True, verbose="ERROR")
        raw.rename_channels({k: v for k, v in {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.items() if k in raw.ch_names})
        raw.pick([c for c in CHANNEL_NAMES if c in raw.ch_names]); raw.reorder_channels(list(CHANNEL_NAMES))
        raw.set_channel_types({c: "eeg" for c in raw.ch_names})
        # share of the analyzed span (first 235 s) marked as photic stimulation or eyes open, from the annotations
        ann = [(float(a["onset"]), str(a["description"]).strip().lower()) for a in raw.annotations]
        photic_on = sorted(t for t, d in ann if d.startswith("photo") and "hz" in d)
        eyes = sorted((t, "open" if "open eyes" in d else "closed") for t, d in ann if d in ("open eyes", "closed eyes"))
        out, qc = harmonise(raw, run_ica=True)
        np.save(npy, out.get_data().astype(np.float32))
        def state_at(t):
            s = "unmarked"
            for tt, st in eyes:
                if tt <= t:
                    s = st
            return s
        grid = np.arange(0, TRUNC_S, 1.0)
        side.write_text(json.dumps({"subject": f"{VERSIONS[version]['prefix']}_{sid}", "source_id": sid,
                                    "cohort": version, "group": grp, "channels": out.ch_names, "qc": qc,
                                    "first_photic_s": photic_on[0] if photic_on else None,
                                    "eyes_open_fraction_235s": float(np.mean([state_at(t) == "open" for t in grid])),
                                    "eyes_closed_fraction_235s": float(np.mean([state_at(t) == "closed" for t in grid]))},
                                   indent=1, default=float))
        return sid, "done"
    except Exception as e:                                                   # noqa: BLE001
        return sid, f"fail {type(e).__name__}: {str(e)[:100]}"


def stage_harmonise():
    from multiprocessing import Pool
    P = participants()
    jobs = [(v, sid, P[sid]["group"]) for v in VERSIONS for sid in sorted(P)]
    t0 = time.time(); n = 0
    with Pool(12) as pool:
        for sid, st in pool.imap_unordered(_harmonise_one, jobs, chunksize=1):
            n += 1
            if st.startswith("fail") or st == "missing":
                print(f"  [{st}] {sid}", flush=True)
            if n % 20 == 0 or n == len(jobs):
                print(f"  harmonise {n}/{len(jobs)} | {(time.time() - t0) / 60:.1f} min", flush=True)


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
    from src.data.scaling import robust_scale
    from src.data.window_features import spectral_stream
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    from src.models.v6 import ConceptExtractor
    ex = ConceptExtractor(emg_free=True); nodes = SpectralNodeFeatures(sfreq=SFREQ).eval()
    for version, spec in VERSIONS.items():
        tag = f"protocol_{version}"
        FEAT = C.FEAT / tag; WIN = C.ALZ / "harmonised_windows" / tag
        if (FEAT / "meta.npz").exists():
            print(f"[{version}] already extracted"); continue
        FEAT.mkdir(parents=True, exist_ok=True); WIN.mkdir(parents=True, exist_ok=True)
        sides = sorted((C.ALZ / "harmonised").glob(f"{spec['prefix']}_sub-*.json"))
        W_all, C_rows, S_rows, N_rows, groups, labels, qc = [], [], [], [], [], [], {}
        for side in sides:
            meta = json.loads(side.read_text()); sid, grp = meta["source_id"], meta["group"]
            data = np.load(side.with_suffix(".npy"))
            dur_full = data.shape[1] / SFREQ
            data = data[:, :int(TRUNC_S * SFREQ)]
            keep, n_rej = [], 0
            for s in window_starts(data.shape[1], sid):
                w = data[:, s:s + WINDOW]
                if float(np.ptp(w, axis=1).max()) > REJECT_V:
                    n_rej += 1; continue
                keep.append((s, w))
            qc[sid] = {"group": grp, "duration_s": dur_full, "n_kept": len(keep), "n_rejected": n_rej,
                       "eyes_open_fraction_235s": meta.get("eyes_open_fraction_235s")}
            if len(keep) < K:
                continue
            r = np.random.default_rng(int(hashlib.md5(("q" + str(sid)).encode("utf-8")).hexdigest()[:8], 16))
            keep = [keep[i] for i in sorted(r.choice(len(keep), K, replace=False))]
            W = np.stack([robust_scale(w, "channel") for _, w in keep]).astype(np.float32)
            with torch.no_grad():
                Wt = torch.from_numpy(W)
                C_rows.append(ex.window_concepts(Wt).cpu().numpy()); N_rows.append(nodes(Wt).cpu().numpy())
            S_rows.append(np.stack([spectral_stream(w, SFREQ) for w in W])); W_all.append(W)
            groups += [sid] * K; labels += [grp] * K
        np.save(WIN / "windows.npy", np.concatenate(W_all))
        np.save(FEAT / "window_concepts.npy", np.concatenate(C_rows).astype(np.float32))
        np.save(FEAT / "spectral.npy", np.concatenate(S_rows).astype(np.float32))
        np.save(FEAT / "node12.npy", np.concatenate(N_rows).astype(np.float32))
        np.savez_compressed(FEAT / "meta.npz", groups=np.array(groups), labels=np.array(labels),
                            cohort=np.array([version] * len(groups)))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / f"extract_{version}.json").write_text(json.dumps(
            {"n_subjects": len(set(groups)), "retained_by_group": {g: sum(1 for k, v in qc.items() if v["group"] == g and v["n_kept"] >= K) for g in ("CN", "AD", "FTD")},
             "qc": qc}, indent=1, default=float))
        print(f"[{version}] {len(set(groups))} subjects extracted", flush=True)


def stage_score():
    import torch
    C = _load("c211", "211_canonical_benchmark.py")
    tb = _load("tb", "157_task_battery.py")
    e161 = C.load_161()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = False
    from src.models.v6 import CONCEPT_NAMES, aggregate_subject
    c212 = _load("c212", "212_classical_benchmark.py")
    T = tb.subject_table("channel_clean")
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    tr_mask = T["cohort"] == "CAUEEG"
    for version in VERSIONS:
        tag = f"protocol_{version}"
        FEAT = C.FEAT / tag
        meta = np.load(FEAT / "meta.npz", allow_pickle=False)
        pg = meta["groups"]; plab = np.array([str(x) for x in meta["labels"]])
        Xp = np.load(C.ALZ / "harmonised_windows" / tag / "windows.npy", mmap_mode="r")
        subs = np.unique(pg); grp_of = {s: plab[np.flatnonzero(pg == s)[0]] for s in subs}
        Cw = np.load(FEAT / "window_concepts.npy"); Sw = np.load(FEAT / "spectral.npy"); Nw = np.load(FEAT / "node12.npy")
        Cp = np.stack([aggregate_subject(Cw[pg == s]) for s in subs])[:, cols]
        Cp = np.where(np.isfinite(Cp), Cp, np.nanmedian(T["X_concept"][tr_mask], 0))
        Sp = np.stack([np.nanmedian(Sw[pg == s], 0) for s in subs]); Sp = np.where(np.isfinite(Sp), Sp, np.nanmedian(T["X_spectral"][tr_mask], 0))
        Np_ = np.stack([np.nanmedian(Nw[pg == s].reshape(-1, 228), 0) for s in subs])
        feats = {"concept": Cp, "rf123": Sp, "gb": Sp, "svm": Sp, "logistic": Sp, "rf_node": np.nan_to_num(Np_)}
        for task, pos in TASKS.items():
            keep = np.array([grp_of[s] in pos + ("CN",) for s in subs]); sp = subs[keep]
            yp = np.array([int(grp_of[s] in pos) for s in sp])
            od = OUTDIR / version / task; od.mkdir(parents=True, exist_ok=True)
            for shard in sorted((C.OUT / task).glob("*__seed*.npz")):
                name, seed = shard.stem.rsplit("__seed", 1); seed = int(seed)
                dst = od / f"{name}__seed{seed}.npz"
                if dst.exists():
                    continue
                ck = C.CKPT / task / name / f"seed{seed}"
                score = None
                if (ck / "fold4.pt").exists() and name != "spectralite16":
                    m_idx = np.flatnonzero(np.isin(pg, sp)); X = np.asarray(Xp[m_idx]); g = pg[m_idx]
                    probs = []
                    for k in range(5):
                        model = C.build(name.replace("ladder_", "ladder:"), dev, e161)
                        model.load_state_dict(torch.load(ck / f"fold{k}.pt", map_location=dev, weights_only=True))
                        probs.append(C.predict(torch, model, X, dev, e161)); del model
                        if dev == "cuda":
                            torch.cuda.empty_cache()
                    score = C.subject_mean(np.mean(probs, 0), g, sp)
                elif name in feats:
                    z = dict(np.load(shard, allow_pickle=True))
                    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import StratifiedKFold
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.svm import SVC
                    from src.models.v6 import SignConstrainedLogistic
                    subsT = np.asarray(T["subjects"]); pos_of = {s: i for i, s in enumerate(subsT)}
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
                print(f"   {version} {task:15s} {name:28s} seed {seed}: AUC {C.auc(yp, score):.3f} (n={len(sp)})", flush=True)


# ------------------------------------------------------------------------------------------------- analyse
def _auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s, method="average"); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stage_analyse():
    C = _load("c211", "211_canonical_benchmark.py")
    P = participants()
    res = {"versions": {k: v["label"] for k, v in VERSIONS.items()} | {"raw": "ds004504 raw, eyes closed (published)"},
           "tasks": {}}
    for v in VERSIONS:
        f = OUTDIR / f"extract_{v}.json"
        if f.exists():
            e = json.loads(f.read_text())
            eo = [q.get("eyes_open_fraction_235s") for q in e["qc"].values() if q.get("eyes_open_fraction_235s") is not None]
            res.setdefault("extract", {})[v] = {"n_subjects": e["n_subjects"], "retained_by_group": e["retained_by_group"],
                                                "eyes_open_fraction_235s_median": float(np.median(eo)) if eo else None}
    rng = np.random.default_rng(0); B = 4000
    for task, pos in TASKS.items():
        per_model = {}
        for shard in sorted((C.OUT / task).glob("*__seed42.npz")):
            name = shard.stem.rsplit("__seed", 1)[0]
            ens = {}
            for v in ("raw", "ecd", "pho"):
                sc, subj, yv = [], None, None
                for seed in (42, 43, 44):
                    if v == "raw":
                        p = C.OUT / task / f"{name}__seed{seed}.npz"
                        if not p.exists():
                            continue
                        z = np.load(p, allow_pickle=True)
                        if "ext_subjects" not in z.files:
                            continue
                        s_, y_, x_ = np.asarray(z["ext_subjects"]), np.asarray(z["ext_y"]), np.asarray(z["ext_score"])
                        if task == "CN_vs_AD":
                            pass
                    else:
                        p = OUTDIR / v / task / f"{name}__seed{seed}.npz"
                        if not p.exists():
                            continue
                        z = np.load(p, allow_pickle=True)
                        s_, y_, x_ = np.asarray(z["subjects"]), np.asarray(z["y"]), np.asarray(z["score"])
                    o = np.argsort(s_); s_, y_, x_ = s_[o], y_[o], x_[o]
                    if subj is None:
                        subj, yv = s_, y_
                    if not np.array_equal(s_, subj):
                        continue
                    sc.append(x_)
                if sc:
                    ens[v] = (subj, yv, np.mean(sc, 0), len(sc))
            if len(ens) < 3:
                continue
            common = sorted(set(ens["raw"][0]) & set(ens["ecd"][0]) & set(ens["pho"][0]))
            if len(common) < 20:
                continue
            idx = {v: np.array([list(ens[v][0]).index(s) for s in common]) for v in ens}
            y = ens["raw"][1][idx["raw"]]
            S = {v: ens[v][2][idx[v]] for v in ens}
            aucs = {v: _auc(y, S[v]) for v in S}
            pos_i, neg_i = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            boots = {"pre": [], "cond": [], "total": []}
            for _ in range(B):
                bi = np.concatenate([rng.choice(pos_i, len(pos_i)), rng.choice(neg_i, len(neg_i))])
                a = {v: _auc(y[bi], S[v][bi]) for v in S}
                boots["pre"].append(a["ecd"] - a["raw"]); boots["cond"].append(a["pho"] - a["ecd"]); boots["total"].append(a["pho"] - a["raw"])
            d = {"pre": aucs["ecd"] - aucs["raw"], "cond": aucs["pho"] - aucs["ecd"], "total": aucs["pho"] - aucs["raw"]}
            per_model[name] = {"n": len(common), "n_pos": int(y.sum()), "n_seeds": {v: ens[v][3] for v in ens}, "auc": aucs,
                               "delta": {k: {"point": d[k], "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
                                             "p_two_sided": float(min(1.0, 2 * min(np.mean(np.array(boots[k]) <= 0), np.mean(np.array(boots[k]) >= 0))))}
                                         for k in d}}
        res["tasks"][task] = {"per_model": per_model}
        if per_model:
            for k in ("pre", "cond", "total"):
                v = np.array([m["delta"][k]["point"] for m in per_model.values()])
                res["tasks"][task][f"summary_{k}"] = {"n_models": len(v), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                                                      "n_negative": int((v < 0).sum())}
            for v in ("raw", "ecd", "pho"):
                a = np.array([m["auc"][v] for m in per_model.values()])
                res["tasks"][task][f"auc_{v}_across_models"] = {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "sd": float(a.std(ddof=1))}
            print(f"[{task}] {len(per_model)} models | median AUC raw {res['tasks'][task]['auc_raw_across_models']['median']:.3f} "
                  f"ecd {res['tasks'][task]['auc_ecd_across_models']['median']:.3f} pho {res['tasks'][task]['auc_pho_across_models']['median']:.3f}", flush=True)
    res["thresholds"] = threshold_transport(C)
    (OUTDIR / "protocol_shift.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUTDIR / 'protocol_shift.json'}")


def threshold_transport(C, model="spectral75k", task="CN_vs_dementia", k=20, n_splits=200):
    """Specificity and sensitivity of the CAUEEG threshold and of split-conformal thresholds when the clinic's
    calibration controls were recorded in one condition and patients are scored in another."""
    def ens(v):
        sc, subj, yv = [], None, None
        for seed in (42, 43, 44):
            if v == "raw":
                z = np.load(C.OUT / task / f"{model}__seed{seed}.npz", allow_pickle=True)
                s_, y_, x_ = np.asarray(z["ext_subjects"]), np.asarray(z["ext_y"]), np.asarray(z["ext_score"])
            else:
                p = OUTDIR / v / task / f"{model}__seed{seed}.npz"
                if not p.exists():
                    return None
                z = np.load(p, allow_pickle=True); s_, y_, x_ = np.asarray(z["subjects"]), np.asarray(z["y"]), np.asarray(z["score"])
            o = np.argsort(s_); subj, yv = s_[o], y_[o]; sc.append(x_[o])
        return subj, yv, np.mean(sc, 0)
    E = {v: ens(v) for v in ("raw", "ecd", "pho")}
    if any(e is None for e in E.values()):
        return {}
    common = sorted(set(E["raw"][0]) & set(E["ecd"][0]) & set(E["pho"][0]))
    S = {v: E[v][2][np.array([list(E[v][0]).index(s) for s in common])] for v in E}
    y = E["raw"][1][np.array([list(E["raw"][0]).index(s) for s in common])]
    # CAUEEG transferred threshold: 80th percentile of out-of-fold CAUEEG control scores of the seed ensemble
    oof = []
    for seed in (42, 43, 44):
        z = np.load(C.OUT / task / f"{model}__seed{seed}.npz", allow_pickle=True)
        o = np.argsort(z["train_subjects"]); oof.append(np.asarray(z["oof_score"])[o]); ytr = np.asarray(z["train_y"])[o]
    tau_src = float(np.quantile(np.mean(oof, 0)[ytr == 0], 0.80))
    out = {"model": model, "task": task, "n": len(common), "n_pos": int(y.sum()), "k": k,
           "transferred": {v: {"spec": float(np.mean(S[v][y == 0] < tau_src)), "sens": float(np.mean(S[v][y == 1] >= tau_src))} for v in S}}
    neg = np.flatnonzero(y == 0); posi = np.flatnonzero(y == 1)
    r_k = min(int(np.ceil(0.80 * (k + 1))), k)
    rng = np.random.default_rng(7)
    for cal_v, ev_v in (("ecd", "ecd"), ("pho", "pho"), ("ecd", "pho"), ("raw", "pho")):
        sp_, se_ = [], []
        for _ in range(n_splits):
            cal = rng.choice(neg, k, replace=False); ev = np.setdiff1d(neg, cal)
            tau = np.sort(S[cal_v][cal])[r_k - 1] + 1e-9
            sp_.append(float(np.mean(S[ev_v][ev] < tau))); se_.append(float(np.mean(S[ev_v][posi] >= tau)))
        out[f"conformal_cal_{cal_v}_eval_{ev_v}"] = {"spec_mean": float(np.mean(sp_)), "spec_p2_5_97_5": [float(np.percentile(sp_, 2.5)), float(np.percentile(sp_, 97.5))],
                                                     "sens_mean": float(np.mean(se_)), "sens_p2_5_97_5": [float(np.percentile(se_, 2.5)), float(np.percentile(se_, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="harmonise")
    args = ap.parse_args()
    for st in args.stage.split(","):
        print(f"=== stage {st}", flush=True)
        {"harmonise": stage_harmonise, "extract": stage_extract, "score": stage_score, "analyse": stage_analyse}[st]()


if __name__ == "__main__":
    main()
