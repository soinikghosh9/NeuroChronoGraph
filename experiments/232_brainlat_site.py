"""BrainLat as a third external site: harmonise, match, and score every canonical model.

BrainLat (Prado et al., Sci Data 2023; Synapse syn51549340): AD, bvFTD and HC on 128-ch Biosemi, five
Latin-American sites. `src/data/brainlat_loader.py` subsamples the 128 electrodes to the pipeline's 19-ch
10-20 set and returns a Raw exactly like every other cohort. This script brings the recordings through the
same harmonisation and matched-window pipeline and scores the models already trained on CAUEEG -- no
retraining for the networks (211 checkpoints), identical fold refits for the classical models (212) -- so
BrainLat is a genuinely blind third site. Mirrors experiments/224_padic_site.py.

Classes: CN (controls), AD, FTD (bvFTD). Contrasts scored:
  CN_vs_AD, CN_vs_dementia -> CN vs AD ;  CN_vs_impaired -> CN vs AD+FTD.
(No MCI in BrainLat, so CN_vs_MCI is skipped.)

Stages (resumable):
  inspect    subject counts, ages, sexes per class -> brainlat_inspect.json, brainlat_subject_meta.json
  harmonise  src.data.harmonise.harmonise() on every usable recording -> <Alzfiles>/harmonised
  extract    matched-corpus rules (235 s truncation, 200 uV rejection before scaling, 24 clean 4-s
             windows, md5-seeded selection, per-channel robust scaling); window tensor, node/QEEG/concepts;
             acquisition confound + channel-order sanity check -> brainlat_extract.json
  score      appends brainlat_subjects / brainlat_y / brainlat_score to every canonical shard

    python experiments/232_brainlat_site.py --stage inspect,harmonise,extract,score
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


C = _load("c211", "211_canonical_benchmark.py")
ALZ = C.ALZ
BRAINLAT_ROOT = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles")) / "datasets" / "BrainLat Data"
PREFIX = {"AD": "BRAINLAT_AD", "CN": "BRAINLAT_CN", "FTD": "BRAINLAT_FTD"}
HARM = ALZ / "harmonised"
TAG = "brainlat_channel_clean"
FEAT = C.FEAT / TAG
WIN = ALZ / "harmonised_windows" / TAG
CAN = C.OUT
WINDOW, STRIDE, MAX_WINDOWS, REJECT_V, SFREQ, TRUNC_S, K = 1024, 1024, 50, 200e-6, 256.0, 235.0, 24
TASK_SITE = {"CN_vs_AD": (("AD",), ("CN",)), "CN_vs_dementia": (("AD", "FTD"), ("CN",)),
             "CN_vs_impaired": (("AD", "FTD"), ("CN",))}
GROUPS = ("AD", "CN", "FTD")


def _loader():
    from src.data.brainlat_loader import BrainLatLoader
    return BrainLatLoader(BRAINLAT_ROOT)


def stage_inspect():
    L = _loader()
    rep, subj = {}, {}
    for grp in GROUPS:
        ids = L.get_subject_ids(group=grp)
        ages = np.array([float(L.get_subject_info(s)["age"]) if str(L.get_subject_info(s)["age"]).strip().isdigit()
                         else np.nan for s in ids], float)
        sexes = [L.get_subject_info(s)["sex"] for s in ids]
        rep[grp] = {"n_subjects": len(ids), "age_median": float(np.nanmedian(ages)) if len(ids) else None,
                    "age_min": float(np.nanmin(ages)) if len(ids) else None,
                    "age_max": float(np.nanmax(ages)) if len(ids) else None,
                    "n_age_ge_55": int(np.nansum(ages >= 55)), "sex_counts": {s: sexes.count(s) for s in set(sexes)}}
        for s in ids:
            info = L.get_subject_info(s)
            subj[f"{PREFIX[grp]}_{s}"] = {"group": grp, "age": info["age"], "sex": info["sex"]}
        print(grp, json.dumps(rep[grp]))
    (CAN / "brainlat_inspect.json").write_text(json.dumps(rep, indent=1))
    (CAN / "brainlat_subject_meta.json").write_text(json.dumps(subj, indent=1))
    print(f"wrote metadata for {len(subj)} BrainLat subjects")


def stage_harmonise():
    import mne
    mne.set_log_level("ERROR")
    from src.data import scaling
    from src.data.harmonise import harmonise
    scaling.set_mode("none")
    L = _loader()
    for grp in GROUPS:
        ids = L.get_subject_ids(group=grp); t0 = time.time(); done = fail = 0
        for k, sub in enumerate(ids, 1):
            sid = f"{PREFIX[grp]}_{sub}"
            npy, side = HARM / f"{sid}.npy", HARM / f"{sid}.json"
            if npy.exists() and side.exists():
                continue
            try:
                raw, qc = harmonise(L.load_raw(sub), run_ica=True)
                np.save(npy, raw.get_data().astype(np.float32))
                side.write_text(json.dumps({"subject": sid, "cohort": "BrainLat", "group": grp,
                                            "channels": raw.ch_names, "qc": qc}, indent=1, default=float))
                done += 1
            except Exception as e:                                       # noqa: BLE001
                fail += 1; print(f"  [fail] {sid}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            if k % 5 == 0:
                print(f"  {grp} {k}/{len(ids)} done {done} fail {fail} {(time.time()-t0)/60:.1f} min", flush=True)
        print(f"[{grp}] harmonised {done}, failed {fail}", flush=True)


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
    from src.data.scaling import robust_scale
    from src.data.window_features import spectral_stream
    from src.models.v5.spectral_nodes import SpectralNodeFeatures
    from src.models.v6 import ConceptExtractor
    FEAT.mkdir(parents=True, exist_ok=True); WIN.mkdir(parents=True, exist_ok=True)
    sides = sorted(p for p in HARM.glob("BRAINLAT_*.json"))
    ex = ConceptExtractor(emg_free=True); nodes = SpectralNodeFeatures(sfreq=SFREQ).eval()
    W_all, C_rows, S_rows, N_rows, groups, labels, starts, qc = [], [], [], [], [], [], [], {}
    for side in sides:
        meta = json.loads(side.read_text()); sid, grp = meta["subject"], meta.get("group", "?")
        if grp not in GROUPS:
            continue
        data = np.load(side.with_suffix(".npy"))
        dur_full = data.shape[1] / SFREQ
        data = data[:, :int(TRUNC_S * SFREQ)]
        keep, n_rej = [], 0
        for s in window_starts(data.shape[1], sid):
            w = data[:, s:s + WINDOW]
            if float(np.ptp(w, axis=1).max()) > REJECT_V:
                n_rej += 1; continue
            keep.append((s, w))
        qc[sid] = {"group": grp, "duration_s": dur_full, "n_kept": len(keep), "n_rejected": n_rej}
        if len(keep) < K:
            continue
        r = np.random.default_rng(int(hashlib.md5(("q" + str(sid)).encode("utf-8")).hexdigest()[:8], 16))
        keep = [keep[i] for i in sorted(r.choice(len(keep), K, replace=False))]
        W = np.stack([robust_scale(w, "channel") for _, w in keep]).astype(np.float32)
        with torch.no_grad():
            Wt = torch.from_numpy(W)
            C_rows.append(ex.window_concepts(Wt).cpu().numpy()); N_rows.append(nodes(Wt).cpu().numpy())
        S_rows.append(np.stack([spectral_stream(w, SFREQ) for w in W])); W_all.append(W)
        for s, _ in keep:
            groups.append(sid); labels.append(grp); starts.append(s)
    np.save(WIN / "windows.npy", np.concatenate(W_all))
    np.save(FEAT / "window_concepts.npy", np.concatenate(C_rows).astype(np.float32))
    np.save(FEAT / "spectral.npy", np.concatenate(S_rows).astype(np.float32))
    np.save(FEAT / "node12.npy", np.concatenate(N_rows).astype(np.float32))
    np.savez_compressed(FEAT / "meta.npz", groups=np.array(groups), labels=np.array(labels),
                        cohort=np.array(["BrainLat"] * len(groups)), start=np.array(starts))
    from scipy.stats import rankdata
    def auc(y, s):
        rr = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
        return float((rr[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    ret = {g: {"offered": sum(1 for v in qc.values() if v["group"] == g),
               "retained": sum(1 for v in qc.values() if v["group"] == g and v["n_kept"] >= K)} for g in GROUPS}
    conf = {}
    for pos in ("AD", "FTD"):
        ids = [k for k, v in qc.items() if v["group"] in (pos, "CN")]
        y = np.array([int(qc[k]["group"] == pos) for k in ids])
        conf[f"CN_vs_{pos}"] = {"duration_auc": auc(y, np.array([qc[k]["duration_s"] for k in ids])),
                                "rejection_rate_auc": auc(y, np.array([qc[k]["n_rejected"] / max(1, qc[k]["n_rejected"] + qc[k]["n_kept"]) for k in ids]))}
    N = np.concatenate(N_rows); from src.config.config import CHANNEL_NAMES as CH
    alpha = N[..., 2].mean(0)
    order_ok = bool(alpha[[CH.index("O1"), CH.index("O2")]].mean() > alpha[[CH.index("Fp1"), CH.index("Fp2")]].mean())
    rep = {"retention": ret, "acquisition_confound_pre_matching": conf, "posterior_alpha_check_passed": order_ok,
           "n_subjects": len(set(groups)), "qc": qc}
    (CAN / "brainlat_extract.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("retention", "acquisition_confound_pre_matching", "posterior_alpha_check_passed", "n_subjects")}, indent=1))


def stage_score():
    import torch
    tb = _load("tb", "157_task_battery.py")
    e161 = C.load_161()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = np.load(FEAT / "meta.npz", allow_pickle=False)
    bg = meta["groups"]; blab = np.array([str(x) for x in meta["labels"]])
    Xb = np.load(WIN / "windows.npy", mmap_mode="r")
    subs_b = np.unique(bg); grp_of = {s: blab[np.flatnonzero(bg == s)[0]] for s in subs_b}
    from src.models.v6 import CONCEPT_NAMES, aggregate_subject
    Cw = np.load(FEAT / "window_concepts.npy"); Sw = np.load(FEAT / "spectral.npy"); Nw = np.load(FEAT / "node12.npy")
    T = tb.subject_table("channel_clean")
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    tr_mask = T["cohort"] == "CAUEEG"
    Cb = np.stack([aggregate_subject(Cw[bg == s]) for s in subs_b])[:, cols]
    Cb = np.where(np.isfinite(Cb), Cb, np.nanmedian(T["X_concept"][tr_mask], 0))
    Sb = np.stack([np.nanmedian(Sw[bg == s], 0) for s in subs_b]); Sb = np.where(np.isfinite(Sb), Sb, np.nanmedian(T["X_spectral"][tr_mask], 0))
    Nb = np.stack([np.nanmedian(Nw[bg == s].reshape(-1, 228), 0) for s in subs_b])
    feats = {"concept": Cb, "rf123": Sb, "gb": Sb, "svm": Sb, "logistic": Sb, "rf_node": np.nan_to_num(Nb)}
    c212 = _load("c212", "212_classical_benchmark.py")

    def _compute(name, seed, ck, z, keep, sb):
        """Score one shard's model on BrainLat; returns subject scores or None. Raises on a bad load."""
        if (ck / "fold4.pt").exists():
            m_idx = np.flatnonzero(np.isin(bg, sb)); X = np.asarray(Xb[m_idx]); g = bg[m_idx]
            probs = []
            for k in range(5):
                model = C.build(name.replace("ladder_", "ladder:"), dev, e161)
                model.load_state_dict(torch.load(ck / f"fold{k}.pt", map_location=dev, weights_only=True))
                probs.append(C.predict(torch, model, X, dev, e161)); del model
                if dev == "cuda":
                    torch.cuda.empty_cache()
            return C.subject_mean(np.mean(probs, 0), g, sb)
        if name in feats:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            from src.models.v6 import SignConstrainedLogistic
            ti = np.array([list(T["subjects"]).index(s) for s in z["train_subjects"]]); ytr = z["train_y"]
            if name == "rf_node":
                Fw = np.load(C.FEAT / "channel_clean" / "node12.npy", mmap_mode="r")
                gm = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)["groups"]
                Xtr = np.nan_to_num(np.stack([np.nanmedian(np.asarray(Fw[gm == s]).reshape(-1, 228), 0) for s in z["train_subjects"]]))
            else:
                Xtr = (T["X_concept"] if name == "concept" else T["X_spectral"])[ti]
            Xte = feats[name][keep]
            folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(z["train_subjects"], ytr))
            signs = [c212.DIR[n] for n in T["names"]]
            make = {"rf123": lambda: RandomForestClassifier(400, random_state=seed, n_jobs=6, class_weight="balanced"),
                    "rf_node": lambda: RandomForestClassifier(400, random_state=seed, n_jobs=6, class_weight="balanced"),
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
            return np.mean(out, 0)
        return None

    for task, (pos, neg) in TASK_SITE.items():
        keep = np.array([grp_of[s] in pos + neg for s in subs_b])
        sb = subs_b[keep]; yb = np.array([int(grp_of[s] in pos) for s in sb])
        print(f"   [{task}] BrainLat: {len(sb)} subjects ({int(yb.sum())} positive)", flush=True)
        for shard in sorted((CAN / task).glob("*__seed*.npz")):
            z = dict(np.load(shard, allow_pickle=True))
            if "brainlat_score" in z:
                continue
            name, seed = shard.stem.rsplit("__seed", 1); seed = int(seed)
            ck = C.CKPT / task / name / f"seed{seed}"
            try:
                score = _compute(name, seed, ck, z, keep, sb)
            except Exception as e:                                       # noqa: BLE001
                print(f"   [skip] {task} {name} seed {seed}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                continue
            if score is None:
                continue
            z.update(brainlat_subjects=sb, brainlat_y=yb, brainlat_score=np.asarray(score, np.float32))
            np.savez_compressed(shard, **z)
            info_p = shard.with_suffix(".json")
            if info_p.exists():
                info = json.loads(info_p.read_text())
                info["auc_brainlat"] = C.auc(yb, score); info["n_brainlat"] = int(len(sb))
                info_p.write_text(json.dumps(info, indent=1))
            print(f"   {task:15s} {name:28s} seed {seed}: BrainLat AUC {C.auc(yb, score):.3f} "
                  f"(n={len(sb)}, {int(yb.sum())} pos)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="inspect")
    args = ap.parse_args()
    for st in args.stage.split(","):
        {"inspect": stage_inspect, "harmonise": stage_harmonise, "extract": stage_extract, "score": stage_score}[st]()


if __name__ == "__main__":
    main()
