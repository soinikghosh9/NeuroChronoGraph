"""P-ADIC as a second external site (review C3): harmonise, match, and score every canonical model.

The project previously held only P-ADIC's MCI file and so recorded the corpus as single-class. The Dryad
release also has an Alzheimer's file and a control file (Shor & Benninger et al. 2021). This script
brings them through exactly the pipeline every other cohort went through and scores the models already
trained on CAUEEG -- no retraining for the networks (211 checkpoints), identical fold refits for the
classical models (212) -- so P-ADIC is a genuinely blind second site.

Stages (all resumable):
  inspect    struct names, fields, subject counts, sampling rates, durations; any per-subject age field
  harmonise  src.data.harmonise.harmonise() on every AD / control recording -> <Alzfiles>/harmonised
  extract    the matched-corpus rules (235 s truncation, 200 uV rejection before scaling, exactly 24
             clean 4-s windows, per-subject md5-seeded selection, per-channel robust scaling) for P-ADIC
             AD, CN and the existing MCI subjects; window tensor, node features, QEEG-123 block, concepts;
             acquisition confound check (duration and retention AUC) and a channel-order sanity check
  score      appends padic_subjects / padic_y / padic_score to every canonical shard
             tasks: CN_vs_AD and CN_vs_dementia -> CN vs AD; CN_vs_impaired -> CN vs AD+MCI;
             CN_vs_MCI -> CN vs MCI

    python experiments/224_padic_site.py --stage inspect,harmonise,extract,score
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
PADIC = ALZ / "datasets" / "P-ADIC"
FILES = {"AD": PADIC / "p-adic alz" / "alz_c1_new.mat", "CN": PADIC / "p-adic ctrl" / "controls_c1_new.mat",
         "MCI": PADIC / "p-adic mci" / "mci_c1_new.mat"}
PREFIX = {"AD": "PADIC_AD", "CN": "PADIC_CN", "MCI": "PADIC_MCI"}
HARM = ALZ / "harmonised"
TAG = "padic_channel_clean"
FEAT = C.FEAT / TAG
WIN = ALZ / "harmonised_windows" / TAG
CAN = C.OUT
WINDOW, STRIDE, MAX_WINDOWS, REJECT_V, SFREQ, TRUNC_S, K = 1024, 1024, 50, 200e-6, 256.0, 235.0, 24
TASK_SITE = {"CN_vs_AD": (("AD",), ("CN",)), "CN_vs_dementia": (("AD",), ("CN",)),
             "CN_vs_impaired": (("AD", "MCI"), ("CN",)), "CN_vs_MCI": (("MCI",), ("CN",))}


def stage_inspect():
    import h5py
    from src.data.padic_group_loader import PADICGroupDataset
    rep = {}
    for grp, f in FILES.items():
        if not f.exists():
            print(f"[{grp}] missing: {f}"); continue
        ds = PADICGroupDataset(f, grp, PREFIX[grp])
        info = ds.info_map
        if not info:
            print(f"[{grp}] no usable recordings found in {f.name}"); continue
        fs = [v["sfreq"] for v in info.values()]; dur = [v["n_samples"] / v["sfreq"] for v in info.values()]
        ages = np.array([v["age"] if v["age"] else np.nan for v in info.values()], float)
        sexes = [v["sex"] for v in info.values()]
        with h5py.File(f, "r") as h:
            fields = list(h[ds.struct].keys())
        rep[grp] = {"struct": ds.struct, "fields": fields, "n_subjects": len(info),
                    "sfreq_counts": {str(k): int(v) for k, v in zip(*np.unique(fs, return_counts=True))},
                    "duration_s_median": float(np.median(dur)), "duration_s_min": float(np.min(dur)),
                    "age_median": float(np.nanmedian(ages)), "age_min": float(np.nanmin(ages)),
                    "age_max": float(np.nanmax(ages)), "age_missing": int(np.isnan(ages).sum()),
                    "n_age_ge_55": int((ages >= 55).sum()), "n_age_ge_60": int((ages >= 60).sum()),
                    "sex_counts": {s: sexes.count(s) for s in set(sexes)}}
        print(grp, json.dumps(rep[grp]))
    for readme in PADIC.rglob("AUTHOR_DATASET*.txt"):
        print(f"--- {readme.name} ---\n" + readme.read_text(errors="ignore")[:3000])
    (CAN / "padic_inspect.json").write_text(json.dumps(rep, indent=1))
    # per-subject age and sex, for the age-increment, age-matching and sex checks at this site
    subj = {}
    for grp, f in FILES.items():
        if not f.exists():
            continue
        ds = PADICGroupDataset(f, grp, PREFIX[grp])
        for sid, i in ds.info_map.items():
            subj[sid] = {"group": grp, "age": i["age"], "sex": i["sex"], "sfreq": i["sfreq"],
                         "duration_s": i["n_samples"] / i["sfreq"]}
    (CAN / "padic_subject_meta.json").write_text(json.dumps(subj, indent=1))
    print(f"wrote per-subject metadata for {len(subj)} P-ADIC subjects")


def stage_harmonise():
    import mne
    mne.set_log_level("ERROR")
    from src.data import scaling
    from src.data.harmonise import harmonise
    from src.data.padic_group_loader import PADICGroupDataset
    scaling.set_mode("none")
    for grp in ("AD", "CN"):
        if not FILES[grp].exists():
            raise SystemExit(f"{FILES[grp]} not found -- download it from Dryad first")
        ds = PADICGroupDataset(FILES[grp], grp, PREFIX[grp]); t0 = time.time(); done = fail = 0
        for k, sid in enumerate(ds.get_subject_ids(), 1):
            npy, side = HARM / f"{sid}.npy", HARM / f"{sid}.json"
            if npy.exists() and side.exists():
                continue
            try:
                raw, qc = harmonise(ds.load_raw(sid), run_ica=True)
                np.save(npy, raw.get_data().astype(np.float32))
                side.write_text(json.dumps({"subject": sid, "cohort": "P-ADIC", "group": grp,
                                            "channels": raw.ch_names, "qc": qc}, indent=1, default=float))
                done += 1
            except Exception as e:                                       # noqa: BLE001
                fail += 1; print(f"  [fail] {sid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            if k % 10 == 0:
                print(f"  {grp} {k}/{len(ds.get_subject_ids())} done {done} fail {fail} {(time.time()-t0)/60:.1f} min", flush=True)


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
    sides = sorted(p for p in HARM.glob("PADIC_*.json"))
    ex = ConceptExtractor(emg_free=True); nodes = SpectralNodeFeatures(sfreq=SFREQ).eval()
    W_all, C_rows, S_rows, N_rows, groups, labels, starts, qc = [], [], [], [], [], [], [], {}
    for side in sides:
        meta = json.loads(side.read_text()); sid, grp = meta["subject"], meta.get("group", "?")
        if grp not in ("AD", "CN", "MCI"):
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
                        cohort=np.array(["P-ADIC"] * len(groups)), start=np.array(starts))
    # acquisition confound check and retention by class
    from scipy.stats import rankdata
    def auc(y, s):
        rr = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
        return float((rr[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    ret = {g: {"offered": sum(1 for v in qc.values() if v["group"] == g),
               "retained": sum(1 for v in qc.values() if v["group"] == g and v["n_kept"] >= K)} for g in ("AD", "CN", "MCI")}
    conf = {}
    for pos in ("AD", "MCI"):
        ids = [k for k, v in qc.items() if v["group"] in (pos, "CN")]
        y = np.array([int(qc[k]["group"] == pos) for k in ids])
        conf[f"CN_vs_{pos}"] = {"duration_auc": auc(y, np.array([qc[k]["duration_s"] for k in ids])),
                                "rejection_rate_auc": auc(y, np.array([qc[k]["n_rejected"] / max(1, qc[k]["n_rejected"] + qc[k]["n_kept"]) for k in ids]))}
    # channel-order sanity: relative alpha should be higher posteriorly than frontally
    N = np.concatenate(N_rows); from src.config.config import CHANNEL_NAMES as CH
    alpha = N[..., 2].mean(0)
    order_ok = bool(alpha[[CH.index("O1"), CH.index("O2")]].mean() > alpha[[CH.index("Fp1"), CH.index("Fp2")]].mean())
    rep = {"retention": ret, "acquisition_confound_pre_matching": conf, "posterior_alpha_check_passed": order_ok,
           "n_subjects": len(set(groups)), "qc": qc}
    (CAN / "padic_extract.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("retention", "acquisition_confound_pre_matching", "posterior_alpha_check_passed", "n_subjects")}, indent=1))


def stage_score():
    import torch
    tb = _load("tb", "157_task_battery.py")
    e161 = C.load_161()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = np.load(FEAT / "meta.npz", allow_pickle=False)
    pg = meta["groups"]; plab = np.array([str(x) for x in meta["labels"]])
    Xp = np.load(WIN / "windows.npy", mmap_mode="r")
    subs_p = np.unique(pg); grp_of = {s: plab[np.flatnonzero(pg == s)[0]] for s in subs_p}
    # subject-level classical features for P-ADIC, aggregated exactly as subject_table does
    from src.models.v6 import CONCEPT_NAMES, aggregate_subject
    Cw = np.load(FEAT / "window_concepts.npy"); Sw = np.load(FEAT / "spectral.npy"); Nw = np.load(FEAT / "node12.npy")
    T = tb.subject_table("channel_clean")
    RO = ("artefact_proxy", "log_n_windows", "spectral_entropy")
    cols = [i for i, n in enumerate(CONCEPT_NAMES) if n not in RO]
    tr_mask = T["cohort"] == "CAUEEG"
    Cp = np.stack([aggregate_subject(Cw[pg == s]) for s in subs_p])[:, cols]
    Cp = np.where(np.isfinite(Cp), Cp, np.nanmedian(T["X_concept"][tr_mask], 0))
    Sp = np.stack([np.nanmedian(Sw[pg == s], 0) for s in subs_p]); Sp = np.where(np.isfinite(Sp), Sp, np.nanmedian(T["X_spectral"][tr_mask], 0))
    Np_ = np.stack([np.nanmedian(Nw[pg == s].reshape(-1, 228), 0) for s in subs_p])
    feats = {"concept": Cp, "rf123": Sp, "gb": Sp, "svm": Sp, "logistic": Sp, "rf_node": np.nan_to_num(Np_)}
    c212 = _load("c212", "212_classical_benchmark.py")

    # P-ADIC's controls run from 19 to 80 years while its patients are 55-87, so the primary contrast
    # uses controls aged >= AGE_FLOOR and the unrestricted one is kept as a sensitivity analysis.
    pmeta = json.loads((CAN / "padic_subject_meta.json").read_text())
    age_of = {s: (pmeta.get(s, {}) or {}).get("age") for s in subs_p}
    AGE_FLOOR = 55.0
    for task, (pos, neg) in TASK_SITE.items():
        keep_all = np.array([grp_of[s] in pos + neg for s in subs_p])
        keep_s = np.array([k and (grp_of[s] in pos or (age_of.get(s) or 0) >= AGE_FLOOR)
                           for s, k in zip(subs_p, keep_all)])
        sp = subs_p[keep_s]
        yp = np.array([int(grp_of[s] in pos) for s in sp])
        sp_all = subs_p[keep_all]
        yp_all = np.array([int(grp_of[s] in pos) for s in sp_all])
        print(f"   [{task}] P-ADIC: {len(sp)} subjects ({int(yp.sum())} positive) with controls "
              f">= {AGE_FLOOR:.0f} y; {len(sp_all)} unrestricted", flush=True)
        for shard in sorted((CAN / task).glob("*__seed*.npz")):
            z = dict(np.load(shard, allow_pickle=True))
            if "padic_score" in z:
                continue
            name, seed = shard.stem.rsplit("__seed", 1); seed = int(seed)
            ck = C.CKPT / task / name / f"seed{seed}"
            score = score_all = None
            if (ck / "fold4.pt").exists():
                m_idx = np.flatnonzero(np.isin(pg, sp_all)); X = np.asarray(Xp[m_idx]); g = pg[m_idx]
                probs = []
                for k in range(5):
                    model = C.build(name.replace("ladder_", "ladder:"), dev, e161)
                    model.load_state_dict(torch.load(ck / f"fold{k}.pt", map_location=dev, weights_only=True))
                    probs.append(C.predict(torch, model, X, dev, e161)); del model
                    torch.cuda.empty_cache()
                w = np.mean(probs, 0)
                score_all = C.subject_mean(w, g, sp_all)
                score = np.array([score_all[list(sp_all).index(s)] for s in sp])
            elif name in feats:
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
                Xte_all = feats[name][keep_all]
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
                        sc = StandardScaler().fit(Xtr[a]); out.append(make().fit(sc.transform(Xtr[a]), ytr[a]).predict_proba(sc.transform(Xte_all))[:, 1])
                    else:
                        out.append(make().fit(Xtr[a], ytr[a]).predict_proba(Xte_all)[:, 1])
                score_all = np.mean(out, 0)
                score = np.array([score_all[list(sp_all).index(s)] for s in sp])
            if score is None:
                continue
            z.update(padic_subjects=sp, padic_y=yp, padic_score=np.asarray(score, np.float32),
                     padic_all_subjects=sp_all, padic_all_y=yp_all,
                     padic_all_score=np.asarray(score_all, np.float32))
            np.savez_compressed(shard, **z)
            info_p = shard.with_suffix(".json")
            if info_p.exists():
                info = json.loads(info_p.read_text())
                info["auc_padic"] = C.auc(yp, score); info["n_padic"] = int(len(sp))
                info["auc_padic_all_controls"] = C.auc(yp_all, score_all); info["n_padic_all"] = int(len(sp_all))
                info["padic_control_age_floor"] = AGE_FLOOR
                info_p.write_text(json.dumps(info, indent=1))
            print(f"   {task:15s} {name:28s} seed {seed}: P-ADIC AUC {C.auc(yp, score):.3f} "
                  f"(n={len(sp)}, {int(yp.sum())} pos) | all controls {C.auc(yp_all, score_all):.3f} "
                  f"(n={len(sp_all)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="inspect")
    args = ap.parse_args()
    for st in args.stage.split(","):
        {"inspect": stage_inspect, "harmonise": stage_harmonise, "extract": stage_extract, "score": stage_score}[st]()


if __name__ == "__main__":
    main()
