"""BrainLat scores for the physical-unit encoder probes, with label-free per-recording gain normalisation.

BrainLat controls (`rs_eeg` export) are ~4x smaller in amplitude than every other group in every cohort
(253_amplitude_gain_audit.py); patients come from the `rs-HEP` export. Probes fed physical units (uV/100) are
therefore gain-confounded at BrainLat. Fix, applied to every BrainLat recording without reference to its label:
scale the recording so its broadband RMS (median over matched windows of the across-channel mean square) equals
the median of the CAUEEG development recordings. The probes themselves are refit exactly as in 216 (verified
against the stored ds004504 scores) and only the BrainLat scores change; the gain-confounded scores are kept
in the shard as brainlat_score_unnormalised.

    python experiments/254_brainlat_gain_normalised_probes.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E = _load("e252", "252_fm_external_complete.py")
F, C = E.F, E.C
TAG = "__brainlat_gainnorm"


def rms(X, g):
    return {s: float(np.sqrt(np.median((X[g == s] ** 2).mean(-1)))) for s in np.unique(g)}


def main():
    import torch
    from src.config.config import CHANNEL_NAMES
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    C.set_determinism(torch)
    CH = list(CHANNEL_NAMES)
    meta = np.load(C.FEAT / "channel_clean" / "meta.npz", allow_pickle=False)
    i_c, Xc = F.load_windows(meta, CH, ("CAUEEG",))
    ref = float(np.median(list(rms(Xc, meta["groups"][i_c]).values()))); del Xc
    bm = F.meta_for(E.BL_TAG); b_idx, XB = F.load_windows(bm, CH, ("BrainLat",)); gB = bm["groups"][b_idx]
    r = rms(XB, gB)
    gain = np.array([ref / r[s] for s in gB], np.float32)[:, None, None]
    XB = XB * gain
    print(f"CAUEEG median RMS {ref:.2f} uV; BrainLat gains median {np.median(list(ref / v for v in r.values())):.2f}", flush=True)
    enc = F.Encoders(torch, "cuda", CH)
    fms = ("labram", "eegpt", "cbramod")
    for fm in fms:
        path = F.EMB / f"{fm}_physical{TAG}.npy"
        if path.exists():
            continue
        model, feat, fs, nt = getattr(enc, fm)(); model.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(XB), 256):
                xb = F.resample(XB[i:i + 256], fs)[..., :nt] / 100.0
                outs.append(feat(torch.from_numpy(xb).to("cuda")).float().cpu().numpy())
        np.save(path, np.concatenate(outs).astype(np.float32)); del model; torch.cuda.empty_cache()
    np.save(F.EMB / f"window_index{TAG}.npy", b_idx)

    idx = np.load(F.EMB / "window_index.npy")
    groups = meta["groups"][idx]
    cls = np.array([C.CLASS[str(x).upper()] for x in meta["labels"][idx]])
    coh = np.array([str(c) for c in meta["cohort"][idx]])
    labels_b = E.brainlat_labels()
    report = {"reference_rms_uV": ref, "gain_by_group_median": {
        k: round(float(np.median([ref / r[s] for s in r if labels_b[s] == k])), 2) for k in ("CN", "AD", "FTD")}}
    for fm in fms:
        Eb = np.load(F.EMB / f"{fm}_physical{TAG}.npy")
        SB = {s: Eb[gB == s].mean(0) for s in np.unique(gB)}
        Em = np.load(F.EMB / f"{fm}_physical.npy")
        subs = np.unique(groups); pos_of = {s: np.flatnonzero(groups == s) for s in subs}
        S = {s: Em[pos_of[s]].mean(0) for s in subs}
        for task in ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired"):
            dest = C.OUT / task / f"{fm}_probe_physical__seed42.npz"
            z = dict(np.load(dest, allow_pickle=True))
            if "brainlat_score_unnormalised" in z:
                continue
            pos, neg, site = C.TASKS[task]
            tr_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == "CAUEEG" and cls[pos_of[s][0]] in pos + neg}))
            te_sub = np.array(sorted({s for s in subs if coh[pos_of[s][0]] == site and cls[pos_of[s][0]] in pos + neg}))
            tr_y = np.array([int(cls[pos_of[s][0]] in pos) for s in tr_sub])
            Xtr = np.stack([S[s] for s in tr_sub]); Xte = np.stack([S[s] for s in te_sub])
            b_sub = np.asarray(z["brainlat_subjects"]); b_y = np.asarray(z["brainlat_y"])
            Xb = np.stack([SB[s] for s in b_sub])
            folds = list(StratifiedKFold(5, shuffle=True, random_state=C.SPLIT_SEED).split(tr_sub, tr_y))
            ext, bl = [], []
            for a, _ in folds:
                gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=5000, class_weight="balanced")), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
                    cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(Xtr[a], tr_y[a])
                ext.append(gs.predict_proba(Xte)[:, 1]); bl.append(gs.predict_proba(Xb)[:, 1])
            dev = float(np.max(np.abs(np.mean(ext, 0) - z["ext_score"])))
            # lbfgs on 2048-d EEGPT embeddings differs by ~1e-4 across BLAS thread counts; ranks must not
            assert dev < 2e-3 and abs(C.auc(z["ext_y"], np.mean(ext, 0)) - C.auc(z["ext_y"], z["ext_score"])) < 1e-3,                 f"refit mismatch {task} {fm} {dev}"
            bl = np.mean(bl, 0).astype(np.float32)
            a_old = C.auc(b_y, z["brainlat_score"]); a_new = C.auc(b_y, bl)
            z.update(brainlat_score_unnormalised=z["brainlat_score"], brainlat_score=bl)
            tmp = dest.with_name(dest.stem + ".tmp.npz"); np.savez_compressed(tmp, **z); os.replace(tmp, dest)
            info = json.loads(dest.with_suffix(".json").read_text())
            info.update(auc_brainlat=a_new, auc_brainlat_unnormalised=a_old,
                        brainlat_input="per-recording gain normalised to the CAUEEG median broadband RMS (254)")
            dest.with_suffix(".json").write_text(json.dumps(info, indent=1))
            report[f"{task}/{fm}_probe_physical"] = {"auc_unnormalised": round(a_old, 3), "auc_gain_normalised": round(a_new, 3)}
            print(f"{task:15s} {fm:8s} physical  BrainLat {a_old:.3f} -> {a_new:.3f}", flush=True)
    (C.OUT / "brainlat_gain_normalised_probes.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
