"""Protocol-matched training: SpectraLite and the QEEG forest trained on CAUEEG eyes-closed, photic-free windows.

Written before running (model_paper/PLAN_REFRAME_2026-10-13.md, N3). CAUEEG recordings mix eyes-open, eyes-closed
and photic segments; the external dementia cohorts are eyes closed. Everything except window selection is held
fixed: the same CAUEEG subjects, the canonical folds (fold_of_subject from the 211 shards), seeds 42/43/44, the
211 recipe, the first 235 s, the 200 uV rejection, robust scaling, and 24 windows drawn by an md5 seed. Windows
are kept only if every point is eyes closed (or unmarked) and photic-free. A subject with fewer than 12 such
windows is dropped from training and from the internal comparison. Both arms (all windows, eyes-closed windows)
of the forest are refitted here with identical code; the SpectraLite all-window arm is the canonical 211 shard.

This is a sensitivity analysis. The reported model does not change whatever the result.

    python experiments/261_protocol_matched_training.py --stage windows,train,forest,analyse
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
TASK = "CN_vs_dementia"; POS = ("AD", "FTD")
OUT = C.OUT / "protocol_matched"
WINF = C.ALZ / "harmonised_windows" / "caueeg_eyes_closed"
WINDOW, STRIDE, REJECT_V, SFREQ, TRUNC_S, K, K_MIN = 1024, 1024, 200e-6, 256.0, 235.0, 24, 12
SEEDS = (42, 43, 44)
EXT = {"ds004504": ("channel_clean", "ds004504_raw"), "P-ADIC": ("padic_channel_clean", None),
       "BrainLat": ("brainlat_channel_clean", None), "ecd": ("protocol_ecd", None), "pho": ("protocol_pho", None)}


def canonical(model="spectral75k"):
    z = np.load(C.OUT / TASK / f"{model}__seed42.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def eye_state():
    import mne
    base = C.ALZ / "datasets" / "caueeg-dataset"
    z = canonical(); subs = z["train_subjects"].astype(str)
    st = {}
    for sid in subs:
        serial = sid.split("_")[-1]
        ev = json.loads((base / "event" / f"{serial}.json").read_text())
        sf = float(mne.io.read_raw_edf(str(base / "signal" / "edf" / f"{serial}.edf"), preload=False, verbose="ERROR").info["sfreq"])
        eye, ph = [], []
        for t, name in ev:
            n = str(name).lower(); tt = t / sf
            if n.startswith("eyes open"):
                eye.append((tt, "open"))
            elif n.startswith("eyes closed"):
                eye.append((tt, "closed"))
            elif n.startswith("photic on"):
                ph.append((tt, True))
            elif n.startswith("photic off"):
                ph.append((tt, False))
        st[sid] = (eye, ph)
    return st


def _at(seq, t, default):
    v = default
    for tt, s in seq:
        if tt <= t:
            v = s
        else:
            break
    return v


def stage_windows():
    from src.data.scaling import robust_scale
    from src.data.window_features import spectral_stream
    if (WINF / "meta.npz").exists():
        print("eyes-closed windows already built"); return
    WINF.mkdir(parents=True, exist_ok=True)
    z = canonical(); subs = z["train_subjects"].astype(str)
    st = eye_state()
    W_all, S_all, groups, n_kept = [], [], [], {}
    for sid in subs:
        data = np.load(C.ALZ / "harmonised" / f"{sid}.npy")[:, :int(TRUNC_S * SFREQ)]
        eye, ph = st[sid]; keep = []
        for s in range(0, data.shape[1] - WINDOW + 1, STRIDE):
            pts = np.linspace(s / SFREQ, (s + WINDOW) / SFREQ, 9)
            if any(_at(eye, p, "unmarked") == "open" for p in pts) or any(_at(ph, p, False) for p in pts):
                continue
            w = data[:, s:s + WINDOW]
            if float(np.ptp(w, axis=1).max()) > REJECT_V:
                continue
            keep.append(w)
        n_kept[sid] = len(keep)
        if len(keep) < K_MIN:
            continue
        if len(keep) > K:
            r = np.random.default_rng(int(hashlib.md5(("ec" + sid).encode("utf-8")).hexdigest()[:8], 16))
            keep = [keep[i] for i in sorted(r.choice(len(keep), K, replace=False))]
        W = np.stack([robust_scale(w, "channel") for w in keep]).astype(np.float32)
        W_all.append(W); S_all.append(np.stack([spectral_stream(w, SFREQ) for w in W]).astype(np.float32))
        groups += [sid] * len(W)
    np.save(WINF / "windows.npy", np.concatenate(W_all)); np.save(WINF / "spectral.npy", np.concatenate(S_all))
    np.savez_compressed(WINF / "meta.npz", groups=np.array(groups))
    v = np.array(list(n_kept.values()))
    rep = {"n_subjects": len(subs), "n_retained": int((v >= K_MIN).sum()), "n_with_24": int((v >= K).sum()),
           "eyes_closed_windows_per_subject_median": float(np.median(v)), "n_kept": n_kept}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "windows.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in rep if k != "n_kept"}), flush=True)


def ext_sets(z):
    """External subject lists exactly as in the canonical shards (P-ADIC controls >= 55 y)."""
    sets = {"ds004504": z["ext_subjects"].astype(str), "P-ADIC": z["padic_subjects"].astype(str),
            "BrainLat": z["brainlat_subjects"].astype(str)}
    for v in ("ecd", "pho"):
        p = C.OUT / "protocol_shift" / v / TASK / "spectral75k__seed42.npz"
        if p.exists():
            sets[v] = np.load(p, allow_pickle=True)["subjects"].astype(str)
    return sets


def stage_train():
    import torch
    C.set_determinism(torch)
    e161 = C.load_161(); dev = "cuda"
    z = canonical(); tr_sub = z["train_subjects"].astype(str); tr_y = z["train_y"].astype(int); fold = z["fold_of_subject"].astype(int)
    m = np.load(WINF / "meta.npz"); g = m["groups"].astype(str); X = np.load(WINF / "windows.npy", mmap_mode="r")
    y_of = dict(zip(tr_sub, tr_y)); yw = np.array([y_of[s] for s in g])
    sets = ext_sets(z)
    XE = {}
    for name, (tag, coh) in EXT.items():
        if name not in sets:
            continue
        mt = np.load(C.FEAT / tag / "meta.npz", allow_pickle=False); gg = mt["groups"].astype(str)
        sel = np.isin(gg, sets[name]) & ((mt["cohort"].astype(str) == coh) if coh else True)
        XE[name] = (np.asarray(np.load(C.windows_path(tag), mmap_mode="r")[np.flatnonzero(sel)]), gg[sel])
    for seed in SEEDS:
        dest = OUT / f"spectral75k_ec__seed{seed}.npz"
        if dest.exists():
            continue
        t0 = time.time(); oof = np.full(len(g), np.nan, np.float32); ext = {n: [] for n in XE}
        for k in range(5):
            fit = np.isin(g, tr_sub[fold != k]); held = np.isin(g, tr_sub[fold == k])
            model, _ = C.train_one(torch, "spectral75k", np.asarray(X[np.flatnonzero(fit)]), yw[fit], dev, seed, e161)
            oof[held] = C.predict(torch, model, np.asarray(X[np.flatnonzero(held)]), dev, e161)
            for n, (Xe, _) in XE.items():
                ext[n].append(C.predict(torch, model, Xe, dev, e161))
            del model; torch.cuda.empty_cache()
        subs_in = np.array([s for s in tr_sub if s in set(g)])
        out = {"train_subjects": subs_in, "train_y": np.array([y_of[s] for s in subs_in]),
               "oof_score": C.subject_mean(oof, g, subs_in)}
        for n, (Xe, ge) in XE.items():
            out[f"{n}_subjects"] = sets[n]; out[f"{n}_score"] = C.subject_mean(np.mean(ext[n], 0), ge, sets[n])
        np.savez_compressed(dest, **out)
        print(f"   seed {seed}: OOF AUC {C.auc(out['train_y'], out['oof_score']):.3f} | {time.time() - t0:.0f}s", flush=True)


def _subject_qeeg(tag, coh, subs, spectral_file=None, groups=None):
    S = np.load(spectral_file if spectral_file is not None else C.FEAT / tag / "spectral.npy", mmap_mode="r")
    if groups is None:
        mt = np.load(C.FEAT / tag / "meta.npz", allow_pickle=False); groups = mt["groups"].astype(str)
        if coh:
            ok = mt["cohort"].astype(str) == coh
        else:
            ok = np.ones(len(groups), bool)
    else:
        ok = np.ones(len(groups), bool)
    return np.stack([np.nanmedian(np.asarray(S[np.flatnonzero(ok & (groups == s))]), 0) for s in subs])


def stage_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    z = canonical("rf123"); tr_sub = z["train_subjects"].astype(str); tr_y = z["train_y"].astype(int)
    sets = ext_sets(canonical())
    m = np.load(WINF / "meta.npz"); g = m["groups"].astype(str)
    have = np.array([s in set(g) for s in tr_sub])
    Xall = _subject_qeeg("channel_clean", "CAUEEG", tr_sub)
    Xec = np.full_like(Xall, np.nan); Xec[have] = _subject_qeeg(None, None, tr_sub[have], WINF / "spectral.npy", g)
    XE = {n: _subject_qeeg(tag, coh, sets[n]) for n, (tag, coh) in EXT.items() if n in sets}
    med = np.nanmedian(Xall, 0)
    fix = lambda A: np.where(np.isfinite(A), A, med)
    Xall, Xec = fix(Xall), fix(Xec); XE = {n: fix(v) for n, v in XE.items()}
    for arm, Xtr, msk in (("all", Xall, np.ones(len(tr_sub), bool)), ("ec", Xec, have)):
        for seed in SEEDS:
            dest = OUT / f"rf123_{arm}__seed{seed}.npz"
            if dest.exists():
                continue
            sub, yy, XX = tr_sub[msk], tr_y[msk], Xtr[msk]
            oof = np.zeros(len(sub)); ext = {n: [] for n in XE}
            for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(sub, yy):
                rf = RandomForestClassifier(400, random_state=seed, n_jobs=12, class_weight="balanced").fit(XX[a], yy[a])
                oof[b] = rf.predict_proba(XX[b])[:, 1]
                for n in XE:
                    ext[n].append(rf.predict_proba(XE[n])[:, 1])
            out = {"train_subjects": sub, "train_y": yy, "oof_score": oof}
            for n in XE:
                out[f"{n}_subjects"] = sets[n]; out[f"{n}_score"] = np.mean(ext[n], 0)
            np.savez_compressed(dest, **out)
            print(f"   rf123 {arm} seed {seed}: OOF AUC {C.auc(yy, oof):.3f}", flush=True)


def _labels(n, subs):
    z = canonical()
    if n in ("ds004504", "P-ADIC", "BrainLat"):
        key = {"ds004504": "ext", "P-ADIC": "padic", "BrainLat": "brainlat"}[n]
        d = dict(zip(z[f"{key}_subjects"].astype(str), z[f"{key}_y"].astype(int)))
    else:
        p = np.load(C.OUT / "protocol_shift" / n / TASK / "spectral75k__seed42.npz", allow_pickle=True)
        d = dict(zip(p["subjects"].astype(str), p["y"].astype(int)))
    return np.array([d[s] for s in subs])


def stage_analyse(B=4000):
    from scipy.stats import rankdata
    def auc(y, s):
        r = rankdata(s, method="average"); n1 = y.sum(); n0 = len(y) - n1
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    def load_arm(prefix):
        zs = [np.load(OUT / f"{prefix}__seed{s}.npz", allow_pickle=True) for s in SEEDS]
        return zs
    def canon_arm(model):
        zs = [np.load(C.OUT / TASK / f"{model}__seed{s}.npz", allow_pickle=True) for s in SEEDS]
        out = []
        for z in zs:
            d = {"train_subjects": z["train_subjects"], "train_y": z["train_y"], "oof_score": z["oof_score"],
                 "ds004504_subjects": z["ext_subjects"], "ds004504_score": z["ext_score"],
                 "P-ADIC_subjects": z["padic_subjects"], "P-ADIC_score": z["padic_score"],
                 "BrainLat_subjects": z["brainlat_subjects"], "BrainLat_score": z["brainlat_score"]}
            out.append(d)
        for v in ("ecd", "pho"):
            for z, s in zip(out, SEEDS):
                p = C.OUT / "protocol_shift" / v / TASK / f"{model}__seed{s}.npz"
                if p.exists():
                    q = np.load(p, allow_pickle=True); z[f"{v}_subjects"] = q["subjects"]; z[f"{v}_score"] = q["score"]
        return out
    def ens(zs, key, subs):
        vals = []
        for z in zs:
            d = dict(zip(np.asarray(z[f"{key}_subjects"]).astype(str), np.asarray(z[f"{key}_score"], float)))
            if not all(s in d for s in subs):
                return None
            vals.append([d[s] for s in subs])
        return np.mean(vals, 0)
    rng = np.random.default_rng(261)
    res = {"windows": json.loads((OUT / "windows.json").read_text()) if (OUT / "windows.json").exists() else None, "comparisons": {}}
    if res["windows"]:
        res["windows"].pop("n_kept", None)
    for label, base, ec in (("SpectraLite", canon_arm("spectral75k"), load_arm("spectral75k_ec")),
                            ("QEEG forest", load_arm("rf123_all"), load_arm("rf123_ec"))):
        r = {}
        # internal: subjects present in both arms
        common = sorted(set(np.asarray(ec[0]["train_subjects"]).astype(str)) & set(np.asarray(base[0]["train_subjects"]).astype(str)))
        y_of = dict(zip(np.asarray(base[0]["train_subjects"]).astype(str), np.asarray(base[0]["train_y"]).astype(int)))
        sets = {"internal": (common, np.array([y_of[s] for s in common]))}
        for n in ("ds004504", "P-ADIC", "BrainLat", "ecd", "pho"):
            if f"{n}_subjects" in ec[0].files:
                subs = list(np.asarray(ec[0][f"{n}_subjects"]).astype(str)); sets[n] = (subs, _labels(n, subs))
        for n, (subs, y) in sets.items():
            key = "oof" if n == "internal" else n
            sb = ens(base, "train" if n == "internal" else n, subs) if n != "internal" else None
            if n == "internal":
                def oof(zs):
                    v = []
                    for z in zs:
                        d = dict(zip(np.asarray(z["train_subjects"]).astype(str), np.asarray(z["oof_score"], float))); v.append([d[s] for s in subs])
                    return np.mean(v, 0)
                sb, se = oof(base), oof(ec)
            else:
                se = ens(ec, n, subs)
            if sb is None or se is None:
                continue
            a0, a1 = auc(y, sb), auc(y, se)
            i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0); d = []
            for _ in range(B):
                ii = np.r_[rng.choice(i1, i1.size), rng.choice(i0, i0.size)]
                d.append(auc(y[ii], se[ii]) - auc(y[ii], sb[ii]))
            r[n] = {"n": len(subs), "auc_all_windows": a0, "auc_eyes_closed": a1, "delta": a1 - a0,
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "p_two_sided": float(min(1.0, 2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))))}
            print(f"[{label}] {n:9s} n={len(subs):4d}: all {a0:.3f} -> eyes-closed {a1:.3f} ({a1 - a0:+.3f}, "
                  f"[{r[n]['ci95'][0]:+.3f}, {r[n]['ci95'][1]:+.3f}])", flush=True)
        res["comparisons"][label] = r
    (OUT / "protocol_matched.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT / 'protocol_matched.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="windows")
    a = ap.parse_args()
    for st in a.stage.split(","):
        print(f"=== stage {st}", flush=True)
        {"windows": stage_windows, "train": stage_train, "forest": stage_forest, "analyse": stage_analyse}[st]()


if __name__ == "__main__":
    main()
