"""Review-revision analyses (peer review of 2026-10-11; plan model_paper/REVIEW_AUDIT_AND_FIX_PLAN_2026-10-11.md).

Every analysis reads the canonical seed shards (outputs/results/canonical/<task>/<model>__seed<s>.npz) and the
released metadata; nothing is retrained and no external label enters any model. Sections (--sections):

  clinical     95% intervals for sensitivity, specificity, PPV/NPV (5/10/20% prevalence) and net benefit at the
               transferred and split-conformal (k=20) thresholds. Each of 2,000 resamples draws one of the 200
               calibration splits, then resamples its held-out controls and all cases. Point values are the means
               over the 200 splits, as in 250_clinical_metrics.py.
  crosssite    calibration controls from one recording site, evaluation at another (non-exchangeable by design):
               BrainLat Argentina <-> Chile; P-ADIC 500 Hz <-> 200 Hz control recordings (patients all 500 Hz).
  procedure    the transferred threshold computed per fold model (its own held-out CAUEEG controls) and applied to
               that fold model's external scores, against the published ensemble-vs-pooled-OOF procedure.
  sweep        target specificity 0.60-0.95 for the split-conformal rule: realized specificity, sensitivity, PPV/NPV.
  capacity     Spearman rho of parameter count with AUC over the 11 networks: two-level bootstrap CIs (networks and
               subjects), internal-minus-external difference, leave-family-out, single-seed variant, and the
               internal-versus-external gain over the QEEG forest against parameter count.
  variance     two-way decomposition of external AUC into cohort, model and residual shares, subject bootstrap.
  precision    paired SE of the AUC difference against the forest, minimum detectable difference at 80% power,
               TOST power at each margin, and the cohort size a 0.05 difference would need.
  endpoint     dementia-model AUC for CN vs AD and CN vs FTD separately at ds004504 and BrainLat.
  cohortid     healthy-control cohort decoding repeated on age-matched control sets.
  demographics per-cohort, per-class age, sex, severity (MMSE/MoCA), education and recording years.
  covariates   BrainLat education and sex; sex-stratified AUC at every cohort that releases sex.
  encoders     internal and external AUC of both input scalings for every frozen probe (selection effect).
  seeds        per-seed AUC of SpectraLite-12/16 beside the seed ensemble.

    python experiments/257_review_revision.py [--sections clinical,crosssite,...]
Out: outputs/results/canonical/review_revision.json (sections merged into the existing file)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata, spearmanr, chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from _calib_splits import anchor_splits  # noqa: E402

CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "review_revision.json"
ALZ = Path(os.environ.get("ALZ_ROOT", "C:/Users/user/Downloads/Alzfiles"))
BLAT = ALZ / "datasets" / "BrainLat Data" / "brainlat"
SITES = {"ext": "ds004504", "padic": "P-ADIC", "brainlat": "BrainLat"}
SL = {"spectral75k": "SpectraLite-12", "spectralite16": "SpectraLite-16"}
REF = "rf123"
K, TARGET = 20, 0.80
WINKEYS = {"ext": ("ext_win_folds", "win_ext_groups"), "padic": ("padic_win_folds", "win_padic_groups"),
           "brainlat": ("brainlat_win_folds", "win_brainlat_groups")}


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


CM = _load("c250_for257", "250_clinical_metrics.py")


# ----------------------------------------------------------------------------------------------- helpers
def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def delong(y, s):
    a = auc(y, s); lo, hi = CM.delong_ci(np.asarray(y).astype(int), np.asarray(s, float))
    return {"auc": a, "ci95": [lo, hi], "n": int(len(y)), "n_pos": int(np.sum(y))}


def wilson(k, n):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def pct(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [np.nan, np.nan]


def shards(task, model, seeds=None):
    fs = sorted((CAN / task).glob(f"{model}__seed*.npz"))
    if seeds is not None:
        fs = [f for f in fs if int(f.stem.split("__seed")[1]) in seeds]
    return fs


def ens(task, model, seeds=None, prefixes=("ext", "padic", "padic_all", "brainlat")):
    fs = shards(task, model, seeds)
    if not fs:
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]; z0 = zs[0]

    def al(z, ks, kv):
        o = {str(v): i for i, v in enumerate(z[ks])}
        return np.array([z[kv][o[str(v)]] for v in z0[ks]], float)
    out = {"n_seeds": len(zs), "seeds": [int(f.stem.split("__seed")[1]) for f in fs],
           "train_subjects": z0["train_subjects"].astype(str), "train_y": z0["train_y"].astype(int),
           "oof": np.mean([al(z, "train_subjects", "oof_score") for z in zs], 0)}
    if "fold_of_subject" in z0.files:
        out["fold_of_subject"] = z0["fold_of_subject"].astype(int)
    for p in prefixes:
        if all(f"{p}_score" in z.files for z in zs):
            out[p] = (z0[f"{p}_subjects"].astype(str), z0[f"{p}_y"].astype(int),
                      np.mean([al(z, f"{p}_subjects", f"{p}_score") for z in zs], 0))
    return out


def src_threshold(E, target=TARGET):
    return float(np.quantile(E["oof"][E["train_y"] == 0], target))


def conformal_t(anchor, target):
    a = np.sort(np.asarray(anchor, float)); k = len(a); j = int(np.ceil((k + 1) * target))
    return float(np.nextafter(a[min(j, k) - 1], np.inf))


def summary_models(task):
    s = json.loads((CAN / "summary.json").read_text())["tasks"][task]
    return s


def diag_maps():
    m = np.load(ROOT / "outputs" / "harmonised_features" / "channel_clean" / "meta.npz", allow_pickle=False)
    g = m["groups"].astype(str); lab = m["labels"].astype(str); coh = m["cohort"].astype(str)
    _, first = np.unique(g, return_index=True)
    code = {"A": "AD", "F": "FTD", "C": "CN", "AD": "AD", "FTD": "FTD", "CN": "CN", "CONTROL": "CN", "MCI": "MCI"}
    ds = {g[i]: code.get(lab[i].upper(), lab[i]) for i in first if coh[i] == "ds004504_raw"}
    pa = {k: v["group"] for k, v in json.loads((CAN / "padic_subject_meta.json").read_text()).items()}
    bl = {k: v["group"] for k, v in json.loads((CAN / "brainlat_subject_meta.json").read_text()).items()}
    return {"ext": ds, "padic": pa, "padic_all": pa, "brainlat": bl}


def brainlat_table():
    import pandas as pd
    rows = []
    for g, tag in (("HC", "CN"), ("AD", "AD"), ("bvFTD", "FTD")):
        d = pd.read_csv(BLAT / f"Demographics_{g}_EEG_data.csv").drop_duplicates("id EEG")
        c = pd.read_csv(BLAT / f"Cognition_{g}_EEG_data.csv")[["id EEG", "moca_total"]].drop_duplicates("id EEG")
        d = d.merge(c, on="id EEG", how="left")
        for _, r in d.iterrows():
            ctry = str(r["path"]).split("/")[-1]
            rows.append((f"BRAINLAT_{tag}_{r['id EEG']}",
                         {"country": ctry if ctry in ("AR", "CL") else None,
                          "sex_female": (None if pd.isna(r["sex"]) else int(r["sex"]) == 0),
                          "age": None if pd.isna(r["Age"]) else float(r["Age"]),
                          "education": None if pd.isna(r["years_education"]) else float(r["years_education"]),
                          "moca": None if pd.isna(r["moca_total"]) else float(r["moca_total"])}))
    return dict(rows)


def padic_mmse_dates():
    import h5py
    P = ALZ / "datasets" / "P-ADIC"
    out = {}
    for f, s, grp in (("p-adic alz/alz_c1_new.mat", "alz_r", "AD"), ("p-adic mci/mci_c1_new.mat", "mci_r", "MCI")):
        with h5py.File(P / f, "r") as h:
            g = h[s]; mm, dd = [], []
            for j in range(g["m"].shape[1]):
                a = np.array(h[g["m"][0][j]]).ravel()
                mm.append(float(a[0]) if a.size and a.dtype != np.uint64 else np.nan)
                d = h[g["eegdate"][0][j]]; arr = np.array(d).ravel()
                dd.append("".join(map(chr, arr)) if d.dtype == np.uint16 else "")
        out[grp] = {"mmse": mm, "dates": dd}
    return out


def merge_write(section, payload):
    rep = json.loads(OUT.read_text()) if OUT.exists() else {}
    rep[section] = payload
    OUT.write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[{section}] written", flush=True)


# ------------------------------------------------------------------------------------------------ sections
def sec_clinical(B=2000):
    out = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for model in ("spectral75k", "spectralite16", REF):
            E = ens(task, model)
            src = src_threshold(E)
            for p, site in SITES.items():
                subs, y, s = E[p]
                rng = np.random.default_rng(257)
                ctrl = subs[y == 0]; cases = s[y == 1]; idx = {v: i for i, v in enumerate(subs)}
                splits = list(anchor_splits(ctrl, K))
                point = {"transferred": [], "conformal": []}
                for anc, held in splits:
                    hs = s[[idx[h] for h in held]]; thr_c = CM.conformal(s[[idx[a] for a in anc]])
                    for rule, thr in (("transferred", src), ("conformal", thr_c)):
                        point[rule].append((np.mean(cases >= thr), np.mean(hs < thr)))
                boots = {"transferred": [], "conformal": []}
                for _ in range(B):
                    anc, held = splits[rng.integers(len(splits))]
                    hs = s[[idx[h] for h in held]]; thr_c = CM.conformal(s[[idx[a] for a in anc]])
                    hb = rng.choice(hs, len(hs)); cb = rng.choice(cases, len(cases))
                    for rule, thr in (("transferred", src), ("conformal", thr_c)):
                        se = float(np.mean(cb >= thr)); sp = float(np.mean(hb < thr))
                        row = {"sensitivity": se, "specificity": sp, "balanced_accuracy": (se + sp) / 2}
                        for pi in (0.05, 0.10, 0.20):
                            cm = CM.confusion_metrics(se, sp, pi)
                            row[f"ppv@{pi:.2f}"] = cm["ppv"]; row[f"npv@{pi:.2f}"] = cm["npv"]
                        for pt in (0.05, 0.10, 0.15, 0.20):
                            row[f"net_benefit@pt{pt:.2f}"] = 0.10 * se - 0.90 * (1 - sp) * pt / (1 - pt)
                        boots[rule].append(row)
                res = {"n_controls": int((y == 0).sum()), "n_cases": int(y.sum()), "n_heldout_controls": int((y == 0).sum() - K)}
                for rule in ("transferred", "conformal"):
                    se = float(np.mean([a for a, _ in point[rule]])); sp = float(np.mean([b for _, b in point[rule]]))
                    r = {"sensitivity": {"mean": se}, "specificity": {"mean": sp}, "balanced_accuracy": {"mean": (se + sp) / 2}}
                    for pi in (0.05, 0.10, 0.20):
                        cm = CM.confusion_metrics(se, sp, pi)
                        r[f"ppv@{pi:.2f}"] = {"mean": cm["ppv"]}; r[f"npv@{pi:.2f}"] = {"mean": cm["npv"]}
                        r[f"npv_gain_over_base@{pi:.2f}"] = {"mean": cm["npv"] - (1 - pi)}
                    for pt in (0.05, 0.10, 0.15, 0.20):
                        r[f"net_benefit@pt{pt:.2f}"] = {"mean": 0.10 * se - 0.90 * (1 - sp) * pt / (1 - pt)}
                    for key in boots[rule][0]:
                        r[key]["ci95"] = pct([b[key] for b in boots[rule]])
                    for pi in (0.05, 0.10, 0.20):
                        r[f"npv_gain_over_base@{pi:.2f}"]["ci95"] = pct([b[f"npv@{pi:.2f}"] - (1 - pi) for b in boots[rule]])
                    res[rule] = r
                out.setdefault(task, {}).setdefault(site, {})[model] = res
                c = res["conformal"]
                print(f"{task:15s} {site:9s} {model:14s} conformal sens {c['sensitivity']['mean']:.2f} "
                      f"[{c['sensitivity']['ci95'][0]:.2f},{c['sensitivity']['ci95'][1]:.2f}] spec "
                      f"{c['specificity']['mean']:.2f} [{c['specificity']['ci95'][0]:.2f},{c['specificity']['ci95'][1]:.2f}] "
                      f"NPV10 {c['npv@0.10']['mean']:.3f} gain {c['npv_gain_over_base@0.10']['mean']:+.3f}", flush=True)
    merge_write("clinical", {"B": B, "k": K, "target": TARGET, "results": out,
                             "note": "point = mean over 200 calibration splits; CI = 2,000 resamples, each one split "
                                     "with its held-out controls and all cases resampled; derived metrics from the "
                                     "resampled sensitivity and specificity"})


def sec_crosssite():
    out = {}
    bl = brainlat_table(); pm = json.loads((CAN / "padic_subject_meta.json").read_text())
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for model in ("spectral75k", "spectralite16", REF):
            E = ens(task, model); src = src_threshold(E)
            designs = []
            subs, y, s = E["brainlat"]
            grp = np.array([bl.get(v, {}).get("country") for v in subs], object)
            designs.append(("BrainLat", "AR", "CL", subs, y, s, grp))
            designs.append(("BrainLat", "CL", "AR", subs, y, s, grp))
            for pref, lab in (("padic", "P-ADIC (controls >= 55 y)"), ("padic_all", "P-ADIC (all controls)")):
                if pref in E:
                    subs, y, s = E[pref]
                    grp = np.array([f"{int(pm[v]['sfreq'])} Hz" for v in subs], object)
                    designs.append((lab, "500 Hz", "200 Hz", subs, y, s, grp))
                    designs.append((lab, "200 Hz", "500 Hz", subs, y, s, grp))
            dxm = diag_maps()["brainlat"]
            for site, A, Bs, subs, y, s, grp in designs:
                anc = (y == 0) & (grp == A); tc = (y == 0) & (grp == Bs)
                tcase = (y == 1) & (grp == Bs) if ((y == 1) & (grp == Bs)).sum() >= 5 else (y == 1)
                if anc.sum() < 5 or tc.sum() < 5:
                    continue
                k = int(anc.sum()); thr = CM.conformal(s[anc])
                sp_k = int(np.sum(s[tc] < thr)); se_k = int(np.sum(s[tcase] >= thr))
                st_k = int(np.sum(s[tc] < src))
                # exchangeable reference: random k-control anchors from the pooled controls, held-out controls
                ctrl = subs[y == 0]; idx = {v: i for i, v in enumerate(subs)}; ref_sp = []
                if k < len(ctrl):
                    for a_, h_ in anchor_splits(ctrl, k):
                        t_ = CM.conformal(s[[idx[v] for v in a_]]); ref_sp.append(np.mean(s[[idx[v] for v in h_]] < t_))
                r = {"calibrate_on": A, "evaluate_on": Bs, "k_anchor_controls": k,
                     "expected_specificity_exchangeable": float(min(np.ceil(0.8 * (k + 1)), k) / (k + 1)),
                     "n_test_controls": int(tc.sum()), "n_test_cases": int(tcase.sum()),
                     "cases_from": Bs if tcase is not (y == 1) and ((y == 1) & (grp == Bs)).sum() >= 5 else "all sites",
                     "specificity": sp_k / tc.sum(), "specificity_wilson95": wilson(sp_k, int(tc.sum())),
                     "sensitivity": se_k / tcase.sum(), "sensitivity_wilson95": wilson(se_k, int(tcase.sum())),
                     "transferred_specificity_on_test": st_k / tc.sum(),
                     "transferred_specificity_wilson95": wilson(st_k, int(tc.sum())),
                     "pooled_random_split_specificity_mean": float(np.mean(ref_sp)) if ref_sp else None,
                     "pooled_random_split_specificity_p2.5_97.5": pct(ref_sp) if ref_sp else None,
                     "auc_test_site": auc(np.r_[np.zeros(tc.sum()), np.ones(tcase.sum())], np.r_[s[tc], s[tcase]])}
                if site == "BrainLat":
                    dx = np.array([dxm.get(v) for v in subs], object)
                    r["sensitivity_by_dx"] = {}
                    for dd in ("AD", "FTD"):
                        mk = tcase & (dx == dd)
                        if mk.sum():
                            kk = int(np.sum(s[mk] >= thr))
                            r["sensitivity_by_dx"][dd] = {"k": kk, "n": int(mk.sum()), "rate": kk / int(mk.sum()),
                                                          "wilson95": wilson(kk, int(mk.sum()))}
                    r["cases_without_country"] = int(((y == 1) & (grp == None)).sum())  # noqa: E711
                out.setdefault(task, {}).setdefault(model, []).append({"cohort": site, **r})
                print(f"{task:15s} {model:14s} {site:26s} {A:>6s}->{Bs:6s} k={k:2d} spec {r['specificity']:.2f} "
                      f"[{r['specificity_wilson95'][0]:.2f},{r['specificity_wilson95'][1]:.2f}] (n={int(tc.sum())}) "
                      f"sens {r['sensitivity']:.2f} | transferred spec {r['transferred_specificity_on_test']:.2f}", flush=True)
    merge_write("crosssite", {"note": "calibration controls from one recording site, evaluation controls (and, where "
                                      "available, cases) from the other; Wilson intervals; BrainLat country from the "
                                      "release's path column; P-ADIC split by control sampling rate (patients all 500 Hz)",
                              "results": out})


def sec_procedure():
    out = {}
    extra = CAN / "review_rescore"
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for model in ("spectral75k", "spectralite16", "ncg_v5", REF):
            fs = shards(task, model)
            if not fs:
                continue
            res = {}
            for p, site in SITES.items():
                per_fold, pooled_thr_per_fold, sd_single, sd_ens = [], [], [], []
                ok = True
                for f in fs:
                    z = np.load(f, allow_pickle=True)
                    wk, gk = WINKEYS[p]
                    if wk in z.files:
                        W = z[wk].astype(float); G = z[gk].astype(str)
                    else:
                        alt = extra / f"{task}__{f.stem}__{p}.npz"
                        if not alt.exists():
                            ok = False; break
                        a = np.load(alt, allow_pickle=True); W = a["win_folds"].astype(float); G = a["win_groups"].astype(str)
                    subs = z[f"{p}_subjects"].astype(str); y = z[f"{p}_y"].astype(int)
                    oof = z["oof_score"].astype(float); ty = z["train_y"].astype(int); fos = z["fold_of_subject"].astype(int)
                    pooled = float(np.quantile(oof[ty == 0], TARGET))
                    S = np.stack([np.array([W[k][G == v].mean() for v in subs]) for k in range(5)])
                    for k in range(5):
                        tk = float(np.quantile(oof[(ty == 0) & (fos == k)], TARGET))
                        per_fold.append((np.mean(S[k][y == 1] >= tk), np.mean(S[k][y == 0] < tk)))
                        pooled_thr_per_fold.append((np.mean(S[k][y == 1] >= pooled), np.mean(S[k][y == 0] < pooled)))
                        sd_single.append(np.std(S[k][y == 0]))
                    sd_ens.append(np.std(S.mean(0)[y == 0]))
                if not ok:
                    continue
                E = ens(task, model); subs, y, s = E[p]; thr = src_threshold(E)
                res[site] = {
                    "published_ensemble": {"sensitivity": float(np.mean(s[y == 1] >= thr)), "specificity": float(np.mean(s[y == 0] < thr))},
                    "per_fold_matched": {"sensitivity": float(np.mean([a for a, _ in per_fold])),
                                         "specificity": float(np.mean([b for _, b in per_fold])),
                                         "specificity_range_over_fold_models": [float(min(b for _, b in per_fold)), float(max(b for _, b in per_fold))]},
                    "single_fold_scores_pooled_threshold": {"sensitivity": float(np.mean([a for a, _ in pooled_thr_per_fold])),
                                                            "specificity": float(np.mean([b for _, b in pooled_thr_per_fold]))},
                    "control_score_sd_single_fold_model": float(np.mean(sd_single)),
                    "control_score_sd_fold_ensemble": float(np.mean(sd_ens)),
                    "n_controls": int((y == 0).sum())}
                r = res[site]
                print(f"{task:15s} {model:14s} {site:9s} spec published {r['published_ensemble']['specificity']:.2f} | "
                      f"per-fold matched {r['per_fold_matched']['specificity']:.2f} | single-fold pooled "
                      f"{r['single_fold_scores_pooled_threshold']['specificity']:.2f} | control SD single "
                      f"{r['control_score_sd_single_fold_model']:.3f} vs ens {r['control_score_sd_fold_ensemble']:.3f}", flush=True)
            if res:
                out.setdefault(task, {})[model] = res
    merge_write("procedure", {"note": "transferred threshold = 80th percentile of CAUEEG control scores. published: "
                                      "pooled out-of-fold controls, external scored by the fold (and seed) ensemble. "
                                      "per_fold_matched: each fold model's threshold from its own held-out CAUEEG controls, "
                                      "applied to that fold model's external scores; averaged over folds and seeds.",
                              "results": out})


def sec_sweep():
    out = {}
    targets = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for model in ("spectral75k", "spectralite16", "ncg_v3"):
            E = ens(task, model)
            for p, site in SITES.items():
                subs, y, s = E[p]; ctrl = subs[y == 0]; idx = {v: i for i, v in enumerate(subs)}
                rows = []
                for t in targets:
                    se_l, sp_l = [], []
                    for anc, held in anchor_splits(ctrl, K):
                        thr = conformal_t(s[[idx[a] for a in anc]], t)
                        se_l.append(np.mean(s[y == 1] >= thr)); sp_l.append(np.mean(s[[idx[h] for h in held]] < thr))
                    se, sp = float(np.mean(se_l)), float(np.mean(sp_l))
                    # label-using reference: best sensitivity at >= t specificity on this cohort's own controls
                    c_sorted = np.sort(s[y == 0]); thr_o = c_sorted[min(int(np.ceil(t * len(c_sorted))), len(c_sorted) - 1)]
                    row = {"target": t, "specificity": sp, "sensitivity": se,
                           "sensitivity_p2.5_97.5": pct(se_l), "specificity_p2.5_97.5": pct(sp_l),
                           "oracle_sensitivity_at_target": float(np.mean(s[y == 1] > thr_o))}
                    for pi in (0.05, 0.10, 0.20):
                        cm = CM.confusion_metrics(se, sp, pi); row[f"ppv@{pi:.2f}"] = cm["ppv"]; row[f"npv@{pi:.2f}"] = cm["npv"]
                    rows.append(row)
                out.setdefault(task, {}).setdefault(site, {})[model] = rows
                print(f"{task:15s} {site:9s} {model:14s} " + " ".join(f"{r['target']:.2f}:{r['specificity']:.2f}/{r['sensitivity']:.2f}" for r in rows), flush=True)
    merge_write("sweep", {"k": K, "note": "split-conformal at each target; spec/sens = mean of 200 splits", "results": out})


def _auc_fast(y, s):
    return auc(y, s)


def sec_capacity(B=2000):
    task = "CN_vs_dementia"; S = summary_models(task)
    nets = list(S["capacity_correlation"]["internal"]["models"])
    params = {m: S["evaluation_sets"]["internal"][m]["params"] for m in nets}
    params["spectral75k"] = 75459; params["spectralite16"] = 75971
    fam = {m: ("graph" if m.startswith("ncg") else "spectralite" if m.startswith("spectral") else "published") for m in nets + ["spectral75k", "spectralite16"]}
    out = {"networks": nets, "params": params, "family": fam}

    def build(variant):
        names = list(nets)
        if variant.get("swap16"):
            names = [("spectralite16" if m == "spectral75k" else m) for m in names]
        seeds = variant.get("seeds")
        data = {}
        for m in names + [REF]:
            E = ens(task, m, seeds=seeds if m != REF else None)
            data[m] = {"internal": (E["train_y"], E["oof"])}
            for p, site in SITES.items():
                data[m][site] = (E[p][1], E[p][2])
        return names, data

    variants = {"published_seeds": {}, "seed42_only": {"seeds": [42]}, "variant_swapped": {"swap16": True}}
    for vname, v in variants.items():
        names, data = build(v)
        sets = ["internal"] + list(SITES.values())
        P = np.log10([params[m] for m in names])
        A = {st: np.array([_auc_fast(*data[m][st]) for m in names]) for st in sets}
        R = {st: np.array([_auc_fast(*data[REF][st])]) for st in sets}
        res = {"n_networks": len(names), "auc": {st: dict(zip(names, A[st].tolist())) for st in sets}}
        for st in sets:
            rho, pv = spearmanr(P, A[st]); res.setdefault("rho", {})[st] = {"rho": float(rho), "p": float(pv)}
            keep = [i for i, m in enumerate(names) if fam[m] != "graph"]
            r2, p2 = spearmanr(P[keep], A[st][keep]); res.setdefault("rho_without_graph_family", {})[st] = {"rho": float(r2), "p": float(p2), "n": len(keep)}
        # gain over the forest internally minus externally, against parameter count
        for st in sets[1:]:
            did = (A["internal"] - R["internal"][0]) - (A[st] - R[st][0])
            r, pv = spearmanr(P, did); res.setdefault("did_vs_params", {})[st] = {"rho": float(r), "p": float(pv),
                                                                                    "did": dict(zip(names, did.tolist()))}
        # two-level bootstrap: networks and subjects (stratified, identical subject resample for all networks)
        rng = np.random.default_rng(2571)
        boot = {st: [] for st in sets}; dboot = {st: [] for st in sets[1:]}; nboot = {st: [] for st in sets}
        ys = {st: data[names[0]][st][0] for st in sets}
        for b in range(B):
            ni = rng.integers(0, len(names), len(names))
            if len(set(ni)) < 4:
                continue
            Ab = {}
            for st in sets:
                y = ys[st]; i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0)
                ii = np.r_[rng.choice(i1, i1.size), rng.choice(i0, i0.size)]
                Ab[st] = np.array([_auc_fast(y[ii], data[names[j]][st][1][ii]) for j in ni])
                boot[st].append(spearmanr(P[ni], Ab[st])[0])
                nboot[st].append(spearmanr(P[ni], A[st][ni])[0])
            for st in sets[1:]:
                dboot[st].append(boot["internal"][-1] - boot[st][-1])
        res["rho_ci95_networks_and_subjects"] = {st: pct(boot[st]) for st in sets}
        res["rho_ci95_networks_only"] = {st: pct(nboot[st]) for st in sets}
        res["rho_internal_minus_external"] = {st: {"point": res["rho"]["internal"]["rho"] - res["rho"][st]["rho"],
                                                   "ci95": pct(dboot[st]),
                                                   "p_one_sided_le0": float(np.mean(np.asarray(dboot[st]) <= 0))}
                                              for st in sets[1:]}
        out[vname] = res
        print(f"[capacity:{vname}] " + " | ".join(
            f"{st} rho {res['rho'][st]['rho']:+.2f} CI [{res['rho_ci95_networks_and_subjects'][st][0]:+.2f},{res['rho_ci95_networks_and_subjects'][st][1]:+.2f}]"
            for st in sets), flush=True)
        print("   int-ext: " + " | ".join(f"{st} {v['point']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] P(<=0)={v['p_one_sided_le0']:.3f}"
                                         for st, v in res["rho_internal_minus_external"].items()), flush=True)
        print("   without graph: " + " | ".join(f"{st} {v['rho']:+.2f} (P={v['p']:.2f})" for st, v in res["rho_without_graph_family"].items()), flush=True)
    merge_write("capacity", out)


def sec_variance(B=1000):
    out = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        S = summary_models(task)["evaluation_sets"]
        models = [m for m in S["internal"] if all(m in S[k] for k in ("external", "padic", "brainlat"))]
        data = {}
        for m in models:
            E = ens(task, m)
            if E is None or not all(p in E for p in SITES):
                continue
            data[m] = {site: (E[p][1], E[p][2]) for p, site in SITES.items()}
        models = list(data); sites = list(SITES.values())

        def decompose(M):
            gm = M.mean(); cm = M.mean(0); mm = M.mean(1)
            ss_c = M.shape[0] * np.sum((cm - gm) ** 2); ss_m = M.shape[1] * np.sum((mm - gm) ** 2)
            ss_t = np.sum((M - gm) ** 2); ss_r = ss_t - ss_c - ss_m
            return {"share_cohort": ss_c / ss_t, "share_model": ss_m / ss_t, "share_residual": ss_r / ss_t,
                    "cohort_means": dict(zip(sites, cm.tolist())), "sd_between_models_within_cohort": dict(zip(sites, M.std(0, ddof=1).tolist()))}
        M = np.array([[auc(*data[m][st]) for st in sites] for m in models])
        res = {"n_models": len(models), "models": models, **decompose(M)}
        rng = np.random.default_rng(2572); bs = []
        for _ in range(B):
            Mb = np.zeros_like(M)
            for j, st in enumerate(sites):
                y = data[models[0]][st][0]; i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0)
                ii = np.r_[rng.choice(i1, i1.size), rng.choice(i0, i0.size)]
                Mb[:, j] = [auc(y[ii], data[m][st][1][ii]) for m in models]
            bs.append(decompose(Mb))
        for k in ("share_cohort", "share_model", "share_residual"):
            res[k + "_ci95"] = pct([b[k] for b in bs])
        out[task] = res
        print(f"[variance {task}] {len(models)} models: cohort {res['share_cohort']:.2f} {res['share_cohort_ci95']} | "
              f"model {res['share_model']:.2f} {res['share_model_ci95']} | residual {res['share_residual']:.2f}", flush=True)
    merge_write("variance", out)


def sec_precision(B=4000):
    out = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        R = ens(task, REF)
        for model in ("spectral75k", "spectralite16"):
            E = ens(task, model)
            for p, site in SITES.items():
                subs, y, s = E[p]; rs, ry, rsc = R[p]
                o = {v: i for i, v in enumerate(rs)}; r = np.array([rsc[o[v]] for v in subs])
                rng = np.random.default_rng(2573); i1 = np.flatnonzero(y == 1); i0 = np.flatnonzero(y == 0); d = []
                for _ in range(B):
                    ii = np.r_[rng.choice(i1, i1.size), rng.choice(i0, i0.size)]
                    d.append(auc(y[ii], s[ii]) - auc(y[ii], r[ii]))
                se = float(np.std(d, ddof=1)); dl = delong(y, s)
                res = {"n": int(y.size), "n_pos": int(y.sum()), "delta_vs_forest": auc(y, s) - auc(y, r), "paired_se": se,
                       "mdd_80pct_power_alpha05": 2.80 * se,
                       "tost_power_if_true_delta_0": {str(m): float(max(0.0, 2 * norm.cdf(m / se - 1.645) - 1)) for m in (0.05, 0.075, 0.10)},
                       "n_needed_for_mdd_0.05": int(np.ceil(y.size * (2.80 * se / 0.05) ** 2)),
                       "auc": dl["auc"], "auc_ci95": dl["ci95"], "auc_ci_halfwidth": (dl["ci95"][1] - dl["ci95"][0]) / 2}
                out.setdefault(task, {}).setdefault(site, {})[model] = res
                print(f"{task:15s} {site:9s} {model:14s} n={res['n']} SE(dAUC) {se:.3f} MDD80 {res['mdd_80pct_power_alpha05']:.3f} "
                      f"TOST power@0.05 {res['tost_power_if_true_delta_0']['0.05']:.2f} @0.10 {res['tost_power_if_true_delta_0']['0.1']:.2f} "
                      f"n for MDD 0.05: {res['n_needed_for_mdd_0.05']}", flush=True)
    merge_write("precision", out)


def sec_endpoint(B=4000):
    dm = diag_maps(); out = {}
    S = summary_models("CN_vs_dementia")["evaluation_sets"]
    models = sorted(set(S["external"]) | {"spectral75k", "spectralite16"})
    for m in models:
        E = ens("CN_vs_dementia", m)
        if E is None:
            continue
        for p in ("ext", "brainlat"):
            if p not in E:
                continue
            subs, y, s = E[p]; dx = np.array([dm[p].get(v, "?") for v in subs])
            res = {}
            for lab in ("AD", "FTD"):
                keep = (dx == "CN") | (dx == lab)
                res[f"CN_vs_{lab}"] = delong((dx[keep] == lab).astype(int), s[keep])
            if m in SL or m == REF:
                rng = np.random.default_rng(2574); c = s[dx == "CN"]; a = s[dx == "AD"]; f = s[dx == "FTD"]; d = []
                for _ in range(B):
                    cb = rng.choice(c, c.size); ab = rng.choice(a, a.size); fb = rng.choice(f, f.size)
                    d.append(auc(np.r_[np.zeros(c.size), np.ones(a.size)], np.r_[cb, ab]) - auc(np.r_[np.zeros(c.size), np.ones(f.size)], np.r_[cb, fb]))
                res["AD_minus_FTD"] = {"point": res["CN_vs_AD"]["auc"] - res["CN_vs_FTD"]["auc"], "ci95": pct(d)}
            out.setdefault(SITES[p], {})[m] = res
    for site, ms in out.items():
        for m in ("spectral75k", "spectralite16", REF):
            if m in ms:
                r = ms[m]
                print(f"[endpoint] {site:9s} {m:14s} AD {r['CN_vs_AD']['auc']:.3f} {np.round(r['CN_vs_AD']['ci95'], 3)} (n={r['CN_vs_AD']['n']}) | "
                      f"FTD {r['CN_vs_FTD']['auc']:.3f} {np.round(r['CN_vs_FTD']['ci95'], 3)} (n={r['CN_vs_FTD']['n']}) | AD-FTD {r['AD_minus_FTD']['point']:+.3f} {np.round(r['AD_minus_FTD']['ci95'], 3)}", flush=True)
        a = [v["CN_vs_AD"]["auc"] for v in ms.values()]; f = [v["CN_vs_FTD"]["auc"] for v in ms.values()]
        print(f"[endpoint] {site}: {len(ms)} models, median CN-vs-AD {np.median(a):.3f}, median CN-vs-FTD {np.median(f):.3f}, "
              f"AD>FTD in {int(np.sum(np.array(a) > np.array(f)))}/{len(a)}", flush=True)
    merge_write("endpoint", out)


def sec_cohortid():
    m229 = _load("s229_for257", "229_site_shift.py"); m248 = _load("a248_for257", "248_age_confound_all_sites.py")
    get = m248.age_maps()
    rows = (m229.subject_features("channel_clean") + m229.subject_features("padic_channel_clean")
            + m229.subject_features("brainlat_channel_clean"))
    seen, keep = set(), []
    for r in sorted(rows, key=lambda r: r[1] != "P-ADIC"):
        if r[0] not in seen:
            seen.add(r[0]); keep.append(r)
    ctrl = [r for r in keep if r[2] in m229.CONTROL_LABELS]
    site_age = {"CAUEEG": "CAUEEG", "ds004504_raw": "ds004504", "P-ADIC": "P-ADIC", "BrainLat": "BrainLat"}
    by = {}
    for r in ctrl:
        if r[1] in site_age:
            a = get(r[0], site_age[r[1]])
            if np.isfinite(a):
                by.setdefault(r[1], []).append((r, float(a)))
    base = by["CAUEEG"]; out = {}
    for site in ("ds004504_raw", "P-ADIC", "BrainLat"):
        other = by.get(site, [])
        ya = np.r_[np.zeros(len(base)), np.ones(len(other))]
        age_auc_unmatched = auc(ya, np.r_[[a for _, a in base], [a for _, a in other]])
        used, pairs = set(), []
        for r, a in sorted(other, key=lambda t: t[1]):
            cand = [(abs(b - a), i) for i, (_, b) in enumerate(base) if i not in used and abs(b - a) <= 3.0]
            if cand:
                _, i = min(cand); used.add(i); pairs.append((base[i], (r, a)))
        B_ = [p[0] for p in pairs]; O_ = [p[1] for p in pairs]
        X = np.array([t[0][4] for t in B_ + O_]); y = np.r_[np.zeros(len(B_)), np.ones(len(O_))].astype(int)
        a_log, ci = m229.cv_auc(X, y, "logistic", ci=True)
        Xu = np.array([t[0][4] for t in base + other]); yu = np.r_[np.zeros(len(base)), np.ones(len(other))].astype(int)
        a_u, ci_u = m229.cv_auc(Xu, yu, "logistic", ci=True)
        out[site] = {"n_external_controls_with_age": len(other), "age_alone_auc_unmatched": age_auc_unmatched,
                     "decodability_unmatched_with_age": {"auc": a_u, "ci95": ci_u},
                     "n_matched_pairs": len(pairs),
                     "age_alone_auc_matched": auc(y, np.r_[[a for _, a in B_], [a for _, a in O_]]),
                     "decodability_age_matched": {"auc": a_log, "ci95": ci},
                     "median_age_external_controls": float(np.median([a for _, a in other])),
                     "median_age_matched": float(np.median([a for _, a in O_]))}
        print(f"[cohortid] {site:13s} age AUC {age_auc_unmatched:.2f} | unmatched {a_u:.3f} | {len(pairs)} age-matched pairs: "
              f"age AUC {out[site]['age_alone_auc_matched']:.2f}, decodability {a_log:.3f} {np.round(ci, 3)}", flush=True)
    merge_write("cohortid", {"note": "logistic on the QEEG-123 subject vector, 5-fold subject CV (as 229); 1:1 greedy "
                                     "nearest-age matching of CAUEEG controls to each external control within 3 years",
                             "results": out})


def _q(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return None if not v.size else {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))], "n": int(v.size)}


def sec_demographics():
    import pandas as pd
    m248 = _load("a248_for257b", "248_age_confound_all_sites.py"); get = m248.age_maps()
    meta = np.load(ROOT / "outputs" / "harmonised_features" / "channel_clean" / "meta.npz", allow_pickle=False)
    g = meta["groups"].astype(str); lab = meta["labels"].astype(str); coh = meta["cohort"].astype(str)
    _, first = np.unique(g, return_index=True)
    code = {"A": "AD", "F": "FTD", "C": "CN", "CONTROL": "CN"}
    retained = {}
    for t in ("CN_vs_dementia", "CN_vs_MCI", "CN_vs_impaired"):
        E = ens(t, "spectralite16")
        for v, yy in zip(E["train_subjects"], E["train_y"]):
            retained.setdefault("CAUEEG", set()).add(v)
        for p in ("ext",):
            if p in E:
                for v in E[p][0]:
                    retained.setdefault("ds004504" if t != "CN_vs_MCI" else "GENEEG", set()).add(v)
    lab_of = {g[i]: code.get(lab[i].upper(), lab[i].upper()) for i in first}
    out = {}
    # CAUEEG: age only (no sex or cognitive score is released per subject)
    for cohort in ("CAUEEG",):
        for cls in ("CN", "MCI", "AD", "FTD"):
            ss = [v for v in retained[cohort] if lab_of.get(v) == cls]
            out.setdefault(cohort, {})[cls] = {"n": len(ss), "age": _q([get(v, "CAUEEG") for v in ss]), "female_pct": None}
    # ds004504
    pt = pd.read_csv(ALZ / "datasets" / "openneuro_ds004504_raw" / "participants.tsv", sep="\t").set_index("participant_id")
    for cls, code_ in (("CN", "C"), ("AD", "A"), ("FTD", "F")):
        ss = [v for v in retained["ds004504"] if v in pt.index and pt.loc[v, "Group"] == code_]
        out.setdefault("ds004504", {})[cls] = {"n": len(ss), "age": _q([float(pt.loc[v, "Age"]) for v in ss]),
                                               "female_pct": 100 * np.mean([pt.loc[v, "Gender"] == "F" for v in ss]),
                                               "mmse": _q([float(pt.loc[v, "MMSE"]) for v in ss])}
    # P-ADIC
    pm = json.loads((CAN / "padic_subject_meta.json").read_text()); md = padic_mmse_dates()
    Ep = ens("CN_vs_impaired", "spectralite16", prefixes=("padic_all",))
    pad_ret = set(Ep["padic_all"][0]) | set(ens("CN_vs_MCI", "spectralite16", prefixes=("padic",))["padic"][0])
    for cls in ("CN", "MCI", "AD"):
        ss = [v for v in pad_ret if pm.get(v, {}).get("group") == cls]
        sx = [pm[v].get("sex") for v in ss]
        e = {"n": len(ss), "age": _q([pm[v]["age"] for v in ss]),
             "female_pct": 100 * np.mean([x == "F" for x in sx if x in ("F", "M")]) if any(x in ("F", "M") for x in sx) else None,
             "sex_missing": int(sum(x not in ("F", "M") for x in sx)),
             "n_age_ge_55": int(sum(pm[v]["age"] >= 55 for v in ss))}
        if cls in md:
            e["mmse_all_released"] = _q(md[cls]["mmse"])
            yrs = sorted({d[-2:] for d in md[cls]["dates"] if d})
            e["recording_years"] = [f"20{y}" for y in yrs]
        e["sampling_rate_counts"] = {f"{int(k)} Hz": int(sum(pm[v]["sfreq"] == k for v in ss)) for k in sorted({pm[v]["sfreq"] for v in ss})}
        out.setdefault("P-ADIC", {})[cls] = e
    # BrainLat
    bl = brainlat_table(); bm = json.loads((CAN / "brainlat_subject_meta.json").read_text())
    for cls in ("CN", "AD", "FTD"):
        ss = [v for v in bm if bm[v]["group"] == cls]
        rows = [bl.get(v, {}) for v in ss]
        out.setdefault("BrainLat", {})[cls] = {
            "n": len(ss), "age": _q([r.get("age") for r in rows]),
            "female_pct": 100 * np.mean([r["sex_female"] for r in rows if r.get("sex_female") is not None]),
            "education_years": _q([r.get("education") for r in rows]), "moca": _q([r.get("moca") for r in rows]),
            "country_counts": {c: int(sum(r.get("country") == c for r in rows)) for c in ("AR", "CL", None) if True}}
    # GENEEG
    for cls, lab_ in (("CN", "CN"), ("MCI", "MCI")):
        ss = [v for v in retained["GENEEG"] if lab_of.get(v) == cls]
        out.setdefault("GENEEG", {})[cls] = {"n": len(ss), "note": "no per-subject demographics released"}
    for c, d in out.items():
        print(f"[demographics] {c}: " + "; ".join(f"{k} n={v['n']} age {v.get('age') and round(v['age']['median'])}"
                                                   f" F%={v.get('female_pct') and round(v['female_pct'])}" for k, v in d.items()), flush=True)
    merge_write("demographics", out)


def _loglik(X, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e8, max_iter=5000).fit(X, y); p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sec_covariates():
    out = {}
    bl = brainlat_table()
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for model in ("spectral75k", "spectralite16"):
            E = ens(task, model); subs, y, s = E["brainlat"]
            ed = np.array([bl.get(v, {}).get("education") for v in subs], float)
            ag = np.array([bl.get(v, {}).get("age") for v in subs], float)
            ok = np.isfinite(ed) & np.isfinite(ag)
            lg = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
            Z = lambda v: (v - v[ok].mean()) / v[ok].std()   # noqa: E731
            base = np.c_[Z(ag)[ok], Z(ed)[ok]]
            ll0 = _loglik(base, y[ok]); ll1 = _loglik(np.c_[base, Z(lg)[ok]], y[ok])
            res = {"n": int(ok.sum()), "education_alone_auc": auc(y[ok], -ed[ok]),
                   "age_alone_auc": auc(y[ok], ag[ok]),
                   "lr_p_score_beyond_age_and_education": float(chi2.sf(2 * (ll1 - ll0), 1)),
                   "model_auc": auc(y[ok], s[ok])}
            # adjusted AUC: score residualised on age and education using controls
            from sklearn.linear_model import LinearRegression
            c = ok & (y == 0); lr = LinearRegression().fit(np.c_[ag[c], ed[c]], lg[c])
            res["age_education_adjusted_auc"] = auc(y[ok], lg[ok] - lr.predict(np.c_[ag[ok], ed[ok]]))
            out.setdefault("BrainLat_education", {}).setdefault(task, {})[model] = res
            print(f"[covariates] BrainLat {task} {model}: education AUC {res['education_alone_auc']:.3f} | adj AUC "
                  f"{res['age_education_adjusted_auc']:.3f} | LR P beyond age+edu {res['lr_p_score_beyond_age_and_education']:.4f}", flush=True)
    # sex-stratified AUC
    pt = __import__("pandas").read_csv(ALZ / "datasets" / "openneuro_ds004504_raw" / "participants.tsv", sep="\t").set_index("participant_id")
    pm = json.loads((CAN / "padic_subject_meta.json").read_text())
    def sex_of(p, v):
        if p == "ext":
            return pt.loc[v, "Gender"] if v in pt.index else None
        if p == "padic":
            return pm.get(v, {}).get("sex")
        f = bl.get(v, {}).get("sex_female"); return None if f is None else ("F" if f else "M")
    for model in ("spectral75k", "spectralite16", REF):
        E = ens("CN_vs_dementia", model)
        for p, site in SITES.items():
            subs, y, s = E[p]; sx = np.array([sex_of(p, v) for v in subs], object)
            res = {"sex_alone_auc_female_higher": auc(y[np.isin(sx, ["F", "M"])], (sx[np.isin(sx, ["F", "M"])] == "F").astype(float))}
            for k in ("F", "M"):
                m_ = sx == k
                res[k] = delong(y[m_], s[m_]) if 0 < y[m_].sum() < m_.sum() else None
            out.setdefault("sex", {}).setdefault(site, {})[model] = res
            if model in SL:
                print(f"[covariates] sex {site:9s} {model:14s} sex AUC {res['sex_alone_auc_female_higher']:.2f} | F "
                      f"{res['F']['auc']:.3f} (n={res['F']['n']}) | M {res['M']['auc']:.3f} (n={res['M']['n']})", flush=True)
    merge_write("covariates", out)


def sec_encoders():
    out = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for enc_ in ("labram", "eegpt", "cbramod", "lead"):
            for scaling in ("physical", "zscore"):
                E = ens(task, f"{enc_}_probe_{scaling}")
                if E is None:
                    continue
                r = {"internal": auc(E["train_y"], E["oof"])}
                for p, site in SITES.items():
                    if p in E:
                        r[site] = auc(E[p][1], E[p][2])
                out.setdefault(task, {}).setdefault(enc_, {})[scaling] = r
    for enc_, d in out.get("CN_vs_dementia", {}).items():
        print(f"[encoders] {enc_}: " + " | ".join(f"{k}: " + ", ".join(f"{s}={v:.3f}" for s, v in r.items()) for k, r in d.items()), flush=True)
    merge_write("encoders", {"note": "input scaling for each frozen probe was chosen on internal out-of-fold AUC",
                             "results": out})


def sec_seeds():
    out = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for model in ("spectral75k", "spectralite16"):
            r = {}
            for f in shards(task, model):
                z = np.load(f, allow_pickle=True); sd = int(f.stem.split("__seed")[1])
                r[sd] = {"internal": auc(z["train_y"], z["oof_score"])}
                for p, site in SITES.items():
                    r[sd][site] = auc(z[f"{p}_y"], z[f"{p}_score"])
            E = ens(task, model); r["ensemble"] = {"internal": auc(E["train_y"], E["oof"]), **{site: auc(E[p][1], E[p][2]) for p, site in SITES.items()}}
            out.setdefault(task, {})[model] = r
            print(f"[seeds] {task} {model}: " + " | ".join(f"{k}: " + ",".join(f"{v:.3f}" for v in d.values()) for k, d in r.items()), flush=True)
    merge_write("seeds", out)


def _delong_paired(y, a, b):
    """Paired DeLong test of AUC(a) - AUC(b) on the same subjects."""
    y = np.asarray(y).astype(int)

    def comps(s):
        p, n = s[y == 1], s[y == 0]
        v10 = np.array([((x > n) + 0.5 * (x == n)).mean() for x in p])
        v01 = np.array([((p > x) + 0.5 * (p == x)).mean() for x in n])
        return v10, v01
    a10, a01 = comps(np.asarray(a, float)); b10, b01 = comps(np.asarray(b, float))
    d = auc(y, a) - auc(y, b)
    var = (np.var(a10 - b10, ddof=1) / len(a10)) + (np.var(a01 - b01, ddof=1) / len(a01))
    se = float(np.sqrt(var)); p = float(2 * norm.sf(abs(d) / se)) if se > 0 else 1.0
    return d, se, p


def sec_headtohead():
    """SpectraLite-16 minus SpectraLite-12 on every task and evaluation set, plus outcomes that were not the
    selection criterion. Selection used external dementia and AD AUC at ds004504, P-ADIC and BrainLat."""
    out = {"note": "delta = SpectraLite-16 minus SpectraLite-12 (seed ensembles); paired DeLong; pooled = fixed-effect "
                   "inverse-variance over the external cohorts. The 16-feature set was chosen on external dementia/AD "
                   "AUC at the same three cohorts, so external deltas are selection-informed; internal deltas are not.",
           "auc": {}, "pooled_external": {}, "loco_decision": {}}
    sets_for = {"CN_vs_dementia": ("internal", "ext", "padic", "brainlat"), "CN_vs_AD": ("internal", "ext", "padic", "brainlat"),
                "CN_vs_impaired": ("internal", "ext", "padic", "brainlat"), "CN_vs_MCI": ("internal", "ext", "padic")}
    for task, sets in sets_for.items():
        A, Bm = ens(task, "spectralite16"), ens(task, "spectral75k")
        rows = {}
        for st in sets:
            if st == "internal":
                o = {v: i for i, v in enumerate(Bm["train_subjects"])}
                y = A["train_y"]; a = A["oof"]; b = np.array([Bm["oof"][o[v]] for v in A["train_subjects"]])
                name = "CAUEEG (internal)"
            else:
                if st not in A or st not in Bm:
                    continue
                sa, y, a = A[st]; sb, _, bb = Bm[st]; o = {v: i for i, v in enumerate(sb)}; b = np.array([bb[o[v]] for v in sa])
                name = "GENEEG" if (task == "CN_vs_MCI" and st == "ext") else SITES[st]
            d, se, p = _delong_paired(y, a, b)
            rows[name] = {"auc16": auc(y, a), "auc12": auc(y, b), "delta": d, "se": se,
                          "ci95": [d - 1.96 * se, d + 1.96 * se], "p": p, "n": int(len(y))}
        if task in ("CN_vs_MCI", "CN_vs_impaired"):
            def _misp(model):
                fs = sorted((CAN / "misp" / "scores" / task).glob(f"{model}__seed*.npz"))
                if not fs:
                    return None
                zs = [np.load(f, allow_pickle=True) for f in fs]; s0 = np.asarray(zs[0]["subjects"]).astype(str)
                sc = []
                for z in zs:
                    o = {v: i for i, v in enumerate(np.asarray(z["subjects"]).astype(str))}
                    sc.append(np.asarray(z["score"], float)[[o[v] for v in s0]])
                return s0, np.asarray(zs[0]["y"]).astype(int), np.mean(sc, 0)
            m16, m12 = _misp("spectralite16"), _misp("spectral75k")
            if m16 is not None and m12 is not None:
                o = {v: i for i, v in enumerate(m12[0])}
                y, a = m16[1], m16[2]; b = np.array([m12[2][o[v]] for v in m16[0]])
                d, se, p_ = _delong_paired(y, a, b)
                rows["MISP"] = {"auc16": auc(y, a), "auc12": auc(y, b), "delta": d, "se": se,
                                "ci95": [d - 1.96 * se, d + 1.96 * se], "p": p_, "n": int(len(y))}
        out["auc"][task] = rows
        ext = {k: v for k, v in rows.items() if k != "CAUEEG (internal)"}
        if len(ext) >= 2:
            w = np.array([1 / v["se"] ** 2 for v in ext.values()]); d = np.array([v["delta"] for v in ext.values()])
            pd_ = float((w * d).sum() / w.sum()); pse = float(np.sqrt(1 / w.sum()))
            q = float(((d - pd_) ** 2 * w).sum())
            out["pooled_external"][task] = {"delta": pd_, "ci95": [pd_ - 1.96 * pse, pd_ + 1.96 * pse],
                                            "p": float(2 * norm.sf(abs(pd_) / pse)), "cochran_q": q,
                                            "q_p": float(chi2.sf(q, len(d) - 1)), "cohorts": list(ext)}
        if task in ("CN_vs_dementia", "CN_vs_AD"):
            dec = {}
            for held in ("ds004504", "P-ADIC", "BrainLat"):
                others = [c for c in ("ds004504", "P-ADIC", "BrainLat") if c != held]
                m16 = np.mean([rows[c]["auc16"] for c in others]); m12 = np.mean([rows[c]["auc12"] for c in others])
                dec[held] = {"chosen_on_other_two": "16" if m16 > m12 else "12",
                             "held_out_delta_16_minus_12": rows[held]["delta"]}
            out["loco_decision"][task] = dec
        for k, v in rows.items():
            print(f"[h2h] {task:15s} {k:18s} 16: {v['auc16']:.3f}  12: {v['auc12']:.3f}  delta {v['delta']:+.3f} "
                  f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] P={v['p']:.3f}", flush=True)
        if task in out["pooled_external"]:
            v = out["pooled_external"][task]
            print(f"[h2h] {task:15s} pooled external delta {v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
                  f"P={v['p']:.3f}  heterogeneity Q P={v['q_p']:.2f}", flush=True)
    # outcomes that were not the selection criterion
    cm = json.loads((CAN / "clinical_metrics.json").read_text())["tasks"]
    sec = {}
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for site in ("ds004504", "P-ADIC", "BrainLat"):
            r = {}
            for m in ("spectralite16", "spectral75k"):
                e = cm[task][site][m]; op = e["operating"]
                r[m] = {"brier": e["brier"], "ece": e["ece"], "cal_slope": e["cal_slope"],
                        "transferred_specificity": op["transferred"]["specificity"]["mean"],
                        "transferred_sensitivity": op["transferred"]["sensitivity"]["mean"],
                        "conformal_sensitivity": op["conformal"]["sensitivity"]["mean"]}
            sec.setdefault(task, {})[site] = r
    try:
        aa = json.loads((CAN / "age_adjusted_net_benefit.json").read_text())
        sec["age_adjusted_source"] = "age_adjusted_net_benefit.json"
        sec["age_adjusted"] = {k: v for k, v in aa.items() if k not in ("note",)} if isinstance(aa, dict) else None
    except FileNotFoundError:
        pass
    out["secondary"] = sec
    for task in ("CN_vs_dementia", "CN_vs_AD"):
        for site, r in sec[task].items():
            a, b = r["spectralite16"], r["spectral75k"]
            print(f"[h2h-secondary] {task:15s} {site:9s} transferred spec 16 {a['transferred_specificity']:.2f} vs 12 "
                  f"{b['transferred_specificity']:.2f} | conformal sens {a['conformal_sensitivity']:.2f} vs {b['conformal_sensitivity']:.2f} "
                  f"| Brier {a['brier']:.3f} vs {b['brier']:.3f} | ECE {a['ece']:.3f} vs {b['ece']:.3f}", flush=True)
    merge_write("headtohead", out)


def sec_caueeg_events():
    """CAUEEG releases event markers (eyes open/closed, photic stimulation); the pipeline uses the first 235 s
    regardless of condition. Label each analysed 4-s window, compare the mix between classes, and recompute
    internal out-of-fold AUC from eyes-closed (or unmarked), photic-free windows only."""
    import mne
    base = ALZ / "datasets" / "caueeg-dataset"
    meta = np.load(ROOT / "outputs" / "harmonised_features" / "channel_clean" / "meta.npz", allow_pickle=False)
    g = meta["groups"].astype(str); coh = meta["cohort"].astype(str); st = meta["start"].astype(int)
    lab = np.array([str(x).upper() for x in meta["labels"]])
    idx = np.flatnonzero(coh == "CAUEEG")
    sf_cache, state = {}, {}
    for sid in np.unique(g[idx]):
        serial = sid.split("_")[-1]
        ev = json.loads((base / "event" / f"{serial}.json").read_text())
        if serial not in sf_cache:
            r = mne.io.read_raw_edf(str(base / "signal" / "edf" / f"{serial}.edf"), preload=False, verbose="ERROR")
            sf_cache[serial] = float(r.info["sfreq"])
        sf = sf_cache[serial]; eye = []; photic = []
        cur_eye, cur_ph = None, False
        for t, name in ev:
            n = str(name).lower(); tt = t / sf
            if n.startswith("eyes open"):
                cur_eye = "open"; eye.append((tt, "open"))
            elif n.startswith("eyes closed"):
                cur_eye = "closed"; eye.append((tt, "closed"))
            elif n.startswith("photic on"):
                photic.append((tt, True))
            elif n.startswith("photic off"):
                photic.append((tt, False))
        state[sid] = (eye, photic)

    def label(sid, t0, t1):
        eye, photic = state[sid]
        def at(seq, t, default):
            v = default
            for tt, s in seq:
                if tt <= t:
                    v = s
                else:
                    break
            return v
        # a window counts as eyes-open / photic if any part of it is
        pts = np.linspace(t0, t1, 9)
        e = {at(eye, p, "unmarked") for p in pts}; ph = any(at(photic, p, False) for p in pts)
        return ("open" if "open" in e else ("closed" if "closed" in e else "unmarked")), ph
    win = {}
    for i in idx:
        t0 = st[i] / 256.0; win[i] = label(g[i], t0, t0 + 4.0)
    rep = {"sampling_rates": sorted({v for v in sf_cache.values()}), "tasks": {}}
    code = {"C": "CN", "CN": "CN", "CONTROL": "CN", "A": "AD", "AD": "AD", "F": "FTD", "FTD": "FTD", "MCI": "MCI"}
    for task, pos in (("CN_vs_dementia", ("AD", "FTD")), ("CN_vs_MCI", ("MCI",))):
        subs = {}
        for i in idx:
            c = code.get(lab[i], lab[i])
            if c in pos + ("CN",):
                subs.setdefault(g[i], []).append(i)
        S = sorted(subs); y = np.array([int(code.get(lab[subs[s][0]], "") in pos) for s in S])
        f_open = np.array([np.mean([win[i][0] == "open" for i in subs[s]]) for s in S])
        f_ph = np.array([np.mean([win[i][1] for i in subs[s]]) for s in S])
        f_un = np.array([np.mean([win[i][0] == "unmarked" for i in subs[s]]) for s in S])
        r = {"n": len(S), "eyes_open_fraction_mean": {"CN": float(f_open[y == 0].mean()), "case": float(f_open[y == 1].mean())},
             "photic_fraction_mean": {"CN": float(f_ph[y == 0].mean()), "case": float(f_ph[y == 1].mean())},
             "unmarked_fraction_mean": {"CN": float(f_un[y == 0].mean()), "case": float(f_un[y == 1].mean())},
             "auc_eyes_open_fraction": auc(y, f_open), "auc_photic_fraction": auc(y, f_ph), "auc_unmarked_fraction": auc(y, f_un)}
        # internal OOF AUC from clean windows only (window-level OOF probabilities in the shards)
        for model in ("spectralite16", "spectral75k"):
            vals = []
            for f in shards(task, model):
                z = np.load(f, allow_pickle=True)
                pos_w, neg_w, site = ({"CN_vs_dementia": (("AD", "FTD"), ("CN",), None), "CN_vs_MCI": (("MCI",), ("CN",), None)})[task]
                keep = np.isin(np.array([code.get(l, l) for l in lab]), pos_w + neg_w) & (coh == "CAUEEG")
                idx_tr = np.flatnonzero(keep)
                assert len(idx_tr) == len(z["oof_win"]), (len(idx_tr), len(z["oof_win"]))
                ow = z["oof_win"].astype(float); gw = z["win_train_groups"].astype(str)
                assert np.array_equal(gw, g[idx_tr])
                clean = np.array([(win[i][0] != "open") and (not win[i][1]) for i in idx_tr])
                ts = z["train_subjects"].astype(str); ty = z["train_y"].astype(int)
                sc_all = np.array([ow[gw == s].mean() for s in ts])
                sc_clean = np.array([ow[(gw == s) & clean].mean() if np.any((gw == s) & clean) else np.nan for s in ts])
                ok = np.isfinite(sc_clean)
                vals.append((auc(ty, sc_all), auc(ty[ok], sc_clean[ok]), int(ok.sum()), auc(ty[ok], sc_all[ok])))
            r[model] = {"auc_all_windows_seed_mean": float(np.mean([v[0] for v in vals])),
                        "auc_clean_windows_seed_mean": float(np.mean([v[1] for v in vals])),
                        "auc_all_windows_same_subjects": float(np.mean([v[3] for v in vals])),
                        "n_subjects_with_clean_windows": vals[0][2]}
        rep["tasks"][task] = r
        print(f"[caueeg_events] {task}: eyes-open window fraction CN {r['eyes_open_fraction_mean']['CN']:.2f} vs case "
              f"{r['eyes_open_fraction_mean']['case']:.2f} (AUC {r['auc_eyes_open_fraction']:.2f}); photic CN "
              f"{r['photic_fraction_mean']['CN']:.2f} vs {r['photic_fraction_mean']['case']:.2f} (AUC {r['auc_photic_fraction']:.2f}); "
              f"unmarked AUC {r['auc_unmarked_fraction']:.2f}", flush=True)
        for model in ("spectralite16", "spectral75k"):
            m = r[model]
            print(f"   {model}: OOF AUC all windows {m['auc_all_windows_same_subjects']:.3f} vs eyes-closed/photic-free only "
                  f"{m['auc_clean_windows_seed_mean']:.3f} (n={m['n_subjects_with_clean_windows']})", flush=True)
    merge_write("caueeg_events", rep)


def _wauc(y, s, w):
    """AUC with case weights w (controls unweighted): weighted share of case-control pairs ranked correctly."""
    pos, neg = s[y == 1], s[y == 0]; wp = w[y == 1]
    gt = (pos[:, None] > neg[None, :]).sum(1) + 0.5 * (pos[:, None] == neg[None, :]).sum(1)
    return float((wp * gt).sum() / (wp.sum() * len(neg)))


def sec_severity(B=2000):
    """Case mix: does the score track cognitive severity, how well are mild versus moderate cases detected, and
    what would P-ADIC's AUC be with ds004504's MMSE distribution? MMSE: ds004504 participants.tsv, P-ADIC field m
    (column-indexed, as are the subject ids); BrainLat releases MoCA only. Mild = MMSE >= 21 (NICE); BrainLat is
    split at its own case median MoCA, so its strata are cohort-relative."""
    import pandas as pd
    from scipy.stats import spearmanr
    pt = pd.read_csv(ALZ / "datasets" / "openneuro_ds004504_raw" / "participants.tsv", sep="\t").set_index("participant_id")
    md = padic_mmse_dates()["AD"]["mmse"]
    bl = brainlat_table()
    ad_only = {"on": False}
    def cog(prefix, sid):
        if prefix == "ext":
            if ad_only["on"] and (sid not in pt.index or pt.loc[sid, "Group"] != "A"):
                return np.nan
            return float(pt.loc[sid, "MMSE"]) if sid in pt.index else np.nan
        if ad_only["on"] and prefix == "brainlat" and "_AD_" not in sid:
            return np.nan
        if prefix == "padic":
            c = int(sid.rsplit("_", 1)[1]); return md[c] if "_AD_" in sid and c < len(md) else np.nan
        return (bl.get(sid) or {}).get("moca") or np.nan
    s = summary_models("CN_vs_dementia")["evaluation_sets"]
    models = sorted(set(s["external"]) & set(s["padic"]) & set(s["brainlat"]))
    rng = np.random.default_rng(11)
    rng_ncg = np.random.default_rng(1101)   # NCG-v3 (Fig. 3a) on its own stream; the other models' draws are unchanged
    bins = [-1, 15.5, 20.5, 24.5, 31]
    out = {"definition": "mild = MMSE >= 21 (ds004504, P-ADIC); BrainLat MoCA split at the case median", "cohorts": {}, "cohorts_ad_only": {}}
    for (prefix, name), only in [(pn, o) for o in (False, True) for pn in (("ext", "ds004504"), ("padic", "P-ADIC"), ("brainlat", "BrainLat"))]:
        ad_only["on"] = only
        res = {}
        for m in models:
            E = ens("CN_vs_dementia", m, prefixes=(prefix,))
            if not E or prefix not in E:
                continue
            sid, y, sc = E[prefix]
            c = np.array([cog(prefix, v) if yy == 1 else np.nan for v, yy in zip(sid, y)], float)
            have = (y == 1) & np.isfinite(c)
            if have.sum() < 10:
                continue
            cut = 20.5 if prefix != "brainlat" else float(np.median(c[have]))
            mild = have & (c > cut); mod = have & (c <= cut)
            r = spearmanr(sc[have], c[have])
            e = {"n_cases_with_score": int(have.sum()), "rho_score_cognition": float(r.statistic), "p": float(r.pvalue),
                 "n_mild": int(mild.sum()), "n_moderate": int(mod.sum()), "cut": cut,
                 "auc_mild_vs_cn": auc(np.r_[np.ones(mild.sum()), np.zeros((y == 0).sum())], np.r_[sc[mild], sc[y == 0]]) if mild.sum() >= 3 else None,
                 "auc_moderate_vs_cn": auc(np.r_[np.ones(mod.sum()), np.zeros((y == 0).sum())], np.r_[sc[mod], sc[y == 0]]) if mod.sum() >= 3 else None}
            if m in ("spectral75k", "rf123", "ncg_v3") and mild.sum() >= 3 and mod.sum() >= 3:
                rg = rng_ncg if m == "ncg_v3" else rng
                bd, br, bm, bo = [], [], [], []
                ii_m, ii_d, ii_c = np.flatnonzero(mild), np.flatnonzero(mod), np.flatnonzero(y == 0)
                for _ in range(B):
                    a, b_, cc = rg.choice(ii_m, len(ii_m)), rg.choice(ii_d, len(ii_d)), rg.choice(ii_c, len(ii_c))
                    am = auc(np.r_[np.ones(len(a)), np.zeros(len(cc))], np.r_[sc[a], sc[cc]])
                    ad = auc(np.r_[np.ones(len(b_)), np.zeros(len(cc))], np.r_[sc[b_], sc[cc]])
                    bd.append(ad - am); bm.append(am); bo.append(ad)
                    hh = np.r_[a, b_]; br.append(spearmanr(sc[hh], c[hh]).statistic)
                e["moderate_minus_mild_ci95"] = pct(bd); e["rho_ci95"] = pct(br)
                e["auc_mild_ci95"] = pct(bm); e["auc_moderate_ci95"] = pct(bo)
            res[m] = e
        summ = {k: {"median": float(np.median([v[k] for v in res.values() if v.get(k) is not None])),
                    "min": float(np.min([v[k] for v in res.values() if v.get(k) is not None])),
                    "max": float(np.max([v[k] for v in res.values() if v.get(k) is not None]))}
                for k in ("rho_score_cognition", "auc_mild_vs_cn", "auc_moderate_vs_cn")}
        out["cohorts_ad_only" if only else "cohorts"][name] = {"per_model": res, "across_models": summ, "n_models": len(res)}
        sl = res.get("spectral75k", {})
        print(f"[severity{' AD only' if only else ''}] {name}: {len(res)} models | median rho {summ['rho_score_cognition']['median']:+.2f} | AUC mild "
              f"{summ['auc_mild_vs_cn']['median']:.3f} moderate {summ['auc_moderate_vs_cn']['median']:.3f} | SpectraLite rho "
              f"{sl.get('rho_score_cognition', float('nan')):+.2f}, mild {sl.get('auc_mild_vs_cn')} (n={sl.get('n_mild')}), "
              f"moderate {sl.get('auc_moderate_vs_cn')} (n={sl.get('n_moderate')})", flush=True)
    # Does the any-impairment model (trained with CAUEEG's MCI cases as positives) detect mild AD better than the
    # dementia model? Same AD strata and controls; prespecified as a question about training spectrum, not a selection.
    ad_only["on"] = True
    cmp = {}
    for prefix, name in (("ext", "ds004504"), ("padic", "P-ADIC"), ("brainlat", "BrainLat")):
        for m in ("spectral75k", "rf123", "spectralite16", "ncg_v5", "atcnet", "eegnet"):
            row = {}
            for task in ("CN_vs_dementia", "CN_vs_impaired"):
                E = ens(task, m, prefixes=(prefix,))
                if not E or prefix not in E:
                    continue
                sid, y, sc = E[prefix]
                c = np.array([cog(prefix, v) if yy == 1 else np.nan for v, yy in zip(sid, y)], float)
                have = (y == 1) & np.isfinite(c)
                cut = 20.5 if prefix != "brainlat" else float(np.median(c[have]))
                mild = have & (c > cut); mod = have & (c <= cut); cn = y == 0
                f = lambda msk: auc(np.r_[np.ones(msk.sum()), np.zeros(cn.sum())], np.r_[sc[msk], sc[cn]]) if msk.sum() >= 3 else None
                row[task] = {"auc_mild_ad": f(mild), "auc_moderate_ad": f(mod), "n_mild": int(mild.sum()), "n_moderate": int(mod.sum()), "n_cn": int(cn.sum())}
            if len(row) == 2:
                cmp.setdefault(name, {})[m] = row
        if name in cmp:
            print(f"[severity] {name} mild AD, dementia vs any-impairment model: " + "; ".join(
                f"{m} {r['CN_vs_dementia']['auc_mild_ad']:.3f}->{r['CN_vs_impaired']['auc_mild_ad']:.3f}" for m, r in cmp[name].items()
                if r['CN_vs_dementia']['auc_mild_ad'] is not None and r['CN_vs_impaired']['auc_mild_ad'] is not None), flush=True)
    out["mild_ad_by_training_task"] = cmp
    # P-ADIC cases reweighted to ds004504's MMSE distribution (AD cases), controls unchanged
    Ed = ens("CN_vs_dementia", "spectral75k", prefixes=("ext",))
    ref = np.array([cog("ext", v) for v, yy in zip(*Ed["ext"][:2]) if yy == 1], float)
    ref_p = np.histogram(ref[np.isfinite(ref)], bins)[0] / np.isfinite(ref).sum()
    std = {}
    for m in models:
        E = ens("CN_vs_dementia", m, prefixes=("padic",))
        if not E or "padic" not in E:
            continue
        sid, y, sc = E["padic"]
        c = np.array([cog("padic", v) if yy == 1 else np.nan for v, yy in zip(sid, y)], float)
        keep = (y == 0) | np.isfinite(c)
        y2, s2, c2 = y[keep], sc[keep], c[keep]
        cb = np.digitize(c2, bins) - 1
        own = np.bincount(cb[y2 == 1], minlength=len(bins) - 1) / (y2 == 1).sum()
        w = np.ones(len(y2))
        for b in range(len(bins) - 1):
            w[(y2 == 1) & (cb == b)] = ref_p[b] / own[b] if own[b] > 0 else 0.0
        std[m] = {"auc_cases_with_mmse": auc(y2, s2), "auc_standardized_to_ds004504_mmse": _wauc(y2, s2, w),
                  "n_cases": int((y2 == 1).sum())}
    v = np.array([d["auc_standardized_to_ds004504_mmse"] - d["auc_cases_with_mmse"] for d in std.values()])
    out["padic_standardized"] = {"ref_mmse_bins": bins, "ref_distribution": ref_p.tolist(), "per_model": std,
                                 "shift_median": float(np.median(v)), "shift_min": float(v.min()), "shift_max": float(v.max())}
    print(f"[severity] P-ADIC standardized to ds004504 MMSE: AUC shift median {np.median(v):+.3f} "
          f"({v.min():+.3f} to {v.max():+.3f}); SpectraLite {std['spectral75k']}", flush=True)
    merge_write("severity", out)


SECTIONS = {"caueeg_events": sec_caueeg_events, "severity": sec_severity, "headtohead": sec_headtohead, "clinical": sec_clinical, "crosssite": sec_crosssite, "procedure": sec_procedure, "sweep": sec_sweep,
            "capacity": sec_capacity, "variance": sec_variance, "precision": sec_precision, "endpoint": sec_endpoint,
            "cohortid": sec_cohortid, "demographics": sec_demographics, "covariates": sec_covariates,
            "encoders": sec_encoders, "seeds": sec_seeds}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    for name in [s for s in a.sections.split(",") if s]:
        SECTIONS[name]()
