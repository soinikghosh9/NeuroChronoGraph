"""Experiment 264: validation of SpectraLite's 12 spectral markers across cohorts (V3-style: verification is in S2;
this script covers analytical and clinical validity) and a test of why a transferred threshold fails.

Every definition and prediction was fixed in model_paper/PREDICTIONS_264_MARKERS_2026-10-23.md before this script
was written (SHA-256 logged in STATUS_REVISION_2026-10-12.md). Nothing here selects a model, feature, or threshold.
CPU only; reads the cached node features (24 clean windows x 19 electrodes x 12 features per subject) and the saved
SpectraLite seed scores. No retraining.

    python experiments/264_marker_validation.py
Out: outputs/results/canonical/marker_validation.json
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "outputs" / "harmonised_features"
CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "marker_validation.json"
B = 2000
NAMES = ["rel delta", "rel theta", "rel alpha", "rel beta", "rel gamma", "IAF", "alpha prominence", "spectral edge",
         "aperiodic exponent", "theta/alpha", "slowing", "spectral entropy"]
CH = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
REGIONS = {"frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"], "central": ["C3", "C4", "Cz"],
           "temporal": ["T3", "T4", "T5", "T6"], "parietal": ["P3", "P4", "Pz"], "occipital": ["O1", "O2"]}
REGIONS["posterior"] = REGIONS["parietal"] + REGIONS["occipital"]
CODE = {"A": "AD", "AD": "AD", "F": "FTD", "FTD": "FTD", "C": "CN", "CN": "CN", "CONTROL": "CN", "HC": "CN", "MCI": "MCI"}
DEM_COH = ("CAUEEG", "ds004504", "P-ADIC", "BrainLat")
MCI_COH = ("CAUEEG", "GENEEG", "P-ADIC", "MISP")


def _mod(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


# ------------------------------------------------------------------------------------------- data
def load_tag(tag, cohort=None):
    """Per subject: class, [19,12] median over windows, and the two half-medians (windows 1-12, 13-24)."""
    m = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    g = m["groups"].astype(str); lab = m["labels"].astype(str); coh = m["cohort"].astype(str)
    N = np.load(FEAT / tag / "node12.npy", mmap_mode="r")
    keep = np.ones(len(g), bool) if cohort is None else coh == cohort
    out = {}
    for s in np.unique(g[keep]):
        idx = np.flatnonzero(g == s)
        x = np.asarray(N[idx], float)
        h = len(idx) // 2
        out[s] = {"cls": CODE.get(lab[idx[0]].upper(), lab[idx[0]]), "med": np.nanmedian(x, 0),
                  "h1": np.nanmedian(x[:h], 0), "h2": np.nanmedian(x[h:], 0)}
    return out


def region_mean(med, region=None):
    """[..., 19, 12] -> [..., 12]: mean over the region's electrodes (whole head if None)."""
    if region is None:
        return np.nanmean(med, -2)
    idx = [CH.index(c) for c in REGIONS[region]]
    return np.nanmean(med[..., idx, :], -2)


def shard_sets(task):
    """Subject id -> label (1 case, 0 control) per evaluation set of SpectraLite's score files, and seed-mean scores."""
    fs = sorted((CAN / task).glob("spectral75k__seed*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    out = {}
    for key, subj, yk, sk in (("internal", "train_subjects", "train_y", "oof_score"),
                              ("ext", "ext_subjects", "ext_y", "ext_score"),
                              ("padic", "padic_subjects", "padic_y", "padic_score"),
                              ("brainlat", "brainlat_subjects", "brainlat_y", "brainlat_score")):
        if subj not in zs[0].files:
            continue
        ids = [str(v) for v in zs[0][subj]]
        sc = []
        for z in zs:
            o = {str(v): i for i, v in enumerate(z[subj])}
            sc.append(np.array([z[sk][o[v]] for v in ids], float))
        out[key] = {"ids": ids, "y": np.asarray(zs[0][yk]).astype(int), "score": np.mean(sc, 0)}
    return out


# ------------------------------------------------------------------------------------------- statistics
def hedges(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)), na, nb


def g_boot(A, Bm, rng):
    """Hedges g per column of A (cases) vs Bm (controls) with subject-bootstrap 95% CIs. Returns list of dicts."""
    res = []
    ia = rng.integers(0, len(A), (B, len(A))); ib = rng.integers(0, len(Bm), (B, len(Bm)))
    for j in range(A.shape[1]):
        g, na, nb = hedges(A[:, j], Bm[:, j])
        a, b = A[:, j][ia], Bm[:, j][ib]
        sp = np.sqrt(((na - 1) * a.var(1, ddof=1) + (nb - 1) * b.var(1, ddof=1)) / (na + nb - 2))
        bs = (a.mean(1) - b.mean(1)) / sp * (1 - 3 / (4 * (na + nb) - 9))
        res.append({"g": float(g), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                    "n_case": int(na), "n_control": int(nb)})
    return res


def dl_pool(gs, ns):
    """DerSimonian-Laird random-effects pooling of Hedges g; ns = [(n_case, n_control)]."""
    g = np.asarray(gs, float)
    v = np.array([(a + b) / (a * b) + gi ** 2 / (2 * (a + b)) for gi, (a, b) in zip(g, ns)])
    w = 1 / v; fe = (w * g).sum() / w.sum()
    Q = (w * (g - fe) ** 2).sum(); df = len(g) - 1
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum())) if df > 0 else 0.0
    ws = 1 / (v + tau2); pooled = (ws * g).sum() / ws.sum(); se = math.sqrt(1 / ws.sum())
    i2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return {"g": float(pooled), "ci95": [float(pooled - 1.96 * se), float(pooled + 1.96 * se)], "I2": float(i2),
            "tau2": float(tau2), "Q": float(Q), "k": int(len(g))}


def icc31(x, y):
    """ICC(3,1), two sessions, consistency."""
    X = np.c_[x, y]; X = X[np.isfinite(X).all(1)]
    n, k = X.shape
    gm = X.mean(); msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse)), int(n)


def auc(y, s):
    y = np.asarray(y).astype(int); r = rankdata(s, method="average")
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pct(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


# ------------------------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(264)
    out = {"definitions": "model_paper/PREDICTIONS_264_MARKERS_2026-10-23.md", "names": NAMES, "bootstrap": B}

    F = {"CAUEEG": load_tag("channel_clean", "CAUEEG"), "ds004504": load_tag("channel_clean", "ds004504_raw"),
         "GENEEG": load_tag("channel_clean", "GENEEG"), "P-ADIC": load_tag("padic_channel_clean"),
         "BrainLat": load_tag("brainlat_channel_clean"), "MISP": load_tag("misp_all")}
    dem = shard_sets("CN_vs_dementia"); mci = shard_sets("CN_vs_MCI")
    zm = [np.load(f, allow_pickle=True) for f in sorted((CAN / "misp" / "scores" / "CN_vs_MCI").glob("spectral75k__seed*.npz"))]
    misp_ids = [str(v) for v in zm[0]["subjects"]]; misp_y = np.asarray(zm[0]["y"]).astype(int)

    dem_sets = {"CAUEEG": dem["internal"], "ds004504": dem["ext"], "P-ADIC": dem["padic"], "BrainLat": dem["brainlat"]}
    mci_sets = {"CAUEEG": mci["internal"], "GENEEG": mci["ext"], "P-ADIC": mci["padic"],
                "MISP": {"ids": misp_ids, "y": misp_y}}

    def mat(cohort, ids, region=None, key="med"):
        return np.array([region_mean(F[cohort][s][key], region) for s in ids])

    def split(cohort, S, case_cls):
        """AD-vs-CN (or MCI-vs-CN) subject ids within a task's evaluated set, using the feature-file class."""
        ids = [s for s in S["ids"] if s in F[cohort]]
        case = [s for s in ids if F[cohort][s]["cls"] in case_cls]
        ctrl = [s for s in ids if F[cohort][s]["cls"] == "CN"]
        return case, ctrl

    # ---- clinical validity: AD vs CN, by region; MCI vs CN
    out["disease_AD"] = {}; out["disease_MCI"] = {}
    for region in (None, "frontal", "central", "temporal", "parietal", "occipital", "posterior"):
        rk = region or "whole_head"
        out["disease_AD"][rk] = {}
        for c in DEM_COH:
            case, ctrl = split(c, dem_sets[c], ("AD",))
            out["disease_AD"][rk][c] = dict(zip(NAMES, g_boot(mat(c, case, region), mat(c, ctrl, region), rng)))
    for c in MCI_COH:
        case, ctrl = split(c, mci_sets[c], ("MCI",))
        out["disease_MCI"][c] = dict(zip(NAMES, g_boot(mat(c, case), mat(c, ctrl), rng)))
    out["pooled_AD"] = {n: dl_pool([out["disease_AD"]["whole_head"][c][n]["g"] for c in DEM_COH],
                                   [(out["disease_AD"]["whole_head"][c][n]["n_case"],
                                     out["disease_AD"]["whole_head"][c][n]["n_control"]) for c in DEM_COH]) for n in NAMES}
    out["pooled_MCI"] = {n: dl_pool([out["disease_MCI"][c][n]["g"] for c in MCI_COH],
                                    [(out["disease_MCI"][c][n]["n_case"], out["disease_MCI"][c][n]["n_control"])
                                     for c in MCI_COH]) for n in NAMES}
    consistent = [n for n in NAMES if len({np.sign(out["disease_AD"]["whole_head"][c][n]["g"]) for c in DEM_COH}) == 1]
    out["sign_consistent_AD"] = consistent

    # ---- healthy-baseline shift against CAUEEG controls (all five external cohorts)
    _, ref_ids = split("CAUEEG", dem_sets["CAUEEG"], ("AD",))
    REF = mat("CAUEEG", ref_ids)
    out["control_shift"] = {}
    ctrl_sets = {"ds004504": dem_sets["ds004504"], "P-ADIC": dem_sets["P-ADIC"], "BrainLat": dem_sets["BrainLat"],
                 "GENEEG": mci_sets["GENEEG"], "MISP": mci_sets["MISP"]}
    for c, S in ctrl_sets.items():
        _, ctrl = split(c, S, ("AD",))
        out["control_shift"][c] = dict(zip(NAMES, g_boot(mat(c, ctrl), REF, rng)))
    out["shift_to_effect"] = {n: {"max_abs_control_shift": max(abs(out["control_shift"][c][n]["g"]) for c in ctrl_sets),
                                  "abs_pooled_AD_g": abs(out["pooled_AD"][n]["g"])} for n in NAMES}

    # ---- analytical validity: split-half reliability (Spearman-Brown) at every cohort; cross-session ICC
    out["split_half"] = {}
    for c in ("CAUEEG", "ds004504", "P-ADIC", "BrainLat", "GENEEG", "MISP"):
        S = dem_sets.get(c) or mci_sets[c]
        ids = [s for s in S["ids"] if s in F[c]]
        H1, H2 = mat(c, ids, key="h1"), mat(c, ids, key="h2")
        rr = {}
        for j, n in enumerate(NAMES):
            ok = np.isfinite(H1[:, j]) & np.isfinite(H2[:, j])
            r = float(np.corrcoef(H1[ok, j], H2[ok, j])[0, 1]); rr[n] = 2 * r / (1 + r)
        out["split_half"][c] = {"n": len(ids), "spearman_brown": rr}
    E = load_tag("protocol_ecd"); P = load_tag("protocol_pho")
    common = sorted(set(E) & set(P))
    XE = np.array([region_mean(E[s]["med"]) for s in common]); XP = np.array([region_mean(P[s]["med"]) for s in common])
    out["cross_session"] = {"n": len(common), "icc31": {n: icc31(XE[:, j], XP[:, j])[0] for j, n in enumerate(NAMES)}}
    # Added 2026-10-23 after the prediction file (descriptive, not a prediction): the same eyes-closed recordings
    # through our pipeline on the raw files and through the authors' preprocessing (ds004504 derivatives).
    Rw = F["ds004504"]; both = sorted(set(E) & set(Rw))
    XR = np.array([region_mean(Rw[s]["med"]) for s in both]); XD = np.array([region_mean(E[s]["med"]) for s in both])
    out["preprocessing_agreement"] = {"n": len(both), "note": "descriptive; added after the prediction file",
                                      "icc31": {n: icc31(XR[:, j], XD[:, j])[0] for j, n in enumerate(NAMES)}}
    # Descriptive: does each photic recording match its own eyes-closed recording better than other subjects'?
    Zf = lambda X: (X - X.mean(0)) / X.std(0)  # noqa: E731
    Dm = ((Zf(XE)[:, None, :] - Zf(XP)[None, :, :]) ** 2).sum(-1)
    out["cross_session"]["median_rank_of_own_recording"] = float(np.median([(Dm[:, i] < Dm[i, i]).sum() for i in range(len(common))]))
    out["cross_session"]["note"] = ("photic session is eyes open with stepped 5-30 Hz stimulation; descriptive rank check "
                                    "added after the prediction file")

    # ---- association with cognition within AD cases (MMSE at ds004504 and P-ADIC; MoCA at BrainLat)
    R257 = _mod("r257_m", "257_review_revision.py")
    import pandas as pd
    pt = pd.read_csv(R257.ALZ / "datasets" / "openneuro_ds004504_raw" / "participants.tsv", sep="\t").set_index("participant_id")
    md = R257.padic_mmse_dates()["AD"]["mmse"]; bl = R257.brainlat_table()

    def cog(c, s):
        if c == "ds004504":
            return float(pt.loc[s, "MMSE"]) if s in pt.index else np.nan
        if c == "P-ADIC":
            k = int(s.rsplit("_", 1)[1]); return float(md[k]) if k < len(md) and md[k] is not None else np.nan
        return float((bl.get(s) or {}).get("moca") or np.nan)
    out["cognition"] = {}
    for c in ("ds004504", "P-ADIC", "BrainLat"):
        case, _ = split(c, dem_sets[c], ("AD",))
        cv = np.array([cog(c, s) for s in case]); X = mat(c, case); ok = np.isfinite(cv)
        out["cognition"][c] = {"score": "MoCA" if c == "BrainLat" else "MMSE", "n": int(ok.sum()),
                               "rho": {n: float(spearmanr(X[ok, j], cv[ok]).statistic) for j, n in enumerate(NAMES)}}
    pooled_cog = {}
    for n in NAMES:
        zs = [(math.atanh(out["cognition"][c]["rho"][n]), out["cognition"][c]["n"] - 3) for c in ("ds004504", "P-ADIC")]
        zbar = sum(z * w for z, w in zs) / sum(w for _, w in zs); se = 1 / math.sqrt(sum(w for _, w in zs))
        pooled_cog[n] = {"rho": math.tanh(zbar), "ci95": [math.tanh(zbar - 1.96 * se), math.tanh(zbar + 1.96 * se)]}
    out["cognition_pooled_mmse"] = pooled_cog

    # ---- mechanism: CAUEEG disease axis, external controls, SpectraLite scores, false referral
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    ci = dem_sets["CAUEEG"]; ids_c = [s for s in ci["ids"] if s in F["CAUEEG"]]
    yc = np.array([ci["y"][ci["ids"].index(s)] for s in ids_c]); Xc = mat("CAUEEG", ids_c)
    mu, sd = np.nanmean(Xc[yc == 0], 0), np.nanstd(Xc[yc == 0], 0, ddof=1)
    Zc = np.nan_to_num((Xc - mu) / sd)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(Zc, yc)
    oof_axis = np.zeros(len(yc))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=264).split(Zc, yc):
        oof_axis[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Zc[tr], yc[tr]).decision_function(Zc[te])
    out["axis"] = {"coef_standardized": dict(zip(NAMES, map(float, lr.coef_[0]))), "caueeg_oof_auc": auc(yc, oof_axis)}
    sc_c = np.array([ci["score"][ci["ids"].index(s)] for s in ids_c])
    tau = float(np.quantile(dem["internal"]["score"][dem["internal"]["y"] == 0], 0.80))
    ref_axis_ctrl = oof_axis[yc == 0]
    pooled = {"axis": [], "score": [], "cohort": [], "referred": []}
    out["mechanism_by_cohort"] = {}
    for c in ("ds004504", "P-ADIC", "BrainLat"):
        S = dem_sets[c]; ids = [s for s, yy in zip(S["ids"], S["y"]) if yy == 0 and s in F[c]]
        ax = lr.decision_function(np.nan_to_num((mat(c, ids) - mu) / sd))
        scr = np.array([S["score"][S["ids"].index(s)] for s in ids]); ref = scr >= tau
        cases = [s for s, yy in zip(S["ids"], S["y"]) if yy == 1]
        ax_case = lr.decision_function(np.nan_to_num((mat(c, [s for s in cases if s in F[c]]) - mu) / sd))
        r = spearmanr(ax, scr).statistic
        out["mechanism_by_cohort"][c] = {
            "n_controls": len(ids), "transferred_specificity": float(1 - ref.mean()),
            "mean_control_axis_shift": float(ax.mean() - ref_axis_ctrl.mean()),
            "control_axis_shift_g": float(hedges(ax, ref_axis_ctrl)[0]),
            "within_cohort_spearman_axis_score": float(r),
            "axis_auc_cases_vs_controls": auc(np.r_[np.ones(len(ax_case)), np.zeros(len(ax))], np.r_[ax_case, ax])}
        pooled["axis"] += list(ax); pooled["score"] += list(scr); pooled["cohort"] += [c] * len(ids)
        pooled["referred"] += list(ref)
    A_ = np.array(pooled["axis"]); S_ = np.array(pooled["score"]); R_ = np.array(pooled["referred"], int)
    C_ = np.array(pooled["cohort"])
    bs_r, bs_a = [], []
    strata = [np.flatnonzero(C_ == c) for c in ("ds004504", "P-ADIC", "BrainLat")]
    for _ in range(B):
        ii = np.concatenate([rng.choice(s, len(s)) for s in strata])
        bs_r.append(spearmanr(A_[ii], S_[ii]).statistic)
        if 0 < R_[ii].sum() < len(ii):
            bs_a.append(auc(R_[ii], A_[ii]))
    out["mechanism_pooled"] = {"n_controls": int(len(A_)), "spearman_axis_score": float(spearmanr(A_, S_).statistic),
                               "spearman_ci95": pct(bs_r), "n_falsely_referred": int(R_.sum()),
                               "auc_axis_referred_vs_not": auc(R_, A_), "auc_ci95": pct(bs_a),
                               "transferred_threshold": tau,
                               "points": {"axis": A_.tolist(), "score": S_.tolist(), "cohort": C_.tolist()}}
    order_shift = sorted(out["mechanism_by_cohort"], key=lambda c: -out["mechanism_by_cohort"][c]["mean_control_axis_shift"])
    order_spec = sorted(out["mechanism_by_cohort"], key=lambda c: out["mechanism_by_cohort"][c]["transferred_specificity"])
    out["mechanism_cohort_order"] = {"by_axis_shift_desc": order_shift, "by_specificity_asc": order_spec}

    # ---- marker-level threshold transport: theta/alpha cutoff from CAUEEG controls vs 20 local controls
    j = NAMES.index("theta/alpha"); cut = float(np.quantile(REF[:, j], 0.80))
    out["marker_threshold"] = {"marker": "theta/alpha", "caueeg_cutoff": cut}
    rk = min(math.ceil(0.80 * 21), 20)
    for c in ("ds004504", "P-ADIC", "BrainLat"):
        case, ctrl = split(c, dem_sets[c], ("AD",))
        xc, xa = mat(c, ctrl)[:, j], mat(c, case)[:, j]
        sp_l, se_l = [], []
        for _ in range(200):
            cal = rng.choice(len(xc), 20, replace=False); held = np.setdiff1d(np.arange(len(xc)), cal)
            t = np.sort(xc[cal])[rk - 1] + 1e-12
            sp_l.append(np.mean(xc[held] < t)); se_l.append(np.mean(xa >= t))
        out["marker_threshold"][c] = {"transferred_spec": float(np.mean(xc < cut)), "transferred_sens_AD": float(np.mean(xa >= cut)),
                                      "local_spec_mean": float(np.mean(sp_l)), "local_spec_p2_5_97_5": pct(sp_l),
                                      "local_sens_AD_mean": float(np.mean(se_l))}

    # ---- reliance against consistency (descriptive, 12 markers)
    ig = json.loads((CAN / "attribution12.json").read_text())["feature_importance_share"]
    v_ig = [ig.get(n, 0.0) for n in NAMES]; v_g = [abs(out["pooled_AD"][n]["g"]) for n in NAMES]
    out["reliance_vs_effect"] = {"ig_share": dict(zip(NAMES, v_ig)), "spearman": float(spearmanr(v_ig, v_g).statistic)}

    # ---- predictions, evaluated against the fixed rules
    P = {}
    pa = out["pooled_AD"]
    P["C1"] = {"held": bool(all(pa[n]["ci95"][0] > 0 for n in ("rel theta", "theta/alpha", "slowing")) and pa["IAF"]["ci95"][1] < 0),
               "detail": {n: [round(pa[n]["g"], 2)] + [round(v, 2) for v in pa[n]["ci95"]] for n in ("rel theta", "theta/alpha", "slowing", "IAF")}}
    n_shift = sum(any(abs(out["control_shift"][c][n]["g"]) >= 0.5 for c in ctrl_sets) for n in NAMES)
    P["C2"] = {"held": bool(n_shift >= 6), "detail": f"{n_shift} of 12 markers"}
    pm = out["pooled_MCI"]
    viol = [n for n in NAMES if abs(pm[n]["g"]) >= 0.3 and (pm[n]["ci95"][0] > 0 or pm[n]["ci95"][1] < 0)]
    P["C3"] = {"held": bool(not viol), "detail": viol}
    sh = {c: float(np.median([out["split_half"][c]["spearman_brown"][n] for n in consistent])) for c in out["split_half"]}
    P["N1"] = {"held": bool(all(v >= 0.80 for v in sh.values())), "detail": sh}
    cs = float(np.median([out["cross_session"]["icc31"][n] for n in consistent]))
    P["N2"] = {"held": bool(cs >= 0.60), "detail": cs}
    post = {c: (out["disease_AD"]["posterior"][c]["theta/alpha"]["g"], out["disease_AD"]["frontal"][c]["theta/alpha"]["g"]) for c in DEM_COH}
    P["N3"] = {"held": bool(sum(p > f for p, f in post.values()) >= 3), "detail": post}
    P["N4"] = {"held": bool(pooled_cog["theta/alpha"]["rho"] < 0 and pooled_cog["slowing"]["rho"] < 0 and pooled_cog["IAF"]["rho"] > 0),
               "detail": {n: pooled_cog[n] for n in ("theta/alpha", "slowing", "IAF")}}
    mp = out["mechanism_pooled"]
    P["N5"] = {"held": bool(mp["spearman_ci95"][0] > 0 and mp["auc_ci95"][0] > 0.5),
               "detail": {"rho": mp["spearman_axis_score"], "rho_ci": mp["spearman_ci95"], "auc": mp["auc_axis_referred_vs_not"], "auc_ci": mp["auc_ci95"]}}
    P["N6"] = {"held": bool(order_shift == order_spec), "detail": out["mechanism_cohort_order"]}
    mt = out["marker_threshold"]
    P["N7"] = {"held": bool(sum(mt[c]["transferred_spec"] < 0.80 for c in ("ds004504", "P-ADIC", "BrainLat")) >= 2
                            and all(abs(mt[c]["local_spec_mean"] - 0.80) <= 0.05 for c in ("ds004504", "P-ADIC", "BrainLat"))),
               "detail": {c: (round(mt[c]["transferred_spec"], 2), round(mt[c]["local_spec_mean"], 2)) for c in ("ds004504", "P-ADIC", "BrainLat")}}
    P["N8"] = {"held": bool(out["reliance_vs_effect"]["spearman"] > 0), "detail": out["reliance_vs_effect"]["spearman"]}
    out["predictions"] = P
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for k, v in P.items():
        print(f"{k}: {'HELD' if v['held'] else 'NOT HELD'} | {v['detail']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
