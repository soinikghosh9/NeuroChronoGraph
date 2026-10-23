"""Transporting the operating point with k local healthy volunteers (replaces the in-sample age claim).

Review C1. A screener trained at a memory clinic is deployed at a new site. At a threshold fixed to 80%
specificity on source out-of-fold controls it does not transport. The question a deploying clinic can
act on: if it records k of its own healthy volunteers, which use of those recordings restores the
intended operating point, and how large must k be? Every method is scored on the SAME 200 random
splits per k (_calib_splits): the k anchor controls may be used for calibration; specificity is
measured only on the remaining (held-out) controls and sensitivity on all cases.

Methods
  source_threshold   80th percentile of SOURCE OOF control scores (no local data)
  label_free_quota   refer the top fraction equal to the training-cohort prevalence (no local data)
  local_quantile     80th percentile of the k anchor controls' own scores (any model; no retraining)
  age_norm_anchor    age-normative deviation re-anchored on the k controls, source threshold
                     (concept logistic, as in exp 204; Spectral-75k, from exp 217)
  age_norm_only      age-normative deviation, no anchor (k-independent)
  age_norm_anchor+local_quantile   both: anchored deviations AND the local 80th-percentile threshold

    python experiments/221_local_calibration.py
Out: outputs/results/canonical/calibration/local_calibration.json (+ per-split npz)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from _calib_splits import K_VALUES, N_SPLITS, anchor_splits  # noqa: E402

CAN = ROOT / "outputs" / "results" / "canonical"
OUT = CAN / "calibration"
TASKS = ("CN_vs_dementia", "CN_vs_AD", "CN_vs_impaired")
TARGET_SPEC = 0.80


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def seed_ensemble(task, model):
    fs = sorted((CAN / task).glob(f"{model}__seed*.npz"))
    if not fs:
        return None
    zs = [np.load(f, allow_pickle=True) for f in fs]
    z0 = zs[0]
    def align(z, ks, kv):
        o = {s: i for i, s in enumerate(z[ks])}
        return np.array([z[kv][o[s]] for s in z0[ks]], float)
    out = {"train_subjects": z0["train_subjects"], "train_y": z0["train_y"],
           "oof": np.mean([align(z, "train_subjects", "oof_score") for z in zs], 0),
           "ext_subjects": z0["ext_subjects"], "ext_y": z0["ext_y"],
           "ext": np.mean([align(z, "ext_subjects", "ext_score") for z in zs], 0), "n_seeds": len(zs)}
    if all("padic_score" in z for z in zs):
        out.update(padic_subjects=z0["padic_subjects"], padic_y=z0["padic_y"],
                   padic=np.mean([align(z, "padic_subjects", "padic_score") for z in zs], 0))
    return out


def conformal_threshold(anchor_scores, target=TARGET_SPEC):
    """Split-conformal threshold: refer a subject only if it scores above the ceil((k+1)*target)-th
    smallest anchor-control score. Under exchangeability of healthy volunteers this guarantees an
    expected specificity >= target on new controls, for any k (Vovk; Angelopoulos & Bates 2023)."""
    a = np.sort(np.asarray(anchor_scores, float)); k = len(a)
    j = int(np.ceil((k + 1) * target))
    return float(np.nextafter(a[min(j, k) - 1], np.inf))


def rates(score, y, subjects, thr, heldout_ctrl):
    hc = np.isin(subjects, heldout_ctrl)
    spec = float(np.mean(score[hc] < thr)); sens = float(np.mean(score[y == 1] >= thr))
    return spec, sens


def net_benefit_at(sens, spec, prev):
    """Net benefit of referring at this operating point, per 1,000 people screened, against
    referring nobody, at a service prevalence `prev` and a threshold probability equal to it
    (Vickers & Elkin 2006). Positive means the rule beats referring nobody; we also give the
    comparison against referring everyone."""
    pt = prev; w = pt / (1 - pt)
    nb = sens * prev - (1 - spec) * (1 - prev) * w
    nb_all = prev - (1 - prev) * w
    return {"per_1000": round(1000 * nb, 1), "vs_refer_all_per_1000": round(1000 * (nb - nb_all), 1)}


def summarise(vals):
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}


def operating_points(E, subs, y, s):
    """Specificity/sensitivity of the four label-free threshold rules at one site, over the shared
    anchor splits. `E` supplies the source out-of-fold controls (for the transferred threshold) and
    the training prevalence (for the quota); `subs/y/s` are that site's subjects, labels and scores."""
    ctrl = subs[y == 0]
    thr_src = float(np.quantile(E["oof"][E["train_y"] == 0], TARGET_SPEC))
    prev = float(np.mean(E["train_y"]))
    thr_quota = float(np.quantile(s, 1 - prev))
    res = {"n_seeds": E["n_seeds"], "n_subjects": int(len(subs)), "n_controls": int(len(ctrl)),
           "all_controls": {
               "source_threshold": dict(zip(("spec", "sens"), rates(s, y, subs, thr_src, ctrl))),
               "label_free_quota": dict(zip(("spec", "sens"), rates(s, y, subs, thr_quota, ctrl)))}, "by_k": {}}
    for k in K_VALUES:
        if k >= len(ctrl):
            continue
        acc = {"source_threshold": [], "label_free_quota": [], "local_quantile": [], "local_conformal": []}
        for anc, held in anchor_splits(ctrl, k):
            thr_loc = float(np.quantile(s[np.isin(subs, anc)], TARGET_SPEC))
            thr_conf = conformal_threshold(s[np.isin(subs, anc)])
            for meth, thr in (("source_threshold", thr_src), ("label_free_quota", thr_quota),
                              ("local_quantile", thr_loc), ("local_conformal", thr_conf)):
                acc[meth].append(rates(s, y, subs, thr, held))
        res["by_k"][str(k)] = {
            meth: {"spec": summarise([a for a, _ in v]), "sens": summarise([b for _, b in v]),
                   "net_benefit": {f"{p:.2f}": net_benefit_at(float(np.mean([b for _, b in v])),
                                                              float(np.mean([a for a, _ in v])), p)
                                   for p in (0.05, 0.10, 0.20)}}
            for meth, v in acc.items()}
    return res


def concept_agedev(task, A204, tb):
    """Exp 204's estimator (L2 logistic on concept features) with raw / age-dev / anchored arms,
    returning source OOF scores and a function that scores the external site for a given anchor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    T = tb.subject_table("channel_clean")
    subs, coh, cls_, X = np.asarray(T["subjects"]), T["cohort"], T["cls"], T["X_concept"]
    amap = A204.load_ages()
    ages = np.array([amap.get(A204.keyf(s, c), np.nan) for s, c in zip(subs, coh)], float)
    pos, neg = {"CN_vs_dementia": (("AD", "FTD"), ("CN",)), "CN_vs_AD": (("AD",), ("CN",)),
                "CN_vs_impaired": (("AD", "FTD", "MCI"), ("CN",))}[task]
    cau = np.isfinite(ages) & (coh == "CAUEEG") & np.isin(cls_, pos + neg)
    ext = np.isfinite(ages) & (coh == "ds004504_raw") & np.isin(cls_, pos + neg)
    y = np.isin(cls_, pos).astype(int)
    Xc, yc, ac, Xe, ye, ae = X[cau], y[cau], ages[cau], X[ext], y[ext], ages[ext]
    m = A204.norm_fit(Xc[yc == 0], ac[yc == 0])

    def fit(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        f = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return lambda Z: f.predict_proba(sc.transform(Z))[:, 1]

    out = {}
    for arm in ("raw", "dev"):
        Xtr = Xc if arm == "raw" else A204.norm_dev(m, Xc, ac)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, yc))
        fits = [fit(Xtr[a], yc[a]) for a, _ in folds]
        oof = np.full(len(yc), np.nan)
        for (a, b), f in zip(folds, fits):
            oof[b] = f(Xtr[b])
        out[arm] = {"fits": fits, "oof": oof}
    e_subs = subs[ext]
    s_age = dict(zip(e_subs, ae)); s_X = dict(zip(e_subs, Xe))

    def score(arm, anchor_subjects=None):
        if arm == "raw":
            Z = Xe
        else:
            anc = None
            if anchor_subjects is not None:
                anc = A204.fit_anchor(m, np.stack([s_X[s] for s in anchor_subjects]),
                                      np.array([s_age[s] for s in anchor_subjects]))
            Z = A204.norm_dev(m, Xe, ae, anchor=anc)
        return np.mean([f(Z) for f in out[arm]["fits"]], 0)
    return {"train_y": yc, "oof": out, "ext_subjects": e_subs, "ext_y": ye, "score": score}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A204 = _load("a204", "204_age_conditioned.py")
    tb = _load("tb", "157_task_battery.py")
    report = {"target_specificity": TARGET_SPEC, "k_values": list(K_VALUES), "n_splits": N_SPLITS, "tasks": {}}
    for task in TASKS:
        tr = {}
        # ---------- score-level methods for every canonical model with scores ----------
        for model in ("rf123", "rf_node", "concept", "spectralite16", "spectral75k", "ncg_v5", "atcnet", "eegnet"):
            E = seed_ensemble(task, model)
            if E is None:
                continue
            mres = operating_points(E, E["ext_subjects"], E["ext_y"], E["ext"])
            if "padic" in E:
                # the same four rules at the second external site, whose case/control mix, amplifier
                # and age structure differ again -- the guarantee is over clinics, so it is only
                # testable with more than one
                mres["padic"] = operating_points(E, E["padic_subjects"], E["padic_y"], E["padic"])
            tr[model] = mres
            k20 = mres["by_k"]["20"]
            extra = ""
            if "padic" in mres and "20" in mres["padic"]["by_k"]:
                p20 = mres["padic"]["by_k"]["20"]
                extra = (f" || P-ADIC source spec {p20['source_threshold']['spec']['mean']:.2f} "
                         f"conformal spec {p20['local_conformal']['spec']['mean']:.2f} "
                         f"sens {p20['local_conformal']['sens']['mean']:.2f}")
            print(f"{task:15s} {model:12s} source spec {k20['source_threshold']['spec']['mean']:.2f} "
                  f"sens {k20['source_threshold']['sens']['mean']:.2f} | local20 spec "
                  f"{k20['local_quantile']['spec']['mean']:.2f} sens {k20['local_quantile']['sens']['mean']:.2f}"
                  + extra, flush=True)

        # ---------- concept logistic with age-normative arms (exp 204, cross-fitted anchors) ----------
        CA = concept_agedev(task, A204, tb)
        y, subs = CA["ext_y"], CA["ext_subjects"]; ctrl = subs[y == 0]
        cres = {"by_k": {}}
        s_raw = CA["score"]("raw"); s_dev = CA["score"]("dev")
        thr_raw = float(np.quantile(CA["oof"]["raw"]["oof"][CA["train_y"] == 0], TARGET_SPEC))
        thr_dev = float(np.quantile(CA["oof"]["dev"]["oof"][CA["train_y"] == 0], TARGET_SPEC))
        from sklearn.metrics import roc_auc_score
        cres["auc"] = {"raw": float(roc_auc_score(y, s_raw)), "dev": float(roc_auc_score(y, s_dev))}
        for k in K_VALUES:
            acc = {"source_threshold": [], "local_quantile": [], "local_conformal": [], "age_norm_only": [],
                   "age_norm_anchor": [], "age_norm_anchor+local_quantile": []}
            aucs_anchor = []
            for anc, held in anchor_splits(ctrl, k):
                s_anc = CA["score"]("dev", anc); aucs_anchor.append(roc_auc_score(y, s_anc))
                acc["source_threshold"].append(rates(s_raw, y, subs, thr_raw, held))
                acc["local_quantile"].append(rates(s_raw, y, subs, float(np.quantile(s_raw[np.isin(subs, anc)], TARGET_SPEC)), held))
                acc["local_conformal"].append(rates(s_raw, y, subs, conformal_threshold(s_raw[np.isin(subs, anc)]), held))
                acc["age_norm_only"].append(rates(s_dev, y, subs, thr_dev, held))
                acc["age_norm_anchor"].append(rates(s_anc, y, subs, thr_dev, held))
                acc["age_norm_anchor+local_quantile"].append(
                    rates(s_anc, y, subs, float(np.quantile(s_anc[np.isin(subs, anc)], TARGET_SPEC)), held))
            cres["by_k"][str(k)] = {meth: {"spec": summarise([a for a, _ in v]), "sens": summarise([b for _, b in v])}
                                    for meth, v in acc.items()}
            cres["by_k"][str(k)]["age_norm_anchor_auc"] = summarise(aucs_anchor)
        tr["concept_logistic_204"] = cres

        report["tasks"][task] = tr
        (OUT / "local_calibration.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "local_calibration.json")


if __name__ == "__main__":
    main()
